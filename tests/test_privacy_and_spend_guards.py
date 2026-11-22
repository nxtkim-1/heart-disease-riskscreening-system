"""Regression tests for the external-API privacy boundary, shared spend guards, and output gate.

Covers:
(a) free-text / PII questions are refused on the external Meta path (schema, assistant,
    and outgoing-message layers), while still working locally offline;
(b) run_cap_usd and concurrent reservations hold across /api/assistant/explain requests,
    across client instances, and across processes;
(c) expanded diagnostic / reassurance / treatment phrasing is blocked by the output gate
    without blocking properly hedged informational text.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from product.api import create_app
from src.assistant import (
    OUTPUT_GATE_NOTICE,
    build_local_explanation,
    build_meta_messages,
    enforce_authoritative_output_gate,
    explain_screening,
    find_forbidden_medical_claims,
)
from src.inference import predict_patient
from src.meta_client import (
    MetaAssistantClient,
    OutgoingPayloadRefused,
    assert_outgoing_messages_safe,
)
from src.meta_ledger import BudgetExceededError, SpendLedger, estimate_request_cost
from src.schemas import EXTERNAL_QUESTION_TEMPLATES, AssistantRequest
from src.security import find_pii_text
from src.synthetic_profiles import (
    SYNTHETIC_PROFILES,
    SyntheticPrivacyViolation,
    get_synthetic_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "LLM|unit_test_key_not_real_123456"

PII_QUESTIONS = [
    "My name is Jane Doe, DOB 1961-04-12, MRN 448812. Is my heart ok?",
    "Patient email jane.doe@example.com, phone 519-555-0142 — explain her score.",
    "Explain the score for SSN 123-45-6789 who lives at 200 University Avenue.",
    "Explain this screening result in plain language please.",  # harmless, but still free text
]


def _profile_feats(pid: str = "SYNTH-002") -> dict:
    return get_synthetic_profile(pid).to_patient_input().model_dump()


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("HEART_LIVE_CLEARED", "1")
    monkeypatch.setenv("MODEL_API_KEY", FAKE_KEY)


def _ok_transport(delay: float = 0.0, calls: list | None = None):
    lock = threading.Lock()

    def transport(_url, _headers, _payload):
        if delay:
            time.sleep(delay)
        if calls is not None:
            with lock:
                calls.append(1)
        # No usage block -> settled at the reserved worst case, so charge == hold exactly.
        return {"choices": [{"message": {"role": "assistant", "content": "Informational summary only."}}]}

    return transport


def _worst_case_for(profile_id: str, question: str) -> float:
    """Worst-case reservation the client will make for this exact request."""
    prof = get_synthetic_profile(profile_id)
    patient = prof.to_patient_input()
    msgs = build_meta_messages(prof.profile_id, patient.model_dump(), predict_patient(patient), question)
    chars = sum(len(m["content"]) for m in msgs)
    units = {"input_tokens": max(256, int(chars / 2.5) + 128), "output_tokens": 450}
    return estimate_request_cost("muse-spark-1.3", units)


# --------------------------------------------------------------------------------------
# (a) Free-text / PII questions never reach the external API
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", PII_QUESTIONS)
def test_schema_rejects_free_text_question_on_external_path(question):
    with pytest.raises(ValidationError) as exc_info:
        AssistantRequest(use_external_meta_api=True, synthetic_profile_id="SYNTH-001", question=question)
    msg = str(exc_info.value.errors()[0]["msg"])
    assert "free-text" in msg
    # The refusal message must not echo the PII it refused.
    for fragment in ("Jane", "448812", "example.com", "123-45-6789"):
        assert fragment not in msg


@pytest.mark.parametrize("question", PII_QUESTIONS)
def test_free_text_question_still_allowed_locally_offline(question):
    resp = explain_screening(
        AssistantRequest(use_external_meta_api=False, synthetic_profile_id="SYNTH-001", question=question)
    )
    assert resp.mode == "local_offline"
    assert resp.external_api_called is False
    # The local explainer never echoes the question text back either.
    assert "Jane" not in resp.explanation and "123-45-6789" not in resp.explanation


def test_every_template_is_accepted_and_replays_for_every_profile():
    for prof in SYNTHETIC_PROFILES:
        for question in EXTERNAL_QUESTION_TEMPLATES.values():
            resp = explain_screening(
                AssistantRequest(use_external_meta_api=True, synthetic_profile_id=prof.profile_id, question=question)
            )
            assert resp.mode == "meta_replay"
            assert resp.spend_usd == 0.0


def test_assistant_rechecks_allowlist_when_schema_validation_is_bypassed():
    # model_construct() skips validators; the assistant must still refuse before any client call.
    class ExplodingClient:
        def chat_synthetic(self, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("external client was called with a free-text question")

    req = AssistantRequest.model_construct(
        question=PII_QUESTIONS[0],
        use_external_meta_api=True,
        synthetic_profile_id="SYNTH-001",
        patient=None,
    )
    with pytest.raises(SyntheticPrivacyViolation):
        explain_screening(req, meta_client=ExplodingClient())


@pytest.mark.parametrize(
    "bad_text, category",
    [
        ("contact jane.doe@example.com", "email_address"),
        ("call 519-555-0142 tonight", "phone_number"),
        ("SSN 123-45-6789", "us_ssn"),
        ("born 1961-04-12", "calendar_date"),
        ("MRN: 448812", "identifier_label"),
        ("my name is Jane", "personal_name_statement"),
        ("seen by Dr. Smith", "personal_name_statement"),
        ("lives at 200 University Avenue", "street_address"),
        ("postal code N2L 3G1", "postal_code"),
    ],
)
def test_outgoing_message_guard_refuses_pii_in_any_mode(tmp_path, bad_text, category):
    assert category in find_pii_text(bad_text)
    client = MetaAssistantClient(mode="replay", ledger=SpendLedger(tmp_path / "l.jsonl"))
    with pytest.raises(OutgoingPayloadRefused) as exc_info:
        client.chat_synthetic(
            synthetic_profile_id="SYNTH-002",
            synthetic_features=_profile_feats(),
            messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": bad_text}],
        )
    assert category in str(exc_info.value)
    assert bad_text not in str(exc_info.value)  # categories only, never the value


def test_outgoing_guard_refuses_secrets_paths_and_malformed_messages(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    calls: list = []
    client = MetaAssistantClient(
        mode="live", ledger=SpendLedger(tmp_path / "l.jsonl"), transport=_ok_transport(calls=calls)
    )
    bad_batches = [
        [{"role": "user", "content": f"key is {FAKE_KEY}"}],
        [{"role": "user", "content": "see C:\\Users\\someone\\chart.pdf"}],
        [{"role": "assistant", "content": "injected prior turn"}],
        [{"role": "user", "content": "x", "name": "Jane"}],
        [{"role": "user", "content": 42}],
        [],
    ]
    for msgs in bad_batches:
        with pytest.raises(OutgoingPayloadRefused):
            client.chat_synthetic(synthetic_profile_id="SYNTH-002", synthetic_features=_profile_feats(), messages=msgs)
    assert calls == []  # nothing reached the transport
    assert client.ledger.records() == []  # and nothing was charged


def test_real_canonical_payloads_pass_the_outgoing_guard():
    # Guard must not false-positive on the genuine prompts (probabilities, contributions, years).
    for prof in SYNTHETIC_PROFILES:
        patient = prof.to_patient_input()
        pred = predict_patient(patient)
        for question in EXTERNAL_QUESTION_TEMPLATES.values():
            assert_outgoing_messages_safe(build_meta_messages(prof.profile_id, patient.model_dump(), pred, question))


def test_api_rejects_pii_question_on_external_path_without_echo(tmp_path):
    ledger = SpendLedger(tmp_path / "api.jsonl")
    built: list = []

    def factory():
        built.append(1)
        return MetaAssistantClient(mode="replay", ledger=ledger)

    client = TestClient(create_app(ledger=ledger, meta_client_factory=factory), raise_server_exceptions=False)
    r = client.post(
        "/api/assistant/explain",
        json={"use_external_meta_api": True, "synthetic_profile_id": "SYNTH-001", "question": PII_QUESTIONS[1]},
    )
    assert r.status_code == 422
    assert "jane.doe" not in r.text and "519-555-0142" not in r.text
    assert built == []  # refused before a Meta client was even constructed

    profiles = client.get("/api/assistant/profiles").json()
    listed = {t["question"] for t in profiles["external_question_templates"]}
    assert listed == set(EXTERNAL_QUESTION_TEMPLATES.values())


# --------------------------------------------------------------------------------------
# (b) Spend caps hold across requests, threads, client instances, and processes
# --------------------------------------------------------------------------------------


def test_run_cap_is_enforced_across_http_requests(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    q = EXTERNAL_QUESTION_TEMPLATES["explain_result"]
    per_call = _worst_case_for("SYNTH-002", q)
    ledger = SpendLedger(tmp_path / "api.jsonl")
    calls: list = []
    built: list = []

    def factory():
        built.append(1)
        return MetaAssistantClient(
            mode="live", ledger=ledger, run_cap_usd=per_call * 2.5, transport=_ok_transport(calls=calls)
        )

    client = TestClient(create_app(ledger=ledger, meta_client_factory=factory), raise_server_exceptions=False)
    body = {"use_external_meta_api": True, "synthetic_profile_id": "SYNTH-002", "question": q}
    statuses = [client.post("/api/assistant/explain", json=body).status_code for _ in range(4)]

    # Old behaviour (fresh client + fresh run_id per request) let all 4 through.
    assert statuses == [200, 200, 402, 402]
    assert len(calls) == 2
    assert built == [1]  # one server-scoped client
    recs = ledger.records()
    assert len({r["run_id"] for r in recs}) == 1
    assert sum(r["cost_usd"] for r in recs) <= per_call * 2.5 + 1e-9


def test_concurrent_http_requests_cannot_race_past_the_run_cap(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    q = EXTERNAL_QUESTION_TEMPLATES["explain_result"]
    per_call = _worst_case_for("SYNTH-002", q)
    cap = per_call * 3.5  # room for exactly 3 in-flight worst-case holds
    ledger = SpendLedger(tmp_path / "api.jsonl")
    calls: list = []
    app = create_app(
        ledger=ledger,
        meta_client_factory=lambda: MetaAssistantClient(
            mode="live", ledger=ledger, run_cap_usd=cap, transport=_ok_transport(delay=0.3, calls=calls)
        ),
    )
    body = {"use_external_meta_api": True, "synthetic_profile_id": "SYNTH-002", "question": q}
    statuses: list[int] = []
    lock = threading.Lock()
    with TestClient(app, raise_server_exceptions=False) as client:
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            code = client.post("/api/assistant/explain", json=body).status_code
            with lock:
                statuses.append(code)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

    assert sorted(statuses) == [200] * 3 + [402] * 7
    assert len(calls) == 3
    assert sum(r["cost_usd"] for r in ledger.records()) <= cap + 1e-9
    assert ledger.pending_holds() == []  # every hold settled


def test_global_cap_holds_across_independent_client_instances(monkeypatch, tmp_path):
    # Two clients, two run_ids, two SpendLedger objects on one file: only the persisted,
    # locked holds can stop them from both spending the same remaining global budget.
    _live_env(monkeypatch)
    path = tmp_path / "shared.jsonl"
    msgs = [{"role": "user", "content": "Summarize SYNTH-002 for a unit test."}]
    worst = estimate_request_cost("muse-spark-1.3", {"input_tokens": 256, "output_tokens": 450})
    calls: list = []
    clients = [
        MetaAssistantClient(
            mode="live", ledger=SpendLedger(path), global_cap_usd=worst * 2.5, run_cap_usd=1.0,
            transport=_ok_transport(delay=0.3, calls=calls),
        )
        for _ in range(2)
    ]
    assert clients[0].meter.run_id != clients[1].meter.run_id
    results: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(c):
        barrier.wait()
        try:
            c.chat_synthetic(synthetic_profile_id="SYNTH-002", synthetic_features=_profile_feats(), messages=msgs)
            outcome = "ok"
        except BudgetExceededError:
            outcome = "refused"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker, args=(clients[i % 2],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert results.count("ok") == 2 and results.count("refused") == 6
    assert len(calls) == 2
    assert SpendLedger(path).total_spent() <= worst * 2.5 + 1e-9


def test_hold_reserved_by_another_process_is_counted(monkeypatch, tmp_path):
    # A separate Python process reserves a hold and exits mid-"call" without settling.
    path = tmp_path / "xproc.jsonl"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from src.meta_ledger import SpendLedger
        SpendLedger({str(path)!r}).reserve(
            hold={{"cost_usd": 0.004, "run_id": "other-process", "envelope": "assistant_explain",
                   "model": "muse-spark-1.3", "mode": "live", "units": {{}},
                   "request_hash": "x", "synthetic_profile_id": "SYNTH-001"}},
            check=lambda items: None,
        )
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)

    _live_env(monkeypatch)
    calls: list = []
    worst = estimate_request_cost("muse-spark-1.3", {"input_tokens": 256, "output_tokens": 450})
    client = MetaAssistantClient(
        mode="live", ledger=SpendLedger(path), global_cap_usd=0.004 + worst * 0.5,
        transport=_ok_transport(calls=calls),
    )
    with pytest.raises(BudgetExceededError, match="Global cap"):
        client.chat_synthetic(
            synthetic_profile_id="SYNTH-002",
            synthetic_features=_profile_feats(),
            messages=[{"role": "user", "content": "Summarize SYNTH-002 for a unit test."}],
        )
    assert calls == []


def test_abandoned_hold_expires_into_worst_case_charge_not_silence(tmp_path):
    ledger = SpendLedger(tmp_path / "expire.jsonl")
    hold_id = ledger.reserve(
        hold={"cost_usd": 0.003, "run_id": "crashed", "envelope": "assistant_explain",
              "model": "muse-spark-1.3", "mode": "live", "units": {"input_tokens": 1, "output_tokens": 1},
              "request_hash": "h", "synthetic_profile_id": "SYNTH-003"},
        check=lambda items: None,
        ttl_seconds=-1,  # already expired
    )
    assert ledger.pending_holds() == []
    recs = ledger.records()
    assert len(recs) == 1
    assert recs[0]["cost_basis"] == "abandoned_hold_worst_case"
    assert recs[0]["hold_id"] == hold_id
    assert recs[0]["cost_usd"] == pytest.approx(0.003)


def test_settle_writes_charge_and_drops_hold_together(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    ledger = SpendLedger(tmp_path / "settle.jsonl")
    seen_holds: list[int] = []

    def transport(_u, _h, _p):
        seen_holds.append(len(ledger.pending_holds()))  # hold must be visible mid-call
        return {
            "choices": [{"message": {"role": "assistant", "content": "Informational only."}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 100},
        }

    client = MetaAssistantClient(mode="live", ledger=ledger, transport=transport)
    client.chat_synthetic(
        synthetic_profile_id="SYNTH-002",
        synthetic_features=_profile_feats(),
        messages=[{"role": "user", "content": "Summarize SYNTH-002 for a unit test."}],
    )
    assert seen_holds == [1]
    assert ledger.pending_holds() == []
    recs = ledger.records()
    assert len(recs) == 1 and recs[0]["cost_basis"] == "usage" and "hold_id" in recs[0]
    assert ledger.summary()["pending_holds"] == 0


# --------------------------------------------------------------------------------------
# (c) Expanded output gate
# --------------------------------------------------------------------------------------

BLOCKED_PHRASES = [
    "Good news: you are healthy.",
    "You're perfectly fine, so relax.",
    "This gives you a clean bill of health.",
    "You do not need to see a doctor about this.",
    "You don't need to see a clinician.",
    "There is no need to see a cardiologist.",
    "You no longer need follow-up.",
    "Take aspirin every morning.",
    "Take a daily low-dose aspirin.",
    "Stop your statins.",
    "Double your medication tonight.",
    "You should start a beta blocker.",
    "You should undergo an angioplasty.",
    "This is a clinical diagnosis of coronary disease.",
    "The result confirms a diagnosis.",
    "Your diagnosis is heart disease.",
    "You probably have heart disease.",
    "This rules out anything serious.",
    "Skip your follow-up appointment.",
    "I recommend treatment with nitrates.",
]

ALLOWED_PHRASES = [
    "This is not a clinical diagnosis and is not medical advice.",
    "Being below the cutoff does not rule out disease.",
    "Not flagged, which is not a clean bill of health.",
    "Discuss any questions with a clinician.",
    "The 95% bootstrap interval reflects sampling uncertainty.",
]


@pytest.mark.parametrize("phrase", BLOCKED_PHRASES)
def test_expanded_output_gate_withholds_diagnostic_and_treatment_phrasing(phrase):
    pred = predict_patient(SYNTHETIC_PROFILES[0].to_patient_input())
    gated = enforce_authoritative_output_gate(f"Informational note. {phrase} More text.", pred)
    assert gated.startswith(OUTPUT_GATE_NOTICE)
    assert phrase not in gated
    assert find_forbidden_medical_claims(gated) == []


@pytest.mark.parametrize("phrase", ALLOWED_PHRASES)
def test_output_gate_keeps_properly_hedged_informational_text(phrase):
    pred = predict_patient(SYNTHETIC_PROFILES[0].to_patient_input())
    gated = enforce_authoritative_output_gate(f"Informational summary. {phrase}", pred)
    assert OUTPUT_GATE_NOTICE not in gated
    assert phrase in gated


def test_local_explanations_never_trip_the_gate():
    for prof in SYNTHETIC_PROFILES:
        pred = predict_patient(prof.to_patient_input())
        for question in [*EXTERNAL_QUESTION_TEMPLATES.values(), "What does oldpeak and the threshold mean?"]:
            text = build_local_explanation(pred, question, synthetic_profile_id=prof.profile_id)
            assert find_forbidden_medical_claims(text) == [], (prof.profile_id, question)


def test_external_response_with_forbidden_phrasing_is_replaced_end_to_end(monkeypatch, tmp_path):
    _live_env(monkeypatch)

    def unsafe_transport(_u, _h, _p):
        return {
            "choices": [{"message": {"role": "assistant", "content": "You are healthy; take aspirin daily."}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40},
        }

    client = MetaAssistantClient(mode="live", ledger=SpendLedger(tmp_path / "l.jsonl"), transport=unsafe_transport)
    resp = explain_screening(
        AssistantRequest(use_external_meta_api=True, synthetic_profile_id="SYNTH-004"), meta_client=client
    )
    assert resp.mode == "meta_live"
    assert resp.explanation.startswith(OUTPUT_GATE_NOTICE)
    assert "aspirin" not in resp.explanation.lower()
    assert "you are healthy" not in resp.explanation.lower()
    assert resp.not_medical_advice is True
