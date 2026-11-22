"""Assistant privacy boundary, authoritative output gate, and Meta spend-ledger tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.assistant import enforce_authoritative_output_gate, explain_screening
from src.inference import predict_patient
from src.meta_client import (
    LiveNotClearedError,
    MetaAssistantClient,
    ReplayFixtureMissingError,
    SecretLeakError,
)
from src.meta_ledger import (
    BudgetExceededError,
    ContributorModelError,
    LedgerCorruptError,
    SpendLedger,
    UnknownCostError,
    estimate_request_cost,
)
from src.schemas import AssistantRequest
from src.synthetic_profiles import (
    SYNTHETIC_PROFILES,
    SyntheticPrivacyViolation,
    assert_registered_synthetic,
    get_synthetic_profile,
)


def test_synthetic_privacy_gate_rejects_unknown_or_mutated_profiles():
    p = get_synthetic_profile("SYNTH-001")
    valid_feats = p.to_patient_input().model_dump()
    assert assert_registered_synthetic("SYNTH-001", valid_feats).profile_id == "SYNTH-001"

    # Unregistered ID refused.
    with pytest.raises(SyntheticPrivacyViolation):
        assert_registered_synthetic("REAL-PATIENT-99", valid_feats)

    # Even a 1-year age mutation (attempting to smuggle real patient data under a synthetic ID) is refused.
    tampered = dict(valid_feats, age=valid_feats["age"] + 1)
    with pytest.raises(SyntheticPrivacyViolation, match="digest"):
        assert_registered_synthetic("SYNTH-001", tampered)


def test_live_and_record_modes_refused_without_clearance_and_key(monkeypatch, tmp_path):
    monkeypatch.delenv("HEART_LIVE_CLEARED", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    with pytest.raises(LiveNotClearedError, match="HEART_LIVE_CLEARED=1"):
        MetaAssistantClient(mode="live", ledger=SpendLedger(tmp_path / "ledger.jsonl"))

    monkeypatch.setenv("HEART_LIVE_CLEARED", "1")
    with pytest.raises(LiveNotClearedError, match="MODEL_API_KEY"):
        MetaAssistantClient(mode="record", ledger=SpendLedger(tmp_path / "ledger.jsonl"))


def test_contributor_model_and_unknown_cost_fail_closed():
    with pytest.raises(ContributorModelError):
        estimate_request_cost("muse-spark-1.3-contributor", {"input_tokens": 100, "output_tokens": 50})

    with pytest.raises(UnknownCostError):
        estimate_request_cost("unknown-model-x", {"input_tokens": 100, "output_tokens": 50})

    with pytest.raises(UnknownCostError):
        estimate_request_cost("muse-spark-1.3", {"input_tokens": -5, "output_tokens": 10})


def test_metered_live_call_logs_hash_chained_ledger_and_enforces_caps(monkeypatch, tmp_path):
    monkeypatch.setenv("HEART_LIVE_CLEARED", "1")
    monkeypatch.setenv("MODEL_API_KEY", "LLM|dummy_unit_test_secret_999")

    ledger_file = tmp_path / "spend.jsonl"
    fixtures_dir = tmp_path / "fixtures"
    ledger = SpendLedger(ledger_file)

    captured_headers = {}

    def fake_transport(url, headers, payload):
        captured_headers.update(headers)
        return {
            "id": "mock-resp-1",
            "choices": [{"message": {"role": "assistant", "content": "Informational summary (not medical advice)."}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 120},
        }

    client = MetaAssistantClient(
        mode="record",
        ledger=ledger,
        fixtures_dir=fixtures_dir,
        run_cap_usd=0.01,
        transport=fake_transport,
    )
    prof = get_synthetic_profile("SYNTH-002")
    feats = prof.to_patient_input().model_dump()
    res = client.chat_synthetic(
        synthetic_profile_id="SYNTH-002",
        synthetic_features=feats,
        messages=[{"role": "user", "content": "Summarize SYNTH-002"}],
        envelope="assistant_explain",
    )

    assert captured_headers["Authorization"] == "Bearer LLM|dummy_unit_test_secret_999"
    assert res["cost_usd"] == pytest.approx((400 * 1.25 + 120 * 4.25) / 1e6, rel=1e-5)
    recs = ledger.records()
    assert len(recs) == 1
    assert recs[0]["synthetic_profile_id"] == "SYNTH-002"
    assert recs[0]["cost_basis"] == "usage"

    # Key must NEVER appear in ledger file or recorded fixture!
    assert "LLM|dummy_unit_test_secret_999" not in ledger_file.read_text(encoding="utf-8")
    fixture_files = list(fixtures_dir.glob("*.json"))
    assert len(fixture_files) == 1
    assert "LLM|dummy_unit_test_secret_999" not in fixture_files[0].read_text(encoding="utf-8")

    # Replay mode serves that exact fixture with $0 spend and no network.
    replay_client = MetaAssistantClient(mode="replay", ledger=ledger, fixtures_dir=fixtures_dir)
    rep = replay_client.chat_synthetic(
        synthetic_profile_id="SYNTH-002",
        synthetic_features=feats,
        messages=[{"role": "user", "content": "Summarize SYNTH-002"}],
        envelope="assistant_explain",
    )
    assert rep["mode"] == "replay"
    assert rep["cost_usd"] == 0.0
    assert len(ledger.records()) == 1  # No new spend line in replay mode.

    # Tight run cap must refuse before transport is ever called.
    called_after_cap = False

    def fail_if_called(*_args):
        nonlocal called_after_cap
        called_after_cap = True
        return {}

    tight = MetaAssistantClient(
        mode="live",
        ledger=ledger,
        run_cap_usd=0.0001,
        transport=fail_if_called,
    )
    with pytest.raises(BudgetExceededError):
        tight.chat_synthetic(
            synthetic_profile_id="SYNTH-002",
            synthetic_features=feats,
            messages=[{"role": "user", "content": "Summarize SYNTH-002"}],
            envelope="assistant_explain",
        )
    assert called_after_cap is False


def test_failed_http_call_still_logs_worst_case_and_redacts_error(monkeypatch, tmp_path):
    secret = "LLM|secret_in_transport_err_777"
    monkeypatch.setenv("HEART_LIVE_CLEARED", "1")
    monkeypatch.setenv("MODEL_API_KEY", secret)
    ledger = SpendLedger(tmp_path / "spend.jsonl")

    def boom(_url, _headers, _payload):
        raise ConnectionError(f"Dial failed with {secret} at C:\\Users\\private\\net.py")

    client = MetaAssistantClient(mode="live", ledger=ledger, transport=boom)
    prof = get_synthetic_profile("SYNTH-001")
    with pytest.raises(RuntimeError) as exc_info:
        client.chat_synthetic(
            synthetic_profile_id="SYNTH-001",
            synthetic_features=prof.to_patient_input().model_dump(),
            messages=[{"role": "user", "content": "Test failure logging"}],
            envelope="assistant_explain",
        )

    assert secret not in str(exc_info.value)
    assert "private" not in str(exc_info.value)
    recs = ledger.records()
    assert len(recs) == 1
    assert recs[0]["cost_basis"] == "failed_request_worst_case"
    assert recs[0]["cost_usd"] > 0


def test_corrupt_or_tampered_ledger_fails_closed(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    ledger = SpendLedger(path)
    ledger.append(
        model="muse-spark-1.3",
        envelope="assistant_explain",
        units={"input_tokens": 100, "output_tokens": 50},
        cost_usd=0.000338,
        run_id="r1",
        cost_basis="usage",
        mode="live",
        request_hash="abc",
        synthetic_profile_id="SYNTH-001",
        latency_ms=100,
    )
    # Tamper with cost_usd -> hash verification must fail closed.
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    raw["cost_usd"] = 0.0
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(LedgerCorruptError, match="SHA-256"):
        ledger.records()


def test_record_mode_refuses_saving_fixture_if_response_echoes_key(monkeypatch, tmp_path):
    secret = "LLM|echoed_secret_in_body_888"
    monkeypatch.setenv("HEART_LIVE_CLEARED", "1")
    monkeypatch.setenv("MODEL_API_KEY", secret)

    def leaky_transport(_u, _h, _p):
        return {
            "choices": [{"message": {"role": "assistant", "content": f"Oops {secret}"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }

    client = MetaAssistantClient(
        mode="record",
        ledger=SpendLedger(tmp_path / "l.jsonl"),
        fixtures_dir=tmp_path / "fx",
        transport=leaky_transport,
    )
    prof = get_synthetic_profile("SYNTH-001")
    with pytest.raises(SecretLeakError):
        client.chat_synthetic(
            synthetic_profile_id="SYNTH-001",
            synthetic_features=prof.to_patient_input().model_dump(),
            messages=[{"role": "user", "content": "Check key leak guard"}],
        )


def test_replay_missing_fixture_fails_loudly(tmp_path):
    client = MetaAssistantClient(mode="replay", fixtures_dir=tmp_path / "empty")
    prof = get_synthetic_profile("SYNTH-001")
    with pytest.raises(ReplayFixtureMissingError):
        client.chat_synthetic(
            synthetic_profile_id="SYNTH-001",
            synthetic_features=prof.to_patient_input().model_dump(),
            messages=[{"role": "user", "content": "Unrecorded custom prompt"}],
        )


def test_authoritative_output_gate_blocks_prescriptive_and_enforces_informational():
    pred = predict_patient(SYNTHETIC_PROFILES[2].to_patient_input())
    unsafe = "You definitely have heart disease. Start taking aspirin 81mg daily and I prescribe statins."
    gated = enforce_authoritative_output_gate(unsafe, pred)
    assert "definitely have heart disease" not in gated.lower()
    assert "start taking aspirin" not in gated.lower()
    assert "i prescribe" not in gated.lower()
    assert "not medical advice" in gated.lower() or "informational" in gated.lower()


def test_replay_assistant_end_to_end_on_all_synthetic_profiles():
    for prof in SYNTHETIC_PROFILES:
        resp = explain_screening(
            AssistantRequest(use_external_meta_api=True, synthetic_profile_id=prof.profile_id)
        )
        assert resp.mode == "meta_replay"
        assert resp.is_synthetic_profile is True
        assert resp.external_api_called is False
        assert resp.spend_usd == 0.0
        assert resp.informational_only is True
        assert resp.not_medical_advice is True
        assert len(resp.uncertainty_reminders) >= 3
        assert "not medical advice" in resp.disclaimer.lower()


def test_zero_gemini_sdk_or_endpoint_in_src():
    root = Path(__file__).resolve().parents[1]
    for py_file in (root / "src").glob("*.py"):
        if py_file.name == "security.py":
            continue  # security.py only lists GEMINI_API_KEY as an env name to redact if present.
        text = py_file.read_text(encoding="utf-8").lower()
        assert "google.generativeai" not in text
        assert "google.genai" not in text
        assert "generativelanguage.googleapis.com" not in text
