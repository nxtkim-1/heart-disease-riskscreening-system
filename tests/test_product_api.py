"""FastAPI product endpoint, session hygiene, privacy, and error-surfacing tests."""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from product.api import create_app
from src.meta_client import MetaAssistantClient
from src.meta_ledger import SpendLedger


def _make_client(tmp_path):
    ledger = SpendLedger(tmp_path / "api_ledger.jsonl")
    app = create_app(
        ledger=ledger,
        meta_client_factory=lambda: MetaAssistantClient(mode="replay", ledger=ledger),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_static_frontend_and_health_probe(tmp_path):
    client = _make_client(tmp_path)

    idx = client.get("/")
    assert idx.status_code == 200
    assert "Heart-Disease Risk Screening System" in idx.text
    assert "Not Medical Advice" in idx.text

    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ready"] is True
    assert body["baseline_sha256_valid"] is True
    assert body["assistant"]["synthetic_only_enforced"] is True
    assert body["assistant"]["gemini_touched"] is False
    assert body["privacy"]["patient_inputs_logged"] is False
    assert body["privacy"]["patient_inputs_persisted"] is False

    # Session hygiene headers
    assert "no-store" in health.headers["cache-control"]
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["referrer-policy"] == "no-referrer"
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["x-patient-data-persisted"] == "false"


def test_predict_endpoint_valid_high_missingness_and_validation_failures(tmp_path, caplog):
    client = _make_client(tmp_path)

    # 1. Valid complete patient (verify zero patient measurements appear in logs).
    payload = {
        "age": 63,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 149,
        "chol": 277,
        "fbs": True,
        "restecg": "lv hypertrophy",
        "thalch": 121,
        "exang": True,
        "oldpeak": 2.3,
        "slope": "flat",
    }
    with caplog.at_level(logging.DEBUG):
        r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold"] == 0.39
    assert data["flagged"] is True
    assert data["uncertainty"]["bootstrap_samples"] == 200
    assert data["uncertainty"]["bootstrap_ci_low"] <= data["probability"] <= data["uncertainty"]["bootstrap_ci_high"]
    assert data["calibrated_reference_probability"] is not None
    assert data["informational_only"] is True
    assert len(data["top_contributions"]) >= 3
    # Ensure raw patient values were not written to logs.
    assert "277" not in caplog.text and "149" not in caplog.text

    # 2. High-missingness input produces explicit imputation warning.
    sparse = {"age": 58, "sex": "Female", "cp": "typical angina"}
    r_sparse = client.post("/api/predict", json=sparse)
    assert r_sparse.status_code == 200
    u = r_sparse.json()["uncertainty"]
    assert u["missing_field_count"] == 8
    assert u["imputation_warning"] is not None
    assert "High missingness" in u["imputation_warning"]

    # 3. Validation failures: sentinel zero cholesterol, PII field, hospital column.
    r_zero = client.post("/api/predict", json={**payload, "chol": 0})
    assert r_zero.status_code == 422
    assert r_zero.json()["error"] == "validation_error"

    r_pii = client.post("/api/predict", json={**payload, "patient_name": "John Smith"})
    assert r_pii.status_code == 422

    r_hosp = client.post("/api/predict", json={**payload, "dataset": "Switzerland"})
    assert r_hosp.status_code == 422


def test_model_info_statistics_and_session_clear(tmp_path):
    client = _make_client(tmp_path)

    m = client.get("/api/model/info")
    assert m.status_code == 200
    assert m.json()["baseline"]["hospital_id_dropped"] is True
    assert m.json()["versioned_candidates"]["harmonized_calibrated"] is not None

    s = client.get("/api/statistics")
    assert s.status_code == 200
    stats = s.json()
    assert "frozen_baseline_historical_audit_n184" in stats
    assert "loho_zero_site_overlap_n920" in stats
    assert "harmonization_comparison" in stats

    clr = client.post("/api/session/clear")
    assert clr.status_code == 200
    assert clr.json()["cleared"] is True
    assert clr.json()["server_state_retained"] is False


def test_assistant_endpoints_privacy_and_replay(tmp_path):
    client = _make_client(tmp_path)

    profs = client.get("/api/assistant/profiles")
    assert profs.status_code == 200
    assert len(profs.json()["profiles"]) == 6

    led = client.get("/api/assistant/ledger")
    assert led.status_code == 200
    assert led.json()["ledger_summary"]["total_spent_usd"] == 0.0

    # Local offline explanation with custom de-identified patient works with zero network.
    local_r = client.post(
        "/api/assistant/explain",
        json={
            "use_external_meta_api": False,
            "patient": {"age": 52, "sex": "Female", "cp": "atypical angina", "oldpeak": 0.4},
            "question": "What does oldpeak mean and how uncertain is this score?",
        },
    )
    assert local_r.status_code == 200
    lb = local_r.json()
    assert lb["mode"] == "local_offline"
    assert lb["external_api_called"] is False
    assert lb["spend_usd"] == 0.0
    assert lb["not_medical_advice"] is True
    assert "ST depression" in lb["explanation"]

    # Synthetic replay via Meta adapter succeeds with $0 spend.
    rep_r = client.post(
        "/api/assistant/explain",
        json={"use_external_meta_api": True, "synthetic_profile_id": "SYNTH-002"},
    )
    assert rep_r.status_code == 200
    rb = rep_r.json()
    assert rb["mode"] == "meta_replay"
    assert rb["is_synthetic_profile"] is True
    assert rb["spend_usd"] == 0.0

    # Custom patient attempting external Meta API is blocked (422).
    bad_ext = client.post(
        "/api/assistant/explain",
        json={
            "use_external_meta_api": True,
            "synthetic_profile_id": "SYNTH-001",
            "patient": {"age": 52, "sex": "Female", "cp": "non-anginal"},
        },
    )
    assert bad_ext.status_code == 422

    # Unregistered profile ID attempting external Meta API is blocked (403).
    bad_id = client.post(
        "/api/assistant/explain",
        json={"use_external_meta_api": True, "synthetic_profile_id": "REAL-USER-1"},
    )
    assert bad_id.status_code == 403
    assert bad_id.json()["error"] == "privacy_boundary_violation"

    # Custom free-text question on the external path is refused before any client is used (422).
    miss = client.post(
        "/api/assistant/explain",
        json={
            "use_external_meta_api": True,
            "synthetic_profile_id": "SYNTH-001",
            "question": "Brand new unrecorded question never seen in fixtures?",
        },
    )
    assert miss.status_code == 422
    assert miss.json()["error"] == "validation_error"


def test_unhandled_error_redacts_secret_and_private_path(monkeypatch, tmp_path):
    secret = "LLM|secret_never_leak_555444"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    app = create_app(ledger=SpendLedger(tmp_path / "l.jsonl"))

    @app.get("/api/_test_boom")
    async def _boom():
        raise RuntimeError(f"Failed at C:\\Users\\secretuser\\app.py using {secret}")

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/_test_boom")
    assert r.status_code == 500
    msg = r.json()["message"]
    assert secret not in msg
    assert "secretuser" not in msg
    assert "[REDACTED_PATH]" in msg
    assert "RuntimeError" in msg
