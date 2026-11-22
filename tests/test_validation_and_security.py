"""Input validation schema, PII rejection, and secret/path redaction tests."""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from src.schemas import AssistantRequest, PatientInput
from src.security import (
    RedactingLogFilter,
    contains_sensitive_material,
    find_pii_keys,
    format_safe_error,
    redact_text,
)


def _valid_payload(**overrides):
    base = {
        "age": 56,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 135,
        "chol": 245,
        "fbs": False,
        "restecg": "normal",
        "thalch": 148,
        "exang": True,
        "oldpeak": 1.4,
        "slope": "flat",
    }
    base.update(overrides)
    return base


def test_valid_patient_and_all_unknown_optionals():
    p = PatientInput(**_valid_payload())
    assert p.missing_fields() == []

    p_unk = PatientInput(
        age=60,
        sex="female",
        cp="ATYPICAL ANGINA",
        trestbps="Unknown",
        chol=None,
        fbs="unknown",
        restecg="Unknown",
        thalch="",
        exang=None,
        oldpeak=None,
        slope="N/A",
    )
    assert p_unk.sex == "Female"
    assert p_unk.cp == "atypical angina"
    assert len(p_unk.missing_fields()) == 8


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("age", 10),
        ("age", 150),
        ("age", float("nan")),
        ("age", True),
        ("trestbps", 0),       # sentinel zero rejected via API; must send null
        ("trestbps", 350),
        ("chol", 0),           # sentinel zero rejected via API; must send null
        ("chol", 1200),
        ("thalch", 20),
        ("oldpeak", float("inf")),
        ("oldpeak", 25.0),
        ("sex", "Other"),
        ("cp", "sharp pain"),
        ("slope", "vertical"),
        ("restecg", "arrhythmia"),
        ("fbs", "maybe"),
    ],
)
def test_invalid_clinical_inputs_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        PatientInput(**_valid_payload(**{field: bad_value}))


def test_extra_fields_hospital_id_and_pii_rejected():
    with pytest.raises(ValidationError, match="Direct personal identifiers"):
        PatientInput(**_valid_payload(patient_name="Jane Doe"))

    with pytest.raises(ValidationError, match="Direct personal identifiers"):
        PatientInput(**_valid_payload(mrn="MRN-12345"))

    with pytest.raises(ValidationError, match="Hospital/site identifiers"):
        PatientInput(**_valid_payload(dataset="Switzerland"))

    with pytest.raises(ValidationError):
        PatientInput(**_valid_payload(unexpected_col=123))


def test_assistant_request_enforces_synthetic_only_external_rule():
    # Custom patient + external Meta API must be refused at schema level.
    with pytest.raises(ValidationError, match="Privacy boundary violation"):
        AssistantRequest(
            use_external_meta_api=True,
            synthetic_profile_id="SYNTH-001",
            patient=PatientInput(**_valid_payload()),
        )

    with pytest.raises(ValidationError, match="registered synthetic_profile_id"):
        AssistantRequest(use_external_meta_api=True)

    # Local offline with custom patient is allowed.
    req = AssistantRequest(use_external_meta_api=False, patient=PatientInput(**_valid_payload()))
    assert req.use_external_meta_api is False


def test_secret_token_and_path_redaction(monkeypatch):
    fake_secret = "LLM|supersecretmeta9876543210"
    monkeypatch.setenv("MODEL_API_KEY", fake_secret)

    raw = (
        f"Failed loading C:\\Users\\alice\\Documents\\secret\\file.json and /home/alice/private.key "
        f"with Authorization: Bearer tok_abcdef1234567890 and key={fake_secret} and sk-live1234567890abcd"
    )
    cleaned = redact_text(raw)
    assert fake_secret not in cleaned
    assert "alice" not in cleaned
    assert "tok_abcdef1234567890" not in cleaned
    assert "sk-live1234567890abcd" not in cleaned
    assert "[REDACTED_PATH]" in cleaned
    assert "[REDACTED_SECRET]" in cleaned or "[REDACTED_TOKEN]" in cleaned
    assert not contains_sensitive_material(cleaned)

    err_text = format_safe_error(RuntimeError(raw), context="Request error at /Users/bob/app.py")
    assert fake_secret not in err_text
    assert "bob" not in err_text
    assert "RuntimeError" in err_text


def test_redacting_log_filter_scrubs_records(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "LLM|testkeyvalue000111222")
    flt = RedactingLogFilter()
    rec = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="x.py",
        lineno=1,
        msg="Token %s failed at C:\\Users\\thomas\\keys.txt",
        args=("LLM|testkeyvalue000111222",),
        exc_info=None,
    )
    assert flt.filter(rec) is True
    assert "LLM|testkeyvalue000111222" not in rec.getMessage()
    assert "thomas" not in rec.getMessage()


def test_find_pii_keys_nested():
    hits = find_pii_keys({"patient": {"age": 55, "email": "a@b.com"}, "items": [{"ssn": "123"}]})
    assert "patient.email" in hits
    assert "items[0].ssn" in hits
