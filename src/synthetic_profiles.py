"""Documented synthetic patient profiles for external Meta API testing and demos.

STRICT PRIVACY BOUNDARY:
- Every profile in this module is 100% synthetically constructed from clinical textbook
  archetypes to span the feature space (low risk, borderline near 0.39, high risk,
  missing-data stress test, female presentation, etc.).
- Zero rows from the UCI dataset and zero real-person records are included.
- The Meta API adapter verifies every outgoing request against this registry's SHA-256
  digest so no real patient data can ever cross the wire.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.schemas import PatientInput

PROVENANCE_NOTE = (
    "Synthetic archetype profiles authored manually for software verification and plain-language "
    "assistant testing (2026-11-22). Not derived from any individual patient or UCI row."
)


@dataclass(frozen=True)
class SyntheticProfile:
    profile_id: str
    title: str
    clinical_scenario: str
    features: dict[str, Any]
    is_synthetic: bool = True

    def canonical_payload(self) -> dict[str, Any]:
        validated = PatientInput(**self.features)
        return {
            "profile_id": self.profile_id,
            "is_synthetic": True,
            "features": validated.model_dump(),
        }

    def digest(self) -> str:
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_patient_input(self) -> PatientInput:
        return PatientInput(**self.features)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "title": self.title,
            "clinical_scenario": self.clinical_scenario,
            "is_synthetic": True,
            "provenance": PROVENANCE_NOTE,
            "digest_sha256": self.digest(),
            "features": PatientInput(**self.features).model_dump(),
        }


SYNTHETIC_PROFILES: tuple[SyntheticProfile, ...] = (
    SyntheticProfile(
        profile_id="SYNTH-001",
        title="Low-Risk Younger Female with Non-Anginal Pain",
        clinical_scenario=(
            "Synthetic 41-year-old female with non-anginal chest discomfort, high peak exercise "
            "heart rate, normal resting ECG, and no ST depression."
        ),
        features={
            "age": 41,
            "sex": "Female",
            "cp": "non-anginal",
            "trestbps": 116,
            "chol": 195,
            "fbs": False,
            "restecg": "normal",
            "thalch": 176,
            "exang": False,
            "oldpeak": 0.0,
            "slope": "upsloping",
        },
    ),
    SyntheticProfile(
        profile_id="SYNTH-002",
        title="Borderline Middle-Aged Male Near Screening Threshold",
        clinical_scenario=(
            "Synthetic 54-year-old male with atypical angina, mild blood pressure elevation, "
            "no exercise angina, and modest ST depression — designed to sit near the 0.39 cutoff."
        ),
        features={
            "age": 54,
            "sex": "Male",
            "cp": "atypical angina",
            "trestbps": 134,
            "chol": 238,
            "fbs": False,
            "restecg": "normal",
            "thalch": 152,
            "exang": False,
            "oldpeak": 0.8,
            "slope": "flat",
        },
    ),
    SyntheticProfile(
        profile_id="SYNTH-003",
        title="High-Risk Older Male with Exercise Angina & ST Depression",
        clinical_scenario=(
            "Synthetic 64-year-old male with asymptomatic presentation on angiography workup, "
            "exercise-induced angina, flat ST slope, and 2.4 mm ST depression."
        ),
        features={
            "age": 64,
            "sex": "Male",
            "cp": "asymptomatic",
            "trestbps": 154,
            "chol": 286,
            "fbs": True,
            "restecg": "lv hypertrophy",
            "thalch": 118,
            "exang": True,
            "oldpeak": 2.4,
            "slope": "flat",
        },
    ),
    SyntheticProfile(
        profile_id="SYNTH-004",
        title="High-Missingness Synthetic Intake (No Stress ECG)",
        clinical_scenario=(
            "Synthetic 59-year-old male with typical angina where cholesterol, blood sugar, "
            "exercise heart rate, ST depression, and ST slope were not yet measured."
        ),
        features={
            "age": 59,
            "sex": "Male",
            "cp": "typical angina",
            "trestbps": 142,
            "chol": None,
            "fbs": None,
            "restecg": None,
            "thalch": None,
            "exang": None,
            "oldpeak": None,
            "slope": None,
        },
    ),
    SyntheticProfile(
        profile_id="SYNTH-005",
        title="Elevated-Risk Older Female with Asymptomatic Presentation",
        clinical_scenario=(
            "Synthetic 63-year-old female with asymptomatic presentation, exercise-induced angina, "
            "ST-T abnormality on resting ECG, and 1.8 mm ST depression."
        ),
        features={
            "age": 63,
            "sex": "Female",
            "cp": "asymptomatic",
            "trestbps": 148,
            "chol": 265,
            "fbs": True,
            "restecg": "st-t abnormality",
            "thalch": 124,
            "exang": True,
            "oldpeak": 1.8,
            "slope": "flat",
        },
    ),
    SyntheticProfile(
        profile_id="SYNTH-006",
        title="Healthy-Profile 30-Year-Old Reference (Population Shift Demo)",
        clinical_scenario=(
            "Synthetic 30-year-old male with atypical angina and normal exercise markers — "
            "used to demonstrate angiography-referral cohort base-rate shift."
        ),
        features={
            "age": 30,
            "sex": "Male",
            "cp": "atypical angina",
            "trestbps": 112,
            "chol": 175,
            "fbs": False,
            "restecg": "normal",
            "thalch": 188,
            "exang": False,
            "oldpeak": 0.0,
            "slope": "upsloping",
        },
    ),
)

_REGISTRY_BY_ID: dict[str, SyntheticProfile] = {p.profile_id: p for p in SYNTHETIC_PROFILES}
_REGISTRY_DIGESTS: dict[str, str] = {p.profile_id: p.digest() for p in SYNTHETIC_PROFILES}


class SyntheticPrivacyViolation(PermissionError):
    """Raised when non-synthetic or unregistered patient data attempts to hit an external API."""


def list_synthetic_profiles() -> list[dict[str, Any]]:
    """Return metadata and canonical features for all registered synthetic profiles."""
    return [p.to_public_dict() for p in SYNTHETIC_PROFILES]


def get_synthetic_profile(profile_id: str) -> SyntheticProfile:
    """Look up a registered synthetic profile by ID, failing closed if unknown."""
    key = str(profile_id or "").strip().upper()
    profile = _REGISTRY_BY_ID.get(key)
    if profile is None:
        raise SyntheticPrivacyViolation(
            f"Unknown synthetic profile ID {profile_id!r}. External API calls only accept "
            f"registered synthetic IDs: {', '.join(sorted(_REGISTRY_BY_ID))}."
        )
    return profile


def assert_registered_synthetic(profile_id: str, features: dict[str, Any]) -> SyntheticProfile:
    """Authoritative privacy gate: confirm profile_id and exact features match a synthetic fixture."""
    profile = get_synthetic_profile(profile_id)
    validated = PatientInput(**features)
    candidate_payload = {
        "profile_id": profile.profile_id,
        "is_synthetic": True,
        "features": validated.model_dump(),
    }
    raw = json.dumps(candidate_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if digest != _REGISTRY_DIGESTS[profile.profile_id]:
        raise SyntheticPrivacyViolation(
            f"Payload for {profile.profile_id} does not match the registered synthetic digest. "
            "Refusing to send modified or real patient measurements to an external API."
        )
    return profile
