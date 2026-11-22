"""Strict Pydantic v2 schemas for patient screening inputs, predictions, and assistant queries.

Enforces clinical range bounds, canonical category enums, rejection of unknown/PII fields,
and explicit separation between measured values and missing ("Unknown" / null) entries.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.security import FORBIDDEN_PII_FIELDS

SexValue = Literal["Male", "Female"]
ChestPainValue = Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
RestECGValue = Literal["normal", "st-t abnormality", "lv hypertrophy"]
SlopeValue = Literal["upsloping", "flat", "downsloping"]

FIELD_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 110.0),
    "trestbps": (60.0, 260.0),
    "chol": (80.0, 700.0),
    "thalch": (40.0, 230.0),
    "oldpeak": (-3.0, 10.0),
}

MEASURED_OPTIONAL_FIELDS: tuple[str, ...] = (
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
)


def _normalize_unknown(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("NaN and infinite numeric values are not permitted; use null for unknown.")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"unknown", "na", "n/a", "none", "null"}:
            return None
        return stripped
    return value


class PatientInput(BaseModel):
    """Validated clinical input matching the 11 hospital-ID-free model features."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: float = Field(..., ge=18.0, le=110.0, description="Patient age in years (18–110).")
    sex: SexValue = Field(..., description="Biological sex recorded in the UCI schema ('Male' or 'Female').")
    cp: ChestPainValue = Field(
        ...,
        description="Chest pain category: typical angina, atypical angina, non-anginal, or asymptomatic.",
    )
    trestbps: float | None = Field(
        default=None,
        ge=60.0,
        le=260.0,
        description="Resting systolic blood pressure in mm Hg (60–260), or null if not measured.",
    )
    chol: float | None = Field(
        default=None,
        ge=80.0,
        le=700.0,
        description="Serum total cholesterol in mg/dl (80–700), or null if not measured.",
    )
    fbs: bool | None = Field(
        default=None,
        description="Whether fasting blood sugar exceeds 120 mg/dl, or null if not measured.",
    )
    restecg: RestECGValue | None = Field(
        default=None,
        description="Resting ECG category, or null if not measured.",
    )
    thalch: float | None = Field(
        default=None,
        ge=40.0,
        le=230.0,
        description="Maximum heart rate reached during exercise test in bpm (40–230), or null if not measured.",
    )
    exang: bool | None = Field(
        default=None,
        description="Exercise-induced angina (true/false), or null if not measured.",
    )
    oldpeak: float | None = Field(
        default=None,
        ge=-3.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest in mm (-3.0 to 10.0), or null if not measured.",
    )
    slope: SlopeValue | None = Field(
        default=None,
        description="Peak exercise ST segment slope, or null if not measured.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_pii_and_hospital(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        lower_keys = {str(k).strip().lower(): k for k in data.keys()}
        pii_hits = sorted(set(lower_keys).intersection(FORBIDDEN_PII_FIELDS))
        if pii_hits:
            raise ValueError(
                f"Direct personal identifiers are not allowed ({', '.join(pii_hits)}). "
                "Enter de-identified clinical measurements only."
            )
        if "dataset" in lower_keys or "hospital" in lower_keys:
            raise ValueError(
                "Hospital/site identifiers are intentionally excluded from the model to prevent site-prevalence leakage."
            )
        return data

    @field_validator("age", mode="before")
    @classmethod
    def _check_age_finite(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Age must be a number, not a boolean.")
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Age must be a finite number.")
        return v

    @field_validator("trestbps", "chol", "thalch", "oldpeak", mode="before")
    @classmethod
    def _check_optional_number(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Measurement must be a number or null, not a boolean.")
        v = _normalize_unknown(v)
        if v is None:
            return None
        # Reject sentinel 0 for resting BP / cholesterol with a clear message.
        if isinstance(v, (int, float)) and v == 0:
            return v  # Let ge= bounds produce a clear clinical range error (oldpeak=0.0 is valid).
        return v

    @field_validator("fbs", "exang", mode="before")
    @classmethod
    def _check_optional_bool(cls, v: Any) -> Any:
        v = _normalize_unknown(v)
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            low = v.lower()
            if low in {"yes", "true", "1"}:
                return True
            if low in {"no", "false", "0"}:
                return False
        raise ValueError("Expected true/false, Yes/No, or null for unknown.")

    @field_validator("sex", mode="before")
    @classmethod
    def _normalize_sex(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower()
            if low == "male":
                return "Male"
            if low == "female":
                return "Female"
        return v

    @field_validator("cp", "restecg", "slope", mode="before")
    @classmethod
    def _normalize_category(cls, v: Any) -> Any:
        v = _normalize_unknown(v)
        if isinstance(v, str):
            return v.lower()
        return v

    def missing_fields(self) -> list[str]:
        """Return list of optional clinical measurements left unmeasured (None)."""
        return [name for name in MEASURED_OPTIONAL_FIELDS if getattr(self, name) is None]

    def to_feature_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame matching src.data.clean_features input."""
        row = {
            "age": float(self.age),
            "sex": self.sex,
            "cp": self.cp,
            "trestbps": np.nan if self.trestbps is None else float(self.trestbps),
            "chol": np.nan if self.chol is None else float(self.chol),
            "fbs": np.nan if self.fbs is None else bool(self.fbs),
            "restecg": np.nan if self.restecg is None else self.restecg,
            "thalch": np.nan if self.thalch is None else float(self.thalch),
            "exang": np.nan if self.exang is None else bool(self.exang),
            "oldpeak": np.nan if self.oldpeak is None else float(self.oldpeak),
            "slope": np.nan if self.slope is None else self.slope,
        }
        return pd.DataFrame([row])


class FeatureContribution(BaseModel):
    feature: str
    label: str
    log_odds_contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]


class UncertaintySummary(BaseModel):
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    bootstrap_std: float | None = None
    bootstrap_samples: int = 0
    missing_field_count: int
    missing_fields: list[str]
    imputation_warning: str | None = None
    calibration_note: str
    population_shift_warning: str
    threshold_proximity_note: str


class PredictionResponse(BaseModel):
    model_name: str
    model_artifact: str
    probability: float
    threshold: float
    flagged: bool
    risk_band: Literal["below_threshold", "borderline_near_threshold", "elevated_above_threshold"]
    uncertainty: UncertaintySummary
    top_contributions: list[FeatureContribution]
    calibrated_reference_probability: float | None = None
    calibrated_reference_artifact: str | None = None
    informational_only: Literal[True] = True
    disclaimer: str


DEFAULT_ASSISTANT_QUESTION = (
    "Explain this screening result, the main drivers, and its uncertainty in plain language."
)

# The ONLY question texts allowed to leave this machine on the external Meta API path.
# Free text could carry real patient details or PII, so the external path accepts a fixed
# set of canonical educational prompts (each has a recorded replay fixture per synthetic
# profile). Free-text questions stay on the local_offline path, which makes no network call.
EXTERNAL_QUESTION_TEMPLATES: dict[str, str] = {
    "explain_result": DEFAULT_ASSISTANT_QUESTION,
    "explain_uncertainty": (
        "Explain how uncertain this synthetic screening estimate is, including the bootstrap "
        "interval, any imputed fields, and population shift."
    ),
    "explain_threshold": (
        "Explain what the 0.39 screening threshold means for this synthetic result and why being "
        "flagged or not flagged is not a diagnosis."
    ),
}
_ALLOWED_EXTERNAL_QUESTIONS: frozenset[str] = frozenset(EXTERNAL_QUESTION_TEMPLATES.values())


def is_allowed_external_question(question: str) -> bool:
    """True only for an exact canonical template text (after outer whitespace is stripped)."""
    return isinstance(question, str) and question.strip() in _ALLOWED_EXTERNAL_QUESTIONS


class AssistantRequest(BaseModel):
    """Request to the explanation assistant.

    External Meta API calls (`use_external_meta_api=True`) STRICTLY require a registered
    synthetic_profile_id, forbid arbitrary/real patient inputs, and only accept a question
    from EXTERNAL_QUESTION_TEMPLATES (no free text).
    Local offline explanations (`use_external_meta_api=False`) work on any validated PatientInput
    and any free-text question without making any network call.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        default=DEFAULT_ASSISTANT_QUESTION,
        min_length=3,
        max_length=600,
    )
    use_external_meta_api: bool = Field(
        default=False,
        description="If true, route through the metered Meta API adapter using a registered SYNTHETIC profile ONLY.",
    )
    synthetic_profile_id: str | None = Field(
        default=None,
        description="Registered synthetic profile ID (e.g., SYNTH-001). Required when use_external_meta_api=true.",
    )
    patient: PatientInput | None = Field(
        default=None,
        description="De-identified measurements for LOCAL offline explanation only. Never sent externally.",
    )

    @model_validator(mode="after")
    def _enforce_privacy_mode(self) -> "AssistantRequest":
        if self.use_external_meta_api:
            if not self.synthetic_profile_id:
                raise ValueError(
                    "External Meta API calls require a registered synthetic_profile_id. "
                    "Arbitrary or real patient inputs are never sent to external APIs."
                )
            if self.patient is not None:
                raise ValueError(
                    "Privacy boundary violation: custom patient payloads cannot be submitted alongside "
                    "use_external_meta_api=true. Use a registered synthetic_profile_id for external API calls, "
                    "or set use_external_meta_api=false for 100% local offline explanation."
                )
            if not is_allowed_external_question(self.question):
                # Deliberately does not echo the submitted text back (it may contain PII).
                raise ValueError(
                    "Privacy boundary: custom free-text questions are never sent to the external Meta API. "
                    "Choose one of the canonical question templates "
                    f"({', '.join(sorted(EXTERNAL_QUESTION_TEMPLATES))}) or set "
                    "use_external_meta_api=false to ask a free-text question locally offline."
                )
        else:
            if self.synthetic_profile_id is None and self.patient is None:
                raise ValueError("Provide either a synthetic_profile_id or patient measurements.")
        return self


class AssistantResponse(BaseModel):
    mode: Literal["local_offline", "meta_replay", "meta_record", "meta_live"]
    is_synthetic_profile: bool
    synthetic_profile_id: str | None = None
    external_api_called: bool
    prediction: PredictionResponse
    explanation: str
    key_takeaways: list[str]
    uncertainty_reminders: list[str]
    spend_usd: float = 0.0
    informational_only: Literal[True] = True
    not_medical_advice: Literal[True] = True
    disclaimer: str
