"""Inference, feature attribution, and honest epistemic uncertainty service.

Always loads the frozen baseline artifact `models/heart_model.joblib` (SHA-256 verified)
as the primary screening model (threshold 0.39), paired with:
- Epistemic 95% predictive uncertainty intervals from the separately versioned bootstrap
  ensemble (`models/candidates/uncertainty_bootstrap_v1.joblib`, B=200 models fit on X_train only).
- Reference calibrated probability from the separately versioned harmonized candidate
  (`models/candidates/harmonized_calibrated_lr_v1.joblib`).
- Exact per-patient log-odds feature contributions from the baseline logistic regression.
- Explicit missingness/imputation, threshold-proximity, and 1981-1988 angiography cohort
  population-shift warnings.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from src.data import LABELS, PROJECT_ROOT, clean_features
from src.harmonization import (
    BASELINE_EXPECTED_SHA256,
    BASELINE_MODEL_PATH,
    CANDIDATES_DIR,
    apply_platt,
    verify_baseline_intact,
)
from src.schemas import (
    FeatureContribution,
    PatientInput,
    PredictionResponse,
    UncertaintySummary,
)

UNCERTAINTY_MODEL_PATH = CANDIDATES_DIR / "uncertainty_bootstrap_v1.joblib"
HARMONIZED_MODEL_PATH = CANDIDATES_DIR / "harmonized_calibrated_lr_v1.joblib"
STATISTICAL_REPORT_PATH = PROJECT_ROOT / "reports" / "statistical_deepening.json"

INFORMATIONAL_DISCLAIMER = (
    "Informational prototype only — NOT a medical device, NOT clinical validation, and NOT medical advice. "
    "The underlying model was trained on 920 patients referred for coronary angiography across four hospitals "
    "(1981–1988; 55% overall disease prevalence), so probabilities do not reflect the general population. "
    "Consult a qualified clinician for any medical decisions."
)


class ModelLoadError(RuntimeError):
    """Raised when the frozen baseline artifact cannot be loaded or fails integrity checks."""


@lru_cache(maxsize=1)
def load_baseline_bundle() -> dict[str, Any]:
    """Load and SHA-256 verify the immutable baseline model bundle."""
    try:
        sha = verify_baseline_intact(BASELINE_MODEL_PATH)
        bundle = joblib.load(BASELINE_MODEL_PATH)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load frozen baseline artifact: {type(exc).__name__}") from exc
    bundle["_verified_sha256"] = sha
    return bundle


@lru_cache(maxsize=1)
def load_uncertainty_bundle() -> dict[str, Any] | None:
    """Load the versioned bootstrap uncertainty ensemble if present."""
    if not UNCERTAINTY_MODEL_PATH.exists():
        return None
    return joblib.load(UNCERTAINTY_MODEL_PATH)


@lru_cache(maxsize=1)
def load_harmonized_bundle() -> dict[str, Any] | None:
    """Load the versioned harmonized calibrated candidate if present."""
    if not HARMONIZED_MODEL_PATH.exists():
        return None
    return joblib.load(HARMONIZED_MODEL_PATH)


def load_statistical_report() -> dict[str, Any] | None:
    """Load generated statistical deepening JSON report if present."""
    if not STATISTICAL_REPORT_PATH.exists():
        return None
    return json.loads(STATISTICAL_REPORT_PATH.read_text(encoding="utf-8"))


def compute_linear_contributions(pipeline: Any, X: Any, top_k: int = 6) -> list[FeatureContribution]:
    """Compute exact per-patient log-odds contributions for the baseline LogisticRegression pipeline."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []

    transformed = pre.transform(X)
    if hasattr(transformed, "toarray"):
        z = transformed.toarray()[0]
    else:
        z = np.asarray(transformed)[0]

    raw_names = pre.get_feature_names_out()
    coefs = model.coef_[0]
    contribs: list[FeatureContribution] = []
    for raw_name, z_val, coef_val in zip(raw_names, z, coefs):
        c = float(z_val * coef_val)
        if abs(c) < 1e-4:
            continue
        short = raw_name.split("__", 1)[1] if "__" in raw_name else str(raw_name)
        label = LABELS.get(short, short)
        direction = "increases_risk" if c > 0 else ("decreases_risk" if c < 0 else "neutral")
        contribs.append(
            FeatureContribution(
                feature=short,
                label=label,
                log_odds_contribution=round(c, 4),
                direction=direction,
            )
        )
    contribs.sort(key=lambda item: abs(item.log_odds_contribution), reverse=True)
    return contribs[:top_k]


def predict_patient(patient: PatientInput) -> PredictionResponse:
    """Score one validated PatientInput with the frozen baseline + honest uncertainty summary."""
    bundle = load_baseline_bundle()
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    features = bundle["features"]

    raw_df = patient.to_feature_dataframe()
    X = clean_features(raw_df)[features]
    prob = float(pipeline.predict_proba(X)[0, 1])
    flagged = prob >= threshold

    # Risk band reflects threshold proximity transparently.
    if abs(prob - threshold) <= 0.08:
        risk_band = "borderline_near_threshold"
        proximity_note = (
            f"Close to the {threshold:.2f} screening cutoff (within ±8 percentage points). "
            "Small measurement changes or missing-value imputation can flip the flag."
        )
    elif flagged:
        risk_band = "elevated_above_threshold"
        proximity_note = (
            f"Above the {threshold:.2f} screening cutoff (set to prioritize sensitivity over precision)."
        )
    else:
        risk_band = "below_threshold"
        proximity_note = (
            f"Below the {threshold:.2f} screening cutoff. 'Not flagged' means below threshold, NOT 'healthy'."
        )

    # Bootstrap epistemic uncertainty interval from versioned ensemble.
    unc_bundle = load_uncertainty_bundle()
    ci_low: float | None = None
    ci_high: float | None = None
    boot_std: float | None = None
    boot_n = 0
    if unc_bundle is not None:
        pipes = unc_bundle.get("pipelines", [])
        boot_n = len(pipes)
        if boot_n > 0:
            preds = np.array([float(p.predict_proba(X)[0, 1]) for p in pipes], dtype=float)
            ci_low = round(float(np.percentile(preds, 2.5)), 4)
            ci_high = round(float(np.percentile(preds, 97.5)), 4)
            boot_std = round(float(np.std(preds, ddof=1)), 4)
            if ci_low <= threshold <= ci_high and abs(prob - threshold) > 0.08:
                proximity_note += (
                    f" Note: the 95% model-sampling interval [{ci_low:.0%}, {ci_high:.0%}] spans the "
                    f"{threshold:.0%} threshold."
                )

    missing = patient.missing_fields()
    missing_count = len(missing)
    if missing_count == 0:
        imputation_warning = None
    elif missing_count <= 2:
        imputation_warning = (
            f"{missing_count} unmeasured field(s) ({', '.join(missing)}) were filled using training-set "
            "medians/modes, which adds extra uncertainty beyond the bootstrap interval."
        )
    else:
        imputation_warning = (
            f"High missingness: {missing_count} of 8 optional measurements ({', '.join(missing)}) were "
            "imputed from training-set defaults. Treat this estimate as heavily uncertain."
        )

    # Reference calibrated probability from harmonized candidate (never overrides baseline).
    harm_bundle = load_harmonized_bundle()
    cal_prob: float | None = None
    cal_artifact: str | None = None
    if harm_bundle is not None:
        raw_h = harm_bundle["pipeline"].predict_proba(X)[:, 1]
        cal_p = apply_platt(raw_h, float(harm_bundle["platt_a"]), float(harm_bundle["platt_b"]))
        cal_prob = round(float(cal_p[0]), 4)
        cal_artifact = harm_bundle["provenance"]["artifact"]

    calibration_note = (
        "Probabilities are calibrated to the 4-hospital UCI angiography cohort (55% disease rate), "
        "and LOHO site-shift audits show mean site bias up to ±25 percentage points when moving across hospitals."
    )
    population_shift_warning = (
        "Trained on patients already referred for coronary angiography (1981–1988). "
        "In lower-prevalence general screening populations, raw probabilities substantially overestimate absolute risk."
    )

    contributions = compute_linear_contributions(pipeline, X, top_k=6)

    return PredictionResponse(
        model_name=str(bundle["model_name"]),
        model_artifact="models/heart_model.joblib",
        probability=round(prob, 4),
        threshold=round(threshold, 4),
        flagged=flagged,
        risk_band=risk_band,
        uncertainty=UncertaintySummary(
            bootstrap_ci_low=ci_low,
            bootstrap_ci_high=ci_high,
            bootstrap_std=boot_std,
            bootstrap_samples=boot_n,
            missing_field_count=missing_count,
            missing_fields=missing,
            imputation_warning=imputation_warning,
            calibration_note=calibration_note,
            population_shift_warning=population_shift_warning,
            threshold_proximity_note=proximity_note,
        ),
        top_contributions=contributions,
        calibrated_reference_probability=cal_prob,
        calibrated_reference_artifact=cal_artifact,
        informational_only=True,
        disclaimer=INFORMATIONAL_DISCLAIMER,
    )


def readiness_probe() -> dict[str, Any]:
    """Probe REAL runtime dependencies and immutable baseline integrity."""
    checks: dict[str, Any] = {
        "sklearn_version": sklearn.__version__,
        "sklearn_pinned_1_7_2": sklearn.__version__ == "1.7.2",
        "baseline_exists": BASELINE_MODEL_PATH.exists(),
        "baseline_sha256_valid": False,
        "baseline_loads": False,
        "uncertainty_ensemble_ready": False,
        "harmonized_candidate_ready": False,
        "statistical_report_ready": STATISTICAL_REPORT_PATH.exists(),
    }
    try:
        sha = verify_baseline_intact(BASELINE_MODEL_PATH)
        checks["baseline_sha256_valid"] = sha == BASELINE_EXPECTED_SHA256
        bundle = load_baseline_bundle()
        checks["baseline_loads"] = "pipeline" in bundle and "threshold" in bundle
        checks["baseline_threshold"] = float(bundle["threshold"])
    except Exception as exc:
        checks["baseline_error"] = type(exc).__name__

    try:
        checks["uncertainty_ensemble_ready"] = load_uncertainty_bundle() is not None
    except Exception:
        checks["uncertainty_ensemble_ready"] = False

    try:
        checks["harmonized_candidate_ready"] = load_harmonized_bundle() is not None
    except Exception:
        checks["harmonized_candidate_ready"] = False

    checks["ready"] = bool(
        checks["sklearn_pinned_1_7_2"]
        and checks["baseline_exists"]
        and checks["baseline_sha256_valid"]
        and checks["baseline_loads"]
    )
    return checks
