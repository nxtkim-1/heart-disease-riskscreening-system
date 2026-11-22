"""Baseline model immutability, historical reproduction, and preprocessing tests."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import DATA_PATH, FEATURES, clean_features, load_dataset
from src.harmonization import (
    BASELINE_EXPECTED_SHA256,
    BASELINE_MODEL_PATH,
    CANDIDATES_DIR,
    file_sha256,
    verify_baseline_intact,
)
from src.inference import load_baseline_bundle, readiness_probe


def test_sklearn_version_pinned():
    assert sklearn.__version__ == "1.7.2"


def test_frozen_baseline_sha256_intact():
    assert BASELINE_MODEL_PATH.exists()
    sha = verify_baseline_intact(BASELINE_MODEL_PATH)
    assert sha == BASELINE_EXPECTED_SHA256
    assert file_sha256(BASELINE_MODEL_PATH) == BASELINE_EXPECTED_SHA256


def test_hospital_id_excluded_from_features():
    assert "dataset" not in FEATURES
    assert "hospital" not in FEATURES
    assert "ca" not in FEATURES
    assert "thal" not in FEATURES
    assert "id" not in FEATURES
    assert len(FEATURES) == 11


def test_clean_features_handles_sentinel_zeros_and_booleans():
    df = pd.DataFrame([
        {
            "age": 58, "sex": "Male", "cp": "asymptomatic",
            "trestbps": 0, "chol": 0, "fbs": "True", "restecg": "normal",
            "thalch": 140, "exang": "FALSE", "oldpeak": 1.2, "slope": "flat",
        },
        {
            "age": 45, "sex": "Female", "cp": "non-anginal",
            "trestbps": 120, "chol": 210, "fbs": False, "restecg": "normal",
            "thalch": 165, "exang": True, "oldpeak": 0.0, "slope": "upsloping",
        },
    ])
    cleaned = clean_features(df)
    assert np.isnan(cleaned.loc[0, "trestbps"])
    assert np.isnan(cleaned.loc[0, "chol"])
    assert cleaned.loc[0, "sex"] == 1
    assert cleaned.loc[1, "sex"] == 0
    assert cleaned.loc[0, "fbs"] == 1
    assert cleaned.loc[0, "exang"] == 0
    assert cleaned.loc[1, "fbs"] == 0
    assert cleaned.loc[1, "exang"] == 1


def test_baseline_reproduces_historical_184_metrics():
    if not DATA_PATH.exists():
        pytest.skip("UCI CSV not downloaded")
    bundle = load_baseline_bundle()
    assert bundle["model_name"] == "Logistic Regression"
    assert float(bundle["threshold"]) == pytest.approx(0.39, abs=1e-6)
    assert int(bundle["n_test"]) == 184

    X, y = load_dataset(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    proba = bundle["pipeline"].predict_proba(X_test)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)

    assert recall_score(y_test, pred) == pytest.approx(0.9216, abs=1e-3)
    assert accuracy_score(y_test, pred) == pytest.approx(0.8098, abs=1e-3)
    assert roc_auc_score(y_test, proba) == pytest.approx(0.8892, abs=1e-3)


def test_versioned_candidate_artifacts_separate_with_provenance():
    cand = CANDIDATES_DIR / "harmonized_calibrated_lr_v1.joblib"
    cand_prov = CANDIDATES_DIR / "harmonized_calibrated_lr_v1.provenance.json"
    unc = CANDIDATES_DIR / "uncertainty_bootstrap_v1.joblib"
    unc_prov = CANDIDATES_DIR / "uncertainty_bootstrap_v1.provenance.json"

    for p in (cand, cand_prov, unc, unc_prov):
        assert p.exists(), f"Missing versioned artifact: {p.name}"

    cp = json.loads(cand_prov.read_text(encoding="utf-8"))
    assert cp["baseline_preserved_sha256"] == BASELINE_EXPECTED_SHA256
    assert cp["hospital_id_used_at_inference"] is False
    assert cp["n_train"] == 736
    assert cp["sklearn_version"] == "1.7.2"

    up = json.loads(unc_prov.read_text(encoding="utf-8"))
    assert up["n_bootstrap_models"] == 200
    assert up["n_train"] == 736

    loaded_cand = joblib.load(cand)
    assert loaded_cand["pipeline"] is not None
    assert "platt_a" in loaded_cand and "platt_b" in loaded_cand


def test_readiness_probe_reports_real_checks():
    probe = readiness_probe()
    assert probe["ready"] is True
    assert probe["sklearn_pinned_1_7_2"] is True
    assert probe["baseline_sha256_valid"] is True
    assert probe["baseline_loads"] is True
    assert probe["uncertainty_ensemble_ready"] is True
    assert probe["harmonized_candidate_ready"] is True
