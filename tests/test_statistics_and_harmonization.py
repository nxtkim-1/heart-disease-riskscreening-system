"""Statistical deepening, calibration math, bootstrap CI, and harmonization tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.data import DATA_PATH, PROJECT_ROOT
from src.harmonization import (
    HarmonizedCalibratedClassifier,
    apply_bayes_prior_shift,
    apply_platt,
    compute_site_weights,
    fit_platt_params,
    load_dataset_with_metadata,
)
from src.statistics import (
    bootstrap_confidence_intervals,
    calibration_slope_and_citl,
    compute_calibration_report,
    wilson_ci,
)


def test_calibration_math_perfect_and_murphy_identity():
    # Perfectly calibrated predictions: y matches bin probabilities exactly.
    y = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80, dtype=int)
    p = np.array([0.2] * 100 + [0.8] * 100, dtype=float)
    cal = compute_calibration_report(y, p, n_bins=10)

    assert cal["ece_uniform"] == pytest.approx(0.0, abs=1e-4)
    assert cal["ece_quantile"] == pytest.approx(0.0, abs=1e-4)
    assert cal["mean_prob_bias"] == pytest.approx(0.0, abs=1e-4)
    assert cal["oe_ratio"] == pytest.approx(1.0, abs=1e-4)

    # Exact Murphy decomposition identity on binned predictions:
    # binned_brier == reliability - resolution + uncertainty
    m = cal["murphy_uniform"]
    assert m["binned_brier"] == pytest.approx(m["reliability"] - m["resolution"] + m["uncertainty"], abs=1e-4)
    assert m["reliability"] == pytest.approx(0.0, abs=1e-4)


def test_citl_detects_overprediction_and_underprediction():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.30, size=400)
    # Over-predicting probabilities (mean ~0.65 vs observed ~0.30) -> CITL must be negative.
    p_over = np.clip(0.50 + 0.30 * y + rng.normal(0, 0.08, size=400), 0.05, 0.95)
    c_over = calibration_slope_and_citl(y, p_over)
    assert c_over["citl"] is not None and c_over["citl"] < -0.5

    # Under-predicting probabilities (mean ~0.15 vs observed ~0.70) -> CITL must be positive.
    y_hi = rng.binomial(1, 0.70, size=400)
    p_under = np.clip(0.10 + 0.25 * y_hi + rng.normal(0, 0.05, size=400), 0.02, 0.90)
    c_under = calibration_slope_and_citl(y_hi, p_under)
    assert c_under["citl"] is not None and c_under["citl"] > 0.5


def test_wilson_ci_and_bootstrap_ci_determinism():
    lo, hi = wilson_ci(94, 102)
    assert lo is not None and hi is not None
    assert 0.84 <= lo < 0.9216 < hi <= 0.97

    assert wilson_ci(0, 0) == (None, None)

    y = np.array([1] * 102 + [0] * 82, dtype=int)
    p = np.array([0.8] * 94 + [0.2] * 8 + [0.6] * 27 + [0.15] * 55, dtype=float)
    ci1 = bootstrap_confidence_intervals(y, p, threshold=0.39, n_bootstrap=400, seed=42)
    ci2 = bootstrap_confidence_intervals(y, p, threshold=0.39, n_bootstrap=400, seed=42)
    assert ci1 == ci2
    assert ci1["recall"]["ci_low"] < 0.9216 < ci1["recall"]["ci_high"]


def test_site_weights_deconfound_hospital_prevalence():
    hospitals = np.array(["A"] * 100 + ["B"] * 100)
    y = np.array([1] * 90 + [0] * 10 + [1] * 30 + [0] * 70)

    w_inv = compute_site_weights(hospitals, y, "inverse_site")
    assert w_inv[hospitals == "A"].sum() == pytest.approx(w_inv[hospitals == "B"].sum(), rel=1e-4)

    w_dec = compute_site_weights(hospitals, y, "site_deconfounded")
    # Weighted prevalence in site A and site B should move dramatically closer to balanced 0.50
    prev_a = w_dec[(hospitals == "A") & (y == 1)].sum() / w_dec[hospitals == "A"].sum()
    prev_b = w_dec[(hospitals == "B") & (y == 1)].sum() / w_dec[hospitals == "B"].sum()
    assert abs(prev_a - prev_b) < 0.20


def test_platt_and_bayes_prior_shift_math():
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1.5, size=500)
    true_p = 1.0 / (1.0 + np.exp(-(0.5 + 0.8 * z)))
    y = rng.binomial(1, true_p)
    raw_p = 1.0 / (1.0 + np.exp(-z))
    a, b = fit_platt_params(raw_p, y)
    cal_p = apply_platt(raw_p, a, b)
    assert np.mean(cal_p) == pytest.approx(np.mean(y), abs=0.03)

    # Bayes prior shift from 0.50 to 0.90 must raise probabilities monotonically.
    shifted = apply_bayes_prior_shift(np.array([0.2, 0.5, 0.8]), 0.50, 0.90)
    assert shifted[0] > 0.2 and shifted[1] == pytest.approx(0.90, abs=1e-4) and shifted[2] > 0.8


def test_harmonized_classifier_requires_no_hospital_id_at_inference():
    if not DATA_PATH.exists():
        pytest.skip("UCI CSV not downloaded")
    X, y, meta = load_dataset_with_metadata(DATA_PATH)
    clf = HarmonizedCalibratedClassifier(weight_mode="site_deconfounded", calibration="platt", cv_folds=3)
    clf.fit(X.iloc[:300], y.iloc[:300], hospitals=meta["dataset"].iloc[:300])
    # Predict using ONLY the 11 clinical features (no hospital column).
    proba = clf.predict_proba(X.iloc[300:320])
    assert proba.shape == (20, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_statistical_deepening_artifacts_exist_and_verify_key_findings():
    report_path = PROJECT_ROOT / "reports" / "statistical_deepening.json"
    md_path = PROJECT_ROOT / "reports" / "STATISTICAL_REPORT.md"
    assert report_path.exists()
    assert md_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for rel in report["generated_plots"]:
        assert (PROJECT_ROOT / rel).exists(), f"Missing plot: {rel}"

    # Verify LOHO covers all 4 hospitals with zero site overlap.
    loho_lr = report["loho_zero_site_overlap_n920"]["model_families"]["Logistic Regression"]
    held_sites = {r["held_out_hospital"] for r in loho_lr["per_hospital"]}
    assert held_sites == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}
    for row in loho_lr["per_hospital"]:
        assert row["held_out_hospital"] not in row["train_hospitals"]

    # Verify headline recall CI bounds span the point estimate honestly.
    hb = report["frozen_baseline_historical_audit_n184"]
    rec = hb["point_metrics"]["recall"]
    ci = hb["bootstrap_95ci"]["recall"]
    assert ci["ci_low"] < rec < ci["ci_high"]

    # Verify prior-shift bound substantially reduces LOHO macro ECE vs unadjusted models.
    by_id = {s["id"]: s for s in report["harmonization_comparison"]}
    assert (
        by_id["prior_shift_oracle_bound"]["loho_zero_site_overlap_n920"]["macro_ece_quantile"]
        < by_id["site_deconfounded_platt_lr"]["loho_zero_site_overlap_n920"]["macro_ece_quantile"]
    )
