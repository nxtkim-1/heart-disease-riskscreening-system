"""Statistical deepening suite: calibration, 4-hospital source-shift, harmonization,
subgroup evaluation, and bootstrap confidence intervals.

Evaluation discipline:
1. Frozen baseline `models/heart_model.joblib` is audited READ-ONLY (SHA-256 verified before and after).
2. All model comparisons, calibration diagnostics, and harmonization selections are run on:
   (a) Leave-One-Hospital-Out (LOHO) group CV across all 4 hospitals (zero hospital overlap;
       primary source-shift evaluation), AND
   (b) Nested out-of-fold 5-fold CV strictly inside the 736 training rows (184 historical test
       rows untouched).
3. The historical 184-patient test set is audited strictly READ-ONLY for the already-frozen
   baseline model (bootstrap CIs + calibration + subgroup audit) and clearly disclosed as
   historical, row-stratified (not hospital-grouped), and non-independent for new tuning.

This analysis is statistical evaluation of an archival 1981-1988 angiography dataset,
NOT clinical validation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.data import DATA_PATH, FEATURES, PROJECT_ROOT
from src.harmonization import (
    BASELINE_MODEL_PATH,
    RANDOM_STATE,
    TARGET_RECALL,
    HarmonizedCalibratedClassifier,
    apply_bayes_prior_shift,
    file_sha256,
    load_dataset_with_metadata,
    pick_recall_threshold,
    verify_baseline_intact,
)
from src.train import build_models

REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_SEED = RANDOM_STATE

BLUE, RED, ORANGE, GREEN, PURPLE, GREY = (
    "#3B6EA5",
    "#C44E52",
    "#DD8452",
    "#4A9D6F",
    "#8172B3",
    "#7A7A7A",
)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Exact Wilson score 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return round(float(max(0.0, center - half)), 4), round(float(min(1.0, center + half)), 4)


def _bin_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[float, float, list[dict[str, Any]], dict[str, float]]:
    """Compute ECE, MCE, reliability bin table, and Murphy Brier decomposition."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), [], {}

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0] = 0.0
        edges[-1] = 1.0
        # Ensure strictly increasing edges if ties collapse quantiles.
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError(f"Unknown binning strategy: {strategy!r}")

    overall_rate = float(np.mean(y))
    ece = 0.0
    mce = 0.0
    reliability = 0.0
    resolution = 0.0
    bins_out: list[dict[str, Any]] = []

    num_intervals = len(edges) - 1
    for i in range(num_intervals):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == num_intervals - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_p = float(np.mean(p[mask]))
        obs_y = float(np.mean(y[mask]))
        pos_cnt = int(np.sum(y[mask]))
        gap = abs(obs_y - mean_p)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        reliability += (cnt / n) * ((mean_p - obs_y) ** 2)
        resolution += (cnt / n) * ((obs_y - overall_rate) ** 2)
        ci_lo, ci_hi = wilson_ci(pos_cnt, cnt)
        bins_out.append({
            "bin_index": i,
            "lower": round(lo, 4),
            "upper": round(hi, 4),
            "count": cnt,
            "positives": pos_cnt,
            "mean_predicted": round(mean_p, 4),
            "observed_rate": round(obs_y, 4),
            "abs_gap": round(gap, 4),
            "observed_ci_low": ci_lo,
            "observed_ci_high": ci_hi,
        })

    uncertainty = overall_rate * (1.0 - overall_rate)
    murphy = {
        "reliability": round(float(reliability), 5),
        "resolution": round(float(resolution), 5),
        "uncertainty": round(float(uncertainty), 5),
        "binned_brier": round(float(reliability - resolution + uncertainty), 5),
    }
    return float(ece), float(mce), bins_out, murphy


def calibration_slope_and_citl(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    """Compute Van Calster calibration-in-the-large (CITL) and Cox calibration slope/intercept."""
    y = np.asarray(y_true, dtype=float)
    eps = 1e-6
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    if len(np.unique(y)) < 2:
        return {"citl": None, "calibration_intercept": None, "calibration_slope": None}

    eta = logit(p)
    target_mean = float(np.mean(y))

    # CITL: intercept alpha such that mean(sigmoid(alpha + eta)) == mean(y).
    try:
        citl = float(brentq(lambda a: float(np.mean(expit(np.clip(a + eta, -30.0, 30.0)))) - target_mean, -15.0, 15.0))
    except ValueError:
        citl = float("nan")

    # Cox logistic recalibration: y ~ a + b * eta (tiny ridge safeguard for near-separation).
    def nll(params: np.ndarray) -> float:
        a, b = params
        q = expit(np.clip(a + b * eta, -30.0, 30.0))
        loss = -np.mean(y * np.log(np.clip(q, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - q, eps, 1.0)))
        reg = 1e-5 * (a * a + (b - 1.0) ** 2)
        return float(loss + reg)

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return {
        "citl": None if np.isnan(citl) else round(citl, 4),
        "calibration_intercept": round(float(res.x[0]), 4),
        "calibration_slope": round(float(res.x[1]), 4),
    }


def compute_calibration_report(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10) -> dict[str, Any]:
    """Full calibration diagnostic bundle (ECE uniform/quantile, MCE, Brier + Murphy, Log-loss, CITL/Slope)."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)

    ece_u, mce_u, bins_u, murphy_u = _bin_calibration(y, p, n_bins=n_bins, strategy="uniform")
    ece_q, mce_q, bins_q, murphy_q = _bin_calibration(y, p, n_bins=n_bins, strategy="quantile")
    cox = calibration_slope_and_citl(y, p)

    mean_p = float(np.mean(p))
    obs_rate = float(np.mean(y))
    oe_ratio = float(obs_rate / mean_p) if mean_p > 1e-9 else None
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6), labels=[0, 1]))

    return {
        "n": int(len(y)),
        "observed_prevalence": round(obs_rate, 4),
        "mean_predicted_prob": round(mean_p, 4),
        "mean_prob_bias": round(mean_p - obs_rate, 4),
        "oe_ratio": None if oe_ratio is None else round(oe_ratio, 4),
        "ece_uniform": round(ece_u, 4),
        "mce_uniform": round(mce_u, 4),
        "ece_quantile": round(ece_q, 4),
        "mce_quantile": round(mce_q, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "murphy_quantile": murphy_q,
        "murphy_uniform": murphy_u,
        **cox,
        "bins_uniform": bins_u,
        "bins_quantile": bins_q,
    }


def compute_point_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """Discrimination, thresholded classification, and calibration point metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)
    specificity = float(tn / n_neg) if n_neg > 0 else None
    auc = float(roc_auc_score(y, p)) if (n_pos > 0 and n_neg > 0) else None
    ece_u, _, _, _ = _bin_calibration(y, p, n_bins=10, strategy="uniform")
    ece_q, _, _, _ = _bin_calibration(y, p, n_bins=10, strategy="quantile")
    rec_w = wilson_ci(int(tp), n_pos)
    prec_w = wilson_ci(int(tp), int(tp + fp))
    spec_w = wilson_ci(int(tn), n_neg)

    return {
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": round(float(np.mean(y)), 4),
        "threshold": round(float(threshold), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4) if n_pos > 0 else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4) if (tp + fp) > 0 else None,
        "specificity": None if specificity is None else round(specificity, 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4) if n_pos > 0 else None,
        "roc_auc": None if auc is None else round(auc, 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "ece_uniform": round(float(ece_u), 4),
        "ece_quantile": round(float(ece_q), 4),
        "mean_predicted_prob": round(float(np.mean(p)), 4),
        "mean_prob_bias": round(float(np.mean(p) - np.mean(y)), 4),
        "wilson_95ci": {
            "recall": list(rec_w),
            "precision": list(prec_w),
            "specificity": list(spec_w),
        },
        "small_sample_warning": bool(len(y) < 30 or n_pos < 15 or n_neg < 15),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | int | None]]:
    """Patient-level nonparametric percentile bootstrap CIs across headline metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    n = len(y)
    rng = np.random.default_rng(seed)

    keys = ("recall", "precision", "specificity", "accuracy", "f1", "roc_auc", "brier_score", "ece_quantile", "ece_uniform")
    samples: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        pred = (pb >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
        pos = tp + fn
        neg = tn + fp
        flagged = tp + fp

        samples["accuracy"].append(float((tp + tn) / n))
        samples["brier_score"].append(float(np.mean((pb - yb) ** 2)))
        if pos > 0:
            rec = float(tp / pos)
            samples["recall"].append(rec)
            if flagged > 0:
                prec = float(tp / flagged)
                samples["precision"].append(prec)
                samples["f1"].append(float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
        if neg > 0:
            samples["specificity"].append(float(tn / neg))
        if pos > 0 and neg > 0:
            samples["roc_auc"].append(float(roc_auc_score(yb, pb)))

        ece_q, _, _, _ = _bin_calibration(yb, pb, n_bins=10, strategy="quantile")
        ece_u, _, _, _ = _bin_calibration(yb, pb, n_bins=10, strategy="uniform")
        samples["ece_quantile"].append(float(ece_q))
        samples["ece_uniform"].append(float(ece_u))

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    out: dict[str, dict[str, float | int | None]] = {}
    for k, vals in samples.items():
        if not vals:
            out[k] = {"ci_low": None, "ci_high": None, "std": None, "valid_draws": 0}
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "ci_low": round(float(np.percentile(arr, lo_q)), 4),
            "ci_high": round(float(np.percentile(arr, hi_q)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "valid_draws": int(len(arr)),
        }
    return out


def evaluate_subgroups(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
    *,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Compute subgroup performance & bootstrap recall CIs across Hospital, Sex, Age, Missingness, and Chest Pain."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(proba, dtype=float)

    age = meta["age_raw"].to_numpy(dtype=float)
    age_band = np.where(age < 50, "<50", np.where(age < 60, "50-59", ">=60"))

    miss = meta["missing_count"].to_numpy(dtype=int)
    miss_band = np.where(miss == 0, "0 missing", np.where(miss <= 2, "1-2 missing", "3+ missing"))

    groupings: dict[str, tuple[np.ndarray, list[str]]] = {
        "hospital": (meta["dataset"].to_numpy(dtype=str), ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]),
        "sex": (meta["sex_label"].to_numpy(dtype=str), ["Female", "Male"]),
        "age_band": (age_band, ["<50", "50-59", ">=60"]),
        "missingness": (miss_band, ["0 missing", "1-2 missing", "3+ missing"]),
        "chest_pain": (
            meta["cp_label"].to_numpy(dtype=str),
            ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
        ),
    }

    results: dict[str, list[dict[str, Any]]] = {}
    for group_key, (labels, order) in groupings.items():
        rows: list[dict[str, Any]] = []
        for g_idx, name in enumerate(order):
            mask = labels == name
            if int(mask.sum()) == 0:
                continue
            pm = compute_point_metrics(y_arr[mask], p_arr[mask], threshold)
            ci = bootstrap_confidence_intervals(
                y_arr[mask],
                p_arr[mask],
                threshold,
                n_bootstrap=n_bootstrap,
                seed=seed + 100 + g_idx,
            )
            cal = calibration_slope_and_citl(y_arr[mask], p_arr[mask])
            rows.append({
                "subgroup": name,
                **pm,
                "citl": cal["citl"],
                "calibration_slope": cal["calibration_slope"],
                "bootstrap_95ci": {
                    "recall": [ci["recall"]["ci_low"], ci["recall"]["ci_high"]],
                    "precision": [ci["precision"]["ci_low"], ci["precision"]["ci_high"]],
                    "roc_auc": [ci["roc_auc"]["ci_low"], ci["roc_auc"]["ci_high"]],
                    "ece_quantile": [ci["ece_quantile"]["ci_low"], ci["ece_quantile"]["ci_high"]],
                },
            })
        results[group_key] = rows
    return results


def dataset_shift_summary(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize prevalence, demographic, and missingness shifts across the 4 UCI hospitals."""
    rows: list[dict[str, Any]] = []
    for site in ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]:
        mask = meta["dataset"] == site
        xs = X.loc[mask]
        ys = y.loc[mask]
        ms = meta.loc[mask]
        miss_by_col = {col: round(float(xs[col].isna().mean() * 100.0), 1) for col in FEATURES}
        rows.append({
            "hospital": site,
            "n": int(len(ys)),
            "positives": int(ys.sum()),
            "prevalence": round(float(ys.mean()), 4),
            "female_pct": round(float((ms["sex_label"] == "Female").mean() * 100.0), 1),
            "mean_age": round(float(ms["age_raw"].mean()), 1),
            "asymptomatic_pct": round(float((ms["cp_label"] == "asymptomatic").mean() * 100.0), 1),
            "mean_missing_features": round(float(ms["missing_count"].mean()), 2),
            "pct_rows_with_any_missing": round(float((ms["missing_count"] > 0).mean() * 100.0), 1),
            "missing_pct_by_feature": miss_by_col,
        })
    return rows


def run_loho_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Leave-One-Hospital-Out (LOHO) evaluation across all 4 hospitals (zero hospital overlap).

    Evaluates:
    1. Baseline 3 model families (LR, RF, XGB) under LOHO.
    2. Harmonization & calibration candidates under LOHO (plus stealth missingness-proxy ablation
       and target-prevalence Bayes prior-shift diagnostic bound).
    """
    hospitals = meta["dataset"].to_numpy(dtype=str)
    sites = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    y_arr = y.to_numpy(dtype=int)

    # Part 1: Model families under LOHO (matching src/train.py specs).
    model_family_loho: dict[str, Any] = {}
    for model_name in ("Logistic Regression", "Random Forest", "XGBoost"):
        pooled_proba = np.zeros(len(y_arr), dtype=float)
        pooled_pred_tuned = np.zeros(len(y_arr), dtype=int)
        per_site: list[dict[str, Any]] = []
        for s_idx, site in enumerate(sites):
            te_mask = hospitals == site
            tr_mask = ~te_mask
            pipe = build_models()[model_name]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof = cross_val_predict(pipe, X.loc[tr_mask], y_arr[tr_mask], cv=cv, method="predict_proba")[:, 1]
            thr = pick_recall_threshold(y_arr[tr_mask], oof, TARGET_RECALL)
            pipe.fit(X.loc[tr_mask], y_arr[tr_mask])
            p_site = pipe.predict_proba(X.loc[te_mask])[:, 1]
            pooled_proba[te_mask] = p_site
            pooled_pred_tuned[te_mask] = (p_site >= thr).astype(int)

            pm_tuned = compute_point_metrics(y_arr[te_mask], p_site, thr)
            pm_039 = compute_point_metrics(y_arr[te_mask], p_site, 0.39)
            cal_site = compute_calibration_report(y_arr[te_mask], p_site)
            ci_site = bootstrap_confidence_intervals(
                y_arr[te_mask], p_site, thr, n_bootstrap=n_bootstrap, seed=seed + 200 + s_idx
            )
            per_site.append({
                "held_out_hospital": site,
                "train_hospitals": [s for s in sites if s != site],
                "train_n": int(tr_mask.sum()),
                "train_prevalence": round(float(y_arr[tr_mask].mean()), 4),
                "cv_tuned_threshold": thr,
                "at_cv_threshold": pm_tuned,
                "at_baseline_039": pm_039,
                "calibration": {
                    "ece_uniform": cal_site["ece_uniform"],
                    "ece_quantile": cal_site["ece_quantile"],
                    "brier_score": cal_site["brier_score"],
                    "citl": cal_site["citl"],
                    "calibration_slope": cal_site["calibration_slope"],
                    "mean_prob_bias": cal_site["mean_prob_bias"],
                    "oe_ratio": cal_site["oe_ratio"],
                },
                "bootstrap_95ci_at_cv_threshold": {
                    "recall": [ci_site["recall"]["ci_low"], ci_site["recall"]["ci_high"]],
                    "roc_auc": [ci_site["roc_auc"]["ci_low"], ci_site["roc_auc"]["ci_high"]],
                    "ece_quantile": [ci_site["ece_quantile"]["ci_low"], ci_site["ece_quantile"]["ci_high"]],
                },
            })

        pooled_cal = compute_calibration_report(y_arr, pooled_proba)
        pooled_pm_039 = compute_point_metrics(y_arr, pooled_proba, 0.39)
        pooled_ci_039 = bootstrap_confidence_intervals(y_arr, pooled_proba, 0.39, n_bootstrap=n_bootstrap, seed=seed + 300)
        macro_auc = round(float(np.mean([r["at_cv_threshold"]["roc_auc"] for r in per_site])), 4)
        macro_recall = round(float(np.mean([r["at_cv_threshold"]["recall"] for r in per_site])), 4)
        macro_ece_q = round(float(np.mean([r["calibration"]["ece_quantile"] for r in per_site])), 4)
        macro_brier = round(float(np.mean([r["calibration"]["brier_score"] for r in per_site])), 4)
        pooled_tuned_recall = round(float(recall_score(y_arr, pooled_pred_tuned)), 4)
        pooled_tuned_prec = round(float(precision_score(y_arr, pooled_pred_tuned)), 4)

        model_family_loho[model_name] = {
            "per_hospital": per_site,
            "macro_avg": {
                "roc_auc": macro_auc,
                "recall_at_cv_threshold": macro_recall,
                "ece_quantile": macro_ece_q,
                "brier_score": macro_brier,
            },
            "pooled_loho_at_039": pooled_pm_039,
            "pooled_loho_bootstrap_95ci_at_039": {
                k: [v["ci_low"], v["ci_high"]] for k, v in pooled_ci_039.items()
            },
            "pooled_loho_at_per_fold_cv_thresholds": {
                "recall": pooled_tuned_recall,
                "precision": pooled_tuned_prec,
            },
            "pooled_calibration": pooled_cal,
            "_pooled_proba": pooled_proba,  # stripped before JSON dump
        }

    # Part 2: Hospital harmonization & calibration strategies under LOHO AND nested train CV.
    strategies: list[dict[str, Any]] = [
        {
            "id": "pooled_baseline_lr",
            "label": "Pooled Baseline LR (unweighted, uncalibrated)",
            "weight_mode": "uniform",
            "calibration": "none",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "site_weighted_lr",
            "label": "Inverse-Site-Weighted LR",
            "weight_mode": "inverse_site",
            "calibration": "none",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "platt_calibrated_lr",
            "label": "Pooled LR + OOF Platt Scaling",
            "weight_mode": "uniform",
            "calibration": "platt",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "isotonic_calibrated_lr",
            "label": "Pooled LR + OOF Isotonic Calibration",
            "weight_mode": "uniform",
            "calibration": "isotonic",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "site_weighted_platt_lr",
            "label": "Inverse-Site-Weighted LR + OOF Platt",
            "weight_mode": "inverse_site",
            "calibration": "platt",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "site_deconfounded_platt_lr",
            "label": "Site-Deconfounded IPW LR + OOF Platt",
            "weight_mode": "site_deconfounded",
            "calibration": "platt",
            "add_missing_indicators": False,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "missing_indicator_ablation_lr",
            "label": "Ablation: Pooled LR + Missingness Indicators (Stealth Site-Proxy Test)",
            "weight_mode": "uniform",
            "calibration": "none",
            "add_missing_indicators": True,
            "inference_hospital_id_free": True,
            "oracle_prior_shift": False,
        },
        {
            "id": "prior_shift_oracle_bound",
            "label": "Diagnostic Bound: Site-Deconfounded Platt + Target-Site Prevalence Recalibration",
            "weight_mode": "site_deconfounded",
            "calibration": "platt",
            "add_missing_indicators": False,
            "inference_hospital_id_free": False,
            "oracle_prior_shift": True,
        },
    ]

    # Also compute nested 5-fold CV strictly on X_train (n=736) for each strategy.
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    h_train = meta.loc[X_train.index, "dataset"].to_numpy(dtype=str)
    y_train_arr = y_train.to_numpy(dtype=int)

    harmonization_results: list[dict[str, Any]] = []
    for strat in strategies:
        # 1) Nested 5-fold CV inside X_train (n=736) — 184 test rows strictly untouched.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_tr = np.zeros(len(X_train), dtype=float)
        for tr_i, va_i in skf.split(X_train, y_train_arr):
            clf = HarmonizedCalibratedClassifier(
                weight_mode=strat["weight_mode"],
                calibration=strat["calibration"],
                add_missing_indicators=strat["add_missing_indicators"],
                cv_folds=5,
                random_state=seed,
            )
            clf.fit(X_train.iloc[tr_i], y_train_arr[tr_i], hospitals=h_train[tr_i])
            p_va = clf.predict_proba(X_train.iloc[va_i])[:, 1]
            if strat["oracle_prior_shift"]:
                # Adjust each validation patient by their true hospital prevalence (diagnostic bound only).
                for site in sites:
                    sm = h_train[va_i] == site
                    if sm.any():
                        site_prev = float(y_arr[hospitals == site].mean())
                        p_va[sm] = apply_bayes_prior_shift(p_va[sm], clf.train_prevalence_, site_prev)
            oof_tr[va_i] = p_va

        tr_thr = pick_recall_threshold(y_train_arr, oof_tr, TARGET_RECALL)
        tr_pm = compute_point_metrics(y_train_arr, oof_tr, tr_thr)
        tr_cal = compute_calibration_report(y_train_arr, oof_tr)

        # 2) Leave-One-Hospital-Out (LOHO) across all 4 hospitals (zero hospital leakage).
        loho_p = np.zeros(len(y_arr), dtype=float)
        site_rows: list[dict[str, Any]] = []
        for site in sites:
            te_mask = hospitals == site
            tr_mask = ~te_mask
            clf = HarmonizedCalibratedClassifier(
                weight_mode=strat["weight_mode"],
                calibration=strat["calibration"],
                add_missing_indicators=strat["add_missing_indicators"],
                cv_folds=5,
                random_state=seed,
            )
            clf.fit(X.loc[tr_mask], y_arr[tr_mask], hospitals=hospitals[tr_mask])
            p_site = clf.predict_proba(X.loc[te_mask])[:, 1]
            if strat["oracle_prior_shift"]:
                p_site = apply_bayes_prior_shift(
                    p_site,
                    source_prevalence=float(y_arr[tr_mask].mean()),
                    target_prevalence=float(y_arr[te_mask].mean()),
                )
            loho_p[te_mask] = p_site
            sc = compute_calibration_report(y_arr[te_mask], p_site)
            sp = compute_point_metrics(y_arr[te_mask], p_site, 0.39)
            site_rows.append({
                "hospital": site,
                "roc_auc": sp["roc_auc"],
                "recall_at_039": sp["recall"],
                "precision_at_039": sp["precision"],
                "brier_score": sc["brier_score"],
                "ece_quantile": sc["ece_quantile"],
                "ece_uniform": sc["ece_uniform"],
                "citl": sc["citl"],
                "mean_prob_bias": sc["mean_prob_bias"],
            })

        loho_cal = compute_calibration_report(y_arr, loho_p)
        loho_pm = compute_point_metrics(y_arr, loho_p, 0.39)
        harmonization_results.append({
            "id": strat["id"],
            "label": strat["label"],
            "inference_hospital_id_free": strat["inference_hospital_id_free"],
            "oracle_prior_shift": strat["oracle_prior_shift"],
            "nested_train_cv_n736": {
                "cv_threshold_for_90pct_recall": tr_thr,
                "roc_auc": tr_pm["roc_auc"],
                "recall": tr_pm["recall"],
                "precision": tr_pm["precision"],
                "brier_score": tr_cal["brier_score"],
                "ece_quantile": tr_cal["ece_quantile"],
                "ece_uniform": tr_cal["ece_uniform"],
                "log_loss": tr_cal["log_loss"],
                "citl": tr_cal["citl"],
                "calibration_slope": tr_cal["calibration_slope"],
            },
            "loho_zero_site_overlap_n920": {
                "macro_roc_auc": round(float(np.mean([r["roc_auc"] for r in site_rows])), 4),
                "macro_brier": round(float(np.mean([r["brier_score"] for r in site_rows])), 4),
                "macro_ece_quantile": round(float(np.mean([r["ece_quantile"] for r in site_rows])), 4),
                "macro_abs_prob_bias": round(float(np.mean([abs(r["mean_prob_bias"]) for r in site_rows])), 4),
                "pooled_roc_auc": loho_pm["roc_auc"],
                "pooled_brier": loho_cal["brier_score"],
                "pooled_ece_quantile": loho_cal["ece_quantile"],
                "pooled_ece_uniform": loho_cal["ece_uniform"],
                "pooled_log_loss": loho_cal["log_loss"],
                "pooled_recall_at_039": loho_pm["recall"],
                "pooled_precision_at_039": loho_pm["precision"],
                "per_hospital": site_rows,
            },
        })

    return {
        "model_families": model_family_loho,
        "harmonization_strategies": harmonization_results,
    }


def _plot_calibration_curves(
    oof_train_models: dict[str, dict[str, Any]],
    loho_lr_cal: dict[str, Any],
    baseline_hist_cal: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    # Panel 1: Unspent OOF 5-fold CV on X_train (n=736)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.1, label="Perfect calibration")
    palette = {"Logistic Regression": BLUE, "Random Forest": GREEN, "XGBoost": ORANGE}
    for name, info in oof_train_models.items():
        bins = info["calibration"]["bins_quantile"]
        xs = [b["mean_predicted"] for b in bins]
        ys = [b["observed_rate"] for b in bins]
        ax.plot(
            xs, ys, "o-", color=palette[name], lw=1.8, markersize=5,
            label=f"{name}\n(ECE={info['calibration']['ece_quantile']:.3f}, Brier={info['calibration']['brier_score']:.3f})",
        )
    ax.set_title("Unspent 5-Fold OOF CV\n(Train n=736, quantile bins)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disease frequency")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel 2: Leave-One-Hospital-Out (n=920, zero hospital overlap)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.1, label="Perfect calibration")
    bins = loho_lr_cal["bins_quantile"]
    xs = [b["mean_predicted"] for b in bins]
    ys = [b["observed_rate"] for b in bins]
    yerr_lo = [max(0.0, b["observed_rate"] - (b["observed_ci_low"] or b["observed_rate"])) for b in bins]
    yerr_hi = [max(0.0, (b["observed_ci_high"] or b["observed_rate"]) - b["observed_rate"]) for b in bins]
    ax.errorbar(
        xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-", color=RED, lw=1.8, capsize=3,
        label=f"LR LOHO (n=920)\nECE={loho_lr_cal['ece_quantile']:.3f}, Slope={loho_lr_cal['calibration_slope']:.2f}",
    )
    ax.set_title("Leave-One-Hospital-Out (LOHO)\n(Zero site overlap, 95% Wilson bars)")
    ax.set_xlabel("Mean predicted probability")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel 3: Frozen Baseline Read-Only Audit on Historical Test (n=184)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.1, label="Perfect calibration")
    bins = baseline_hist_cal["bins_quantile"]
    xs = [b["mean_predicted"] for b in bins]
    ys = [b["observed_rate"] for b in bins]
    yerr_lo = [max(0.0, b["observed_rate"] - (b["observed_ci_low"] or b["observed_rate"])) for b in bins]
    yerr_hi = [max(0.0, (b["observed_ci_high"] or b["observed_rate"]) - b["observed_rate"]) for b in bins]
    ax.errorbar(
        xs, ys, yerr=[yerr_lo, yerr_hi], fmt="s-", color=PURPLE, lw=1.8, capsize=3,
        label=f"Frozen Baseline (hist n=184)\nECE={baseline_hist_cal['ece_quantile']:.3f}, Slope={baseline_hist_cal['calibration_slope']:.2f}",
    )
    ax.set_title("Frozen Baseline Historical Audit\n(Read-only n=184, 95% Wilson bars)")
    ax.set_xlabel("Mean predicted probability")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    for a in axes:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.grid(alpha=0.2)
    fig.suptitle("Calibration Reliability Curves (Not Clinical Validation)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_hospital_source_shift(loho_lr_per_site: list[dict[str, Any]], out_path: Path) -> None:
    sites = [r["held_out_hospital"] for r in loho_lr_per_site]
    aucs = [r["at_cv_threshold"]["roc_auc"] for r in loho_lr_per_site]
    recalls = [r["at_cv_threshold"]["recall"] for r in loho_lr_per_site]
    rec_lo = [r["bootstrap_95ci_at_cv_threshold"]["recall"][0] for r in loho_lr_per_site]
    rec_hi = [r["bootstrap_95ci_at_cv_threshold"]["recall"][1] for r in loho_lr_per_site]
    biases = [r["calibration"]["mean_prob_bias"] * 100.0 for r in loho_lr_per_site]
    eces = [r["calibration"]["ece_quantile"] for r in loho_lr_per_site]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    y_pos = np.arange(len(sites))

    # Panel 1: LOHO ROC-AUC & Recall with 95% CI
    ax = axes[0]
    ax.barh(y_pos - 0.16, aucs, height=0.30, color=BLUE, label="ROC-AUC")
    ax.barh(y_pos + 0.16, recalls, height=0.30, color=RED, label="Recall (at 3-site CV thr)")
    ax.errorbar(
        recalls, y_pos + 0.16,
        xerr=[[max(0.0, r - lo) for r, lo in zip(recalls, rec_lo)],
              [max(0.0, hi - r) for r, hi in zip(recalls, rec_hi)]],
        fmt="none", ecolor="black", elinewidth=1.2, capsize=3,
    )
    ax.axvline(0.90, color=GREY, ls="--", lw=1, label="90% recall target")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sites)
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("LOHO Discrimination & Recall\n(Unseen Hospital)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)

    # Panel 2: Mean probability bias (Predicted - Observed, percentage points)
    ax = axes[1]
    colors = [RED if b < 0 else ORANGE for b in biases]
    ax.barh(y_pos, biases, color=colors, height=0.52)
    ax.axvline(0.0, color="black", lw=0.9)
    for i, b in enumerate(biases):
        offset = 1.5 if b >= 0 else -1.5
        ha = "left" if b >= 0 else "right"
        ax.text(b + offset, i, f"{b:+.1f} pp", va="center", ha=ha, fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(sites))
    ax.set_xlim(-45, 35)
    ax.set_xlabel("Mean predicted − observed prevalence (pp)")
    ax.set_title("Base-Rate / Calibration Bias\n(+ over-predicts, − under-predicts)")

    # Panel 3: Quantile ECE per unseen hospital
    ax = axes[2]
    ax.barh(y_pos, eces, color=PURPLE, height=0.52)
    for i, e in enumerate(eces):
        ax.text(e + 0.008, i, f"{e:.3f}", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(sites))
    ax.set_xlim(0, max(eces) + 0.12)
    ax.set_xlabel("Expected Calibration Error (quantile ECE)")
    ax.set_title("Calibration Error by Unseen Hospital\n(Lower is better)")

    fig.suptitle("4-Hospital Source-Shift Analysis (Leave-One-Hospital-Out LR)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_subgroup_forest(loho_subgroups: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    rows: list[tuple[str, float, float, float, int, bool]] = []
    for section, title in [
        ("hospital", "Hospital"),
        ("sex", "Sex"),
        ("age_band", "Age"),
        ("missingness", "Missing"),
    ]:
        for item in loho_subgroups.get(section, []):
            rec = item["recall"]
            lo, hi = item["bootstrap_95ci"]["recall"]
            if rec is None or lo is None or hi is None:
                continue
            label = f"{title}: {item['subgroup']} (n={item['n']}, pos={item['n_pos']})"
            rows.append((label, rec, lo, hi, item["n"], item["small_sample_warning"]))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y_pos = np.arange(len(rows))[::-1]
    for y_i, (label, rec, lo, hi, _n, warn) in zip(y_pos, rows):
        col = ORANGE if warn else BLUE
        ax.errorbar(
            rec, y_i,
            xerr=[[max(0.0, rec - lo)], [max(0.0, hi - rec)]],
            fmt="o", color=col, ecolor=col, capsize=3.5, markersize=6, lw=1.6,
        )
        suffix = " *" if warn else ""
        ax.text(min(1.02, hi + 0.015), y_i, f"{rec:.1%} [{lo:.1%}, {hi:.1%}]{suffix}", va="center", fontsize=8.5)

    ax.axvline(0.90, color=RED, ls="--", lw=1.1, label="90% screening recall target")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.35, 1.24)
    ax.set_xlabel("Recall (sensitivity) with 95% bootstrap CI at threshold 0.39")
    ax.set_title("Subgroup Recall & Uncertainty Under Leave-One-Hospital-Out (LOHO, n=920)\n(* small-sample warning: n<30 or n_pos<15 or n_neg<15)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=8.5)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_harmonization_comparison(harmonization_strategies: list[dict[str, Any]], out_path: Path) -> None:
    short_names = {
        "pooled_baseline_lr": "Pooled Baseline LR",
        "site_weighted_lr": "Inverse-Site Weighted",
        "platt_calibrated_lr": "Pooled + Platt",
        "isotonic_calibrated_lr": "Pooled + Isotonic",
        "site_weighted_platt_lr": "Site-Weighted + Platt",
        "site_deconfounded_platt_lr": "Site-Deconfounded + Platt",
        "missing_indicator_ablation_lr": "Ablation: +Missing Flags",
        "prior_shift_oracle_bound": "Oracle Prior-Shift Bound*",
    }
    labels = [short_names.get(s["id"], s["id"]) for s in harmonization_strategies]
    cv_auc = [s["nested_train_cv_n736"]["roc_auc"] for s in harmonization_strategies]
    loho_auc = [s["loho_zero_site_overlap_n920"]["macro_roc_auc"] for s in harmonization_strategies]
    loho_ece = [s["loho_zero_site_overlap_n920"]["macro_ece_quantile"] for s in harmonization_strategies]
    loho_brier = [s["loho_zero_site_overlap_n920"]["macro_brier"] for s in harmonization_strategies]

    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    # Panel 1: Nested Train CV vs LOHO Macro ROC-AUC (exposes missingness-proxy gap!)
    ax = axes[0]
    ax.barh(y + 0.16, cv_auc, height=0.30, color=BLUE, label="Pooled 5-fold CV (train n=736)")
    ax.barh(y - 0.16, loho_auc, height=0.30, color=RED, label="LOHO Macro AUC (unseen site)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.75, 0.94)
    ax.set_xlabel("ROC-AUC (higher is better)")
    ax.set_title("Pooled CV vs Unseen-Hospital LOHO Discrimination")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8.5, frameon=False)
    ax.grid(axis="x", alpha=0.2)

    # Panel 2: LOHO Macro ECE & Macro Brier
    ax = axes[1]
    ax.barh(y + 0.16, loho_ece, height=0.30, color=PURPLE, label="LOHO Macro Quantile ECE")
    ax.barh(y - 0.16, loho_brier, height=0.30, color=ORANGE, label="LOHO Macro Brier Score")
    for yi, e, b in zip(y, loho_ece, loho_brier):
        ax.text(e + 0.004, yi + 0.16, f"{e:.3f}", va="center", fontsize=8)
        ax.text(b + 0.004, yi - 0.16, f"{b:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.0, max(max(loho_ece), max(loho_brier)) + 0.08)
    ax.set_xlabel("Calibration / Probability Error (lower is better)")
    ax.set_title("LOHO Calibration Error Across Harmonization Strategies")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8.5, frameon=False)
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Hospital Harmonization Comparison (*Oracle bound requires target-site prevalence)", y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_statistical_deepening(
    *,
    data_path: Path = DATA_PATH,
    reports_dir: Path = REPORTS_DIR,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute the full statistical deepening pipeline and write JSON, Markdown, and PNG reports."""
    baseline_sha_before = verify_baseline_intact()
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_dataset_with_metadata(data_path)
    data_sha = file_sha256(data_path)

    # Exact baseline 80/20 split (matching src/train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    meta_train = meta.loc[X_train.index]
    meta_test = meta.loc[X_test.index]

    # 1. Read-only audit of frozen baseline artifact on historical n=184 split.
    baseline_bundle = joblib.load(BASELINE_MODEL_PATH)
    baseline_pipe = baseline_bundle["pipeline"]
    baseline_thr = float(baseline_bundle["threshold"])
    baseline_test_proba = baseline_pipe.predict_proba(X_test)[:, 1]

    baseline_hist_pm = compute_point_metrics(y_test.to_numpy(), baseline_test_proba, baseline_thr)
    baseline_hist_ci = bootstrap_confidence_intervals(
        y_test.to_numpy(), baseline_test_proba, baseline_thr, n_bootstrap=n_bootstrap, seed=seed
    )
    baseline_hist_cal = compute_calibration_report(y_test.to_numpy(), baseline_test_proba)
    baseline_hist_subgroups = evaluate_subgroups(
        y_test, baseline_test_proba, meta_test, baseline_thr, n_bootstrap=min(1000, n_bootstrap), seed=seed + 10
    )

    # 2. Unspent 5-fold out-of-fold CV calibration & bootstrap CIs strictly on X_train (n=736).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_train_models: dict[str, Any] = {}
    lr_oof_proba = None
    for m_idx, (name, pipe) in enumerate(build_models().items()):
        oof_p = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = pick_recall_threshold(y_train.to_numpy(), oof_p, TARGET_RECALL)
        pm = compute_point_metrics(y_train.to_numpy(), oof_p, thr)
        ci = bootstrap_confidence_intervals(
            y_train.to_numpy(), oof_p, thr, n_bootstrap=n_bootstrap, seed=seed + 50 + m_idx
        )
        cal = compute_calibration_report(y_train.to_numpy(), oof_p)
        oof_train_models[name] = {
            "cv_threshold": thr,
            "point_metrics": pm,
            "bootstrap_95ci": ci,
            "calibration": cal,
        }
        if name == "Logistic Regression":
            lr_oof_proba = oof_p

    oof_train_subgroups = evaluate_subgroups(
        y_train, lr_oof_proba, meta_train, oof_train_models["Logistic Regression"]["cv_threshold"],
        n_bootstrap=min(1000, n_bootstrap), seed=seed + 80,
    )

    # 3. 4-Hospital Dataset Shift + Leave-One-Hospital-Out (LOHO) + Harmonization study.
    shift_table = dataset_shift_summary(X, y, meta)
    loho = run_loho_evaluation(X, y, meta, n_bootstrap=min(1000, n_bootstrap), seed=seed)
    loho_lr_proba = loho["model_families"]["Logistic Regression"]["_pooled_proba"]
    loho_subgroups = evaluate_subgroups(
        y, loho_lr_proba, meta, baseline_thr, n_bootstrap=min(1000, n_bootstrap), seed=seed + 400
    )

    # Strip internal numpy arrays before serialization.
    for m_data in loho["model_families"].values():
        m_data.pop("_pooled_proba", None)

    # 4. Generate publication-grade plots in reports/.
    _plot_calibration_curves(
        oof_train_models,
        loho["model_families"]["Logistic Regression"]["pooled_calibration"],
        baseline_hist_cal,
        reports_dir / "calibration_curves.png",
    )
    _plot_hospital_source_shift(
        loho["model_families"]["Logistic Regression"]["per_hospital"],
        reports_dir / "hospital_source_shift.png",
    )
    _plot_subgroup_forest(
        loho_subgroups,
        reports_dir / "subgroup_recall_ci.png",
    )
    _plot_harmonization_comparison(
        loho["harmonization_strategies"],
        reports_dir / "harmonization_comparison.png",
    )

    # Verify frozen baseline untouched after full run.
    baseline_sha_after = verify_baseline_intact()
    assert baseline_sha_before == baseline_sha_after

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "metadata": {
            "generated_utc": now_iso,
            "dataset": "UCI Heart Disease (Janosi et al. 1988 / Detrano et al. 1989; CC BY 4.0)",
            "n_total": int(len(X)),
            "n_hospitals": 4,
            "hospitals": ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
            "dataset_sha256": data_sha,
            "baseline_artifact": "models/heart_model.joblib",
            "baseline_sha256": baseline_sha_after,
            "baseline_threshold": baseline_thr,
            "sklearn_version": sklearn.__version__,
            "bootstrap_draws": n_bootstrap,
            "seed": seed,
            "repro_command": "python -m src.statistics",
            "clinical_validation_disclaimer": (
                "Statistical audit of archival 1981-1988 angiography-referral patients. "
                "This is NOT clinical validation and NOT a medical device."
            ),
            "evaluation_discipline_note": (
                "All model/calibration/harmonization comparisons use unspent 5-fold nested CV on "
                "n=736 training rows and zero-hospital-overlap Leave-One-Hospital-Out (LOHO) CV on "
                "all 4 hospitals. The historical n=184 test split is audited read-only for the "
                "already-frozen baseline artifact and is never used for tuning, calibration, or selection."
            ),
        },
        "dataset_shift_by_hospital": shift_table,
        "frozen_baseline_historical_audit_n184": {
            "split": "Historical 80/20 row-stratified test split (read-only audit; not hospital-grouped)",
            "point_metrics": baseline_hist_pm,
            "bootstrap_95ci": baseline_hist_ci,
            "calibration": baseline_hist_cal,
            "subgroups": baseline_hist_subgroups,
        },
        "unspent_oof_train_cv_n736": {
            "split": "5-fold stratified OOF CV strictly inside n=736 training rows",
            "models": oof_train_models,
            "lr_subgroups": oof_train_subgroups,
        },
        "loho_zero_site_overlap_n920": {
            "grouping_rule": "Group by hospital (`dataset`): train on 3 hospitals, score on 1 unseen hospital",
            "model_families": loho["model_families"],
            "lr_subgroups_at_039": loho_subgroups,
        },
        "harmonization_comparison": loho["harmonization_strategies"],
        "generated_plots": [
            "reports/calibration_curves.png",
            "reports/hospital_source_shift.png",
            "reports/subgroup_recall_ci.png",
            "reports/harmonization_comparison.png",
        ],
    }

    json_path = reports_dir / "statistical_deepening.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = reports_dir / "STATISTICAL_REPORT.md"
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return report


def _fmt_ci(val: float | None, lo: float | None, hi: float | None, pct: bool = True) -> str:
    if val is None:
        return "N/A"
    if pct:
        if lo is None or hi is None:
            return f"{val:.1%}"
        return f"{val:.1%} [{lo:.1%}, {hi:.1%}]"
    if lo is None or hi is None:
        return f"{val:.3f}"
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"


def _render_markdown_report(r: dict[str, Any]) -> str:
    meta = r["metadata"]
    hb = r["frozen_baseline_historical_audit_n184"]
    hp = hb["point_metrics"]
    hci = hb["bootstrap_95ci"]
    hcal = hb["calibration"]
    oof_lr = r["unspent_oof_train_cv_n736"]["models"]["Logistic Regression"]
    loho_lr = r["loho_zero_site_overlap_n920"]["model_families"]["Logistic Regression"]

    lines = [
        "# Statistical Deepening Report: Calibration, Source Shift, Harmonization & Uncertainty",
        "",
        f"> **Disclaimer**: {meta['clinical_validation_disclaimer']}",
        "",
        "## Provenance & Evaluation Discipline",
        f"- **Generated (UTC)**: `{meta['generated_utc']}`",
        f"- **Dataset**: {meta['dataset']} (`n={meta['n_total']}`, SHA-256 `{meta['dataset_sha256'][:16]}...`)",
        f"- **Frozen baseline artifact**: `{meta['baseline_artifact']}` (threshold `{meta['baseline_threshold']:.2f}`, SHA-256 `{meta['baseline_sha256'][:16]}...`, scikit-learn `{meta['sklearn_version']}`) — verified untouched before and after.",
        f"- **Bootstrap**: `B={meta['bootstrap_draws']}` nonparametric draws, `seed={meta['seed']}`.",
        f"- **Reproduce**: `{meta['repro_command']}`",
        f"- **Split & leakage policy**: {meta['evaluation_discipline_note']}",
        "",
        "## 1. Headline Recall & Calibration with 95% Bootstrap Confidence Intervals",
        "",
        "Turning the single historical point estimate (`92.2%` recall on `n=184`, `102` positive patients) into honest uncertainty intervals:",
        "",
        "| Evaluation Regime | Grouping | n (pos) | Recall (95% CI) | Precision (95% CI) | ROC-AUC (95% CI) | Quantile ECE | Brier | CITL | Cal Slope |",
        "|---|---|---:|---|---|---|---|---|---|---|",
        (
            f"| **Frozen Baseline (Historical Audit, Read-Only)** | Row-stratified (all 4 sites mixed) | "
            f"{hp['n']} ({hp['n_pos']}) | "
            f"{_fmt_ci(hp['recall'], hci['recall']['ci_low'], hci['recall']['ci_high'])} | "
            f"{_fmt_ci(hp['precision'], hci['precision']['ci_low'], hci['precision']['ci_high'])} | "
            f"{_fmt_ci(hp['roc_auc'], hci['roc_auc']['ci_low'], hci['roc_auc']['ci_high'], pct=False)} | "
            f"{hcal['ece_quantile']:.3f} | {hcal['brier_score']:.3f} | {hcal['citl']:+.2f} | {hcal['calibration_slope']:.2f} |"
        ),
        (
            f"| **Unspent 5-Fold OOF CV (Train only, LR)** | Row-stratified inside train | "
            f"{oof_lr['point_metrics']['n']} ({oof_lr['point_metrics']['n_pos']}) | "
            f"{_fmt_ci(oof_lr['point_metrics']['recall'], oof_lr['bootstrap_95ci']['recall']['ci_low'], oof_lr['bootstrap_95ci']['recall']['ci_high'])} | "
            f"{_fmt_ci(oof_lr['point_metrics']['precision'], oof_lr['bootstrap_95ci']['precision']['ci_low'], oof_lr['bootstrap_95ci']['precision']['ci_high'])} | "
            f"{_fmt_ci(oof_lr['point_metrics']['roc_auc'], oof_lr['bootstrap_95ci']['roc_auc']['ci_low'], oof_lr['bootstrap_95ci']['roc_auc']['ci_high'], pct=False)} | "
            f"{oof_lr['calibration']['ece_quantile']:.3f} | {oof_lr['calibration']['brier_score']:.3f} | {oof_lr['calibration']['citl']:+.2f} | {oof_lr['calibration']['calibration_slope']:.2f} |"
        ),
        (
            f"| **Leave-One-Hospital-Out (LOHO, LR @ 0.39)** | Hospital-grouped (zero site overlap) | "
            f"{loho_lr['pooled_loho_at_039']['n']} ({loho_lr['pooled_loho_at_039']['n_pos']}) | "
            f"{_fmt_ci(loho_lr['pooled_loho_at_039']['recall'], loho_lr['pooled_loho_bootstrap_95ci_at_039']['recall'][0], loho_lr['pooled_loho_bootstrap_95ci_at_039']['recall'][1])} | "
            f"{_fmt_ci(loho_lr['pooled_loho_at_039']['precision'], loho_lr['pooled_loho_bootstrap_95ci_at_039']['precision'][0], loho_lr['pooled_loho_bootstrap_95ci_at_039']['precision'][1])} | "
            f"{_fmt_ci(loho_lr['pooled_loho_at_039']['roc_auc'], loho_lr['pooled_loho_bootstrap_95ci_at_039']['roc_auc'][0], loho_lr['pooled_loho_bootstrap_95ci_at_039']['roc_auc'][1], pct=False)} | "
            f"{loho_lr['pooled_calibration']['ece_quantile']:.3f} | {loho_lr['pooled_calibration']['brier_score']:.3f} | {loho_lr['pooled_calibration']['citl']:+.2f} | {loho_lr['pooled_calibration']['calibration_slope']:.2f} |"
        ),
        "",
        f"Analytical Wilson 95% CI cross-check on frozen baseline historical recall: `{hp['wilson_95ci']['recall'][0]:.1%}` to `{hp['wilson_95ci']['recall'][1]:.1%}` (closely agrees with bootstrap `{hci['recall']['ci_low']:.1%}` to `{hci['recall']['ci_high']:.1%}`).",
        "",
        "![Calibration curves](calibration_curves.png)",
        "",
        "## 2. 4-Hospital Source Shift (Dataset Shift & Leave-One-Hospital-Out)",
        "",
        "| Held-Out Hospital | n (prev) | Train prev | CV thr | LOHO Recall (95% CI) | LOHO Precision | LOHO ROC-AUC | Prob Bias (pp) | Quantile ECE | CITL |",
        "|---|---:|---:|---:|---|---|---|---|---:|---:|",
    ]

    for row in loho_lr["per_hospital"]:
        at_cv = row["at_cv_threshold"]
        ci = row["bootstrap_95ci_at_cv_threshold"]
        cal = row["calibration"]
        citl_val = cal["citl"]
        citl_s = "N/A" if citl_val is None else f"{citl_val:+.2f}"
        lines.append(
            f"| {row['held_out_hospital']} | {at_cv['n']} ({at_cv['prevalence']:.1%}) | "
            f"{row['train_prevalence']:.1%} | {row['cv_tuned_threshold']:.2f} | "
            f"{_fmt_ci(at_cv['recall'], ci['recall'][0], ci['recall'][1])} | "
            f"{at_cv['precision']:.1%} | "
            f"{_fmt_ci(at_cv['roc_auc'], ci['roc_auc'][0], ci['roc_auc'][1], pct=False)} | "
            f"{cal['mean_prob_bias']*100:+.1f} pp | {cal['ece_quantile']:.3f} | "
            f"{citl_s} |"
        )

    lines.extend([
        "",
        "![Hospital source shift](hospital_source_shift.png)",
        "",
        "## 3. Subgroup Analysis (Leave-One-Hospital-Out LR @ threshold 0.39, n=920)",
        "",
        "| Dimension | Subgroup | n (pos) | Recall (95% CI) | Precision | Specificity | ROC-AUC | Quantile ECE | Note |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ])
    for dim, items in r["loho_zero_site_overlap_n920"]["lr_subgroups_at_039"].items():
        for it in items:
            ci_r = it["bootstrap_95ci"]["recall"]
            note = "Small sample (interpret cautiously)" if it["small_sample_warning"] else ""
            spec_s = "N/A" if it["specificity"] is None else f"{it['specificity']:.1%}"
            auc_s = "N/A" if it["roc_auc"] is None else f"{it['roc_auc']:.3f}"
            prec_s = "N/A" if it["precision"] is None else f"{it['precision']:.1%}"
            lines.append(
                f"| {dim} | {it['subgroup']} | {it['n']} ({it['n_pos']}) | "
                f"{_fmt_ci(it['recall'], ci_r[0], ci_r[1])} | {prec_s} | {spec_s} | {auc_s} | "
                f"{it['ece_quantile']:.3f} | {note} |"
            )

    lines.extend([
        "",
        "![Subgroup recall CIs](subgroup_recall_ci.png)",
        "",
        "## 4. Hospital Harmonization & Stealth Missingness-Proxy Ablation",
        "",
        "| Strategy | Inference Site-ID Free? | Nested Train CV AUC (n=736) | Nested Train CV ECE | LOHO Macro AUC | LOHO Macro ECE | LOHO Macro Brier | LOHO Pooled ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for s in r["harmonization_comparison"]:
        tr = s["nested_train_cv_n736"]
        lo = s["loho_zero_site_overlap_n920"]
        lines.append(
            f"| {s['label']} | {'Yes' if s['inference_hospital_id_free'] else 'No (diagnostic bound)'} | "
            f"{tr['roc_auc']:.4f} | {tr['ece_quantile']:.4f} | {lo['macro_roc_auc']:.4f} | "
            f"{lo['macro_ece_quantile']:.4f} | {lo['macro_brier']:.4f} | {lo['pooled_ece_quantile']:.4f} |"
        )

    lines.extend([
        "",
        "![Harmonization comparison](harmonization_comparison.png)",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    report = run_statistical_deepening()
    hb = report["frozen_baseline_historical_audit_n184"]
    hp = hb["point_metrics"]
    hci = hb["bootstrap_95ci"]
    hcal = hb["calibration"]
    print(f"Baseline SHA-256 verified untouched: {report['metadata']['baseline_sha256']}")
    print(
        f"Frozen Baseline Historical (n={hp['n']}): Recall={hp['recall']:.3f} "
        f"95% CI [{hci['recall']['ci_low']:.3f}, {hci['recall']['ci_high']:.3f}], "
        f"Quantile ECE={hcal['ece_quantile']:.4f}, Brier={hcal['brier_score']:.4f}, "
        f"Slope={hcal['calibration_slope']:.3f}"
    )
    print("\nLOHO Macro Metrics (Zero Hospital Overlap, n=920):")
    for s in report["harmonization_comparison"]:
        lo = s["loho_zero_site_overlap_n920"]
        tr = s["nested_train_cv_n736"]
        print(
            f"  {s['id']:<30} | TrainCV AUC={tr['roc_auc']:.4f} ECE={tr['ece_quantile']:.4f} "
            f"| LOHO Macro AUC={lo['macro_roc_auc']:.4f} ECE={lo['macro_ece_quantile']:.4f} "
            f"Brier={lo['macro_brier']:.4f}"
        )


if __name__ == "__main__":
    main()
