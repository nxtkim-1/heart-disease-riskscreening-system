"""Hospital harmonization, calibration wrappers, and versioned candidate training.

STRICT BASELINE PRESERVATION:
- NEVER overwrites or modifies `models/heart_model.joblib`.
- All candidate models and uncertainty ensembles are saved under `models/candidates/`
  with complete JSON provenance sidecars (data hash, split, grouping rule, seed, date,
  scikit-learn version, and exact command).
- All fitting, calibration, and threshold selection use training / out-of-fold data only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    TARGET,
    build_preprocessor,
    clean_features,
    load_raw,
)

RANDOM_STATE = 42
TARGET_RECALL = 0.90
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.joblib"
BASELINE_EXPECTED_SHA256 = "aa7f0be5903c16ce7070858b91f8b8ac1764fec0c385894ea8a959dbb291d4ec"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "candidates"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_baseline_intact(path: Path = BASELINE_MODEL_PATH) -> str:
    """Confirm the frozen baseline model exists and matches its immutable SHA-256."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen baseline model missing: {path.name}")
    digest = file_sha256(path)
    if digest != BASELINE_EXPECTED_SHA256:
        raise RuntimeError(
            f"Frozen baseline model SHA-256 mismatch! Expected {BASELINE_EXPECTED_SHA256[:12]}, "
            f"got {digest[:12]}."
        )
    return digest


def load_dataset_with_metadata(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load cleaned X, binary target y, and metadata DataFrame (dataset/hospital, raw sex, raw age, missing count)."""
    raw = load_raw(path)
    cleaned = clean_features(raw)
    y = (cleaned["num"] > 0).astype(int).rename(TARGET)
    X = cleaned[FEATURES].copy()
    meta = pd.DataFrame(
        {
            "dataset": raw["dataset"].astype(str),
            "sex_label": raw["sex"].astype(str),
            "age_raw": pd.to_numeric(raw["age"], errors="coerce"),
            "cp_label": raw["cp"].fillna("unknown").astype(str),
            "missing_count": X.isna().sum(axis=1).astype(int),
        },
        index=X.index,
    )
    return X, y, meta


def compute_site_weights(hospitals: np.ndarray | pd.Series, y: np.ndarray | pd.Series, mode: str) -> np.ndarray:
    """Compute training-only sample weights for hospital harmonization.

    Modes:
    - 'uniform': all 1.0 (pooled baseline).
    - 'inverse_site': balances total weight across hospitals (sum per site equal),
      preventing larger hospitals (Cleveland, Hungary) from dominating smaller ones.
    - 'site_deconfounded': inverse-site-and-class weighting w_i ~ 1 / n_{h(i), y_i},
      making hospital identity statistically independent of the label Y in the weighted
      training distribution (P_w(Y=1 | H=h) = 0.50 for all sites) without ever passing
      hospital ID as an input feature.
    """
    h = np.asarray(hospitals, dtype=object)
    y_arr = np.asarray(y, dtype=int)
    n = len(h)
    if mode == "uniform":
        return np.ones(n, dtype=float)

    unique_h = np.unique(h)
    weights = np.ones(n, dtype=float)
    if mode == "inverse_site":
        for site in unique_h:
            mask = h == site
            weights[mask] = n / (len(unique_h) * mask.sum())
        return weights

    if mode == "site_deconfounded":
        # Stabilize extreme minority cells (e.g. Switzerland negatives n=8) with mild cap
        # so variance does not explode while still removing hospital-label confounding.
        for site in unique_h:
            for cls in (0, 1):
                mask = (h == site) & (y_arr == cls)
                cnt = int(mask.sum())
                if cnt > 0:
                    weights[mask] = n / (2.0 * len(unique_h) * cnt)
        # Normalize mean weight to 1.0 and cap extreme ratios at 6x median to preserve stability.
        med = float(np.median(weights))
        weights = np.clip(weights, 0.1 * med, 6.0 * med)
        weights = weights * (n / weights.sum())
        return weights

    raise ValueError(f"Unknown site weighting mode: {mode!r}")


def build_preprocessor_with_missing_indicators() -> ColumnTransformer:
    """Diagnostic ablation preprocessor adding explicit missingness indicator columns.

    Used strictly to test whether missingness indicators act as stealth hospital-ID
    proxies (e.g., chol missing 100% in Switzerland) that inflate pooled CV but hurt LOHO.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary = SimpleImputer(strategy="most_frequent", add_indicator=True)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ])


def build_base_lr_pipeline(*, add_missing_indicators: bool = False, C: float = 1.0) -> Pipeline:
    pre = build_preprocessor_with_missing_indicators() if add_missing_indicators else build_preprocessor()
    return Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=1000, C=C)),
    ])


def fit_platt_params(raw_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit Platt logistic recalibration parameters (a, b) on out-of-fold probabilities:
    p_cal = sigmoid(a + b * logit(p_raw)).
    """
    eps = 1e-6
    p = np.clip(np.asarray(raw_probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    z = logit(p)

    def nll(params: np.ndarray) -> float:
        a, b = params
        lin = np.clip(a + b * z, -30.0, 30.0)
        q = expit(lin)
        # Tiny L2 regularization around identity (a=0, b=1) prevents divergence on small folds.
        loss = -np.mean(y * np.log(np.clip(q, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - q, eps, 1.0)))
        reg = 1e-4 * (a * a + (b - 1.0) ** 2)
        return float(loss + reg)

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    return a_hat, b_hat


def apply_platt(raw_probs: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(np.asarray(raw_probs, dtype=float), eps, 1.0 - eps)
    return expit(np.clip(a + b * logit(p), -30.0, 30.0))


def apply_bayes_prior_shift(
    probs: np.ndarray,
    source_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """Exact Bayes prior-odds intercept adjustment (Saerens et al., 2002).

    logit(p_adj) = logit(p) + logit(pi_target) - logit(pi_source).
    Used as a diagnostic bound in LOHO to decompose calibration error into
    base-rate (prevalence) shift vs slope/covariate shift.
    """
    eps = 1e-6
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    pi_s = float(np.clip(source_prevalence, eps, 1.0 - eps))
    pi_t = float(np.clip(target_prevalence, eps, 1.0 - eps))
    delta = float(logit(pi_t) - logit(pi_s))
    return expit(np.clip(logit(p) + delta, -30.0, 30.0))


def pick_recall_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold in [0.01, 0.95] achieving at least `target_recall` on validation data."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(proba, dtype=float)
    for t in np.arange(0.95, 0.0, -0.01):
        if recall_score(y_arr, (p_arr >= t).astype(int), zero_division=0) >= target_recall:
            return round(float(t), 2)
    return 0.01


class HarmonizedCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Hospital-ID-free classifier supporting site weighting and out-of-fold calibration.

    At inference time, `predict_proba(X)` requires ONLY the 11 clinical features — never
    a hospital identifier. During `fit(X, y, hospitals=...)`, training weights harmonize
    across hospitals and an out-of-fold Platt or Isotonic calibrator maps raw scores to
    calibrated probabilities without overfitting the calibration step.
    """

    def __init__(
        self,
        *,
        weight_mode: str = "uniform",
        calibration: str = "none",
        add_missing_indicators: bool = False,
        C: float = 1.0,
        cv_folds: int = 5,
        random_state: int = RANDOM_STATE,
    ):
        self.weight_mode = weight_mode
        self.calibration = calibration
        self.add_missing_indicators = add_missing_indicators
        self.C = C
        self.cv_folds = cv_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, hospitals: pd.Series | np.ndarray | None = None):
        y_arr = np.asarray(y, dtype=int)
        if hospitals is None:
            h_arr = np.full(len(y_arr), "pooled", dtype=object)
        else:
            h_arr = np.asarray(hospitals, dtype=object)

        base_template = build_base_lr_pipeline(
            add_missing_indicators=self.add_missing_indicators,
            C=self.C,
        )

        # Out-of-fold raw probabilities strictly inside training data for calibration fitting.
        self.platt_a_: float = 0.0
        self.platt_b_: float = 1.0
        self.isotonic_: IsotonicRegression | None = None

        if self.calibration in {"platt", "isotonic"}:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_raw = np.zeros(len(y_arr), dtype=float)
            for tr_idx, va_idx in skf.split(X, y_arr):
                fold_pipe = clone(base_template)
                fold_w = compute_site_weights(h_arr[tr_idx], y_arr[tr_idx], self.weight_mode)
                fold_pipe.fit(X.iloc[tr_idx], y_arr[tr_idx], model__sample_weight=fold_w)
                oof_raw[va_idx] = fold_pipe.predict_proba(X.iloc[va_idx])[:, 1]
            if self.calibration == "platt":
                # Recalibrate back to the unweighted training marginal prevalence.
                self.platt_a_, self.platt_b_ = fit_platt_params(oof_raw, y_arr)
            elif self.calibration == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
                iso.fit(oof_raw, y_arr)
                self.isotonic_ = iso
        elif self.calibration != "none":
            raise ValueError(f"Unsupported calibration method: {self.calibration!r}")

        # Fit final base pipeline on full provided training fold.
        full_w = compute_site_weights(h_arr, y_arr, self.weight_mode)
        self.pipeline_ = clone(base_template)
        self.pipeline_.fit(X, y_arr, model__sample_weight=full_w)
        self.classes_ = np.array([0, 1], dtype=int)
        self.train_prevalence_ = float(np.mean(y_arr))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline_.predict_proba(X)[:, 1]
        if self.calibration == "platt":
            cal = apply_platt(raw, self.platt_a_, self.platt_b_)
        elif self.calibration == "isotonic" and self.isotonic_ is not None:
            cal = np.asarray(self.isotonic_.predict(raw), dtype=float)
        else:
            cal = raw
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_and_save_versioned_candidates(
    *,
    n_bootstrap_ensemble: int = 200,
    seed: int = RANDOM_STATE,
    data_path: Path = DATA_PATH,
    out_dir: Path = CANDIDATES_DIR,
) -> dict[str, Any]:
    """Train separately versioned candidate & uncertainty artifacts strictly on X_train (n=736).

    Never overwrites or modifies `models/heart_model.joblib`.
    """
    baseline_sha = verify_baseline_intact()
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_dataset_with_metadata(data_path)
    data_sha = file_sha256(data_path)

    # Exact same train/test partition as baseline src/train.py so X_test (n=184) is NEVER seen during fit.
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    h_train = meta.loc[X_train.index, "dataset"].to_numpy()
    y_train_arr = y_train.to_numpy()

    # 1. Fit site-deconfounded + Platt-calibrated candidate strictly on X_train (n=736).
    # Threshold selected strictly via out-of-fold CV inside X_train.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_cal = np.zeros(len(X_train), dtype=float)
    for tr_idx, va_idx in skf.split(X_train, y_train_arr):
        fold_clf = HarmonizedCalibratedClassifier(
            weight_mode="site_deconfounded",
            calibration="platt",
            cv_folds=5,
            random_state=seed,
        )
        fold_clf.fit(X_train.iloc[tr_idx], y_train_arr[tr_idx], hospitals=h_train[tr_idx])
        oof_cal[va_idx] = fold_clf.predict_proba(X_train.iloc[va_idx])[:, 1]

    cand_threshold = pick_recall_threshold(y_train_arr, oof_cal, TARGET_RECALL)
    candidate = HarmonizedCalibratedClassifier(
        weight_mode="site_deconfounded",
        calibration="platt",
        cv_folds=5,
        random_state=seed,
    )
    candidate.fit(X_train, y_train_arr, hospitals=h_train)

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cand_path = out_dir / "harmonized_calibrated_lr_v1.joblib"
    cand_prov_path = out_dir / "harmonized_calibrated_lr_v1.provenance.json"
    cand_provenance = {
        "artifact": "models/candidates/harmonized_calibrated_lr_v1.joblib",
        "model_name": "Site-Deconfounded Platt-Calibrated Logistic Regression (v1)",
        "weight_mode": "site_deconfounded",
        "calibration": "platt",
        "platt_params": {"intercept_a": round(candidate.platt_a_, 6), "slope_b": round(candidate.platt_b_, 6)},
        "threshold": cand_threshold,
        "target_recall_cv": TARGET_RECALL,
        "features": FEATURES,
        "hospital_id_used_at_inference": False,
        "dataset": "UCI Heart Disease (4 hospitals: Cleveland, Hungary, Switzerland, VA Long Beach)",
        "dataset_sha256": data_sha,
        "split": "80/20 stratified train_test_split (fit strictly on n=736 training rows; 184 test rows excluded)",
        "grouping_note": "Evaluated under Leave-One-Hospital-Out (LOHO) CV in reports/statistical_deepening.json",
        "n_train": int(len(X_train)),
        "train_prevalence": round(float(y_train_arr.mean()), 4),
        "seed": seed,
        "created_utc": now_iso,
        "sklearn_version": sklearn.__version__,
        "script": "src/harmonization.py",
        "repro_command": "python -m src.harmonization",
        "baseline_preserved_sha256": baseline_sha,
    }
    joblib.dump(
        {
            "pipeline": candidate.pipeline_,
            "platt_a": float(candidate.platt_a_),
            "platt_b": float(candidate.platt_b_),
            "calibration": "platt",
            "weight_mode": "site_deconfounded",
            "threshold": cand_threshold,
            "model_name": cand_provenance["model_name"],
            "features": FEATURES,
            "provenance": cand_provenance,
        },
        cand_path,
    )
    cand_prov_path.write_text(json.dumps(cand_provenance, indent=2) + "\n", encoding="utf-8")

    # 2. Fit bootstrap epistemic-uncertainty ensemble (B=200 resamples strictly on X_train).
    # Mirrors baseline LR pipeline recipe so uncertainty intervals reflect sampling variance of the baseline.
    rng = np.random.default_rng(seed)
    boot_pipes: list[Pipeline] = []
    n_tr = len(X_train)
    for b in range(n_bootstrap_ensemble):
        # Stratified resample within X_train to ensure both classes appear in every bootstrap fit.
        pos_idx = np.where(y_train_arr == 1)[0]
        neg_idx = np.where(y_train_arr == 0)[0]
        sampled = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        pipe = build_base_lr_pipeline(add_missing_indicators=False)
        pipe.fit(X_train.iloc[sampled], y_train_arr[sampled])
        boot_pipes.append(pipe)

    unc_path = out_dir / "uncertainty_bootstrap_v1.joblib"
    unc_prov_path = out_dir / "uncertainty_bootstrap_v1.provenance.json"
    unc_provenance = {
        "artifact": "models/candidates/uncertainty_bootstrap_v1.joblib",
        "purpose": "Bootstrap epistemic-uncertainty ensemble mirroring the baseline LR recipe",
        "n_bootstrap_models": n_bootstrap_ensemble,
        "n_train": int(n_tr),
        "dataset": "UCI Heart Disease (4 hospitals)",
        "dataset_sha256": data_sha,
        "split": "Fit strictly on n=736 training rows (184 historical test rows never touched)",
        "seed": seed,
        "created_utc": now_iso,
        "sklearn_version": sklearn.__version__,
        "script": "src/harmonization.py",
        "repro_command": "python -m src.harmonization",
        "baseline_preserved_sha256": baseline_sha,
    }
    joblib.dump(
        {
            "pipelines": boot_pipes,
            "n_bootstrap_models": n_bootstrap_ensemble,
            "features": FEATURES,
            "provenance": unc_provenance,
        },
        unc_path,
    )
    unc_prov_path.write_text(json.dumps(unc_provenance, indent=2) + "\n", encoding="utf-8")

    # Double-check baseline was not touched.
    verify_baseline_intact()
    return {
        "candidate_path": str(cand_path),
        "uncertainty_path": str(unc_path),
        "candidate_provenance": cand_provenance,
        "uncertainty_provenance": unc_provenance,
    }


def main() -> None:
    res = train_and_save_versioned_candidates()
    print("Saved versioned candidate artifacts (frozen baseline untouched):")
    print(f"  Candidate:   {res['candidate_provenance']['artifact']} (threshold={res['candidate_provenance']['threshold']:.2f})")
    print(f"  Uncertainty: {res['uncertainty_provenance']['artifact']} (B={res['uncertainty_provenance']['n_bootstrap_models']})")
    print(f"  Baseline SHA-256 verified: {res['candidate_provenance']['baseline_preserved_sha256']}")


if __name__ == "__main__":
    main()
