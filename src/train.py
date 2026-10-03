"""
Train and compare logistic regression, random forest and XGBoost, tune each
model's decision threshold with cross-validation, and save the chosen model.

Outputs:
    reports/model_comparison.csv   all metrics for all three models
    reports/confusion_matrices.png
    reports/roc_curves.png
    reports/feature_importance.png
    models/heart_model.joblib      the chosen model + its decision threshold (used by app.py)
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import FEATURES, LABELS, PROJECT_ROOT, build_preprocessor, load_dataset

RANDOM_STATE = 42
TARGET_RECALL = 0.90  # catch at least 90% of sick patients (on validation data)

REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"
RED, BLUE = "#C44E52", "#4C72B0"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def make_models() -> dict[str, Pipeline]:
    """Each model gets its own copy of the preprocessing steps."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
        for name, clf in classifiers.items()
    }


def metrics_at(y_true, proba, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "missed_sick (FN)": fn,
        "false_alarms (FP)": fp,
    }


def pick_threshold(y_true, proba, target_recall: float) -> float:
    """Highest threshold that still catches `target_recall` of sick patients.

    Lowering the threshold below 0.5 trades more false alarms for fewer
    missed patients. We choose it on cross-validation predictions from the
    training set, never on the test set.
    """
    for t in np.arange(0.95, 0.0, -0.01):
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            return round(float(t), 2)
    return 0.0


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    X, y = load_dataset()
    # stratify=y keeps the sick/healthy ratio the same in both splits.
    # The test set is locked away until the very end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} patients | Test: {len(X_test)} patients\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, test_probas = [], {}, {}

    for name, pipe in make_models().items():
        # 1) Validation: out-of-fold probabilities on the training set
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = pick_threshold(y_train, oof_proba, TARGET_RECALL)
        cv_tuned = metrics_at(y_train, oof_proba, threshold)

        # 2) Final fit on all training data, score once on the untouched test set
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name], test_probas[name] = pipe, proba

        for label, thr in [("default 0.50", 0.5), (f"tuned {threshold:.2f}", threshold)]:
            rows.append({"model": name, "threshold": label, **metrics_at(y_test, proba, thr)})
        rows[-1]["cv_precision_at_target"] = cv_tuned["precision"]
        rows[-1]["_threshold_value"] = threshold

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("Test-set results (higher is better except FN / FP):")
    print(results.drop(columns=["cv_precision_at_target", "_threshold_value"])
                 .round(3).to_string(index=False))
    results.drop(columns=["_threshold_value"]).round(4).to_csv(
        REPORTS / "model_comparison.csv", index=False)

    # ---- Choose the final model ----------------------------------------
    # Rule decided BEFORE looking at the test set: among models tuned to catch
    # 90% of sick patients in cross-validation, keep the one that raises the
    # fewest false alarms there (highest CV precision).
    tuned = results.dropna(subset=["cv_precision_at_target"])
    thresholds = dict(zip(tuned["model"], tuned["_threshold_value"]))
    best = tuned.loc[tuned["cv_precision_at_target"].idxmax()]
    best_name, best_threshold = best["model"], float(best["_threshold_value"])
    print(f"\nChosen model: {best_name} (threshold {best_threshold:.2f})")

    joblib.dump(
        {
            "pipeline": fitted[best_name],
            "threshold": best_threshold,
            "model_name": best_name,
            "features": FEATURES,
            "n_test": len(X_test),
            "test_metrics": best.drop(["model", "threshold", "_threshold_value"]).to_dict(),
        },
        MODELS / "heart_model.joblib",
    )
    print(f"Saved {MODELS / 'heart_model.joblib'}")

    # ---- Plots -----------------------------------------------------------
    # Confusion matrices at each model's tuned threshold
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, proba) in zip(axes, test_probas.items()):
        thr = thresholds[name]
        ConfusionMatrixDisplay.from_predictions(
            y_test, (proba >= thr).astype(int),
            display_labels=["No disease", "Disease"], cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"{name}\n(threshold {thr:.2f})")
    fig.suptitle(f"Test set ({len(y_test)} patients): bottom-left = sick patients the model missed",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrices.png", dpi=150, bbox_inches="tight")

    # ROC curves, with a dot where each model's tuned threshold puts it
    y_true = y_test.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        (line,) = ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, proba):.2f})")
        flagged = proba >= thresholds[name]
        ax.plot(flagged[y_true == 0].mean(), flagged[y_true == 1].mean(), "o",
                color=line.get_color(), markersize=7)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Random guessing")
    ax.set_xlabel("False positive rate (healthy patients flagged)")
    ax.set_ylabel("True positive rate (sick patients caught)")
    ax.set_title("ROC curves on the test set\n(dots = each model's tuned threshold)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_curves.png", dpi=150)

    # Feature importance from the chosen model
    pipe = fitted[best_name]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    # "cat__cp_asymptomatic" -> "cp_asymptomatic" -> "No chest pain (asymptomatic)"
    clean_names = [LABELS.get(n.split("__", 1)[1], n.split("__", 1)[1]) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if hasattr(model, "coef_"):
        # Logistic regression: sign tells direction. Positive = raises predicted risk.
        coef = pd.Series(model.coef_[0], index=clean_names)
        coef = coef.reindex(coef.abs().sort_values().index).tail(12)
        coef.plot.barh(ax=ax, color=[RED if c > 0 else BLUE for c in coef], width=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Effect on predicted risk (red raises it, blue lowers it)")
    else:
        imp = pd.Series(model.feature_importances_, index=clean_names).sort_values().tail(12)
        imp.plot.barh(ax=ax, color=RED, width=0.7)
        ax.set_xlabel("Importance")
    ax.set_title(f"What pushes the {best_name} prediction up or down")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    print(f"Saved plots to {REPORTS}/")


if __name__ == "__main__":
    main()
