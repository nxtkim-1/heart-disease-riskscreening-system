"""
Data report: missing values, impossible zeros, class balance and disease rate
by hospital. Saves two charts to reports/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import LABELS, PROJECT_ROOT, load_raw

REPORTS = PROJECT_ROOT / "reports"
BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#8C8C8C"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_raw()

    print(f"Shape: {df.shape[0]} patients x {df.shape[1]} columns\n")
    print("Column types:")
    print(df.dtypes.to_string(), "\n")

    # --- Missing values ---
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    print("Missing values (count, %):")
    for col in df.columns:
        if missing[col]:
            print(f"  {col:<10} {missing[col]:>4}  ({missing_pct[col]}%)")

    # --- Hidden missing values: zeros that can't be real ---
    print("\nImpossible zeros (really 'not measured'):")
    for col in ["chol", "trestbps"]:
        print(f"  {col:<10} {(df[col] == 0).sum():>4} rows equal 0")

    # --- Target ---
    has_disease = (df["num"] > 0)
    print(f"\nPatients with heart disease: {has_disease.sum()} / {len(df)} "
          f"({has_disease.mean():.0%})")
    print("\nDisease rate by hospital:")
    print(has_disease.groupby(df["dataset"]).mean().map("{:.0%}".format).to_string())

    # --- Plot: missing values per column, counting the disguised zeros too ---
    blank_pct = df.isna().mean() * 100
    zero_pct = pd.Series({col: (df[col] == 0).mean() * 100 for col in ["chol", "trestbps"]})
    zero_pct = zero_pct.reindex(df.columns, fill_value=0)
    total_pct = blank_pct + zero_pct
    order = total_pct[total_pct > 0].sort_values().index
    labels = [LABELS.get(col, col) for col in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, blank_pct[order], color=BLUE, label="Blank in the data")
    ax.barh(labels, zero_pct[order], left=blank_pct[order], color=ORANGE,
            label="Recorded as 0 (really not measured)")
    for i, col in enumerate(order):
        pct = f"{total_pct[col]:.0f}%" if total_pct[col] >= 1 else "<1%"
        ax.text(total_pct[col] + 1, i, pct, va="center", fontsize=9)
    ax.set_xlim(0, 75)
    ax.set_xlabel("% of the 920 patients")
    ax.set_title("How much of each column is missing")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "missing_values.png", dpi=150)
    print(f"\nSaved {REPORTS / 'missing_values.png'}")

    # --- Plot: disease rate by hospital ---
    rate = has_disease.groupby(df["dataset"]).mean().sort_values() * 100
    patients = df["dataset"].value_counts()

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(rate.index, rate, color=BLUE)
    ax.axvline(has_disease.mean() * 100, color=GREY, ls="--", lw=1,
               label=f"All four hospitals together ({has_disease.mean():.0%})")
    for i, site in enumerate(rate.index):
        ax.text(rate[site] + 1, i, f"{rate[site]:.0f}%  ({patients[site]} patients)",
                va="center", fontsize=9)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlabel("% of patients with heart disease")
    ax.set_title("The hospital alone says a lot about the diagnosis")
    fig.tight_layout()
    fig.savefig(REPORTS / "disease_by_hospital.png", dpi=150)
    print(f"Saved {REPORTS / 'disease_by_hospital.png'}")


if __name__ == "__main__":
    main()
