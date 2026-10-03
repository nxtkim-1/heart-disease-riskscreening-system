"""
Download the four UCI Heart Disease files (Cleveland, Hungary, Switzerland,
VA Long Beach) and combine them into data/heart_disease_uci.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SITES = {
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hungary",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "VA Long Beach",
}
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# The raw files store categories as numeric codes. Map them to readable labels.
CODE_TO_LABEL = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {1: True, 0: False},
    "restecg": {0: "normal", 1: "st-t abnormality", 2: "lv hypertrophy"},
    "exang": {1: True, 0: False},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},  # sic, as in the UCI docs
}

OUT_PATH = Path(__file__).parent / "heart_disease_uci.csv"


def main() -> None:
    frames = []
    for filename, site in SITES.items():
        # "?" marks a missing value in the raw files
        df = pd.read_csv(BASE_URL + filename, header=None, names=RAW_COLUMNS, na_values="?")
        df.insert(2, "dataset", site)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    for col, mapping in CODE_TO_LABEL.items():
        df[col] = df[col].map(mapping)  # unknown/missing codes stay NaN
    for col in ["age", "trestbps", "chol", "thalch", "ca", "num"]:
        df[col] = df[col].astype("Int64")

    df.insert(0, "id", np.arange(1, len(df) + 1))
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
