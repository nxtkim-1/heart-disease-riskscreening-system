"""Load, clean, and preprocess the UCI Heart Disease data.

Training and the app both use this module, so a row typed into the form is
cleaned the same way as the training rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"

# Continuous: median fill, then scale.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
# Multi-value categories: most-common fill, then one-hot encode.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope"]
# Yes/no columns, already 0/1 after clean_features().
BINARY_FEATURES = ["sex", "fbs", "exang"]

# Left out: id (a row number), dataset (the source hospital; disease rate
# differs by site and a new patient has none), ca and thal (missing on most rows).
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = "target"

# Chart labels. Keys are column names, plus one-hot names like "cp_asymptomatic".
LABELS = {
    "age": "Age",
    "sex": "Male",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "High fasting blood sugar",
    "restecg": "Resting ECG",
    "thalch": "Max heart rate on exercise test",
    "exang": "Chest pain during exercise",
    "oldpeak": "ST depression on exercise ECG",
    "slope": "ST slope on exercise ECG",
    "ca": "Vessels seen on fluoroscopy",
    "thal": "Thallium stress test",
    "cp_asymptomatic": "No chest pain (asymptomatic)",
    "cp_typical angina": "Chest pain: typical angina",
    "cp_atypical angina": "Chest pain: atypical angina",
    "cp_non-anginal": "Chest pain: non-anginal",
    "restecg_normal": "Resting ECG: normal",
    "restecg_st-t abnormality": "Resting ECG: ST-T abnormality",
    "restecg_lv hypertrophy": "Resting ECG: LV hypertrophy",
    "slope_upsloping": "ST slope: upsloping",
    "slope_flat": "ST slope: flat",
    "slope_downsloping": "ST slope: downsloping",
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/download_data.py` first.")
    return pd.read_csv(path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn sentinel zeros into missing values and yes/no columns into 0/1."""
    df = df.copy()

    # A cholesterol or resting blood pressure of 0 is physically impossible.
    # In this dataset 0 means "not measured", so treat it as missing.
    for col in ["chol", "trestbps"]:
        df[col] = df[col].replace(0, np.nan)

    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    for col in ["fbs", "exang"]:
        # Stored as True/False in the CSV; pandas may read them as bools or strings
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0})

    return df


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_features(load_raw(path))
    # num is 0 (no disease) to 4 (severe). We only ask: any disease or not?
    y = (df["num"] > 0).astype(int).rename(TARGET)
    X = df[FEATURES]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Fill, encode, and scale. Fit on training rows only so test-set stats don't leak in."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
    ])
