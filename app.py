"""
Streamlit app: enter a patient's details and get an estimated risk of heart disease.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import PROJECT_ROOT, clean_features

MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.joblib"

st.set_page_config(page_title="Heart disease risk screen", page_icon="🫀")


@st.cache_resource  # load the model once, not on every click
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()
if bundle is None:
    st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
    st.stop()

pipeline, threshold = bundle["pipeline"], bundle["threshold"]
m = bundle["test_metrics"]

# ---- Sidebar: what the model is and how good it is -----------------------
with st.sidebar:
    st.subheader("About this model")
    st.write(f"**{bundle['model_name']}**, trained on 920 patients from the UCI Heart Disease dataset.")
    st.write(f"Results on {bundle['n_test']} held-out test patients the model never saw during training:")
    st.metric("Recall (sick patients caught)", f"{m['recall']:.0%}")
    st.metric("Precision (flags that were right)", f"{m['precision']:.0%}")
    st.caption(
        f"The decision threshold is {threshold:.2f}, not 0.50. It was lowered on purpose: "
        "missing a sick patient is worse than a false alarm."
    )

st.title("Heart disease risk screen")
st.write("Enter a patient's information to estimate the chance of heart disease. "
         "Leave a field as *Unknown* if it wasn't measured.")

UNKNOWN = "Unknown"


def optional_number(label, min_value, max_value, value, step, help_text, key):
    """Number input with an 'unknown' checkbox. Unknown values get filled in by the model."""
    col1, col2 = st.columns([3, 1])
    unknown = col2.checkbox(UNKNOWN, key=f"{key}_unknown")
    num = col1.number_input(label, min_value, max_value, value, step,
                            help=help_text, disabled=unknown, key=key)
    return np.nan if unknown else num


with st.form("patient"):
    st.subheader("Patient")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 55)
    sex = c2.radio("Sex", ["Male", "Female"], horizontal=True)

    st.subheader("Symptoms")
    cp = st.selectbox(
        "Chest pain type",
        ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
        help="Typical angina: chest pain brought on by exertion and relieved by rest.",
    )
    exang = st.radio("Chest pain during exercise?", ["No", "Yes", UNKNOWN], horizontal=True)

    st.subheader("Measurements")
    trestbps = optional_number("Resting blood pressure (mm Hg)", 80, 220, 130, 1,
                               "Systolic pressure at rest.", "trestbps")
    chol = optional_number("Serum cholesterol (mg/dl)", 100, 600, 240, 1, None, "chol")
    fbs = st.radio("Fasting blood sugar above 120 mg/dl?", ["No", "Yes", UNKNOWN], horizontal=True)

    st.subheader("Exercise test and ECG")
    thalch = optional_number("Maximum heart rate reached (bpm)", 60, 220, 150, 1, None, "thalch")
    oldpeak = optional_number("ST depression during exercise (mm)", 0.0, 7.0, 1.0, 0.1,
                              "How far the ST segment drops on the ECG during exercise vs. rest.",
                              "oldpeak")
    slope = st.selectbox("Slope of the peak exercise ST segment",
                         ["flat", "upsloping", "downsloping", UNKNOWN])
    restecg = st.selectbox("Resting ECG result",
                           ["normal", "st-t abnormality", "lv hypertrophy", UNKNOWN])

    submitted = st.form_submit_button("Estimate risk", type="primary")

if submitted:
    yes_no = {"Yes": True, "No": False, UNKNOWN: np.nan}
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": yes_no[fbs],
        "restecg": np.nan if restecg == UNKNOWN else restecg,
        "thalch": thalch, "exang": yes_no[exang], "oldpeak": oldpeak,
        "slope": np.nan if slope == UNKNOWN else slope,
    }])
    # Same cleaning as the training data, then the pipeline handles the rest
    X = clean_features(patient)[bundle["features"]]
    prob = float(pipeline.predict_proba(X)[0, 1])

    st.divider()
    if prob >= threshold:
        st.error(f"**Flagged: likely heart disease.** Estimated probability {prob:.0%}.")
        st.write("This patient's profile resembles patients in the dataset who had heart disease. "
                 "A clinician should follow up.")
    else:
        st.success(f"**Not flagged.** Estimated probability {prob:.0%}.")
        st.write(f"Below the {threshold:.0%} cut-off this model uses to flag a patient.")
    st.progress(prob)

st.divider()
st.caption(
    "Not a medical device and not medical advice. "
    "The model was trained on ~900 patients referred for angiography between 1981 and 1988, "
    "so it does not represent the general population."
)
