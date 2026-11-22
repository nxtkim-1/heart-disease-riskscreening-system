"""Streamlit prototype UI for the Heart-Disease Risk Screening System.

Uses the frozen baseline model (`models/heart_model.joblib`, threshold 0.39) via
`src.inference.predict_patient` with Pydantic validation, honest bootstrap uncertainty
intervals, missing-value warnings, local offline plain-language explanations, and session reset.
"""
import streamlit as st
from pydantic import ValidationError

from src.assistant import explain_screening
from src.inference import (
    INFORMATIONAL_DISCLAIMER,
    ModelLoadError,
    load_baseline_bundle,
    predict_patient,
)
from src.schemas import AssistantRequest, PatientInput
from src.security import format_safe_error

st.set_page_config(page_title="Heart disease risk screen", page_icon="🫀")


@st.cache_resource
def get_baseline_bundle():
    return load_baseline_bundle()


try:
    bundle = get_baseline_bundle()
except ModelLoadError as exc:
    st.error(format_safe_error(exc, context="Baseline model could not be loaded"))
    st.stop()

threshold = float(bundle["threshold"])
metrics = bundle["test_metrics"]

with st.sidebar:
    st.subheader("About this model")
    st.write(f"**{bundle['model_name']}**, trained on 920 patients from the UCI Heart Disease dataset (1981–1988).")
    st.write(f"Historical read-only audit on {bundle['n_test']} test patients:")
    st.metric("Recall (95% bootstrap CI)", f"{metrics['recall']:.1%}", "86.7% – 97.0%")
    st.metric("Precision", f"{metrics['precision']:.1%}")
    st.caption(
        f"Decision threshold: {threshold:.2f} (lowered from 0.50 on training CV to favor sensitivity). "
        "Leave-One-Hospital-Out audit shows strong site calibration shift across the four hospitals."
    )
    if st.button("Clear session inputs"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.title("Heart disease risk screen")
st.warning(INFORMATIONAL_DISCLAIMER)
st.write(
    "Enter de-identified clinical measurements to estimate the model's probability score. "
    "Leave any unmeasured field as *Unknown*. No inputs are logged or persisted."
)

UNKNOWN = "Unknown"


def optional_number(label, min_value, max_value, value, step, help_text, key):
    """Number input with an 'Unknown' checkbox. Unknown values are passed as None."""
    col1, col2 = st.columns([3, 1])
    unknown = col2.checkbox(UNKNOWN, key=f"{key}_unknown")
    num = col1.number_input(
        label, min_value, max_value, value, step, help=help_text, disabled=unknown, key=key
    )
    return None if unknown else num


with st.form("patient"):
    st.subheader("Patient")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 110, 55, key="age")
    sex = c2.radio("Sex", ["Male", "Female"], horizontal=True, key="sex")

    st.subheader("Symptoms")
    cp = st.selectbox(
        "Chest pain type",
        ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
        help="Typical angina: chest pain brought on by exertion and relieved by rest. "
        "Asymptomatic means no chest pain at all.",
        key="cp",
    )
    exang = st.radio(
        "Chest pain during exercise?",
        ["No", "Yes", UNKNOWN],
        horizontal=True,
        help="Whether exercise brought on chest pain (angina).",
        key="exang",
    )

    st.subheader("Measurements")
    trestbps = optional_number(
        "Resting blood pressure (mm Hg)", 60, 260, 130, 1, "Systolic pressure at rest.", "trestbps"
    )
    chol = optional_number(
        "Serum cholesterol (mg/dl)", 80, 700, 240, 1, "Total cholesterol from a blood test.", "chol"
    )
    fbs = st.radio("Fasting blood sugar above 120 mg/dl?", ["No", "Yes", UNKNOWN], horizontal=True, key="fbs")

    st.subheader("Exercise test and ECG")
    thalch = optional_number("Maximum heart rate reached (bpm)", 40, 230, 150, 1, None, "thalch")
    oldpeak = optional_number(
        "ST depression during exercise (mm)",
        -3.0,
        10.0,
        1.0,
        0.1,
        "How far the ST segment drops on the ECG during exercise vs. rest.",
        "oldpeak",
    )
    slope = st.selectbox(
        "Slope of the peak exercise ST segment",
        ["flat", "upsloping", "downsloping", UNKNOWN],
        help="The direction of the ST segment on the ECG at peak exercise.",
        key="slope",
    )
    restecg = st.selectbox(
        "Resting ECG result",
        ["normal", "st-t abnormality", "lv hypertrophy", UNKNOWN],
        help="ST-T abnormality: changes in the ST segment or T wave. "
        "LV hypertrophy: signs of a thickened heart wall.",
        key="restecg",
    )

    submitted = st.form_submit_button("Estimate risk", type="primary")

if submitted:
    try:
        patient = PatientInput(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=None if fbs == UNKNOWN else (fbs == "Yes"),
            restecg=None if restecg == UNKNOWN else restecg,
            thalch=thalch,
            exang=None if exang == UNKNOWN else (exang == "Yes"),
            oldpeak=oldpeak,
            slope=None if slope == UNKNOWN else slope,
        )
        pred = predict_patient(patient)
        local_asst = explain_screening(AssistantRequest(use_external_meta_api=False, patient=patient))
    except ValidationError as exc:
        st.error(format_safe_error(exc, context="Invalid clinical input"))
    except Exception as exc:
        st.error(format_safe_error(exc, context="Prediction failed"))
    else:
        st.divider()
        u = pred.uncertainty
        ci_txt = (
            f" (95% bootstrap interval: {u.bootstrap_ci_low:.1%}–{u.bootstrap_ci_high:.1%})"
            if u.bootstrap_ci_low is not None and u.bootstrap_ci_high is not None
            else ""
        )
        if pred.flagged:
            st.error(f"**Flagged (above {pred.threshold:.0%} cutoff).** Estimated probability {pred.probability:.1%}{ci_txt}.")
        else:
            st.success(f"**Not flagged (below {pred.threshold:.0%} cutoff).** Estimated probability {pred.probability:.1%}{ci_txt}.")
        st.progress(pred.probability)
        st.info(u.threshold_proximity_note)
        if u.imputation_warning:
            st.warning(u.imputation_warning)
        if pred.calibrated_reference_probability is not None:
            st.caption(
                f"Harmonized Platt-calibrated reference probability: {pred.calibrated_reference_probability:.1%} "
                f"({pred.calibrated_reference_artifact}). Frozen baseline remains primary."
            )
        st.subheader("Plain-language local summary (zero external API calls)")
        st.write(local_asst.explanation)

st.divider()
st.caption(INFORMATIONAL_DISCLAIMER)
