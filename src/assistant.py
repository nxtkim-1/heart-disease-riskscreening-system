"""Informational plain-language explanation assistant for heart-disease risk screening.

Strict safety & privacy architecture:
1. TWO SEPARATE PATHS:
   - `local_offline`: 100% deterministic Python explainer using exact model log-odds
     contributions, clinical glossary definitions, and epistemic uncertainty intervals.
     Zero network calls; safe for any validated PatientInput.
   - `meta_replay` / `meta_record` / `meta_live`: Metered Meta API adapter that accepts
     REGISTERED SYNTHETIC PROFILES ONLY (`SYNTH-001`..`SYNTH-006`) and canonical question
     templates ONLY (`EXTERNAL_QUESTION_TEMPLATES`). Custom/real patient inputs and free-text
     questions are refused in code before any external request can be constructed, and the
     client scans the final outgoing messages for PII, secrets, and private paths.
2. AUTHORITATIVE OUTPUT GATE:
   - Every response is verified in code to enforce `informational_only=True`,
     `not_medical_advice=True`, mandatory non-medical-advice disclaimer, honest uncertainty
     reminders, secret/path redaction, and filtering of prescriptive diagnostic language.
3. ZERO GEMINI USAGE:
   - Never imports or touches Gemini credentials, billing, or SDKs.
"""
from __future__ import annotations

import re
from typing import Any

from src.inference import INFORMATIONAL_DISCLAIMER, predict_patient
from src.meta_client import MetaAssistantClient
from src.schemas import (
    AssistantRequest,
    AssistantResponse,
    PatientInput,
    PredictionResponse,
    is_allowed_external_question,
)
from src.security import redact_text
from src.synthetic_profiles import (
    SyntheticPrivacyViolation,
    assert_registered_synthetic,
    get_synthetic_profile,
)

# Plain-language clinical glossary for answering questions about unfamiliar inputs.
INPUT_GLOSSARY: dict[str, str] = {
    "cp": (
        "Chest pain type distinguishes typical angina (exertional chest discomfort relieved by rest), "
        "atypical angina, non-anginal pain, and asymptomatic (no chest pain). In this angiography cohort, "
        "'asymptomatic' raises model risk because patients without chest pain who were still sent for an "
        "angiogram usually had other abnormal tests."
    ),
    "oldpeak": (
        "ST depression (oldpeak, in mm) measures how far the ST segment drops on an exercise ECG "
        "compared with rest. Larger drops indicate stronger ECG stress changes."
    ),
    "slope": (
        "Peak exercise ST slope describes whether the ECG ST segment tilts upsloping, flat, or "
        "downsloping at peak exercise; flat or downsloping patterns are more closely associated "
        "with positive angiograms in this dataset."
    ),
    "thalch": (
        "Maximum heart rate achieved (thalch, in bpm) is the peak heart rate reached during the "
        "exercise stress test."
    ),
    "exang": (
        "Exercise-induced angina (exang) records whether physical exertion during the stress test "
        "brought on chest pain."
    ),
    "restecg": (
        "Resting ECG captures baseline heart electrical findings at rest: normal, ST-T wave "
        "abnormality, or signs of left ventricular hypertrophy."
    ),
    "trestbps": "Resting systolic blood pressure (mm Hg) recorded at hospital admission.",
    "chol": (
        "Serum total cholesterol (mg/dl). Note that in the 1988 UCI data, 100% of Switzerland rows "
        "recorded 0 for missing cholesterol, which our preprocessing treats as Unknown/missing."
    ),
    "fbs": "Fasting blood sugar > 120 mg/dl (a binary indicator of elevated fasting glucose).",
    "threshold": (
        "The baseline screening threshold is 0.39 (39%), lowered from 0.50 on training cross-validation "
        "to target ~90% sensitivity so fewer sick patients are missed, at the cost of more false alarms."
    ),
    "uncertainty": (
        "Uncertainty comes from three sources: finite sample size (the 95% bootstrap interval around "
        "the probability), missing measurements filled by training medians/modes, and population shift "
        "between 1981–1988 angiography referrals (55% disease rate) and general populations."
    ),
}

# Authoritative output gate patterns: block diagnostic verdicts, reassurance that someone is
# healthy, discouraging clinical follow-up, and treatment/medication orders. Hedged phrasing
# ("this is not a clinical diagnosis", "does not rule out disease") must stay allowed.
_CLINICIAN = r"(?:a\s+|your\s+|any\s+)?(?:clinician|doctor|physician|cardiologist|specialist|gp|nurse|health\s*care\s+provider)"
_MEDICATION = (
    r"(?:an?\s+|your\s+|daily\s+|low[-\s]dose\s+|baby\s+)*"
    r"(?:aspirin|statins?|medications?|medicines?|meds|drugs?|pills?|nitroglycerin|beta[-\s]?blockers?|"
    r"blood\s+thinners?|anticoagulants?|ace\s+inhibitors?|insulin|metformin|dose)"
)
_FORBIDDEN_MEDICAL_CLAIMS: tuple[re.Pattern[str], ...] = (
    # Diagnostic verdicts about the reader.
    re.compile(
        r"(?i)\byou\s+(?:definitely\s+|certainly\s+|clearly\s+|probably\s+)?(?:have|do\s+not\s+have|don'?t\s+have|"
        r"are\s+suffering\s+from|are\s+free\s+of)\s+(?:heart|coronary(?:\s+artery)?|cardiac|cardiovascular)\s+disease\b"
    ),
    re.compile(r"(?i)\byou(?:\s+are|'re)\s+(?:completely\s+|perfectly\s+|totally\s+|now\s+)?(?:healthy|disease[-\s]free|in\s+the\s+clear|fine|safe|not\s+at\s+risk)\b"),
    re.compile(r"(?i)(?<!not\sa\s)\bclean\s+bill\s+of\s+health\b"),
    re.compile(
        r"(?i)\b(?:this|the\s+(?:result|score|model|estimate|screen(?:ing)?))\s+(?:is|provides|gives|represents|"
        r"confirms)\s+(?:a|your|the)\s+(?:clinical\s+|medical\s+|definitive\s+)?diagnosis\b"
    ),
    re.compile(r"(?i)\byour\s+(?:clinical\s+|medical\s+)?diagnosis\s+is\b"),
    re.compile(r"(?i)\b(?:this|the\s+(?:result|score|model|estimate))\s+(?:confirms|proves|rules\s+out)\b"),
    re.compile(r"(?i)\bi\s+(?:diagnose|prescribe|recommend\s+(?:medication|treatment|surgery|a\s+stent|an?\s+angioplasty))\b"),
    # Discouraging clinical follow-up.
    re.compile(rf"(?i)\bno\s+need\s+(?:to\s+(?:see|visit|consult|call)|for)\s+{_CLINICIAN}\b"),
    re.compile(
        rf"(?i)\byou\s+(?:do\s+not|don'?t|dont|won'?t|will\s+not|no\s+longer)\s+need\s+(?:to\s+(?:see|visit|consult|call)\s+)?"
        rf"(?:{_CLINICIAN}|medical\s+(?:care|attention|follow[-\s]?up)|follow[-\s]?up|further\s+testing)\b"
    ),
    re.compile(rf"(?i)\b(?:skip|avoid|cancel)\s+(?:seeing\s+|your\s+)?(?:{_CLINICIAN}|appointment|follow[-\s]?up)\b"),
    # Medication / treatment orders.
    re.compile(r"(?i)\b(?:start|stop|keep)\s+taking\s+[a-z0-9\s\-]{2,30}\b"),
    re.compile(rf"(?i)\b(?:take|start|stop|increase|decrease|double|halve|skip|discontinue)\s+(?:taking\s+)?{_MEDICATION}\b"),
    re.compile(
        rf"(?i)\byou\s+(?:should|must|need\s+to|ought\s+to)\s+(?:take|start|stop|undergo|get|have)\s+"
        rf"(?:{_MEDICATION}|surgery|an?\s+(?:angioplasty|stent|bypass|angiogram))\b"
    ),
)

OUTPUT_GATE_NOTICE = (
    "[Output gate: the external assistant's wording was withheld because it contained diagnostic, "
    "reassurance, or treatment phrasing. Showing the deterministic local explanation instead.] "
)


def find_forbidden_medical_claims(text: str) -> list[str]:
    """Return the regex patterns (as strings) that match diagnostic/prescriptive phrasing."""
    return [pat.pattern for pat in _FORBIDDEN_MEDICAL_CLAIMS if pat.search(text or "")]

SYSTEM_PROMPT = (
    "You are an informational assistant explaining output from an archival 1981-1988 UCI heart-disease "
    "logistic regression prototype trained on 920 coronary-angiography referral patients. "
    "STRICT RULES: "
    "(1) This is strictly informational and NEVER medical advice or a diagnosis. "
    "(2) You are only ever shown SYNTHETIC archetype profiles, never real patients. "
    "(3) Always communicate uncertainty honestly: mention the 0.39 screening threshold, the 95% "
    "bootstrap probability interval, any imputed unknown fields, and that a 55% angiography-cohort "
    "base rate overestimates risk in general populations. "
    "(4) Never diagnose, never tell anyone they are healthy, and never prescribe or alter medications. "
    "(5) Respond concisely in plain language (3-5 sentences)."
)


def _uncertainty_reminders(pred: PredictionResponse) -> list[str]:
    u = pred.uncertainty
    reminders = [
        u.threshold_proximity_note,
        u.population_shift_warning,
        u.calibration_note,
    ]
    if u.bootstrap_ci_low is not None and u.bootstrap_ci_high is not None:
        reminders.insert(
            0,
            f"Bootstrap 95% model-sampling interval ({u.bootstrap_samples} training resamples): "
            f"{u.bootstrap_ci_low:.1%} to {u.bootstrap_ci_high:.1%} (point estimate {pred.probability:.1%}).",
        )
    if u.imputation_warning:
        reminders.append(u.imputation_warning)
    if pred.calibrated_reference_probability is not None:
        reminders.append(
            f"Site-deconfounded Platt-calibrated reference probability: "
            f"{pred.calibrated_reference_probability:.1%} (for sensitivity comparison; baseline remains primary)."
        )
    return reminders


def _key_takeaways(pred: PredictionResponse) -> list[str]:
    status = (
        f"Flagged (above the {pred.threshold:.0%} screening cutoff)"
        if pred.flagged
        else f"Not flagged (below the {pred.threshold:.0%} screening cutoff — not a clean bill of health)"
    )
    items = [
        f"Baseline Logistic Regression estimate: {pred.probability:.1%} — {status}.",
    ]
    raisers = [c for c in pred.top_contributions if c.direction == "increases_risk"][:3]
    lowerers = [c for c in pred.top_contributions if c.direction == "decreases_risk"][:3]
    if raisers:
        items.append(
            "Top factors pushing score UP: "
            + ", ".join(f"{c.label} ({c.log_odds_contribution:+.2f} log-odds)" for c in raisers)
            + "."
        )
    if lowerers:
        items.append(
            "Top factors pushing score DOWN: "
            + ", ".join(f"{c.label} ({c.log_odds_contribution:+.2f} log-odds)" for c in lowerers)
            + "."
        )
    if pred.uncertainty.missing_field_count > 0:
        items.append(
            f"{pred.uncertainty.missing_field_count} field(s) left Unknown and filled with training defaults: "
            + ", ".join(pred.uncertainty.missing_fields)
            + "."
        )
    return items


def _match_glossary_notes(question: str) -> list[str]:
    q = question.lower()
    hits: list[str] = []
    keyword_map = {
        "cp": ("chest pain", "angina", "asymptomatic", "cp"),
        "oldpeak": ("oldpeak", "st depression", "st segment"),
        "slope": ("slope", "upsloping", "downsloping", "flat"),
        "thalch": ("thalch", "heart rate", "max hr", "bpm"),
        "exang": ("exang", "exercise angina", "exercise chest"),
        "restecg": ("restecg", "resting ecg", "hypertrophy", "st-t"),
        "trestbps": ("blood pressure", "trestbps", "systolic"),
        "chol": ("cholesterol", "chol"),
        "fbs": ("blood sugar", "glucose", "fbs", "fasting"),
        "threshold": ("threshold", "0.39", "cutoff", "cut-off", "flag"),
        "uncertainty": ("uncertain", "confidence", "interval", "calibrat", "hospital", "shift"),
    }
    for key, terms in keyword_map.items():
        if any(term in q for term in terms):
            hits.append(INPUT_GLOSSARY[key])
    return hits


def build_local_explanation(
    pred: PredictionResponse,
    question: str,
    *,
    synthetic_profile_id: str | None = None,
) -> str:
    """Deterministic plain-language explanation built strictly locally (zero network calls)."""
    u = pred.uncertainty
    ci_phrase = (
        f" (95% bootstrap sampling interval: {u.bootstrap_ci_low:.1%}–{u.bootstrap_ci_high:.1%})"
        if u.bootstrap_ci_low is not None and u.bootstrap_ci_high is not None
        else ""
    )
    flag_phrase = (
        f"above the {pred.threshold:.0%} screening cutoff (flagged for follow-up review)"
        if pred.flagged
        else f"below the {pred.threshold:.0%} screening cutoff (not flagged, which does not rule out disease)"
    )
    prefix = (
        f"[Synthetic profile {synthetic_profile_id}] "
        if synthetic_profile_id
        else "[Local offline explanation — no data left this machine] "
    )

    drivers_up = [c.label for c in pred.top_contributions if c.direction == "increases_risk"][:2]
    drivers_dn = [c.label for c in pred.top_contributions if c.direction == "decreases_risk"][:2]

    parts = [
        f"{prefix}For informational purposes only (not medical advice): the frozen baseline Logistic "
        f"Regression model estimates a {pred.probability:.1%} probability{ci_phrase}, which is {flag_phrase}."
    ]
    if drivers_up:
        parts.append(f"The strongest inputs pushing the estimate higher are {', '.join(drivers_up)}.")
    if drivers_dn:
        parts.append(f"Inputs pulling the estimate lower include {', '.join(drivers_dn)}.")
    if u.imputation_warning:
        parts.append(u.imputation_warning)

    glossary_hits = _match_glossary_notes(question)
    if glossary_hits:
        parts.append("Regarding your question: " + " ".join(glossary_hits[:2]))

    parts.append(
        "Important uncertainty note: this score reflects patterns in 920 patients referred for coronary "
        "angiography (1981–1988, 55% prevalence) and can shift substantially across hospitals or lower-risk "
        "general populations."
    )
    return " ".join(parts)


def build_meta_messages(
    synthetic_profile_id: str,
    features: dict[str, Any],
    pred: PredictionResponse,
    question: str,
) -> list[dict[str, str]]:
    """Build deterministic prompt messages for the Meta API (SYNTHETIC profile only)."""
    summary = {
        "synthetic_profile_id": synthetic_profile_id,
        "is_synthetic": True,
        "features": features,
        "baseline_probability": pred.probability,
        "threshold": pred.threshold,
        "flagged": pred.flagged,
        "risk_band": pred.risk_band,
        "bootstrap_95ci": [pred.uncertainty.bootstrap_ci_low, pred.uncertainty.bootstrap_ci_high],
        "missing_fields": pred.uncertainty.missing_fields,
        "top_contributions": [c.model_dump() for c in pred.top_contributions[:4]],
        "calibrated_reference_probability": pred.calibrated_reference_probability,
        "question": question,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Explain the following SYNTHETIC screening archetype output in plain language "
                "with honest uncertainty and zero medical advice:\n"
                + str(summary)
            ),
        },
    ]


def enforce_authoritative_output_gate(raw_text: str, pred: PredictionResponse) -> str:
    """Authoritative code gate on assistant output.

    Redacts secrets/paths and guarantees an explicit informational/non-medical-advice
    qualifier. If ANY diagnostic, reassurance, or treatment phrasing is found, the whole text
    is withheld (fail closed) and replaced by the deterministic local explanation: patching
    words out of a sentence can leave fragments such as "...aspirin daily" that still read
    as advice.
    """
    cleaned = redact_text(raw_text or "").strip()
    if not cleaned:
        cleaned = build_local_explanation(pred, "Explain this result.")

    if find_forbidden_medical_claims(cleaned):
        fallback = redact_text(build_local_explanation(pred, "Explain this result."))
        if find_forbidden_medical_claims(fallback):  # Should never happen; refuse rather than emit.
            fallback = "Explanation withheld by the informational output gate."
        cleaned = OUTPUT_GATE_NOTICE + fallback

    low = cleaned.lower()
    if "not medical advice" not in low and "informational" not in low:
        cleaned = (
            "Informational summary only (not medical advice): " + cleaned
        )
    return cleaned


def _extract_chat_text(api_resp: dict[str, Any]) -> str:
    choices = api_resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    if isinstance(api_resp.get("output_text"), str):
        return api_resp["output_text"]
    raise ValueError("Unexpected Meta API response shape: missing choices[0].message.content")


def explain_screening(
    req: AssistantRequest,
    *,
    meta_client: MetaAssistantClient | None = None,
) -> AssistantResponse:
    """Explain a screening prediction via local offline mode OR synthetic-only Meta API."""
    if req.use_external_meta_api:
        if not req.synthetic_profile_id:
            raise SyntheticPrivacyViolation("External Meta API requires a registered synthetic_profile_id.")
        # Second check (the schema validator is the first): catches requests built with
        # model_construct() or mutated after validation. Never echoes the question text.
        if req.patient is not None or not is_allowed_external_question(req.question):
            raise SyntheticPrivacyViolation(
                "External Meta API only accepts a registered synthetic profile with a canonical "
                "question template; free-text questions are answered locally offline only."
            )
        profile = get_synthetic_profile(req.synthetic_profile_id)
        patient = profile.to_patient_input()
        features_dict = patient.model_dump()
        # Double-verify digest before building external payload.
        assert_registered_synthetic(profile.profile_id, features_dict)

        pred = predict_patient(patient)
        client = meta_client or MetaAssistantClient()
        messages = build_meta_messages(profile.profile_id, features_dict, pred, req.question)
        result = client.chat_synthetic(
            synthetic_profile_id=profile.profile_id,
            synthetic_features=features_dict,
            messages=messages,
            envelope="assistant_explain",
        )
        raw_text = _extract_chat_text(result["response"])
        safe_text = enforce_authoritative_output_gate(raw_text, pred)
        mode_map = {
            "replay": "meta_replay",
            "record": "meta_record",
            "live": "meta_live",
        }
        return AssistantResponse(
            mode=mode_map[result["mode"]],
            is_synthetic_profile=True,
            synthetic_profile_id=profile.profile_id,
            external_api_called=(result["mode"] != "replay"),
            prediction=pred,
            explanation=safe_text,
            key_takeaways=_key_takeaways(pred),
            uncertainty_reminders=_uncertainty_reminders(pred),
            spend_usd=float(result["cost_usd"]),
            informational_only=True,
            not_medical_advice=True,
            disclaimer=INFORMATIONAL_DISCLAIMER,
        )

    # Local offline path: zero network calls.
    is_synth = False
    synth_id: str | None = None
    if req.synthetic_profile_id is not None:
        profile = get_synthetic_profile(req.synthetic_profile_id)
        patient = profile.to_patient_input()
        is_synth = True
        synth_id = profile.profile_id
    else:
        assert req.patient is not None
        patient = req.patient

    pred = predict_patient(patient)
    local_text = build_local_explanation(pred, req.question, synthetic_profile_id=synth_id)
    safe_text = enforce_authoritative_output_gate(local_text, pred)
    return AssistantResponse(
        mode="local_offline",
        is_synthetic_profile=is_synth,
        synthetic_profile_id=synth_id,
        external_api_called=False,
        prediction=pred,
        explanation=safe_text,
        key_takeaways=_key_takeaways(pred),
        uncertainty_reminders=_uncertainty_reminders(pred),
        spend_usd=0.0,
        informational_only=True,
        not_medical_advice=True,
        disclaimer=INFORMATIONAL_DISCLAIMER,
    )
