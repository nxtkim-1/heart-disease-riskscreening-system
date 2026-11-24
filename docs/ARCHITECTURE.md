# Heart Disease Risk Screening System: architecture and decision record

The constraint: the 184 test patients are read once, at the very end. Every cleaning, tuning and selection decision is made on training data, and the served model is a frozen, hash-verified artifact.

## System walk

The UCI four-hospital file (920 patients) is cleaned once, in one place (src/data.py): impossible zeros become missing, columns missing for most patients (`ca` 66%, `thal` 53%) are dropped, and the hospital column is dropped because disease rate varies by site (36% to 93%) and a new patient comes from none of them. One 80/20 split (seed 42) is made up front. Training fits each model as a full sklearn Pipeline (impute, scale, one-hot, then classifier), tunes the decision threshold by 5-fold cross-validation for recall >= 0.90, and only then reads the test set. Serving loads the frozen baseline artifact, verifies its SHA-256, and returns the probability at threshold 0.39 with a bootstrap 95% interval, per-feature log-odds contributions, and explicit warnings. The Streamlit app and the hardened FastAPI backend share that one inference path.

## Decision records

### D1. The test set is read once

- Choice: one split at the start; all tuning on CV; the final model selected by a rule fixed in advance (among models tuned to 90% recall in CV, the fewest false alarms). Logistic regression won. XGBoost missed one fewer patient on the test set, but switching because of the test set would make the test set useless as an honest check (README).
- Measured (test, 184 patients, threshold 0.39): accuracy 0.81, precision 0.78, recall 0.92, ROC-AUC 0.89; 8 sick patients missed, 27 false alarms. Recall 95% bootstrap CI 86.7-97.0% (shown in the app sidebar).

### D2. Lower the threshold instead of chasing model complexity

- Measured: moving logistic regression from 0.50 to 0.39 halved missed sick patients (16 to 8) for 5 extra false alarms; all three models land within a couple of points of each other on 920 patients, and logistic regression is the easiest to explain (README results table and notes).

### D3. The hospital column is a shortcut, so it is dropped

- Evidence: 93% of Swiss patients vs 36% of Hungarian patients have disease; a model that can see the site leans on it (README, reports/disease_by_hospital.png). The leave-one-hospital-out audit added later shows strong site calibration shift across the four hospitals, and the app says so next to its metrics.

### D4. Serve a frozen artifact, verify it, fail closed

- Choice: inference always loads `models/heart_model.joblib` and verifies its SHA-256 before use; any failure raises ModelLoadError and stops serving (src/inference.py, src/harmonization.py). Improved candidates (a B=200 bootstrap ensemble fitted on training rows only for epistemic intervals, and a harmonized calibrated logistic regression as a probability reference) are versioned separately under models/candidates/ with provenance files, never silently swapped in.
- Alternative: retrain on startup; rejected because a served number must be reproducible from a fixed artifact.

### D5. Honest output over confident output

- Choice: every prediction returns the probability plus a 95% predictive interval from the bootstrap ensemble, exact per-feature log-odds contributions, and warnings for imputed fields, threshold proximity and cohort shift. The Streamlit sidebar shows the decision threshold and the LOHO caveat next to the metrics; a fixed informational disclaimer is always on screen.
- Limits: trained on 920 angiography-referred patients from 1981-1988 (55% prevalence); probabilities do not reflect the general population (stated in code and README).

### D6. The API assumes hostile input

- Choice (product/api.py, src/schemas.py, src/security.py): Pydantic v2 schemas reject out-of-range values, NaN/Inf, extra keys, hospital identifiers and PII; patient input is never logged or persisted; all API routes send Cache-Control: no-store; a redacting log filter strips secrets, tokens and private paths from errors while keeping useful diagnostics.
- Evidence: tests/test_product_api.py, tests/test_privacy_and_spend_guards.py, tests/test_validation_and_security.py.

### D7. The assistant can only spend through a ledger, and only on synthetic patients

- Choice: Meta assistant calls require registered synthetic profiles (src/synthetic_profiles.py); a real patient question cannot be sent. All calls go through a metered spend ledger (src/meta_ledger.py) with a per-process run cap shared across requests, and tests run against recorded replay fixtures (fixtures/recorded/) instead of the live API.
- Not yet measured: live-call behaviour is fixture-covered; the fixture set, not a live budget event, is the current evidence.

## Interview questions this doc should let the owner answer

- XGBoost missed one fewer test patient. Why is it not the shipped model?
- What exactly happens on boot if the model file was modified by one byte?
- Which fields of a typed-in patient row could reach a log line, and what stops them?
