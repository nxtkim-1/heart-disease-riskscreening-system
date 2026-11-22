"""Hardened FastAPI backend for the Heart-Disease Risk Screening System.

Key guarantees:
- Loads the frozen baseline artifact `models/heart_model.joblib` (SHA-256 verified, read-only).
- Strict Pydantic v2 input schemas reject invalid ranges, NaN/Inf, extra keys, hospital IDs, and PII.
- Zero patient input logging or disk persistence; strict Cache-Control: no-store headers on all API routes.
- Error surfacing preserves useful diagnostic details while redacting secrets, tokens, and private paths.
- External Meta API assistant calls strictly require registered SYNTHETIC profiles and go through the
  metered spend ledger (`src/meta_client.py`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.assistant import explain_screening
from src.inference import (
    INFORMATIONAL_DISCLAIMER,
    ModelLoadError,
    load_baseline_bundle,
    load_harmonized_bundle,
    load_statistical_report,
    load_uncertainty_bundle,
    predict_patient,
    readiness_probe,
)
from src.meta_client import (
    LiveNotClearedError,
    MetaAssistantClient,
    ReplayFixtureMissingError,
    SecretLeakError,
    api_key_configured,
    live_cleared,
)
from src.meta_ledger import SpendGuardError, SpendLedger
from src.schemas import AssistantRequest, AssistantResponse, PatientInput, PredictionResponse
from src.security import RedactingLogFilter, format_safe_error, redact_text
from src.synthetic_profiles import SyntheticPrivacyViolation, list_synthetic_profiles

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger("heart_screen.api")
if not any(isinstance(f, RedactingLogFilter) for f in logger.filters):
    logger.addFilter(RedactingLogFilter())


def create_app(
    *,
    meta_client_factory=None,
    ledger: SpendLedger | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Heart-Disease Risk Screening System (Informational Prototype)",
        version="2.0.0",
        description=(
            "Statistical screening prototype on archival UCI coronary-angiography data (1981–1988). "
            "Informational only — NOT a medical device and NOT medical advice."
        ),
    )

    app.state.meta_client_factory = meta_client_factory
    app.state.ledger = ledger

    @app.middleware("http")
    async def session_hygiene_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["X-Session-Ephemeral"] = "true"
            response.headers["X-Patient-Data-Persisted"] = "false"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        sanitized_errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = redact_text(str(err.get("msg", "Invalid input")))
            sanitized_errors.append({"field": loc, "message": msg, "type": str(err.get("type", "value_error"))})
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Input failed clinical schema or privacy validation.",
                "details": sanitized_errors,
            },
        )

    @app.exception_handler(SyntheticPrivacyViolation)
    async def privacy_exception_handler(_request: Request, exc: SyntheticPrivacyViolation):
        return JSONResponse(
            status_code=403,
            content={
                "error": "privacy_boundary_violation",
                "message": redact_text(str(exc)),
            },
        )

    @app.exception_handler(LiveNotClearedError)
    async def clearance_exception_handler(_request: Request, exc: LiveNotClearedError):
        return JSONResponse(
            status_code=403,
            content={
                "error": "live_api_not_cleared",
                "message": redact_text(str(exc)),
            },
        )

    @app.exception_handler(SpendGuardError)
    async def spend_exception_handler(_request: Request, exc: SpendGuardError):
        return JSONResponse(
            status_code=402,
            content={
                "error": "spend_guard_refused",
                "message": redact_text(str(exc)),
            },
        )

    @app.exception_handler(ReplayFixtureMissingError)
    async def replay_missing_handler(_request: Request, exc: ReplayFixtureMissingError):
        return JSONResponse(
            status_code=404,
            content={
                "error": "replay_fixture_missing",
                "message": redact_text(str(exc)),
            },
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(_request: Request, exc: ModelLoadError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "model_unavailable",
                "message": format_safe_error(exc, context="Baseline model unavailable"),
            },
        )

    @app.exception_handler(SecretLeakError)
    async def secret_leak_handler(_request: Request, _exc: SecretLeakError):
        return JSONResponse(
            status_code=500,
            content={
                "error": "secret_guard_triggered",
                "message": "Operation aborted to prevent secret exposure.",
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(_request: Request, exc: Exception):
        logger.error("Unhandled API exception: %s", format_safe_error(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": format_safe_error(exc, context="Unexpected server error"),
            },
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(status_code=404, detail="Frontend index not found.")
        return FileResponse(idx)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        probe = readiness_probe()
        status_code = 200 if probe["ready"] else 503
        payload = {
            **probe,
            "assistant": {
                "external_provider": "meta_api",
                "synthetic_only_enforced": True,
                "live_cleared": live_cleared(),
                "api_key_configured": api_key_configured(),
                "gemini_touched": False,
            },
            "privacy": {
                "patient_inputs_logged": False,
                "patient_inputs_persisted": False,
            },
            "informational_only": True,
        }
        return JSONResponse(status_code=status_code, content=payload)

    @app.get("/api/model/info")
    async def model_info() -> dict[str, Any]:
        bundle = load_baseline_bundle()
        harm = load_harmonized_bundle()
        unc = load_uncertainty_bundle()
        return {
            "baseline": {
                "artifact": "models/heart_model.joblib",
                "model_name": bundle["model_name"],
                "threshold": float(bundle["threshold"]),
                "features": list(bundle["features"]),
                "n_test_historical": int(bundle["n_test"]),
                "historical_test_metrics": bundle["test_metrics"],
                "sha256": bundle["_verified_sha256"],
                "hospital_id_dropped": True,
                "hospital_leakage_note": (
                    "The `dataset` (hospital) column is excluded because disease prevalence ranges "
                    "from 36% (Hungary) to 93.5% (Switzerland); keeping it would let the model cheat "
                    "using hospital shortcuts rather than patient measurements."
                ),
            },
            "versioned_candidates": {
                "harmonized_calibrated": None if harm is None else harm.get("provenance"),
                "uncertainty_bootstrap": None if unc is None else unc.get("provenance"),
            },
            "disclaimer": INFORMATIONAL_DISCLAIMER,
        }

    @app.post("/api/predict", response_model=PredictionResponse)
    async def predict(patient: PatientInput) -> PredictionResponse:
        # Patient features are scored purely in memory and never logged or written to disk.
        return predict_patient(patient)

    @app.get("/api/statistics")
    async def statistics() -> dict[str, Any]:
        report = load_statistical_report()
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="Statistical report not generated yet. Run `python -m src.statistics`.",
            )
        return report

    @app.get("/api/assistant/profiles")
    async def assistant_profiles() -> dict[str, Any]:
        return {
            "synthetic_only_policy": (
                "External Meta API calls only accept these registered synthetic archetype profiles. "
                "Zero real patient data ever leaves this machine."
            ),
            "profiles": list_synthetic_profiles(),
        }

    @app.get("/api/assistant/ledger")
    async def assistant_ledger() -> dict[str, Any]:
        led = app.state.ledger or SpendLedger()
        return {
            "ledger_summary": led.summary(),
            "live_cleared": live_cleared(),
            "api_key_configured": api_key_configured(),
            "operator_instructions": (
                "Live Meta API calls stay blocked until an operator sets HEART_ASSISTANT_MODE=live "
                "(or record), HEART_LIVE_CLEARED=1, and MODEL_API_KEY."
            ),
        }

    @app.post("/api/assistant/explain", response_model=AssistantResponse)
    async def assistant_explain(req: AssistantRequest) -> AssistantResponse:
        client = None
        if req.use_external_meta_api:
            if app.state.meta_client_factory is not None:
                client = app.state.meta_client_factory()
            else:
                client = MetaAssistantClient(ledger=app.state.ledger)
        return explain_screening(req, meta_client=client)

    @app.post("/api/session/clear")
    async def clear_session() -> dict[str, Any]:
        return {
            "cleared": True,
            "server_state_retained": False,
            "message": "Session cleared. No patient measurements are logged or persisted on the server.",
        }

    return app


app = create_app()
