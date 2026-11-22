"""Metered Meta API adapter for the heart-screening informational assistant.

This module is the ONLY code in the repository that reads MODEL_API_KEY or contacts
https://api.meta.ai/v1.

Hard guardrails enforced in code:
1. Synthetic-only privacy boundary: every call MUST pass `assert_registered_synthetic()`
   with a registered `synthetic_profile_id` before any payload is built or sent.
2. Replay-first testing: default mode is `replay` (zero network, zero spend, reads
   recorded fixtures in fixtures/recorded/<hash>.json; fails loudly if missing).
3. Live clearance gate: `record` and `live` modes require BOTH
   `HEART_LIVE_CLEARED=1` AND `MODEL_API_KEY` set in the environment.
4. Spend honesty: every `record` or `live` call reserves worst-case cost before the
   HTTP request and settles actual (or worst-case on failure) cost into the hash-chained
   spend ledger.
5. Secret & path safety: MODEL_API_KEY only enters the Authorization header at send time;
   fixtures, ledger lines, and surfaced errors are scanned and redacted.
6. Zero Gemini touch: this adapter never imports, configures, or touches Gemini APIs.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from src.data import PROJECT_ROOT
from src.meta_ledger import (
    DEFAULT_ASSISTANT_MODEL,
    DEFAULT_RUN_CAP_USD,
    GLOBAL_CAP_USD,
    META_BASE_URL,
    MODEL_PRICES,
    SpendLedger,
    SpendMeter,
    UnknownCostError,
    assert_standard_model,
    estimate_request_cost,
)
from src.security import find_pii_keys, format_safe_error
from src.synthetic_profiles import assert_registered_synthetic

MODES: tuple[str, ...] = ("replay", "record", "live")
MODE_ENV = "HEART_ASSISTANT_MODE"
CLEARANCE_ENV = "HEART_LIVE_CLEARED"
KEY_ENV = "MODEL_API_KEY"
DEFAULT_FIXTURES_DIR = PROJECT_ROOT / "fixtures" / "recorded"


class LiveNotClearedError(RuntimeError):
    """Raised when record/live mode is attempted without explicit operator clearance."""


class ReplayFixtureMissingError(FileNotFoundError):
    """Raised when replay mode is asked for a request with no recorded fixture."""


class SecretLeakError(RuntimeError):
    """Raised if an active API key would be written to a fixture or response."""


def _read_api_key() -> str | None:
    val = os.environ.get(KEY_ENV)
    return val if val else None


def api_key_configured() -> bool:
    return _read_api_key() is not None


def live_cleared() -> bool:
    return os.environ.get(CLEARANCE_ENV) == "1"


def _contains_live_key(text: str) -> bool:
    key = _read_api_key()
    return bool(key) and key in text


def canonical_request_hash(desc: dict[str, Any]) -> str:
    canonical = json.dumps(desc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_http_transport(url: str, headers: dict[str, str], payload: bytes, timeout: float = 60.0) -> dict[str, Any]:
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_token_usage(resp: dict[str, Any]) -> dict[str, int] | None:
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return None
    in_tok = usage.get("prompt_tokens", usage.get("input_tokens"))
    out_tok = usage.get("completion_tokens", usage.get("output_tokens"))
    if isinstance(in_tok, int) and isinstance(out_tok, int) and in_tok >= 0 and out_tok >= 0:
        return {"input_tokens": in_tok, "output_tokens": out_tok}
    return None


class MetaAssistantClient:
    """Metered, privacy-gated Meta API client (replay / record / live)."""

    def __init__(
        self,
        *,
        mode: str | None = None,
        model: str = DEFAULT_ASSISTANT_MODEL,
        ledger: SpendLedger | None = None,
        global_cap_usd: float = GLOBAL_CAP_USD,
        run_cap_usd: float = DEFAULT_RUN_CAP_USD,
        envelope_caps: dict[str, float] | None = None,
        run_id: str | None = None,
        fixtures_dir: Path | None = None,
        base_url: str = META_BASE_URL,
        transport: Callable[[str, dict[str, str], bytes], dict[str, Any]] | None = None,
    ):
        resolved_mode = (mode or os.environ.get(MODE_ENV) or "replay").strip().lower()
        if resolved_mode not in MODES:
            raise ValueError(f"Unsupported mode {resolved_mode!r}; expected one of {MODES}.")
        assert_standard_model(model)
        if resolved_mode != "replay":
            if not live_cleared():
                raise LiveNotClearedError(
                    f"Mode '{resolved_mode}' refused: set {CLEARANCE_ENV}=1 for operator-approved live spend."
                )
            if not _read_api_key():
                raise LiveNotClearedError(
                    f"Mode '{resolved_mode}' refused: {KEY_ENV} is not set."
                )

        self.mode = resolved_mode
        self.model = model
        self.ledger = ledger or SpendLedger()
        self.meter = SpendMeter(
            self.ledger,
            global_cap_usd=global_cap_usd,
            run_cap_usd=run_cap_usd,
            envelope_caps=envelope_caps,
            run_id=run_id,
            prices=MODEL_PRICES,
        )
        self.fixtures_dir = Path(fixtures_dir or DEFAULT_FIXTURES_DIR)
        self.base_url = base_url.rstrip("/")
        self.transport = transport or default_http_transport
        self._lock = threading.Lock()
        self._pending_holds: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"MetaAssistantClient(mode={self.mode!r}, model={self.model!r}, run_id={self.meter.run_id!r})"

    def chat_synthetic(
        self,
        *,
        synthetic_profile_id: str,
        synthetic_features: dict[str, Any],
        messages: list[dict[str, str]],
        envelope: str = "assistant_explain",
        max_output_tokens: int = 450,
        temperature: float = 0.2,
        path: str = "/chat/completions",
    ) -> dict[str, Any]:
        """Send a chat completion request strictly tied to a verified synthetic profile.

        Returns {"response": <api_json>, "mode": ..., "cost_usd": ..., "request_hash": ..., "alarms": [...]}.
        """
        # 1. Authoritative privacy gate: synthetic profiles ONLY, zero PII.
        profile = assert_registered_synthetic(synthetic_profile_id, synthetic_features)
        pii_hits = find_pii_keys(synthetic_features)
        if pii_hits:
            raise PermissionError(f"PII keys refused before external API call: {pii_hits}")

        assert_standard_model(self.model)
        json_body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_output_tokens),
            "temperature": float(temperature),
        }
        desc = {
            "model": self.model,
            "path": path,
            "synthetic_profile_id": profile.profile_id,
            "synthetic_digest": profile.digest(),
            "json": json_body,
        }
        req_hash = canonical_request_hash(desc)

        if self.mode == "replay":
            return {
                "response": self._load_replay(req_hash, profile.profile_id),
                "mode": "replay",
                "cost_usd": 0.0,
                "request_hash": req_hash,
                "alarms": [],
                "live_verified": False,
            }

        # Conservative worst-case input token estimate (~2.5 chars/token + buffer)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        worst_units = {
            "input_tokens": max(256, int(prompt_chars / 2.5) + 128),
            "output_tokens": int(max_output_tokens),
        }

        with self._lock:
            est_cost, env_before = self.meter.preflight(
                model=self.model,
                envelope=envelope,
                worst_units=worst_units,
                pending_holds=self._pending_holds,
            )
            hold = {
                "cost_usd": est_cost,
                "run_id": self.meter.run_id,
                "envelope": envelope,
                "pending": True,
            }
            self._pending_holds.append(hold)

        t0 = time.monotonic()
        try:
            resp = self._send_http(path, json_body)
        except Exception as exc:
            latency_ms = round((time.monotonic() - t0) * 1000)
            self._settle(
                hold,
                envelope=envelope,
                units=worst_units,
                cost_usd=est_cost,
                cost_basis="failed_request_worst_case",
                request_hash=req_hash,
                synthetic_profile_id=profile.profile_id,
                latency_ms=latency_ms,
                error=type(exc).__name__,
            )
            self.meter.check_alarms(envelope, env_before, env_before + est_cost)
            raise RuntimeError(format_safe_error(exc, context="Meta API request failed")) from exc
        finally:
            self._release(hold)

        latency_ms = round((time.monotonic() - t0) * 1000)
        usage_units = _extract_token_usage(resp)
        try:
            if usage_units is None:
                raise UnknownCostError("Response missing token usage")
            actual_cost = estimate_request_cost(self.model, usage_units, MODEL_PRICES)
            cost_basis = "usage"
            logged_units: dict[str, Any] = usage_units
        except UnknownCostError:
            actual_cost = est_cost
            cost_basis = "worst_case_estimate"
            logged_units = worst_units

        self._settle(
            hold,
            envelope=envelope,
            units=logged_units,
            cost_usd=actual_cost,
            cost_basis=cost_basis,
            request_hash=req_hash,
            synthetic_profile_id=profile.profile_id,
            latency_ms=latency_ms,
        )
        alarms = self.meter.check_alarms(envelope, env_before, env_before + actual_cost)

        if self.mode == "record":
            self._save_fixture(req_hash, desc, resp)

        return {
            "response": resp,
            "mode": self.mode,
            "cost_usd": actual_cost,
            "request_hash": req_hash,
            "alarms": alarms,
            "live_verified": True,
        }

    def _send_http(self, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
        key = _read_api_key()
        if not key:
            raise LiveNotClearedError(f"{KEY_ENV} is not set.")
        payload = json.dumps(json_body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        return self.transport(f"{self.base_url}{path}", headers, payload)

    def _settle(self, hold: dict[str, Any], **kwargs: Any) -> None:
        with self._lock:
            self.ledger.append(model=self.model, run_id=self.meter.run_id, mode=self.mode, **kwargs)
            self._pending_holds[:] = [h for h in self._pending_holds if h is not hold]

    def _release(self, hold: dict[str, Any]) -> None:
        with self._lock:
            self._pending_holds[:] = [h for h in self._pending_holds if h is not hold]

    def _fixture_path(self, req_hash: str) -> Path:
        return self.fixtures_dir / f"{req_hash}.json"

    def _load_replay(self, req_hash: str, synthetic_profile_id: str) -> dict[str, Any]:
        path = self._fixture_path(req_hash)
        if not path.exists():
            raise ReplayFixtureMissingError(
                f"No recorded Meta API replay fixture for {synthetic_profile_id} "
                f"(request {req_hash[:12]}). Record a cleared live fixture or use local_offline mode."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["response"]

    def _save_fixture(self, req_hash: str, desc: dict[str, Any], resp: dict[str, Any]) -> None:
        text = json.dumps(
            {
                "recorded_kind": "meta_api_fixture",
                "synthetic_only": True,
                "request": desc,
                "response": resp,
            },
            indent=2,
            sort_keys=True,
        )
        if _contains_live_key(text):
            raise SecretLeakError("Refusing to write a fixture containing MODEL_API_KEY.")
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self._fixture_path(req_hash).write_text(text + "\n", encoding="utf-8")
