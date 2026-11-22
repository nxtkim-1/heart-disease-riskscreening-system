"""Metered spend ledger and budget guards for Meta API assistant calls.

Follows strict spend-honesty rules:
- Append-only JSONL ledger with cross-process file locking and SHA-256 hash chaining.
- Fail closed if the ledger is unreadable, corrupted, or ends in a partial line.
- Worst-case cost estimation BEFORE any live network request; unknown models or units fail closed.
- Enforces global cap, per-run cap, and named envelope caps with 50% / 80% alarm hooks.
- Never stores API keys, authorization headers, or patient data in ledger records.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

META_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_ASSISTANT_MODEL = "muse-spark-1.3"
PRICE_SOURCE = "https://dev.meta.ai/docs/pricing-rate-limits (Standard mode, checked 2026-10-27)"

# Standard-mode text token prices (USD per 1M tokens).
MODEL_PRICES: dict[str, dict[str, float]] = {
    "muse-spark-1.3": {"input_per_1m": 1.25, "output_per_1m": 4.25},
    "muse-wonder-1.0": {"input_per_1m": 4.00, "output_per_1m": 20.00},
}

# Conservative caps for this heart-screening assistant project.
GLOBAL_CAP_USD = 25.00
DEFAULT_RUN_CAP_USD = 1.00
ENVELOPE_CAPS_USD: dict[str, float] = {
    "assistant_explain": 5.00,
    "assistant_record_fixture": 2.00,
    "assistant_smoke": 1.00,
}

DEFAULT_LEDGER_PATH = Path.home() / ".heart_screen" / "meta_spend_ledger.jsonl"
GENESIS_HASH = "0" * 64


class SpendGuardError(RuntimeError):
    """Base error for spend-ledger or budget refusals."""


class LedgerCorruptError(SpendGuardError):
    """Raised when the spend ledger cannot be read or fails integrity verification."""


class UnknownCostError(SpendGuardError):
    """Raised when pricing or usage units are unknown (fail closed)."""


class BudgetExceededError(SpendGuardError):
    """Raised when a projected worst-case call would exceed global, run, or envelope caps."""


class ContributorModelError(ValueError):
    """Raised when a Contributor-mode model ID is requested (Standard mode only)."""


def assert_standard_model(model_id: str) -> str:
    """Refuse Contributor-mode model IDs everywhere."""
    if "-contributor" in str(model_id).lower():
        raise ContributorModelError(f"Refusing Contributor-mode model id: {model_id!r}")
    return str(model_id)


def resolve_ledger_path() -> Path:
    env = os.environ.get("HEART_LEDGER_PATH")
    return Path(env) if env else DEFAULT_LEDGER_PATH


def _positive_number(d: dict[str, Any] | None, key: str, label: str) -> float:
    val = None if d is None else d.get(key)
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
        raise UnknownCostError(f"{label} '{key}' is missing or invalid")
    return float(val)


def estimate_request_cost(
    model: str,
    units: dict[str, Any],
    prices: dict[str, dict[str, float]] = MODEL_PRICES,
) -> float:
    """Compute exact USD cost for text token units; fail closed on unknown model/units."""
    assert_standard_model(model)
    price = prices.get(model)
    if price is None:
        raise UnknownCostError(f"No approved Standard-mode price for model {model!r}")
    in_tok = _positive_number(units, "input_tokens", "units")
    out_tok = _positive_number(units, "output_tokens", "units")
    in_rate = _positive_number(price, "input_per_1m", f"{model} price")
    out_rate = _positive_number(price, "output_per_1m", f"{model} price")
    return round((in_tok * in_rate + out_tok * out_rate) / 1_000_000.0, 8)


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    canonical = json.dumps(record_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    with open(lock_path, "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _parse_and_verify(text: str, path: Path) -> list[dict[str, Any]]:
    if text and not text.endswith("\n"):
        raise LedgerCorruptError(f"Spend ledger {path.name} ends in an incomplete line.")
    records: list[dict[str, Any]] = []
    expected_prev = GENESIS_HASH
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerCorruptError(f"Spend ledger {path.name} line {lineno} is invalid JSON.") from exc
        if not isinstance(rec, dict):
            raise LedgerCorruptError(f"Spend ledger {path.name} line {lineno} is not a JSON object.")
        cost = rec.get("cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise LedgerCorruptError(f"Spend ledger {path.name} line {lineno} has invalid cost_usd.")
        prev_hash = rec.get("prev_hash")
        entry_hash = rec.get("entry_hash")
        if prev_hash != expected_prev:
            raise LedgerCorruptError(
                f"Spend ledger {path.name} line {lineno} failed hash-chain continuity."
            )
        body = {k: v for k, v in rec.items() if k != "entry_hash"}
        if entry_hash != _record_hash(body):
            raise LedgerCorruptError(f"Spend ledger {path.name} line {lineno} failed SHA-256 verification.")
        expected_prev = entry_hash
        records.append(rec)
    return records


class SpendLedger:
    """Append-only hash-chained Meta API spend ledger."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else resolve_ledger_path()

    def records(self) -> list[dict[str, Any]]:
        try:
            with _exclusive_lock(self.path):
                if not self.path.exists():
                    return []
                text = self.path.read_text(encoding="utf-8")
        except LedgerCorruptError:
            raise
        except (OSError, UnicodeDecodeError) as exc:
            raise LedgerCorruptError(f"Spend ledger {self.path.name} unreadable: {type(exc).__name__}") from exc
        return _parse_and_verify(text, self.path)

    def total_spent(self, *, run_id: str | None = None, envelope: str | None = None) -> float:
        total = 0.0
        for rec in self.records():
            if run_id is not None and rec.get("run_id") != run_id:
                continue
            if envelope is not None and rec.get("envelope") != envelope:
                continue
            total += float(rec["cost_usd"])
        return round(total, 6)

    def append(
        self,
        *,
        model: str,
        envelope: str,
        units: dict[str, Any],
        cost_usd: float,
        run_id: str,
        cost_basis: str,
        mode: str,
        request_hash: str,
        synthetic_profile_id: str,
        latency_ms: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        with _exclusive_lock(self.path):
            existing = []
            if self.path.exists():
                existing = _parse_and_verify(self.path.read_text(encoding="utf-8"), self.path)
            prev_hash = existing[-1]["entry_hash"] if existing else GENESIS_HASH
            body: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model,
                "envelope": envelope,
                "units": units,
                "cost_usd": round(float(cost_usd), 6),
                "cost_basis": cost_basis,
                "run_id": run_id,
                "mode": mode,
                "request_hash": request_hash,
                "synthetic_profile_id": synthetic_profile_id,
                "latency_ms": int(latency_ms),
                "prev_hash": prev_hash,
            }
            if error is not None:
                body["error"] = error
            rec = {**body, "entry_hash": _record_hash(body)}
            line = json.dumps(rec, sort_keys=True) + "\n"
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            return rec

    def summary(self) -> dict[str, Any]:
        recs = self.records()
        by_envelope: dict[str, float] = {}
        for r in recs:
            env = str(r.get("envelope", "unknown"))
            by_envelope[env] = round(by_envelope.get(env, 0.0) + float(r["cost_usd"]), 6)
        return {
            "entries": len(recs),
            "total_spent_usd": round(sum(float(r["cost_usd"]) for r in recs), 6),
            "global_cap_usd": GLOBAL_CAP_USD,
            "by_envelope_usd": by_envelope,
            "envelope_caps_usd": dict(ENVELOPE_CAPS_USD),
            "price_source": PRICE_SOURCE,
        }


class SpendMeter:
    """Preflight budget checker enforcing global, per-run, and envelope caps."""

    def __init__(
        self,
        ledger: SpendLedger,
        *,
        global_cap_usd: float = GLOBAL_CAP_USD,
        run_cap_usd: float = DEFAULT_RUN_CAP_USD,
        envelope_caps: dict[str, float] | None = None,
        run_id: str | None = None,
        prices: dict[str, dict[str, float]] = MODEL_PRICES,
    ):
        self.ledger = ledger
        self.global_cap_usd = float(global_cap_usd)
        self.run_cap_usd = float(run_cap_usd)
        self.envelope_caps = dict(ENVELOPE_CAPS_USD if envelope_caps is None else envelope_caps)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.prices = prices
        self.alarms: list[str] = []

    def preflight(
        self,
        *,
        model: str,
        envelope: str,
        worst_units: dict[str, Any],
        pending_holds: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """Validate envelope, ledger integrity, and all caps BEFORE sending a live request.

        Returns (worst_case_cost_usd, prior_envelope_spend_usd).
        """
        if envelope not in self.envelope_caps:
            raise BudgetExceededError(
                f"Unknown spend envelope {envelope!r}; allowed: {sorted(self.envelope_caps)}"
            )
        est = estimate_request_cost(model, worst_units, self.prices)
        records = self.ledger.records()  # Fails closed if corrupt/unreadable.
        all_items = records + list(pending_holds)

        global_spent = sum(float(r["cost_usd"]) for r in all_items)
        if global_spent + est > self.global_cap_usd + 1e-12:
            raise BudgetExceededError(
                f"Global cap exceeded: spent=${global_spent:.4f} + est=${est:.4f} > cap=${self.global_cap_usd:.2f}"
            )

        run_spent = sum(float(r["cost_usd"]) for r in all_items if r.get("run_id") == self.run_id)
        if run_spent + est > self.run_cap_usd + 1e-12:
            raise BudgetExceededError(
                f"Per-run cap exceeded: spent=${run_spent:.4f} + est=${est:.4f} > cap=${self.run_cap_usd:.2f}"
            )

        env_cap = self.envelope_caps[envelope]
        env_spent = sum(float(r["cost_usd"]) for r in all_items if r.get("envelope") == envelope)
        if env_spent + est > env_cap + 1e-12:
            raise BudgetExceededError(
                f"Envelope '{envelope}' cap exceeded: spent=${env_spent:.4f} + est=${est:.4f} > cap=${env_cap:.2f}"
            )

        return est, env_spent

    def check_alarms(self, envelope: str, before: float, after: float) -> list[str]:
        cap = self.envelope_caps.get(envelope, 0.0)
        if cap <= 0:
            return []
        triggered: list[str] = []
        for pct in (0.50, 0.80):
            threshold = cap * pct
            if before < threshold <= after:
                msg = f"Spend alarm: envelope '{envelope}' crossed {int(pct * 100)}% (${after:.4f} / ${cap:.2f})"
                triggered.append(msg)
                self.alarms.append(msg)
        return triggered
