"""Security, redaction, and privacy hygiene utilities.

Preserves useful diagnostic error messages while removing API keys, bearer tokens,
and private local filesystem paths before anything reaches a UI, API response,
log file, or exported artifact.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

# Fields that indicate directly identifying personal health information (PII/PHI).
FORBIDDEN_PII_FIELDS: frozenset[str] = frozenset({
    "name",
    "full_name",
    "first_name",
    "last_name",
    "patient_name",
    "email",
    "phone",
    "phone_number",
    "ssn",
    "social_security_number",
    "mrn",
    "medical_record_number",
    "patient_id",
    "dob",
    "date_of_birth",
    "birth_date",
    "address",
    "street_address",
    "zip_code",
    "postal_code",
    "insurance_id",
    "health_card",
})

# Environment variable names whose values must never be emitted anywhere.
SENSITIVE_ENV_VARS: tuple[str, ...] = (
    "MODEL_API_KEY",
    "META_API_KEY",
    "API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
)

# Ordered secret and path redaction patterns.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization / Bearer headers (ignore already-redacted placeholders)
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)(?!\[REDACTED)([^\s\"',;]+)"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(?i)\bbearer\s+(?!\[REDACTED)[A-Za-z0-9._\-~+/=]{8,}"), "Bearer [REDACTED_TOKEN]"),
    # Common API key prefixes (Meta LLM|, sk-, AIza, ghp_, etc.)
    (re.compile(r"\bLLM\|[A-Za-z0-9|._\-]{8,}"), "[REDACTED_SECRET]"),
    (re.compile(r"\b(?:sk|pk|rk|ak|ghp|gho|github_pat|AIza)[A-Za-z0-9_\-]{10,}"), "[REDACTED_SECRET]"),
    # Key/token/secret query or assignment forms (ignore already-redacted placeholders)
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|token)"
            r"\s*[:=]\s*['\"]?)(?!\[REDACTED)([^\s'\",;}{]+)"
        ),
        r"\1[REDACTED_SECRET]",
    ),
)

_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Windows drive paths, e.g. C:\Users\alice\... or C:/Users/alice/...
    (re.compile(r"[A-Za-z]:\\(?:[^\s\"'<>|;,)}\]]+\\)*[^\s\"'<>|;,)}\]]*"), "[REDACTED_PATH]"),
    (re.compile(r"[A-Za-z]:/(?:[^\s\"'<>|;,)}\]]+/)*[^\s\"'<>|;,)}\]]*"), "[REDACTED_PATH]"),
    # Windows UNC paths \\server\share\...
    (re.compile(r"\\\\[A-Za-z0-9_.\-]+\\[^\s\"'<>|;,)}\]]+"), "[REDACTED_PATH]"),
    # POSIX home/system paths (/Users/..., /home/..., /root/..., /var/..., /etc/..., /tmp/...)
    (
        re.compile(r"(?<![A-Za-z0-9_:/])/(?:Users|home|root|var|etc|tmp|private|opt|mnt)/[^\s\"'<>|;,)}\]]+"),
        "[REDACTED_PATH]",
    ),
)


def _active_secret_values(extra_secrets: tuple[str, ...] = ()) -> list[str]:
    secrets: list[str] = []
    for env_name in SENSITIVE_ENV_VARS:
        val = os.environ.get(env_name)
        if val and len(val) >= 6:
            secrets.append(val)
    for extra in extra_secrets:
        if extra and len(extra) >= 6:
            secrets.append(extra)
    # Longest first to avoid partial replacement leaving fragments.
    secrets.sort(key=len, reverse=True)
    return secrets


def redact_text(text: str, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Redact live secrets, token patterns, and private filesystem paths from text."""
    if not text:
        return text
    cleaned = str(text)
    for secret in _active_secret_values(extra_secrets):
        cleaned = cleaned.replace(secret, "[REDACTED_SECRET]")
    for pattern, replacement in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    for pattern, replacement in _PATH_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def contains_sensitive_material(text: str, *, extra_secrets: tuple[str, ...] = ()) -> bool:
    """Return True if text still contains an active secret value or raw private path."""
    if not text:
        return False
    for secret in _active_secret_values(extra_secrets):
        if secret in text:
            return True
    for pattern, _ in _SECRET_PATTERNS:
        if pattern.search(text):
            return True
    for pattern, _ in _PATH_PATTERNS:
        if pattern.search(text):
            return True
    return False


def format_safe_error(exc: BaseException, *, context: str | None = None) -> str:
    """Return a diagnostic error message with secrets and private paths redacted."""
    err_type = type(exc).__name__
    raw_msg = str(exc).strip() or err_type
    safe_msg = redact_text(raw_msg)
    if context:
        safe_ctx = redact_text(context)
        return f"{safe_ctx}: {err_type}: {safe_msg}"
    return f"{err_type}: {safe_msg}"


def find_pii_keys(payload: Any) -> list[str]:
    """Recursively find any disallowed PII/PHI keys in a nested dict/list payload."""
    found: list[str] = []

    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_str = str(k).strip().lower()
                full = f"{prefix}.{k}" if prefix else str(k)
                if key_str in FORBIDDEN_PII_FIELDS:
                    found.append(full)
                _walk(v, full)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{prefix}[{idx}]")

    _walk(payload)
    return sorted(set(found))


def safe_relative_path(path: Path, root: Path) -> str:
    """Return a POSIX relative path under root, or basename only if outside root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


class RedactingLogFilter(logging.Filter):
    """Logging filter that strips secrets, paths, and accidental patient payload markers."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_text(rendered)
        record.args = ()
        return True
