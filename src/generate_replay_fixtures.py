"""Generate explicitly labeled offline schema-replay fixtures for registered synthetic profiles.

HONESTY NOTICE:
These files are offline replay stubs matching the expected OpenAI/Meta `/v1/chat/completions`
response schema so replay-mode unit tests and UI demos run deterministically with zero network
calls and $0.00 spend. Each file is explicitly marked `"live_api_verified": false` and
`"fixture_origin": "offline_schema_replay_stub"`. To replace a stub with a genuine recorded Meta
API response, an operator must run with `HEART_ASSISTANT_MODE=record`, `HEART_LIVE_CLEARED=1`,
and `MODEL_API_KEY` set.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.assistant import build_local_explanation, build_meta_messages
from src.inference import predict_patient
from src.meta_client import DEFAULT_FIXTURES_DIR, canonical_request_hash
from src.meta_ledger import DEFAULT_ASSISTANT_MODEL
from src.schemas import DEFAULT_ASSISTANT_QUESTION, EXTERNAL_QUESTION_TEMPLATES
from src.synthetic_profiles import SYNTHETIC_PROFILES

DEFAULT_QUESTION = DEFAULT_ASSISTANT_QUESTION


def generate_offline_replay_stubs(fixtures_dir: Path = DEFAULT_FIXTURES_DIR) -> list[Path]:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # One stub per (synthetic profile x canonical question template): these are the only
    # requests the external path will ever accept.
    pairs = [(p, q) for p in SYNTHETIC_PROFILES for q in EXTERNAL_QUESTION_TEMPLATES.values()]
    for profile, question in pairs:
        patient = profile.to_patient_input()
        features = patient.model_dump()
        pred = predict_patient(patient)
        messages = build_meta_messages(profile.profile_id, features, pred, question)
        json_body = {
            "model": DEFAULT_ASSISTANT_MODEL,
            "messages": messages,
            "max_tokens": 450,
            "temperature": 0.2,
        }
        desc = {
            "model": DEFAULT_ASSISTANT_MODEL,
            "path": "/chat/completions",
            "synthetic_profile_id": profile.profile_id,
            "synthetic_digest": profile.digest(),
            "json": json_body,
        }
        req_hash = canonical_request_hash(desc)
        stub_content = (
            "[OFFLINE REPLAY FIXTURE — NOT A LIVE META API CALL] "
            + build_local_explanation(pred, question, synthetic_profile_id=profile.profile_id)
        )
        fixture = {
            "fixture_origin": "offline_schema_replay_stub",
            "live_api_verified": False,
            "synthetic_only": True,
            "synthetic_profile_id": profile.profile_id,
            "request": desc,
            "response": {
                "id": f"replay-stub-{req_hash[:12]}",
                "object": "chat.completion",
                "model": DEFAULT_ASSISTANT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": stub_content,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 380,
                    "completion_tokens": 135,
                    "total_tokens": 515,
                },
            },
        }
        out = fixtures_dir / f"{req_hash}.json"
        out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(out)
    return written


def main() -> None:
    paths = generate_offline_replay_stubs()
    print(f"Wrote {len(paths)} explicitly labeled offline replay stubs to {DEFAULT_FIXTURES_DIR.name}/")


if __name__ == "__main__":
    main()
