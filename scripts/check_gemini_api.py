"""Kiem tra API key/model Gemini bang dung mot request text rat nho.

Chay tu root repo:
    python -u -m scripts.check_gemini_api
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from aic2026.gemini_qa import GeminiQAError, GeminiQAReader  # noqa: E402


def main() -> int:
    reader = GeminiQAReader(model=os.getenv('GEMINI_MODEL', 'gemini-3.6-flash'), max_images=1, retries=0)
    if not reader.configured:
        print("FAIL: thieu GEMINI_API_KEY trong .env")
        return 2

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    "Tra ve dung JSON sau, khong them chu: "
                    '{"status":"ok"}'
                )
            }],
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
        },
    }
    try:
        response = reader._call(payload)
        parsed = reader._extract_json(response)
    except GeminiQAError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"OK: model={reader.model}")
    print(json.dumps(parsed, ensure_ascii=False))
    print("Key khong duoc in ra va khong nam trong cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
