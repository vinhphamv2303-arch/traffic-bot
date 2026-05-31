from __future__ import annotations

import json
import re
from typing import Any


def norm(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw LLM output.

    Supports plain JSON and markdown fenced JSON.
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    # Fallback: find first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    return None
