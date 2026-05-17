from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Union


def strip_vietnamese_accents(text: str, keep_dd: bool = False) -> str:
    if keep_dd:
        text = text.replace("đ", "dd").replace("Đ", "dd")
    else:
        text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text or "").encode("ASCII", "ignore").decode("utf-8")


def normalize_id(text: str) -> str:
    """
    Stable ID slug.
    Example:
      12/2025/TT-BCA -> 12_2025_tt_bca
      12/2025/TTBCA  -> 12_2025_ttbca
    """
    text = normalize_document_number(text) if looks_like_document_number(text) else (text or "")
    text = strip_vietnamese_accents(text, keep_dd=True)
    text = text.replace("/", "_").replace("-", "_")
    text = re.sub(r"\s+", "_", text).lower()
    text = re.sub(r"[^a-z0-9_\.]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def clean_filename(text: str) -> str:
    """
    Safe directory/file name for legal document numbers.

    Examples:
      12/2025/TT-BCA  -> 12_2025_TTBCA
      12/2025/TT-BXD  -> 12_2025_TTBXD
      160/2024/NĐ-CP  -> 160_2024_NDCP
      118/2025/QH15   -> 118_2025_QH15
    """
    text = normalize_document_number(text) if looks_like_document_number(text) else (text or "UNKNOWN")
    text = strip_vietnamese_accents(text, keep_dd=False)

    # Legal doc number path style
    if looks_like_document_number(text):
        text = text.replace("/", "_")
        text = text.replace("-", "")
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"[\\/*?:\"<>|]", "_", text.strip())
        text = re.sub(r"[\s_]+", "_", text)

    text = re.sub(r"[\\/*?:\"<>|]", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "UNKNOWN").upper()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Union[str, Path], data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def looks_like_document_number(text: str) -> bool:
    text = text or ""
    return bool(re.search(r"\d+\s*/\s*\d{4}\s*/", text))


def normalize_document_number(text: str) -> str:
    """
    Normalize Vietnamese legal document numbers without losing agency suffix.

    Handles variants:
      Số: 12/2025/TT-BCA
      12/2025/TT BCA
      12 / 2025 / TT - BCA
      118/2025/QH15

    Returns:
      12/2025/TT-BCA
      118/2025/QH15
    """
    raw = collapse_ws(text or "")
    raw = raw.replace("\u00a0", " ")

    # Remove leading "Số:" if caller passed the whole cell.
    raw = re.sub(r"^\s*Số\s*:\s*", "", raw, flags=re.IGNORECASE)

    # Capture number/year/type/suffix. Suffix may be separated by hyphen or whitespace.
    m = re.search(
        r"(?P<num>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*"
        r"(?P<kind>[A-ZĐ]+)"
        r"(?:\s*[-–—]?\s*(?P<agency>[A-ZĐ0-9]+(?:\s*[-–—]\s*[A-ZĐ0-9]+)*))?"
        r"(?=\s*(?:ngày|,|;|\.|\)|$))",
        raw,
        flags=re.IGNORECASE | re.UNICODE,
    )
    if not m:
        # Fallback: compact obvious spaces around slashes/hyphens.
        raw = re.sub(r"\s*/\s*", "/", raw)
        raw = re.sub(r"\s*[-–—]\s*", "-", raw)
        return raw.strip()

    num = m.group("num")
    year = m.group("year")
    kind = (m.group("kind") or "").upper().replace("Đ", "D")
    agency = (m.group("agency") or "").upper().replace("Đ", "D")
    agency = re.sub(r"[^A-Z0-9]+", "", agency)

    if agency:
        if kind == "QH" and agency.isdigit():
            return f"{num}/{year}/{kind}{agency}"
        return f"{num}/{year}/{kind}-{agency}"
    return f"{num}/{year}/{kind}"
