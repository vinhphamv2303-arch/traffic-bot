
from __future__ import annotations
import hashlib, json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()

def strip_vietnamese_accents(text: str, keep_dd: bool = False) -> str:
    if keep_dd:
        text = text.replace("đ", "dd").replace("Đ", "dd")
    else:
        text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text or "").encode("ASCII", "ignore").decode("utf-8")

def normalize_id(text: str) -> str:
    text = strip_vietnamese_accents(text or "", keep_dd=True)
    text = text.replace("/", "_").replace("-", "_")
    text = re.sub(r"\s+", "_", text).lower()
    text = re.sub(r"[^a-z0-9_\.]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"

def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def find_units_files(input_path: Union[str, Path]) -> List[Path]:
    p = Path(input_path)
    if p.is_file() and p.name == "units.jsonl":
        return [p]
    if not p.is_dir():
        return []

    package_main = p / "main" / "units.jsonl"
    if package_main.exists():
        return [package_main]

    main_units = sorted(p.glob("*/main/units.jsonl"))
    if main_units:
        return main_units

    legacy_units = sorted(
        child / "units.jsonl"
        for child in p.iterdir()
        if child.is_dir() and (child / "units.jsonl").exists()
    )
    if legacy_units:
        return legacy_units

    root_units = p / "units.jsonl"
    if root_units.exists():
        return [root_units]
    return []

def units_file_output_name(units_path: Union[str, Path]) -> str:
    p = Path(units_path)
    if p.name == "units.jsonl" and p.parent.name == "main" and p.parent.parent.name:
        return p.parent.parent.name
    return p.parent.name

def parse_vietnamese_date_to_iso(text: str) -> Optional[str]:
    m = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text or "", re.I | re.U)
    if not m:
        m = re.search(r"(?:ngày\s+)?(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", text or "", re.I | re.U)
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        from datetime import date
        date(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"

def extract_all_vietnamese_dates(text: str) -> List[Dict[str, Any]]:
    items = []
    patterns = [
        r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
        r"(?:ngày\s+)?\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}",
    ]
    seen = set()
    for pattern in patterns:
        for m in re.finditer(pattern, text or "", re.I | re.U):
            iso = parse_vietnamese_date_to_iso(m.group(0))
            if not iso:
                continue
            key = (m.start(), m.end(), iso)
            if key in seen:
                continue
            seen.add(key)
            items.append({"date": iso, "raw": collapse_ws(m.group(0)), "span": [m.start(), m.end()]})
    return sorted(items, key=lambda item: item["span"])

def normalize_document_number(text: str) -> Optional[str]:
    if not text:
        return None
    raw = collapse_ws(text).replace("\u00a0", " ")
    m = re.search(
        r"(?P<num>\d+)\s*/\s*(?:(?P<year>\d{4})\s*/\s*)?(?P<kind>[A-ZĐ]+)"
        r"(?:\s*[-–—]?\s*(?P<agency>[A-ZĐ0-9]{1,20}(?:\s*[-–—]\s*[A-ZĐ0-9]{1,20})*))?",
        raw, re.I | re.U)
    if not m:
        return None
    num, year = m.group("num"), m.group("year")
    kind = (m.group("kind") or "").upper().replace("Đ", "D")
    agency = (m.group("agency") or "").upper().replace("Đ", "D")
    agency = re.sub(r"\s*[-–—]\s*", "-", agency)
    agency = re.sub(r"[^A-Z0-9-]+", "", agency)
    agency = re.sub(r"-+", "-", agency).strip("-")
    if strip_vietnamese_accents(agency, keep_dd=False).upper() in {"NG", "NGAY"}:
        agency = ""
    if year and kind == "QH" and agency.isdigit():
        return f"{num}/{year}/QH{agency}"
    if year:
        return f"{num}/{year}/{kind}-{agency}" if agency else f"{num}/{year}/{kind}"
    return f"{num}/{kind}-{agency}" if agency else f"{num}/{kind}"

def previous_day_iso(date_iso: str) -> Optional[str]:
    try:
        from datetime import date, timedelta
        y, m, d = [int(x) for x in date_iso.split("-")]
        return (date(y, m, d) - timedelta(days=1)).isoformat()
    except Exception:
        return None
