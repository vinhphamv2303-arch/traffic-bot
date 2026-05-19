from __future__ import annotations
import csv, json, random, re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

ALLOWED_LABELS = {
    "ACTOR", "BEHAVIOR", "CONDITION", "DOCUMENT", "INFRASTRUCTURE", "VEHICLE", "VEHICLE_CONDITION_OR_EQUIPMENT"
}

def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def is_reference_like(text: str) -> bool:
    t = collapse_ws(text).lower()
    pats = [
        r"^điều\s+\d+[a-z]?$", r"^khoản\s+\d+$", r"^điểm\s+[a-zđ]$",
        r"^phụ\s+lục\s+([ivxlcdm]+|\d+|[a-z])$", r"^mẫu\s+(số\s+)?[0-9a-z_.\-\/]+$",
        r"^qcvn\s+.+$", r"^\d+/\d{4}/[a-zđ]+.*$",
    ]
    return any(re.match(p, t, flags=re.I | re.U) for p in pats)

def is_too_generic(text: str, label: str) -> bool:
    t = collapse_ws(text).lower()
    generic = {
        "BEHAVIOR": {"vi phạm", "vi phạm quy định", "thực hiện", "tham gia giao thông", "bị xử phạt"},
        "VEHICLE": {"xe", "phương tiện", "loại xe", "phương tiện giao thông"},
        "ACTOR": {"người", "cá nhân", "tổ chức", "đối tượng", "cơ quan có thẩm quyền", "người điều khiển phương tiện"},
        "INFRASTRUCTURE": {"đường", "nơi", "khu vực", "vị trí", "địa điểm", "hệ thống", "thiết bị"},
        "DOCUMENT": {"văn bản", "quy định", "hồ sơ", "giấy tờ"},
        "VEHICLE_CONDITION_OR_EQUIPMENT": {"thiết bị", "bộ phận", "không đạt chuẩn", "không đúng quy định", "không bảo đảm"},
        "CONDITION": {"điều kiện", "yêu cầu", "tiêu chuẩn", "quy định", "phù hợp", "được phép", "không được"},
    }
    return t in generic.get(label, set()) or len(t) <= 2

def normalize_entity(e: Dict[str, Any], text: str, auto_clean: bool = True):
    label = e.get("label")
    if label not in ALLOWED_LABELS:
        return None
    start, end = e.get("start"), e.get("end")
    span_text = e.get("text") or ""
    if start is None or end is None:
        idx = text.find(span_text)
        if idx < 0: idx = text.lower().find(span_text.lower())
        if idx < 0: return None
        start, end = idx, idx + len(span_text)
    try:
        start, end = int(start), int(end)
    except Exception:
        return None
    if start < 0 or end <= start or end > len(text): return None
    span = text[start:end]
    if auto_clean and (is_reference_like(span) or is_too_generic(span, label)):
        return None
    return {"start": start, "end": end, "label": label, "text": span}

def clean_row(row: Dict[str, Any], auto_clean: bool = True) -> Dict[str, Any]:
    text = row.get("text") or ""
    ents = []
    for e in row.get("entities") or []:
        ne = normalize_entity(e, text, auto_clean=auto_clean)
        if ne: ents.append(ne)
    ents.sort(key=lambda x: (x["start"], -(x["end"]-x["start"])))
    kept, last_end = [], -1
    for e in ents:
        if e["start"] < last_end: continue
        kept.append(e); last_end = e["end"]
    out = dict(row); out["entities"] = kept
    return out

def split_train_dev(rows: List[Dict[str, Any]], eval_ratio: float, seed: int):
    rows = list(rows); random.Random(seed).shuffle(rows)
    if eval_ratio <= 0: return rows, []
    n_dev = max(1, int(len(rows) * eval_ratio))
    return rows[n_dev:], rows[:n_dev]

def balance_negatives(rows: List[Dict[str, Any]], negative_ratio: float, seed: int):
    if negative_ratio <= 0: return rows
    pos = [r for r in rows if r.get("entities")]; neg = [r for r in rows if not r.get("entities")]
    if not pos: return rows
    max_neg = int(len(pos) * negative_ratio)
    if len(neg) > max_neg: neg = random.Random(seed).sample(neg, max_neg)
    out = pos + neg; random.Random(seed).shuffle(out); return out

def label_summary(rows):
    counts = {}
    for r in rows:
        for e in r.get("entities") or []:
            counts[e["label"]] = counts.get(e["label"], 0) + 1
    return dict(sorted(counts.items()))
