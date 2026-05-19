
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any, Iterable, Iterator

LABELS = ["ACTOR","BEHAVIOR","CONDITION","DOCUMENT","INFRASTRUCTURE","VEHICLE","VEHICLE_CONDITION_OR_EQUIPMENT"]

LABEL_IMPORTANCE = {
    "BEHAVIOR": 1.15, "CONDITION": 1.05, "VEHICLE": 1.00, "DOCUMENT": 0.95,
    "VEHICLE_CONDITION_OR_EQUIPMENT": 0.95, "ACTOR": 0.85, "INFRASTRUCTURE": 0.80,
}

GENERIC_SURFACES = {
    "đường bộ","xe cơ giới","phương tiện","phương tiện giao thông","người","tổ chức",
    "cá nhân","cơ quan","thiết bị","hệ thống","đường","làn","khu vực","giấy tờ",
    "hồ sơ","văn bản"
}

def ensure_dir(path: str|Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def read_json(path: str|Path) -> Any:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def write_json(path: str|Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def read_jsonl(path: str|Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def write_jsonl(path: str|Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def stable_id(*parts: str, prefix: str="id") -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return prefix + "_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("–","-").replace("—","-")
    text = re.sub(r"[“”\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def iter_sentence_entity_files(root: str|Path) -> list[Path]:
    root = Path(root)
    if (root/"sentence_entities.jsonl").exists():
        return [root/"sentence_entities.jsonl"]
    files = sorted(root.glob("*/sentence_entities.jsonl"))
    if not files:
        raise FileNotFoundError(f"No sentence_entities.jsonl found under {root}")
    return files

def span_overlap(a: dict, b: dict) -> bool:
    try: return int(a["start"]) < int(b["end"]) and int(b["start"]) < int(a["end"])
    except Exception: return False

def span_len(e: dict) -> int:
    try: return int(e["end"]) - int(e["start"])
    except Exception: return len(str(e.get("text","")))

def entity_id(label: str, canonical: str|None, text: str|None) -> str:
    return stable_id(label, normalize_text(canonical or text or ""), prefix="ent")

def surface_is_generic(text: str) -> bool:
    return normalize_text(text) in GENERIC_SURFACES

def base_weight(label: str, text: str, source: str, confidence: float=1.0, scope: str="direct") -> float:
    w = LABEL_IMPORTANCE.get(label, 1.0)
    if source == "hybrid_agree": w *= 1.10
    elif source == "gliner": w *= 0.70 * max(0.0, min(float(confidence), 1.0))
    elif source == "gazetteer": w *= 0.75
    else: w *= 0.60
    if scope == "inherited": w *= 0.35
    if surface_is_generic(text): w *= 0.35
    if len(normalize_text(text).split()) <= 1: w *= 0.65
    return round(float(w), 6)
