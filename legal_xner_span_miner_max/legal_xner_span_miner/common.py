from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


LABELS = [
    "BEHAVIOR",
    "VEHICLE",
    "ACTOR",
    "INFRASTRUCTURE",
    "DOCUMENT",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
    "CONDITION",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def normalize_surface(text: str) -> str:
    text = collapse_ws(text).lower()
    text = re.sub(r"^[\"'“”‘’\(\[\{]+|[\"'“”‘’\)\]\};:,.]+$", "", text)
    return collapse_ws(text)


def normalize_key(text: str) -> str:
    text = strip_accents(normalize_surface(text))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return collapse_ws(text)


def stable_id(*parts: str, prefix: str = "id") -> str:
    raw = "|".join([p or "" for p in parts])
    return prefix + "_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def find_sentence_package_dirs(sentences_root: str | Path) -> List[Path]:
    p = Path(sentences_root)
    if (p / "sentences.jsonl").exists():
        return [p]
    return sorted([x for x in p.iterdir() if x.is_dir() and (x / "sentences.jsonl").exists()])


def is_reference_like(text: str) -> bool:
    t = normalize_surface(text)
    patterns = [
        r"^điều\s+\d+[a-z]?$",
        r"^khoản\s+\d+$",
        r"^điểm\s+[a-zđ]$",
        r"^phụ\s+lục\s+([ivxlcdm]+|\d+|[a-z])$",
        r"^mẫu\s+(số\s+)?[0-9a-z_.\-\/]+$",
        r"^chương\s+([ivxlcdm]+|\d+)$",
        r"^mục\s+([ivxlcdm]+|\d+)$",
        r"^qcvn\s+.+$",
        r"^\d+/\d{4}/[a-zđ]+.*$",
        r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+(này|số\s+.+)$",
    ]
    return any(re.match(p, t, flags=re.I | re.U) for p in patterns)


def token_count(text: str) -> int:
    return len([t for t in collapse_ws(text).split(" ") if t])


def is_too_generic(text: str, label: str | None = None) -> bool:
    t = normalize_surface(text)
    generic = {
        "vi phạm", "vi phạm quy định", "thực hiện", "tham gia giao thông", "bị xử phạt", "xử phạt",
        "xe", "phương tiện", "loại xe", "phương tiện giao thông",
        "người", "cá nhân", "tổ chức", "đối tượng", "người điều khiển phương tiện",
        "đường", "nơi", "khu vực", "vị trí", "địa điểm", "hệ thống", "thiết bị",
        "văn bản", "quy định", "hồ sơ", "giấy tờ", "bộ phận",
        "điều kiện", "yêu cầu", "tiêu chuẩn", "phù hợp", "được phép", "không được",
    }
    if t in generic:
        return True
    if len(t) <= 2:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    return False


def is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def boundary_ok(text: str, start: int, end: int) -> bool:
    return (start == 0 or not is_word_char(text[start - 1])) and (end >= len(text) or not is_word_char(text[end]))


def find_exact_spans(text: str, surface: str) -> List[Tuple[int, int]]:
    low = text.lower()
    needle = surface.lower()
    out = []
    pos = 0
    while True:
        i = low.find(needle, pos)
        if i < 0:
            break
        j = i + len(surface)
        if boundary_ok(text, i, j):
            out.append((i, j))
        pos = i + 1
    return out


def non_overlapping_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mentions = sorted(
        mentions,
        key=lambda x: (
            x.get("start", 0),
            -(int(x.get("end", 0)) - int(x.get("start", 0))),
            -float(x.get("score", 0)),
        ),
    )
    kept = []
    last_end = -1
    for m in mentions:
        s = int(m.get("start", -1))
        e = int(m.get("end", -1))
        if s < last_end:
            continue
        kept.append(m)
        last_end = e
    return kept


def summarize_counts(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    c = Counter([r.get(key) or "UNKNOWN" for r in rows])
    return dict(sorted(c.items()))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default
