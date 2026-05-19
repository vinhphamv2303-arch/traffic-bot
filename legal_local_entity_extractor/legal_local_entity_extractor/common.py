\
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
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


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def normalize_loose(text: str) -> str:
    text = strip_accents((text or "").lower())
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


def is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def boundary_ok(text: str, start: int, end: int) -> bool:
    return (start == 0 or not is_word_char(text[start - 1])) and (end >= len(text) or not is_word_char(text[end]))


def find_all_exact(text: str, needle: str) -> List[Tuple[int, int]]:
    if not text or not needle:
        return []
    out = []
    low = text.lower()
    n = needle.lower()
    pos = 0
    while True:
        idx = low.find(n, pos)
        if idx < 0:
            break
        end = idx + len(needle)
        if boundary_ok(text, idx, end):
            out.append((idx, end))
        pos = idx + 1
    return out


def label_summary(rows: List[Dict[str, Any]], field: str = "entities") -> Dict[str, int]:
    out = {}
    for r in rows:
        for e in r.get(field) or []:
            lab = e.get("label") or "UNKNOWN"
            out[lab] = out.get(lab, 0) + 1
    return dict(sorted(out.items()))


def split_rows(rows: List[Dict[str, Any]], seed: int = 42, dev_ratio: float = 0.1, test_ratio: float = 0.05):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = int(n * test_ratio)
    n_dev = int(n * dev_ratio)
    test = rows[:n_test]
    dev = rows[n_test:n_test+n_dev]
    train = rows[n_test+n_dev:]
    return train, dev, test
