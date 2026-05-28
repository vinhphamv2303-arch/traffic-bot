from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

LABELS = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]

LABEL_DESCRIPTIONS = {
    "ACTOR": "actor",
    "BEHAVIOR": "behavior",
    "CONDITION": "condition",
    "DOCUMENT": "document",
    "INFRASTRUCTURE": "infrastructure",
    "VEHICLE": "vehicle",
    "VEHICLE_CONDITION_OR_EQUIPMENT": "vehicle_condition_or_equipment",
}


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


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stable_id(*parts: str, prefix: str = "id") -> str:
    raw = "|".join([p or "" for p in parts])
    return prefix + "_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def iter_sentence_entity_files(entities_root: str | Path) -> List[Path]:
    root = Path(entities_root)

    candidate_names = [
        "sentences_with_entities.jsonl",  # new canonical name
        "sentence_entities.jsonl",  # legacy name
        "sentences_with_entity_links.jsonl",  # old gazetteer name
    ]

    for name in candidate_names:
        if (root / name).exists():
            return [root / name]

    files = []
    for name in candidate_names:
        files.extend(sorted(root.glob(f"*/{name}")))

    if not files:
        raise FileNotFoundError(
            f"No sentence entity files found under {root}. "
            f"Expected one of: {', '.join(candidate_names)}"
        )

    # Deduplicate in case multiple legacy aliases exist in the same package.
    seen_packages = set()
    selected = []
    for f in files:
        pkg = f.parent
        if pkg in seen_packages:
            continue
        seen_packages.add(pkg)
        selected.append(f)

    return selected


def token_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text or ""):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def char_to_token_span(start: int, end: int, offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if e <= start or s >= end:
            continue
        idxs.append(i)
    if not idxs:
        return None
    return idxs[0], idxs[-1]


def entity_len(e: Dict[str, Any]) -> int:
    try:
        return int(e.get("end", 0)) - int(e.get("start", 0))
    except Exception:
        return 0


def clean_direct_entities(
        text: str,
        entities: list[dict[str, Any]],
        keep_labels: list[str] | None = None,
        min_weight: float = 0.0,
) -> list[dict[str, Any]]:
    keep = set(keep_labels or LABELS)
    n = len(text or "")
    cleaned = []

    for e in entities or []:
        label = e.get("label")
        if label not in keep:
            continue
        try:
            start, end = int(e.get("start")), int(e.get("end"))
        except Exception:
            continue
        if start < 0 or end <= start or end > n:
            continue
        weight = float(e.get("graph_weight", e.get("confidence", 1.0)) or 1.0)
        if weight < min_weight:
            continue
        span_text = text[start:end]
        if not collapse_ws(span_text):
            continue
        cleaned.append({
            "start": start,
            "end": end,
            "text": span_text,
            "label": label,
            "canonical": e.get("canonical") or span_text,
            "entity_id": e.get("entity_id"),
            "weight": weight,
            "source": e.get("source", "gazetteer_v2"),
        })

    # Keep longest non-overlap. This is important for GLiNER span training.
    cleaned = sorted(cleaned, key=lambda x: (-entity_len(x), -float(x.get("weight", 1.0)), int(x["start"])))
    occupied = [False] * n
    kept = []
    for e in cleaned:
        if any(occupied[e["start"]:e["end"]]):
            continue
        for i in range(e["start"], e["end"]):
            occupied[i] = True
        kept.append(e)
    kept.sort(key=lambda x: x["start"])
    return kept


def doc_split(rows: list[dict[str, Any]], seed: int = 42, dev_ratio: float = 0.1, test_ratio: float = 0.05):
    """
    Document/package-aware split to reduce leakage.
    """
    by_doc = defaultdict(list)
    for r in rows:
        doc = r.get("package_id") or r.get("document_number") or r.get("document_id") or "UNKNOWN"
        by_doc[doc].append(r)

    items = list(by_doc.items())
    random.Random(seed).shuffle(items)

    total = len(rows)
    target_test = int(total * test_ratio)
    target_dev = int(total * dev_ratio)

    train, dev, test = [], [], []
    for _, rs in items:
        if len(test) < target_test:
            test.extend(rs)
        elif len(dev) < target_dev:
            dev.extend(rs)
        else:
            train.extend(rs)

    if not train or not dev:
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        n = len(rows)
        nt = max(1, int(n * test_ratio))
        nd = max(1, int(n * dev_ratio))
        return rows[nt + nd:], rows[nt:nt + nd], rows[:nt]

    return train, dev, test


def entity_count_by_label(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter()
    for r in rows:
        for e in r.get("entities") or []:
            c[e.get("label", "UNKNOWN")] += 1
    return dict(sorted(c.items()))
