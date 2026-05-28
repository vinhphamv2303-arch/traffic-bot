from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


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


def stable_id(*parts: str, prefix: str = "id") -> str:
    raw = "|".join([p or "" for p in parts])
    return f"{prefix}_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def find_package_dirs(root: str | Path, marker: str) -> List[Path]:
    p = Path(root)
    if (p / marker).exists():
        return [p]
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.is_dir() and (x / marker).exists()])


def load_jsonl_from_root(root: str | Path, all_file: str, package_file: str) -> List[Dict[str, Any]]:
    root = Path(root)
    all_path = root / all_file
    if all_path.exists():
        return list(read_jsonl(all_path))

    rows: List[Dict[str, Any]] = []
    for d in find_package_dirs(root, package_file):
        rows.extend(read_jsonl(d / package_file))
    return rows


def compact_text(text: str, max_chars: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
