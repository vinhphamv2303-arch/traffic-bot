from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
import types
import unicodedata
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


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(path: str | Path, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _install_legacy_pickle_aliases() -> None:
    """Allow indexes pickled before the retrieval package rename to load."""
    from . import bm25 as bm25_module

    package_names = [
        "retrieval_pipelines",
        "retrieval_pipelines.legal_linearrag_retriever",
        "retrieval_pipelines.legal_linearrag_retriever.legal_linearrag_retriever",
    ]
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    sys.modules[
        "retrieval_pipelines.legal_linearrag_retriever.legal_linearrag_retriever.bm25"
    ] = bm25_module


def load_pickle(path: str | Path) -> Any:
    _install_legacy_pickle_aliases()
    with open(path, "rb") as f:
        return pickle.load(f)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def normalize_for_match(text: str) -> str:
    return collapse_ws(strip_accents((text or "").lower()))


def normalize_for_tokenize(text: str) -> str:
    text = strip_accents((text or "").lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return collapse_ws(text)


def simple_tokenize(text: str) -> List[str]:
    return [t for t in normalize_for_tokenize(text).split(" ") if t]


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def topk_dict(scores: Dict[str, float], k: int) -> Dict[str, float]:
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k])


def cosine_matrix(query_vec, matrix):
    import numpy as np
    q = query_vec.astype("float32")
    qn = np.linalg.norm(q) + 1e-9
    mn = np.linalg.norm(matrix, axis=1) + 1e-9
    return (matrix @ q) / (mn * qn)
