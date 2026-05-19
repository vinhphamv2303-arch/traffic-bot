from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .utils import load_jsonl_from_root, read_jsonl


def load_passages(passages_root: str | Path) -> List[Dict[str, Any]]:
    """
    Supports:
      passages/all_passages.jsonl
      passages/<PACKAGE_ID>/passages.jsonl
      passages/<single package>/passages.jsonl
    """
    return load_jsonl_from_root(passages_root, "all_passages.jsonl", "passages.jsonl")


def load_entity_links(entity_links_root: str | Path) -> List[Dict[str, Any]]:
    """
    Supports:
      entity_links_v1_pruned/all_sentence_entity_links.jsonl
      entity_links_v1_pruned/<PACKAGE_ID>/sentence_entity_links.jsonl
      entity_links_v1_pruned/<single package>/sentence_entity_links.jsonl
    """
    return load_jsonl_from_root(entity_links_root, "all_sentence_entity_links.jsonl", "sentence_entity_links.jsonl")


def load_canonical_entities(gazetteer_root: str | Path) -> List[Dict[str, Any]]:
    p = Path(gazetteer_root) / "canonical_entities.jsonl"
    if not p.exists():
        return []
    return list(read_jsonl(p))
