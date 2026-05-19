\
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import boundary_ok, normalize_loose, read_jsonl


class GazetteerMatcher:
    """
    Accent-insensitive matcher for known aliases.

    It matches on normalized text, so offsets are reliable only when matching exact text.
    For training, we use exact offsets when possible and keep loose matches mainly as labels/candidates.
    """
    def __init__(self, aliases: List[Dict[str, Any]]):
        self.aliases = []
        for a in aliases:
            if a.get("match_mode") == "reject":
                continue
            surface = a.get("surface") or ""
            key = normalize_loose(surface)
            if not key:
                continue
            self.aliases.append({**a, "_key": key})
        self.aliases.sort(key=lambda x: (-len(x["_key"]), x["_key"]))

    @classmethod
    def from_root(cls, root: str | Path):
        p = Path(root) / "aliases.jsonl"
        return cls(list(read_jsonl(p)))

    def match_loose(self, text: str, scope: str = "direct") -> List[Dict[str, Any]]:
        norm_text = normalize_loose(text)
        occupied = [False] * len(norm_text)
        out = []

        for a in self.aliases:
            needle = a["_key"]
            start = 0
            while True:
                idx = norm_text.find(needle, start)
                if idx < 0:
                    break
                end = idx + len(needle)
                # normalized boundary
                if (idx == 0 or not norm_text[idx - 1].isalnum()) and (end >= len(norm_text) or not norm_text[end].isalnum()) and not any(occupied[idx:end]):
                    for i in range(idx, end):
                        occupied[i] = True
                    out.append({
                        "surface": a.get("surface"),
                        "canonical": a.get("canonical"),
                        "label": a.get("label"),
                        "entity_id": a.get("entity_id"),
                        "scope": scope,
                        "match_mode": a.get("match_mode", "keep"),
                        "graph_weight": float(a.get("graph_weight", 1.0)) if scope == "direct" else min(0.45, float(a.get("graph_weight", 1.0))),
                        "source": "gazetteer_loose" if scope == "direct" else "gazetteer_inherited",
                    })
                start = idx + 1
        return out
