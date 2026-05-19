from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import normalize_for_match, read_jsonl, strip_accents


def boundary_ok(text: str, start: int, end: int) -> bool:
    def is_word_char(ch):
        return ch.isalnum() or ch == "_"
    return (start == 0 or not is_word_char(text[start - 1])) and (end >= len(text) or not is_word_char(text[end]))


class GazetteerMatcher:
    def __init__(self, aliases: List[Dict[str, Any]]):
        self.aliases = sorted(aliases, key=lambda x: (-len(x.get("surface") or ""), x.get("surface") or ""))

    @classmethod
    def from_gazetteer_root(cls, gazetteer_root: str | Path):
        aliases_path = Path(gazetteer_root) / "aliases.jsonl"
        aliases = list(read_jsonl(aliases_path))
        aliases = [a for a in aliases if a.get("match_mode") != "reject"]
        return cls(aliases)

    @staticmethod
    def _normalize_with_offsets(text: str) -> Tuple[str, List[Tuple[int, int]]]:
        norm_chars: List[str] = []
        offsets: List[Tuple[int, int]] = []
        last_space = False

        for i, ch in enumerate(text or ""):
            if ch.isspace():
                if norm_chars and not last_space:
                    norm_chars.append(" ")
                    offsets.append((i, i + 1))
                    last_space = True
                continue

            piece = strip_accents(ch).lower()
            if not piece:
                continue

            for out_ch in piece:
                norm_chars.append(out_ch)
                offsets.append((i, i + 1))
                last_space = False

        while norm_chars and norm_chars[-1] == " ":
            norm_chars.pop()
            offsets.pop()

        return "".join(norm_chars), offsets

    def match(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        low, offsets = self._normalize_with_offsets(text)
        occupied = [False] * len(low)
        out = []

        for a in self.aliases:
            needle = normalize_for_match(a.get("surface") or "")
            if not needle:
                continue
            start = 0
            while True:
                idx = low.find(needle, start)
                if idx < 0:
                    break
                end = idx + len(needle)

                if boundary_ok(low, idx, end) and not any(occupied[idx:end]):
                    for i in range(idx, end):
                        occupied[i] = True
                    original_start = offsets[idx][0]
                    original_end = offsets[end - 1][1]
                    out.append({
                        "surface": text[original_start:original_end],
                        "canonical": a.get("canonical"),
                        "label": a.get("label"),
                        "entity_id": a.get("entity_id"),
                        "start": original_start,
                        "end": original_end,
                        "match_mode": a.get("match_mode", "keep"),
                        "graph_weight": float(a.get("graph_weight", 1.0)),
                        "source": "query_gazetteer_exact",
                        "score": 1.0,
                    })
                start = idx + 1

        out.sort(key=lambda x: (x["start"], x["end"]))
        return out
