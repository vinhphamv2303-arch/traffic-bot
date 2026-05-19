\
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def whitespace_tokens_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens = []
    offsets = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        end = i
        tokens.append(text[start:end])
        offsets.append((start, end))
    return tokens, offsets


def char_span_to_token_span(start: int, end: int, offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    token_idxs = []
    for i, (s, e) in enumerate(offsets):
        if e <= start or s >= end:
            continue
        token_idxs.append(i)
    if not token_idxs:
        return None
    return token_idxs[0], token_idxs[-1]


def row_to_gliner(row: Dict[str, Any]) -> Dict[str, Any] | None:
    text = row.get("text") or ""
    tokens, offsets = whitespace_tokens_with_offsets(text)
    ner = []
    for e in row.get("entities") or []:
        span = char_span_to_token_span(int(e["start"]), int(e["end"]), offsets)
        if span is None:
            continue
        ner.append([span[0], span[1], e["label"]])
    if not tokens:
        return None
    return {"tokenized_text": tokens, "ner": ner}
