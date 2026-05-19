\
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import (
    ensure_dir,
    find_all_exact,
    find_sentence_package_dirs,
    label_summary,
    read_jsonl,
    split_rows,
    stable_id,
    write_json,
    write_jsonl,
)
from .gazetteer import GazetteerMatcher


def _exact_entity_mentions(text: str, matcher: GazetteerMatcher, scope: str, source: str) -> List[Dict[str, Any]]:
    """
    Match all aliases. For training BIO/GLiNER, direct entities need start/end in text.
    For inherited context entities, start/end are offsets in context, not the sentence text.
    """
    out = []
    # Use alias surfaces directly to recover exact offsets.
    for a in matcher.aliases:
        surface = a.get("surface") or ""
        for start, end in find_all_exact(text, surface):
            out.append({
                "text": text[start:end],
                "label": a.get("label"),
                "start": start,
                "end": end,
                "canonical": a.get("canonical"),
                "entity_id": a.get("entity_id"),
                "scope": scope,
                "source": source,
                "match_mode": a.get("match_mode", "keep"),
                "graph_weight": float(a.get("graph_weight", 1.0)) if scope == "direct" else min(0.45, float(a.get("graph_weight", 1.0))),
            })
    # longest/non-overlap
    out.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    kept = []
    last_end = -1
    for e in out:
        if e["start"] < last_end:
            continue
        kept.append(e)
        last_end = e["end"]
    return kept


def _dedupe_by_span(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for e in entities:
        key = (e.get("scope"), e.get("label"), e.get("text"), e.get("start"), e.get("end"))
        score = float(e.get("graph_weight", 1.0))
        if key not in best or score > float(best[key].get("graph_weight", 0)):
            best[key] = e
    return list(best.values())


def sentence_to_training_row(row: Dict[str, Any], matcher: GazetteerMatcher, include_inherited: bool = True) -> Dict[str, Any]:
    text = row.get("text") or ""
    context = row.get("path_text") or row.get("context_text") or ""

    direct = _exact_entity_mentions(text, matcher, scope="direct", source="gazetteer_direct")
    inherited = _exact_entity_mentions(context, matcher, scope="inherited", source="gazetteer_inherited") if include_inherited and context else []

    # GLiNER only trains on spans inside text. We'll store direct entities in `entities`.
    # Inherited entities are kept separately for graph/linker and future context-aware model variants.
    out = {
        "id": row.get("sentence_id"),
        "sentence_id": row.get("sentence_id"),
        "passage_id": row.get("passage_id"),
        "source_unit_id": row.get("source_unit_id"),
        "package_id": row.get("package_id"),
        "document_id": row.get("document_id"),
        "document_number": row.get("document_number"),
        "unit_type": row.get("unit_type"),
        "path_text": row.get("path_text"),
        "text": text,
        "context": context,
        "entities": _dedupe_by_span(direct),
        "inherited_entities": _dedupe_by_span(inherited),
    }
    return out


def build_local_ner_dataset(
    sentences_root: str | Path,
    gazetteer_root: str | Path,
    output_dir: str | Path,
    include_inherited: bool = True,
    include_negative: bool = True,
    negative_ratio: float = 0.5,
    seed: int = 42,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.05,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    matcher = GazetteerMatcher.from_root(gazetteer_root)

    rows = []
    for pkg_dir in find_sentence_package_dirs(sentences_root):
        for r in read_jsonl(pkg_dir / "sentences.jsonl"):
            tr = sentence_to_training_row(r, matcher, include_inherited=include_inherited)
            rows.append(tr)

    pos = [r for r in rows if r.get("entities")]
    neg = [r for r in rows if not r.get("entities")]

    if include_negative and negative_ratio > 0:
        import random
        max_neg = int(len(pos) * negative_ratio)
        if len(neg) > max_neg:
            neg = random.Random(seed).sample(neg, max_neg)
        dataset = pos + neg
    else:
        dataset = pos

    train, dev, test = split_rows(dataset, seed=seed, dev_ratio=dev_ratio, test_ratio=test_ratio)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "all_trainable.jsonl", dataset)

    # GLiNER format: token-level text with entity spans as [start, end, label]
    def to_gliner(r):
        return {
            "tokenized_text": r["text"].split(),
            "ner": [],  # placeholder, real conversion by char offsets is handled in trainer using text/entity text
            "text": r["text"],
            "entities": [
                {"start": e["start"], "end": e["end"], "label": e["label"], "text": e["text"]}
                for e in r.get("entities") or []
            ],
            "id": r.get("id"),
        }

    write_jsonl(output_dir / "train_gliner_raw.jsonl", [to_gliner(r) for r in train])
    write_jsonl(output_dir / "dev_gliner_raw.jsonl", [to_gliner(r) for r in dev])
    write_jsonl(output_dir / "test_gliner_raw.jsonl", [to_gliner(r) for r in test])

    summary = {
        "sentences_root": str(sentences_root),
        "gazetteer_root": str(gazetteer_root),
        "total_sentence_rows": len(rows),
        "positive_rows": len(pos),
        "negative_rows_sampled": len(dataset) - len(pos),
        "dataset_rows": len(dataset),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "test_rows": len(test),
        "direct_entity_count": sum(len(r.get("entities") or []) for r in dataset),
        "inherited_entity_count_all_sentences": sum(len(r.get("inherited_entities") or []) for r in rows),
        "by_label": label_summary(dataset, "entities"),
        "include_inherited": include_inherited,
        "include_negative": include_negative,
        "negative_ratio": negative_ratio,
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "dev": str(output_dir / "dev.jsonl"),
            "test": str(output_dir / "test.jsonl"),
            "all_trainable": str(output_dir / "all_trainable.jsonl"),
        },
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary
