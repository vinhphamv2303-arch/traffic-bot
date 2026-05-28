\
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import (
    LABELS,
    char_to_token_span,
    clean_direct_entities,
    doc_split,
    ensure_dir,
    entity_count_by_label,
    iter_sentence_entity_files,
    read_jsonl,
    token_offsets,
    write_json,
    write_jsonl,
)


def to_gliner_record(row: Dict[str, Any]) -> Dict[str, Any] | None:
    text = row.get("text") or ""
    toks, offsets = token_offsets(text)
    if not toks:
        return None

    ner = []
    entities_out = []
    for e in row.get("entities") or []:
        span = char_to_token_span(int(e["start"]), int(e["end"]), offsets)
        if span is None:
            continue
        ner.append([span[0], span[1], e["label"]])
        entities_out.append(e)

    return {
        "id": row.get("sentence_id"),
        "tokenized_text": toks,
        "ner": ner,
        # keep these extra fields for debugging; GLiNER train_model ignores unknown fields in recent versions,
        # and if your installed version is strict, use *_gliner_min.json instead.
        "text": text,
        "entities": entities_out,
        "package_id": row.get("package_id"),
        "document_number": row.get("document_number"),
        "passage_id": row.get("passage_id"),
        "path_text": row.get("path_text"),
    }


def to_gliner_min_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tokenized_text": record["tokenized_text"],
        "ner": record["ner"],
    }


def build_gliner_dataset(
    entities_root: str | Path,
    output_dir: str | Path,
    min_weight: float = 0.0,
    negative_ratio: float = 0.35,
    max_rows: int | None = None,
    seed: int = 42,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.05,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    rows = []
    dropped_empty = 0
    dropped_no_tokens = 0

    for f in iter_sentence_entity_files(entities_root):
        for r in read_jsonl(f):
            text = r.get("text") or ""
            direct = clean_direct_entities(text, r.get("entities") or [], keep_labels=LABELS, min_weight=min_weight)
            new_r = {
                "sentence_id": r.get("sentence_id"),
                "passage_id": r.get("passage_id"),
                "source_unit_id": r.get("source_unit_id"),
                "package_id": r.get("package_id"),
                "document_id": r.get("document_id"),
                "document_number": r.get("document_number"),
                "path_text": r.get("path_text"),
                "text": text,
                "entities": direct,
            }
            if not text.strip():
                dropped_empty += 1
                continue
            rows.append(new_r)

    positives = [r for r in rows if r.get("entities")]
    negatives = [r for r in rows if not r.get("entities")]

    import random
    rnd = random.Random(seed)
    if negative_ratio > 0 and negatives:
        n_neg = min(len(negatives), int(len(positives) * negative_ratio))
        negatives = rnd.sample(negatives, n_neg)
    else:
        negatives = []

    dataset = positives + negatives
    rnd.shuffle(dataset)

    if max_rows and len(dataset) > max_rows:
        # keep positives priority, sample negatives/positives jointly but deterministic.
        dataset = rnd.sample(dataset, max_rows)

    train, dev, test = doc_split(dataset, seed=seed, dev_ratio=dev_ratio, test_ratio=test_ratio)

    def convert(rows):
        out = []
        for r in rows:
            rec = to_gliner_record(r)
            if rec is None:
                continue
            out.append(rec)
        return out

    train_g = convert(train)
    dev_g = convert(dev)
    test_g = convert(test)
    all_g = convert(dataset)

    # Full/debug jsonl.
    write_jsonl(output_dir / "train.jsonl", train_g)
    write_jsonl(output_dir / "dev.jsonl", dev_g)
    write_jsonl(output_dir / "test.jsonl", test_g)
    write_jsonl(output_dir / "all.jsonl", all_g)

    # GLiNER official train.py uses JSON list.
    write_json(output_dir / "train.json", train_g)
    write_json(output_dir / "dev.json", dev_g)
    write_json(output_dir / "test.json", test_g)

    # Minimal JSON list if installed GLiNER version is strict about fields.
    write_json(output_dir / "train_gliner_min.json", [to_gliner_min_record(x) for x in train_g])
    write_json(output_dir / "dev_gliner_min.json", [to_gliner_min_record(x) for x in dev_g])
    write_json(output_dir / "test_gliner_min.json", [to_gliner_min_record(x) for x in test_g])

    summary = {
        "entities_root": str(entities_root),
        "output_dir": str(output_dir),
        "source_sentence_rows": len(rows),
        "positive_sentence_rows": len(positives),
        "negative_sentence_rows_sampled": len(negatives),
        "dataset_rows": len(dataset),
        "train_rows": len(train_g),
        "dev_rows": len(dev_g),
        "test_rows": len(test_g),
        "entity_count_total": sum(len(r.get("entities") or []) for r in dataset),
        "entity_count_by_label": entity_count_by_label(dataset),
        "negative_ratio": negative_ratio,
        "min_weight": min_weight,
        "format": {
            "train_json": str(output_dir / "train.json"),
            "dev_json": str(output_dir / "dev.json"),
            "test_json": str(output_dir / "test.json"),
            "train_gliner_min_json": str(output_dir / "train_gliner_min.json"),
            "dev_gliner_min_json": str(output_dir / "dev_gliner_min.json"),
        },
        "note": "GLiNER trains only on direct entities inside sentence text. inherited_entities/path entities are not used as spans.",
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary
