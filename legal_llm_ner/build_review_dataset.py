from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from legal_llm_ner.schema import ALLOWED_LABELS, ENTITY_SCHEMA, LEGACY_LABEL_ALIASES


TOKEN_CONTEXT = 90
LICENSE_CLASS_CODES = [
    "C1E",
    "D1E",
    "D2E",
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "B2",
    "BE",
    "C1",
    "D1",
    "D2",
    "CE",
    "DE",
    "B",
    "C",
    "D",
]
LICENSE_CLASS_CODE_RE = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(re.escape(code) for code in LICENSE_CLASS_CODES) + r")(?![A-Z0-9])"
)
LICENSE_CLASS_PREFIX_RE = re.compile(r"hạng", flags=re.I)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def iter_sentence_files(inputs: list[Path]):
    seen = set()
    for path in inputs:
        path = path.resolve()
        if path.is_file() and path.name == "sentence_entities.jsonl":
            files = [path]
        elif path.is_dir() and (path / "sentence_entities.jsonl").exists():
            files = [path / "sentence_entities.jsonl"]
        elif path.is_dir():
            files = sorted(path.rglob("sentence_entities.jsonl"))
        else:
            files = []

        for file_path in files:
            key = str(file_path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                yield file_path


def source_name_for(file_path: Path, input_roots: list[Path]) -> str:
    resolved = file_path.resolve()
    best_root = None
    for root in input_roots:
        root = root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if best_root is None or len(str(root)) > len(str(best_root)):
            best_root = root
    if best_root is not None:
        return best_root.name
    return file_path.parent.name


def valid_span(sentence_text: str, start: Any, end: Any, entity_text: str) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(sentence_text)
        and sentence_text[start:end] == entity_text
    )


def detect_row_flags(sentence_id: str, text: str) -> list[str]:
    flags = []
    stripped = (text or "").strip()
    if ".table_" in sentence_id or " | " in stripped:
        flags.append("table_like_sentence")
    if stripped and stripped.upper() == stripped and len(stripped) <= 100:
        flags.append("heading_like_sentence")
    return flags


def find_overlap_pairs(entities: list[dict[str, Any]]) -> list[list[str]]:
    pairs = []
    ordered = sorted(entities, key=lambda e: (e["start"], e["end"], e["label"], e["text"]))
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            if right["start"] >= left["end"]:
                break
            pairs.append([left["entity_id"], right["entity_id"]])
    return pairs


def iter_license_class_spans(text: str):
    seen = set()
    for prefix_match in LICENSE_CLASS_PREFIX_RE.finditer(text or ""):
        segment_start = prefix_match.start()
        max_segment_end = min(len(text), prefix_match.end() + 100)
        stop_positions = [
            pos
            for pos in [
                text.find(";", prefix_match.end()),
                text.find(".", prefix_match.end()),
                text.find("\n", prefix_match.end()),
            ]
            if pos != -1 and pos <= max_segment_end
        ]
        segment_end = min(stop_positions) if stop_positions else max_segment_end
        segment = text[segment_start:segment_end]
        code_matches = list(LICENSE_CLASS_CODE_RE.finditer(segment))
        if not code_matches:
            continue

        first_code = code_matches[0]
        prefix_gap = segment[prefix_match.end() - segment_start : first_code.start()]
        if len(prefix_gap) > 20:
            continue

        for index, code_match in enumerate(code_matches):
            start = segment_start + code_match.start()
            end = segment_start + code_match.end()
            if index == 0:
                start = segment_start
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            yield start, end


def add_license_class_supplements(rows: list[dict[str, Any]], limit: int) -> int:
    if limit <= 0:
        return 0

    added = 0
    for row in rows:
        if added >= limit:
            break
        text = row.get("text") or ""
        existing = {
            (ent.get("label"), ent.get("start"), ent.get("end"), ent.get("text"))
            for ent in row.get("entities", [])
        }
        for start, end in iter_license_class_spans(text):
            entity_text = text[start:end]
            key = ("LICENSE_CLASS", start, end, entity_text)
            if key in existing:
                continue
            entity_id = "ent_" + md5_text(f"{row['sentence_id']}|LICENSE_CLASS|{start}|{end}|{entity_text}")[:16]
            row["entities"].append(
                {
                    "entity_id": entity_id,
                    "text": entity_text,
                    "label": "LICENSE_CLASS",
                    "raw_label": None,
                    "start": start,
                    "end": end,
                    "confidence": 1.0,
                    "alignment_status": "aligned",
                    "source_probes": ["regex_license_class"],
                    "review_status": "needs_review",
                    "quality_flags": ["regex_supplement"],
                }
            )
            existing.add(key)
            added += 1
            if added >= limit:
                break
    return added


def merge_outputs(input_paths: list[Path], min_confidence: float, license_class_supplement_limit: int):
    rows_by_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    entity_keys_by_sentence = defaultdict(set)
    rejected = Counter()
    source_files = []

    for file_path in iter_sentence_files(input_paths):
        source_name = source_name_for(file_path, input_paths)
        source_files.append(str(file_path))
        for row in read_jsonl(file_path):
            sentence_id = row.get("sentence_id")
            text = row.get("text") or ""
            if not sentence_id or not text:
                rejected["missing_sentence_id_or_text"] += 1
                continue

            out_row = rows_by_id.setdefault(
                sentence_id,
                {
                    "sentence_id": sentence_id,
                    "passage_id": row.get("passage_id"),
                    "source_unit_id": row.get("source_unit_id"),
                    "package_id": row.get("package_id"),
                    "document_id": row.get("document_id"),
                    "document_number": row.get("document_number"),
                    "text": text,
                    "entities": [],
                    "source_probes": [],
                    "review_status": "needs_review",
                    "quality_flags": detect_row_flags(sentence_id, text),
                },
            )
            if source_name not in out_row["source_probes"]:
                out_row["source_probes"].append(source_name)

            if out_row["text"] != text:
                rejected["sentence_text_conflict"] += 1
                continue

            for ent in row.get("entities", []):
                raw_label = ent.get("label")
                label = LEGACY_LABEL_ALIASES.get(raw_label, raw_label)
                entity_text = ent.get("text") or ""
                start = ent.get("start")
                end = ent.get("end")
                confidence = ent.get("confidence")

                if label not in ALLOWED_LABELS:
                    rejected[f"invalid_label:{raw_label}"] += 1
                    continue
                if not valid_span(text, start, end, entity_text):
                    rejected["invalid_or_unaligned_span"] += 1
                    continue
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.0
                if confidence < min_confidence:
                    rejected["below_min_confidence"] += 1
                    continue

                dedupe_key = (label, start, end, entity_text)
                if dedupe_key in entity_keys_by_sentence[sentence_id]:
                    for existing in out_row["entities"]:
                        if (
                            existing["label"],
                            existing["start"],
                            existing["end"],
                            existing["text"],
                        ) == dedupe_key:
                            existing["confidence"] = max(existing["confidence"], confidence)
                            if source_name not in existing["source_probes"]:
                                existing["source_probes"].append(source_name)
                            break
                    continue

                entity_keys_by_sentence[sentence_id].add(dedupe_key)
                entity_flags = []
                if raw_label != label:
                    entity_flags.append("legacy_alias_mapped")

                entity_id = "ent_" + md5_text(f"{sentence_id}|{label}|{start}|{end}|{entity_text}")[:16]
                out_row["entities"].append(
                    {
                        "entity_id": entity_id,
                        "text": entity_text,
                        "label": label,
                        "raw_label": raw_label if raw_label != label else None,
                        "start": start,
                        "end": end,
                        "confidence": confidence,
                        "alignment_status": "aligned",
                        "source_probes": [source_name],
                        "review_status": "needs_review",
                        "quality_flags": entity_flags,
                    }
                )

    license_class_supplement_count = add_license_class_supplements(
        list(rows_by_id.values()), license_class_supplement_limit
    )

    for row in rows_by_id.values():
        row["source_probes"].sort()
        row["entities"].sort(key=lambda e: (e["start"], e["end"], e["label"], e["text"]))
        overlap_pairs = find_overlap_pairs(row["entities"])
        if overlap_pairs:
            if "overlap_conflict" not in row["quality_flags"]:
                row["quality_flags"].append("overlap_conflict")
            overlap_ids = {entity_id for pair in overlap_pairs for entity_id in pair}
            for ent in row["entities"]:
                if ent["entity_id"] in overlap_ids and "overlap_conflict" not in ent["quality_flags"]:
                    ent["quality_flags"].append("overlap_conflict")
        row["overlap_conflicts"] = overlap_pairs
        row["entity_count"] = len(row["entities"])

    return list(rows_by_id.values()), sorted(source_files), rejected, license_class_supplement_count


def flatten_mentions(rows: list[dict[str, Any]]):
    mentions = []
    for row in rows:
        text = row["text"]
        for ent in row["entities"]:
            start = ent["start"]
            end = ent["end"]
            mentions.append(
                {
                    **ent,
                    "sentence_id": row["sentence_id"],
                    "passage_id": row.get("passage_id"),
                    "source_unit_id": row.get("source_unit_id"),
                    "package_id": row.get("package_id"),
                    "document_id": row.get("document_id"),
                    "document_number": row.get("document_number"),
                    "sentence_text": text,
                    "left_context": text[max(0, start - TOKEN_CONTEXT) : start],
                    "right_context": text[end : min(len(text), end + TOKEN_CONTEXT)],
                    "row_quality_flags": row.get("quality_flags", []),
                }
            )
    return mentions


def write_review_csv(path: Path, mentions: list[dict[str, Any]]):
    fields = [
        "review_decision",
        "corrected_label",
        "corrected_text",
        "notes",
        "entity_id",
        "sentence_id",
        "package_id",
        "document_number",
        "label",
        "text",
        "start",
        "end",
        "confidence",
        "source_probes",
        "quality_flags",
        "row_quality_flags",
        "left_context",
        "right_context",
        "sentence_text",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mention in mentions:
            writer.writerow(
                {
                    "review_decision": "",
                    "corrected_label": "",
                    "corrected_text": "",
                    "notes": "",
                    "entity_id": mention.get("entity_id"),
                    "sentence_id": mention.get("sentence_id"),
                    "package_id": mention.get("package_id"),
                    "document_number": mention.get("document_number"),
                    "label": mention.get("label"),
                    "text": mention.get("text"),
                    "start": mention.get("start"),
                    "end": mention.get("end"),
                    "confidence": mention.get("confidence"),
                    "source_probes": ";".join(mention.get("source_probes", [])),
                    "quality_flags": ";".join(mention.get("quality_flags", [])),
                    "row_quality_flags": ";".join(mention.get("row_quality_flags", [])),
                    "left_context": mention.get("left_context"),
                    "right_context": mention.get("right_context"),
                    "sentence_text": mention.get("sentence_text"),
                }
            )


def count_labels(rows: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    for row in rows:
        for ent in row.get("entities", []):
            counts[ent["label"]] += 1
    return counts


def write_label_summary_csv(path: Path, all_counts: Counter, flat_counts: Counter):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "description", "review_count", "flat_trainable_count"])
        writer.writeheader()
        for label in sorted(ENTITY_SCHEMA):
            writer.writerow(
                {
                    "label": label,
                    "description": ENTITY_SCHEMA[label],
                    "review_count": all_counts[label],
                    "flat_trainable_count": flat_counts[label],
                }
            )


def write_readme(path: Path):
    path.write_text(
        """# Legal NER Silver Review Dataset

This folder is generated from LLM NER probe outputs.

Files:

- `sentence_entities_review.jsonl`: merged sentence-level silver annotations, including rows with overlap conflicts for manual review.
- `entity_mentions_review.jsonl`: one row per entity mention.
- `entities_review.csv`: Excel-friendly UTF-8-BOM review sheet. Fill `review_decision`, `corrected_label`, `corrected_text`, and `notes`.
- `sentence_entities_trainable_flat.jsonl`: unreviewed flat NER candidate data with overlap-conflict sentences removed.
- `entity_mentions_trainable_flat.jsonl`: mention rows matching the flat candidate data.
- `label_summary.csv`: label coverage summary.
- `dataset_summary.json`: counts, source files, and rejected rows.

Recommended review values:

- `accept`: keep the span and label.
- `reject`: remove the entity.
- `fix`: use `corrected_label` and/or `corrected_text`.

Important: all annotations are silver/unreviewed. Use them for fine-tuning only after review, or keep the model output provenance as weak supervision.
""",
        encoding="utf-8",
    )


def build_dataset(args):
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_files, rejected, license_class_supplement_count = merge_outputs(
        args.input, args.min_confidence, args.license_class_supplement_limit
    )
    review_mentions = flatten_mentions(rows)

    flat_rows = [row for row in rows if "overlap_conflict" not in row.get("quality_flags", [])]
    flat_mentions = flatten_mentions(flat_rows)

    write_jsonl(output_dir / "sentence_entities_review.jsonl", rows)
    write_jsonl(output_dir / "entity_mentions_review.jsonl", review_mentions)
    write_jsonl(output_dir / "sentence_entities_trainable_flat.jsonl", flat_rows)
    write_jsonl(output_dir / "entity_mentions_trainable_flat.jsonl", flat_mentions)
    write_review_csv(output_dir / "entities_review.csv", review_mentions)

    all_counts = count_labels(rows)
    flat_counts = count_labels(flat_rows)
    write_label_summary_csv(output_dir / "label_summary.csv", all_counts, flat_counts)

    summary = {
        "source_files": source_files,
        "sentence_count": len(rows),
        "sentence_with_entities_count": sum(1 for row in rows if row.get("entities")),
        "entity_count": len(review_mentions),
        "flat_trainable_sentence_count": len(flat_rows),
        "flat_trainable_sentence_with_entities_count": sum(1 for row in flat_rows if row.get("entities")),
        "flat_trainable_entity_count": len(flat_mentions),
        "overlap_conflict_sentence_count": len(rows) - len(flat_rows),
        "by_label": {label: all_counts[label] for label in sorted(ENTITY_SCHEMA)},
        "flat_trainable_by_label": {label: flat_counts[label] for label in sorted(ENTITY_SCHEMA)},
        "missing_labels": [label for label in sorted(ENTITY_SCHEMA) if all_counts[label] == 0],
        "flat_trainable_missing_labels": [label for label in sorted(ENTITY_SCHEMA) if flat_counts[label] == 0],
        "rejected": dict(rejected),
        "min_confidence": args.min_confidence,
        "license_class_supplement_count": license_class_supplement_count,
    }
    write_json(output_dir / "dataset_summary.json", summary)
    write_readme(output_dir / "README.md")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LLM NER probe outputs into a reviewable silver dataset.")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="NER output folders or sentence_entities.jsonl files.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop entities below this confidence.")
    parser.add_argument(
        "--license-class-supplement-limit",
        type=int,
        default=0,
        help="Add up to N regex-derived LICENSE_CLASS spans, marked with quality_flags=regex_supplement.",
    )
    return parser.parse_args()


def main():
    summary = build_dataset(parse_args())
    print(
        "Built review dataset: "
        f"{summary['sentence_count']} sentences, "
        f"{summary['entity_count']} entities, "
        f"{summary['flat_trainable_entity_count']} flat-trainable entities."
    )


if __name__ == "__main__":
    main()
