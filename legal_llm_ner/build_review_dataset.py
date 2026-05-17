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
LOW_LABEL_SUPPLEMENT_LABELS = [
    "FINE_AMOUNT",
    "TIME_OR_DURATION",
    "LOCATION_OR_ROAD_CONTEXT",
    "TRAFFIC_SIGNAL_OR_SIGN",
    "CONSEQUENCE_OR_HARM",
    "PLAN_OR_PROJECT",
]
DEPRIORITIZED_REGEX_PACKAGE_IDS = {"100_2015_QH13", "12_2017_QH14"}
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
LICENSE_CLASS_PREFIX_RE = re.compile("h\u1ea1ng", flags=re.I)

DONG = "\u0111\u1ed3ng"
PHAT_TIEN = "ph\u1ea1t\\s+ti\u1ec1n"
MUC_PHAT = "m\u1ee9c\\s+ph\u1ea1t"
TU = "t\u1eeb"
DEN = "\u0111\u1ebfn"
DUOI = "d\u01b0\u1edbi"
AMOUNT_RE = r"\d[\d\.]*\s*" + DONG
FINE_AMOUNT_PATTERNS = [
    re.compile(
        rf"{PHAT_TIEN}\s+(?P<span>(?:{TU}\s+)?{AMOUNT_RE}(?:\s+{DEN}\s+(?:{DUOI}\s+)?{AMOUNT_RE})?)",
        flags=re.I,
    ),
    re.compile(
        rf"{MUC_PHAT}\s+(?:ti\u1ec1n\s+)?(?P<span>(?:{TU}\s+)?{AMOUNT_RE}(?:\s+{DEN}\s+(?:{DUOI}\s+)?{AMOUNT_RE})?)",
        flags=re.I,
    ),
]

TIME_SPAN_RE = re.compile(
    r"(?P<span>\b\d{1,3}\s+(?:ng\u00e0y\s+l\u00e0m\s+vi\u1ec7c|ng\u00e0y|th\u00e1ng|n\u0103m)\b)",
    flags=re.I,
)
TIME_CUES = [
    "th\u1eddi h\u1ea1n",
    "trong th\u1eddi h\u1ea1n",
    "k\u1ec3 t\u1eeb",
    "\u0111\u1ecbnh k\u1ef3",
    "hi\u1ec7u l\u1ef1c",
    "kh\u00f4ng qu\u00e1",
    "t\u1ed1i \u0111a",
    "t\u1ed1i thi\u1ec3u",
    "ni\u00ean h\u1ea1n",
    "tr\u01b0\u1edbc ng\u00e0y",
    "sau ng\u00e0y",
]
TIME_PREVIOUS_BLOCKS = [
    "ph\u1ea1t t\u00f9",
    "t\u00f9",
    "\u00e1n",
    "c\u1ea3i t\u1ea1o kh\u00f4ng giam gi\u1eef",
    "c\u1ea5m \u0111\u1ea3m nhi\u1ec7m",
    "c\u1ea5m h\u00e0nh ngh\u1ec1",
    "t\u1eeb \u0111\u1ee7",
    "d\u01b0\u1edbi",
]


def compile_term_pattern(terms: list[str]):
    terms = sorted(terms, key=len, reverse=True)
    return re.compile("(?P<span>" + "|".join(re.escape(term) for term in terms) + ")", flags=re.I)


LOCATION_TERM_RE = compile_term_pattern(
    [
        "\u0111\u01b0\u1eddng d\u00e0nh cho ng\u01b0\u1eddi \u0111i b\u1ed9",
        "\u0111\u01b0\u1eddng cao t\u1ed1c",
        "khu v\u1ef1c \u0111\u01b0\u1eddng b\u1ed9",
        "\u0111\u01b0\u1eddng \u0111\u00f4 th\u1ecb",
        "\u0111\u01b0\u1eddng ngang",
        "l\u00e0n \u0111\u01b0\u1eddng",
        "ph\u1ea7n \u0111\u01b0\u1eddng",
        "chi\u1ec1u \u0111\u01b0\u1eddng",
        "n\u00fat giao",
        "giao l\u1ed9",
    ]
)
TRAFFIC_SIGNAL_TERM_RE = compile_term_pattern(
    [
        "hi\u1ec7u l\u1ec7nh \u0111i\u1ec1u khi\u1ec3n giao th\u00f4ng",
        "t\u00edn hi\u1ec7u giao th\u00f4ng",
        "\u0111\u00e8n t\u00edn hi\u1ec7u",
        "bi\u1ec3n b\u00e1o hi\u1ec7u",
        "v\u1ea1ch k\u1ebb \u0111\u01b0\u1eddng",
        "t\u1ea5m ph\u1ea3n quang",
        "d\u1ea3i ph\u00e2n c\u00e1ch",
        "bi\u1ec3n b\u00e1o",
        "hi\u1ec7u l\u1ec7nh",
        "c\u1ecdc ti\u00eau",
        "g\u01b0\u01a1ng c\u1ea7u",
    ]
)
CONSEQUENCE_PATTERNS = [
    re.compile(
        r"(?P<span>g\u00e2y\s+(?:tai\s+n\u1ea1n\s+giao\s+th\u00f4ng|th\u01b0\u01a1ng\s+t\u00edch|thi\u1ec7t\s+h\u1ea1i|t\u1ed5n\s+h\u1ea1i)[^,;:.\n]{0,80})",
        flags=re.I,
    ),
    re.compile(
        r"(?P<span>t\u1ed5n\s+th\u01b0\u01a1ng\s+c\u01a1\s+th\u1ec3\s+\d+%[^,;:.\n]{0,50})",
        flags=re.I,
    ),
    re.compile(r"(?P<span>l\u00e0m\s+ch\u1ebft\s+ng\u01b0\u1eddi|ch\u1ebft\s+ng\u01b0\u1eddi)", flags=re.I),
]
PLAN_PATTERNS = [
    re.compile(r"(?P<span>quy\s+ho\u1ea1ch\s+[^,;:.\n]{0,80})", flags=re.I),
    re.compile(r"(?P<span>k\u1ebf\s+ho\u1ea1ch\s+[^,;:.\n]{0,80})", flags=re.I),
    re.compile(r"(?P<span>d\u1ef1\s+\u00e1n\s+[^,;:.\n]{0,80})", flags=re.I),
    re.compile(r"(?P<span>ch\u01b0\u01a1ng\s+tr\u00ecnh\s+\u0111\u00e0o\s+t\u1ea1o[^,;:.\n]{0,80})", flags=re.I),
]


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


def iter_source_sentence_files(sentences_root: Path):
    root = sentences_root.resolve()
    if root.is_file() and root.name == "sentences.jsonl":
        yield root
    elif root.is_dir() and (root / "sentences.jsonl").exists():
        yield root / "sentences.jsonl"
    elif root.is_dir():
        yield from sorted(root.rglob("sentences.jsonl"))


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


def build_base_sentence_row(row: dict[str, Any]) -> dict[str, Any]:
    sentence_id = row.get("sentence_id")
    text = row.get("text") or ""
    return {
        "sentence_id": sentence_id,
        "passage_id": row.get("passage_id"),
        "source_unit_id": row.get("source_unit_id"),
        "package_id": row.get("package_id"),
        "document_id": row.get("document_id"),
        "document_number": row.get("document_number"),
        "source_type": row.get("source_type"),
        "attachment_id": row.get("attachment_id"),
        "attachment_type": row.get("attachment_type"),
        "unit_type": row.get("unit_type"),
        "passage_kind": row.get("passage_kind"),
        "sentence_type": row.get("sentence_type"),
        "path_text": row.get("path_text"),
        "text": text,
        "entities": [],
        "source_probes": [],
        "review_status": "needs_review",
        "quality_flags": detect_row_flags(sentence_id or "", text),
    }


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


def normalize_candidate_span(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ":;,.\"'"):
        end -= 1
    return start, end


def trim_plan_span(text: str, start: int, end: int):
    span = text[start:end]
    lowered = span.lower()
    cut_markers = [
        " t\u1ea1i kho\u1ea3n",
        " t\u1ea1i \u0111i\u1ec1u",
        " quy \u0111\u1ecbnh",
        " c\u00f3 tr\u00e1ch nhi\u1ec7m",
        " \u0111\u1ec3 ",
    ]
    cut_positions = [lowered.find(marker) for marker in cut_markers if lowered.find(marker) > 0]
    if cut_positions:
        end = start + min(cut_positions)
    return normalize_candidate_span(text, start, end)


def has_overlap(entities: list[dict[str, Any]], start: int, end: int) -> bool:
    for ent in entities:
        ent_start = ent.get("start")
        ent_end = ent.get("end")
        if isinstance(ent_start, int) and isinstance(ent_end, int) and start < ent_end and ent_start < end:
            return True
    return False


def is_noisy_regex_source(sentence_id: str, text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if ".table_" in sentence_id or " | " in stripped:
        return True
    return False


def regex_source_priority(row: dict[str, Any]):
    package_id = row.get("package_id") or ""
    title = (row.get("document_title") or "").upper()
    deprioritized = package_id in DEPRIORITIZED_REGEX_PACKAGE_IDS or "H\u00ccNH S\u1ef0" in title
    return (
        1 if deprioritized else 0,
        package_id,
        row.get("passage_order") if isinstance(row.get("passage_order"), int) else 10**9,
        row.get("sentence_order") if isinstance(row.get("sentence_order"), int) else 10**9,
        row.get("sentence_id") or "",
    )


def iter_pattern_spans(text: str, patterns: list[re.Pattern[str]]):
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            start, end = match.span("span")
            start, end = normalize_candidate_span(text, start, end)
            if start < end:
                yield start, end


def iter_time_spans(text: str):
    lowered = (text or "").lower()
    if not any(cue in lowered for cue in TIME_CUES):
        return
    for match in TIME_SPAN_RE.finditer(text or ""):
        start, end = match.span("span")
        previous = lowered[max(0, start - 45) : start]
        following = lowered[end : min(len(lowered), end + 20)]
        if any(block in previous for block in TIME_PREVIOUS_BLOCKS):
            continue
        if "tu\u1ed5i" in following:
            continue
        start, end = normalize_candidate_span(text, start, end)
        if start < end:
            yield start, end


def iter_regex_supplement_spans(label: str, text: str):
    if label == "FINE_AMOUNT":
        yield from iter_pattern_spans(text, FINE_AMOUNT_PATTERNS)
    elif label == "TIME_OR_DURATION":
        yield from iter_time_spans(text)
    elif label == "LOCATION_OR_ROAD_CONTEXT":
        yield from iter_pattern_spans(text, [LOCATION_TERM_RE])
    elif label == "TRAFFIC_SIGNAL_OR_SIGN":
        yield from iter_pattern_spans(text, [TRAFFIC_SIGNAL_TERM_RE])
    elif label == "CONSEQUENCE_OR_HARM":
        yield from iter_pattern_spans(text, CONSEQUENCE_PATTERNS)
    elif label == "PLAN_OR_PROJECT":
        for start, end in iter_pattern_spans(text, PLAN_PATTERNS):
            start, end = trim_plan_span(text, start, end)
            if start < end:
                yield start, end


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


def add_regex_entity(row: dict[str, Any], label: str, start: int, end: int, source_name: str):
    text = row.get("text") or ""
    entity_text = text[start:end]
    entity_id = "ent_" + md5_text(f"{row['sentence_id']}|{label}|{start}|{end}|{entity_text}")[:16]
    row["entities"].append(
        {
            "entity_id": entity_id,
            "text": entity_text,
            "label": label,
            "raw_label": None,
            "start": start,
            "end": end,
            "confidence": 1.0,
            "alignment_status": "aligned",
            "source_probes": [source_name],
            "review_status": "needs_review",
            "quality_flags": ["regex_supplement"],
        }
    )
    if "regex_supplement" not in row["source_probes"]:
        row["source_probes"].append("regex_supplement")


def add_low_label_supplements_from_sentences(
    rows_by_id: OrderedDict[str, dict[str, Any]],
    sentences_root: Path | None,
    labels: list[str],
    limit_per_label: int,
):
    counts = Counter()
    if not sentences_root or limit_per_label <= 0:
        return counts

    labels = [label for label in labels if label in LOW_LABEL_SUPPLEMENT_LABELS]
    if not labels:
        return counts

    source_sentences = []
    for file_path in iter_source_sentence_files(sentences_root):
        source_sentences.extend(read_jsonl(file_path))
    source_sentences.sort(key=regex_source_priority)

    for sentence in source_sentences:
        if all(counts[label] >= limit_per_label for label in labels):
            break
        sentence_id = sentence.get("sentence_id")
        text = sentence.get("text") or ""
        if not sentence_id or is_noisy_regex_source(sentence_id, text):
            continue

        row = rows_by_id.get(sentence_id)
        row_was_created = False
        for label in labels:
            if counts[label] >= limit_per_label:
                continue
            for start, end in iter_regex_supplement_spans(label, text):
                if row is not None and has_overlap(row.get("entities", []), start, end):
                    continue
                if row is None:
                    row = build_base_sentence_row(sentence)
                    row_was_created = True
                if has_overlap(row.get("entities", []), start, end):
                    continue
                add_regex_entity(row, label, start, end, f"regex_{label.lower()}")
                counts[label] += 1
                break
            if row_was_created:
                rows_by_id[sentence_id] = row
                row_was_created = False

    return counts


def merge_outputs(
    input_paths: list[Path],
    min_confidence: float,
    license_class_supplement_limit: int,
    regex_supplement_sentence_root: Path | None,
    regex_supplement_labels: list[str],
    regex_supplement_limit_per_label: int,
):
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

    license_class_supplement_count = add_license_class_supplements(list(rows_by_id.values()), license_class_supplement_limit)
    low_label_supplement_counts = add_low_label_supplements_from_sentences(
        rows_by_id,
        regex_supplement_sentence_root,
        regex_supplement_labels,
        regex_supplement_limit_per_label,
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

    return (
        list(rows_by_id.values()),
        sorted(source_files),
        rejected,
        license_class_supplement_count,
        low_label_supplement_counts,
    )


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

    rows, source_files, rejected, license_class_supplement_count, low_label_supplement_counts = merge_outputs(
        args.input,
        args.min_confidence,
        args.license_class_supplement_limit,
        args.regex_supplement_sentence_root,
        args.regex_supplement_labels,
        args.regex_supplement_limit_per_label,
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
        "regex_supplement_sentence_root": str(args.regex_supplement_sentence_root)
        if args.regex_supplement_sentence_root
        else None,
        "regex_supplement_limit_per_label": args.regex_supplement_limit_per_label,
        "regex_supplement_counts": {
            label: low_label_supplement_counts[label] for label in sorted(LOW_LABEL_SUPPLEMENT_LABELS)
        },
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
    parser.add_argument(
        "--regex-supplement-sentence-root",
        type=Path,
        help="Sentence root used to add small regex-derived candidates for sparse labels.",
    )
    parser.add_argument(
        "--regex-supplement-limit-per-label",
        type=int,
        default=0,
        help="Add up to N regex-derived candidates for each sparse label.",
    )
    parser.add_argument(
        "--regex-supplement-labels",
        nargs="+",
        default=LOW_LABEL_SUPPLEMENT_LABELS,
        choices=LOW_LABEL_SUPPLEMENT_LABELS,
        help="Sparse labels to supplement from --regex-supplement-sentence-root.",
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
