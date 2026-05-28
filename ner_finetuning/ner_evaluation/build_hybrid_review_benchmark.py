from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = {
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
}


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_gold_case(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("text") or ""
    entities = []
    for ent in item.get("entities") or []:
        norm = normalize_entity(text, ent, source="gold")
        if norm:
            entities.append(strip_for_gold(norm))
    return {"text": text, "entities": dedupe_exact(entities)}


def normalize_entity(text: str, ent: dict[str, Any], source: str) -> dict[str, Any] | None:
    label = ent.get("label")
    if label not in LABELS:
        return None
    try:
        start = int(ent.get("start"))
        end = int(ent.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > len(text):
        return None

    span_text = text[start:end]
    if not span_text.strip():
        return None

    confidence = ent.get("confidence", ent.get("score", ent.get("graph_weight", 1.0)))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0

    return {
        "text": span_text,
        "label": label,
        "start": start,
        "end": end,
        "source": source,
        "confidence": confidence,
    }


def strip_for_gold(ent: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": ent["text"],
        "label": ent["label"],
        "start": ent["start"],
        "end": ent["end"],
    }


def dedupe_exact(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for ent in sorted(entities, key=lambda e: (e["start"], e["end"], e["label"], e["text"].lower())):
        key = (ent["start"], ent["end"], ent["label"], ent["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a["start"]) < int(b["end"]) and int(b["start"]) < int(a["end"])


def rank_entity(ent: dict[str, Any]) -> tuple[float, int]:
    source = ent.get("source") or ""
    source_bonus = 0.15 if "hybrid" in source else 0.10 if source == "gazetteer" else 0.0
    length = int(ent["end"]) - int(ent["start"])
    moderate_bonus = 0.05 if 2 <= length <= 80 else 0.0
    return (float(ent.get("confidence") or 0.0) + source_bonus + moderate_bonus, length)


def merge_hybrid(text: str, gazetteer_entities: list[dict[str, Any]], gliner_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_exact: dict[tuple[int, int, str], dict[str, Any]] = {}
    for ent in gazetteer_entities + gliner_entities:
        key = (ent["start"], ent["end"], ent["label"])
        prev = by_exact.get(key)
        if prev is None:
            by_exact[key] = dict(ent)
            continue
        prev_sources = set(str(prev.get("source", "")).split("+"))
        prev_sources.add(str(ent.get("source", "")))
        prev["source"] = "+".join(sorted(s for s in prev_sources if s))
        prev["confidence"] = max(float(prev.get("confidence") or 0.0), float(ent.get("confidence") or 0.0))

    candidates = list(by_exact.values())

    for gaz in gazetteer_entities:
        for gli in gliner_entities:
            if gaz["label"] == gli["label"] and overlap(gaz, gli):
                for cand in candidates:
                    if cand["label"] == gaz["label"] and (overlap(cand, gaz) or overlap(cand, gli)):
                        if "hybrid_agree" not in str(cand.get("source", "")):
                            cand["source"] = f"{cand.get('source', '')}+hybrid_agree".strip("+")
                        cand["confidence"] = max(float(cand.get("confidence") or 0.0), float(gli.get("confidence") or 0.0), float(gaz.get("confidence") or 0.0))

    selected: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda e: (-rank_entity(e)[0], e["start"], -(e["end"] - e["start"]))):
        same_label_overlaps = [i for i, ent in enumerate(selected) if ent["label"] == cand["label"] and overlap(ent, cand)]
        if not same_label_overlaps:
            selected.append(cand)
            continue

        replace_idx = None
        for idx in same_label_overlaps:
            if rank_entity(cand) > rank_entity(selected[idx]):
                replace_idx = idx
                break
        if replace_idx is not None:
            selected[replace_idx] = cand

    return dedupe_exact([strip_for_gold(ent) for ent in selected])


def is_good_review_candidate(text: str, entities: list[dict[str, Any]], min_len: int, max_len: int, max_entities: int) -> bool:
    text = " ".join(text.split())
    if not (min_len <= len(text) <= max_len):
        return False
    if not entities or len(entities) > max_entities:
        return False
    if text.count("|") >= 3:
        return False

    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.75:
            return False
    return True


def load_sentence_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(root.glob("*/sentence_entities.jsonl")):
        for row in iter_jsonl(path):
            sid = row.get("sentence_id")
            if sid:
                rows[sid] = row
    return rows


def build_candidates(
    gazetteer_root: Path,
    gliner_root: Path,
    existing_texts: set[str],
    min_len: int,
    max_len: int,
    max_entities: int,
) -> list[dict[str, Any]]:
    gaz_rows = load_sentence_rows(gazetteer_root)
    gliner_rows = load_sentence_rows(gliner_root)
    candidates = []

    for sid in sorted(set(gaz_rows) | set(gliner_rows)):
        gaz = gaz_rows.get(sid) or {}
        gli = gliner_rows.get(sid) or {}
        base = gaz or gli
        text = base.get("text") or ""
        if not text or text in existing_texts:
            continue

        gaz_ents = [e for e in (normalize_entity(text, ent, "gazetteer") for ent in gaz.get("entities") or []) if e]
        gli_ents = [e for e in (normalize_entity(text, ent, "gliner") for ent in gli.get("entities") or []) if e]
        hybrid_ents = merge_hybrid(text, gaz_ents, gli_ents)
        if not is_good_review_candidate(text, hybrid_ents, min_len=min_len, max_len=max_len, max_entities=max_entities):
            continue

        candidates.append({
            "sentence_id": sid,
            "package_id": base.get("package_id") or (Path(sid).parts[0] if sid else None),
            "document_number": base.get("document_number"),
            "path_text": base.get("path_text"),
            "text": text,
            "entities": hybrid_ents,
            "gazetteer_entities": [strip_for_gold(e) for e in gaz_ents],
            "gliner_entities": [strip_for_gold(e) for e in gli_ents],
            "label_set": sorted({e["label"] for e in hybrid_ents}),
        })
    return candidates


def entity_label_counts(items: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    for item in items:
        counts.update(ent["label"] for ent in item.get("entities") or [])
    return counts


def select_diverse(candidates: list[dict[str, Any]], need: int, existing_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    selected_sentence_ids = set()
    label_counts = entity_label_counts(existing_cases)
    package_counts = Counter()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        groups[cand.get("package_id") or "UNKNOWN"].append(cand)

    def candidate_score(cand: dict[str, Any]) -> tuple[float, float, int]:
        labels = {ent["label"] for ent in cand["entities"]}
        rarity = sum(1.0 / (1 + label_counts[label]) for label in labels)
        pkg_penalty = package_counts[cand.get("package_id") or "UNKNOWN"] * 0.05
        length_penalty = abs(len(cand["text"]) - 180) / 1000
        return (rarity - pkg_penalty - length_penalty, -abs(len(cand["entities"]) - 3), -len(cand["text"]))

    for pkg in groups:
        groups[pkg].sort(key=lambda c: (-len(c["label_set"]), len(c["text"])))

    packages = sorted(groups)
    max_rounds = max((len(v) for v in groups.values()), default=0)

    for _ in range(max_rounds):
        if len(selected) >= need:
            break
        for pkg in packages:
            if len(selected) >= need:
                break
            pool = [c for c in groups[pkg] if c["sentence_id"] not in selected_sentence_ids]
            if not pool:
                continue
            best = max(pool, key=candidate_score)
            selected.append(best)
            selected_sentence_ids.add(best["sentence_id"])
            package_counts[best.get("package_id") or "UNKNOWN"] += 1
            label_counts.update(ent["label"] for ent in best["entities"])

    if len(selected) < need:
        for cand in sorted(candidates, key=candidate_score, reverse=True):
            if len(selected) >= need:
                break
            if cand["sentence_id"] in selected_sentence_ids:
                continue
            selected.append(cand)
            selected_sentence_ids.add(cand["sentence_id"])
            label_counts.update(ent["label"] for ent in cand["entities"])

    return selected[:need]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a 200-case NER review benchmark from current gold cases and cached hybrid predictions.")
    ap.add_argument("--gold-file", default="data/benchmark/ner_gold_benchmark/ner_benchmark_gemini_clean.json")
    ap.add_argument("--gazetteer-pred-root", default="data/preprocessed/gazetteer_pseudo_labels")
    ap.add_argument("--gliner-pred-root", default="data/preprocessed/gliner_predictions_th070")
    ap.add_argument("--output-file", default="data/benchmark/ner_gold_benchmark/ner_benchmark_hybrid_review_200.json")
    ap.add_argument("--new-cases-file", default="data/benchmark/ner_gold_benchmark/ner_benchmark_hybrid_new_cases_128.json")
    ap.add_argument("--debug-file", default="data/benchmark/ner_gold_benchmark/ner_benchmark_hybrid_new_cases_128_debug.json")
    ap.add_argument("--summary-file", default="data/benchmark/ner_gold_benchmark/summary_hybrid_review_200.json")
    ap.add_argument("--target-count", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=650)
    ap.add_argument("--max-entities", type=int, default=12)
    args = ap.parse_args()

    gold_cases_raw = read_json(args.gold_file)
    gold_cases_for_stats = [normalize_gold_case(item) for item in gold_cases_raw]
    existing_texts = {item.get("text") or "" for item in gold_cases_raw}
    need = max(0, args.target_count - len(gold_cases_raw))

    candidates = build_candidates(
        gazetteer_root=Path(args.gazetteer_pred_root),
        gliner_root=Path(args.gliner_pred_root),
        existing_texts=existing_texts,
        min_len=args.min_len,
        max_len=args.max_len,
        max_entities=args.max_entities,
    )
    selected = select_diverse(candidates, need=need, existing_cases=gold_cases_for_stats)
    new_cases = [{"text": item["text"], "entities": item["entities"]} for item in selected]
    combined = gold_cases_raw + new_cases

    write_json(args.output_file, combined)
    write_json(args.new_cases_file, new_cases)
    write_json(args.debug_file, selected)

    summary = {
        "source_gold_file": args.gold_file,
        "output_file": args.output_file,
        "new_cases_file": args.new_cases_file,
        "debug_file": args.debug_file,
        "target_count": args.target_count,
        "existing_gold_cases": len(gold_cases_raw),
        "needed_new_cases": need,
        "available_hybrid_candidates_after_filter": len(candidates),
        "added_hybrid_cases": len(new_cases),
        "total_cases": len(combined),
        "entity_mentions_total": sum(len(item["entities"]) for item in combined),
        "entity_mentions_added": sum(len(item["entities"]) for item in new_cases),
        "label_counts_total": dict(sorted(entity_label_counts(combined).items())),
        "label_counts_added": dict(sorted(entity_label_counts(new_cases).items())),
        "package_counts_added": dict(sorted(Counter(item.get("package_id") or "UNKNOWN" for item in selected).items())),
        "selection_filters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "max_entities": args.max_entities,
            "skip_duplicate_text_from_gold": True,
            "skip_table_like_text_with_3_or_more_pipes": True,
            "skip_mostly_uppercase_text": True,
        },
        "note": (
            "First existing_gold_cases items are the old cleaned gold cases. "
            "The appended cases are hybrid pre-labels built from cached gazetteer_pseudo_labels "
            "and gliner_predictions_th070; review them manually before treating them as gold."
        ),
    }
    write_json(args.summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(combined) < args.target_count:
        raise SystemExit(f"Only built {len(combined)} cases, target was {args.target_count}.")


if __name__ == "__main__":
    main()
