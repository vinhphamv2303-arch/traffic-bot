
from collections import Counter, defaultdict
from pathlib import Path

from .utils import (
    canonical_key,
    collapse_ws,
    default_status,
    ensure_dir,
    normalize_surface,
    read_jsonl,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)

def _example_from_mention(m):
    return {
        "sentence_id": m.get("sentence_id"),
        "passage_id": m.get("passage_id"),
        "package_id": m.get("package_id"),
        "document_number": m.get("document_number"),
        "path_text": m.get("path_text"),
        "source_text": m.get("source_text") or m.get("sentence_text") or m.get("text"),
    }

def aggregate_entity_vocab(entity_mentions_path, output_dir, max_examples=5, min_count_for_summary=1):
    output_dir = ensure_dir(output_dir)
    entity_mentions_path = Path(entity_mentions_path)

    groups = {}
    key_to_labels = defaultdict(Counter)
    key_to_surfaces = defaultdict(Counter)
    total_mentions = 0
    skipped = 0

    for m in read_jsonl(entity_mentions_path):
        total_mentions += 1
        raw_surface = collapse_ws(m.get("text") or m.get("surface") or "")
        label = collapse_ws(m.get("label") or "")
        if not raw_surface or not label:
            skipped += 1
            continue

        surface = normalize_surface(raw_surface)
        norm_key = canonical_key(surface)
        if not norm_key:
            skipped += 1
            continue

        gkey = (norm_key, label)
        if gkey not in groups:
            groups[gkey] = {
                "surface_id": stable_id(norm_key, label, prefix="sf"),
                "surface": surface,
                "normalized_key": norm_key,
                "label": label,
                "count": 0,
                "packages": set(),
                "documents": set(),
                "examples": [],
                "models": Counter(),
                "sources": Counter(),
            }

        g = groups[gkey]
        g["count"] += 1
        if m.get("package_id"):
            g["packages"].add(m.get("package_id"))
        if m.get("document_number"):
            g["documents"].add(m.get("document_number"))
        if len(g["examples"]) < max_examples:
            g["examples"].append(_example_from_mention(m))
        if m.get("model"):
            g["models"][m.get("model")] += 1
        if m.get("source"):
            g["sources"][m.get("source")] += 1

        key_to_labels[norm_key][label] += 1
        key_to_surfaces[norm_key][surface] += 1

    surface_rows = []
    for (norm_key, label), g in groups.items():
        count = g["count"]
        if count < min_count_for_summary:
            continue

        status, reason = default_status(g["surface"], label, count)
        preferred_surface = key_to_surfaces[norm_key].most_common(1)[0][0]
        label_conflict = len(key_to_labels[norm_key]) > 1

        surface_rows.append({
            "surface_id": g["surface_id"],
            "surface": preferred_surface,
            "normalized_key": norm_key,
            "label": label,
            "count": count,
            "status": status,
            "reason": reason,
            "canonical": preferred_surface if status != "reject" else "",
            "label_final": label if status != "reject" else "",
            "label_conflict": label_conflict,
            "labels_seen": dict(key_to_labels[norm_key]),
            "package_count": len(g["packages"]),
            "document_count": len(g["documents"]),
            "packages": sorted(g["packages"]),
            "documents": sorted(g["documents"]),
            "examples": g["examples"],
            "models": dict(g["models"]),
            "sources": dict(g["sources"]),
        })

    surface_rows.sort(key=lambda r: (-r["count"], r["label"], r["surface"]))
    write_jsonl(output_dir / "surface_forms.jsonl", surface_rows)

    fieldnames = [
        "surface_id", "surface", "label", "count", "status", "reason",
        "canonical", "label_final", "label_conflict", "labels_seen",
        "package_count", "document_count", "example_sentence_id",
        "example_text", "example_path",
    ]

    csv_rows = []
    for r in surface_rows:
        ex0 = r["examples"][0] if r["examples"] else {}
        csv_rows.append({
            "surface_id": r["surface_id"],
            "surface": r["surface"],
            "label": r["label"],
            "count": r["count"],
            "status": r["status"],
            "reason": r["reason"],
            "canonical": r["canonical"],
            "label_final": r["label_final"],
            "label_conflict": r["label_conflict"],
            "labels_seen": "; ".join([f"{k}:{v}" for k, v in r["labels_seen"].items()]),
            "package_count": r["package_count"],
            "document_count": r["document_count"],
            "example_sentence_id": ex0.get("sentence_id"),
            "example_text": ex0.get("source_text"),
            "example_path": ex0.get("path_text"),
        })

    write_csv(output_dir / "surface_summary.csv", csv_rows, fieldnames)
    write_csv(output_dir / "reviewed_surface_forms.csv", csv_rows, fieldnames)

    conflict_rows = [r for r in csv_rows if str(r.get("label_conflict")).lower() in {"true", "1"}]
    write_csv(output_dir / "label_conflicts.csv", conflict_rows, fieldnames)

    by_label = Counter()
    by_status = Counter()
    for r in surface_rows:
        by_label[r["label"]] += r["count"]
        by_status[r["status"]] += 1

    summary = {
        "entity_mentions_path": str(entity_mentions_path),
        "total_mentions": total_mentions,
        "skipped_mentions": skipped,
        "surface_form_count": len(surface_rows),
        "label_conflict_count": len(conflict_rows),
        "by_label_mentions": dict(sorted(by_label.items())),
        "by_status_surface_forms": dict(sorted(by_status.items())),
        "outputs": {
            "surface_forms_jsonl": str(output_dir / "surface_forms.jsonl"),
            "surface_summary_csv": str(output_dir / "surface_summary.csv"),
            "reviewed_surface_forms_csv": str(output_dir / "reviewed_surface_forms.csv"),
            "label_conflicts_csv": str(output_dir / "label_conflicts.csv"),
        },
    }
    write_json(output_dir / "vocab_summary.json", summary)
    return summary
