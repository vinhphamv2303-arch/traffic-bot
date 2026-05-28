
from collections import defaultdict
from pathlib import Path
from .common import (
    VALID_LABELS, ensure_dir, read_csv, write_jsonl, write_json, write_csv,
    normalize_surface, stable_id, safe_int, label_to_filename
)

def is_true(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

def build_gazetteer(reviewed_csv, output_dir, include_conflicts=False, min_count=8):
    out = ensure_dir(output_dir)
    rows = read_csv(reviewed_csv)

    aliases = []
    canonical_nodes = {}
    by_label = defaultdict(set)
    skipped = 0
    skipped_conflicts = 0
    skipped_low_count = 0
    conflict_rows = []

    for r in rows:
        status = (r.get("status") or "").strip().lower()
        if status != "accept":
            skipped += 1
            continue
        count = safe_int(r.get("count"), 0)
        if count < min_count:
            skipped += 1
            skipped_low_count += 1
            continue
        if not include_conflicts and is_true(r.get("label_conflict")):
            skipped += 1
            skipped_conflicts += 1
            conflict_rows.append({
                "surface_id": r.get("surface_id"),
                "surface": normalize_surface(r.get("surface") or ""),
                "canonical": normalize_surface(r.get("canonical") or r.get("surface") or ""),
                "label": (r.get("label_final") or r.get("label") or "").strip(),
                "labels_seen": r.get("labels_seen"),
                "count": count,
            })
            continue

        surface = normalize_surface(r.get("surface") or "")
        canonical = normalize_surface(r.get("canonical") or surface)
        label = (r.get("label_final") or r.get("label") or "").strip()

        if not surface or not canonical or label not in VALID_LABELS:
            skipped += 1
            continue

        entity_id = stable_id(label, canonical, prefix="ent")
        item = {
            "entity_id": entity_id,
            "surface": surface,
            "canonical": canonical,
            "label": label,
            "count": count,
            "surface_id": r.get("surface_id"),
        }
        aliases.append(item)
        by_label[label].add(surface)

        key = (label, canonical)
        if key not in canonical_nodes:
            canonical_nodes[key] = {
                "entity_id": entity_id,
                "canonical": canonical,
                "label": label,
                "aliases": set(),
                "count": 0,
            }
        canonical_nodes[key]["aliases"].add(surface)
        canonical_nodes[key]["count"] += count

    aliases = sorted(aliases, key=lambda x: (x["label"], x["canonical"], x["surface"]))
    write_jsonl(out / "aliases.jsonl", aliases)

    canonical_rows = []
    for node in canonical_nodes.values():
        canonical_rows.append({
            "entity_id": node["entity_id"],
            "canonical": node["canonical"],
            "label": node["label"],
            "aliases": sorted(node["aliases"], key=lambda x: (-len(x), x)),
            "count": node["count"],
        })
    canonical_rows.sort(key=lambda x: (x["label"], x["canonical"]))
    write_jsonl(out / "canonical_entities.jsonl", canonical_rows)

    csv_rows = []
    for a in aliases:
        csv_rows.append({
            "entity_id": a["entity_id"],
            "surface": a["surface"],
            "canonical": a["canonical"],
            "label": a["label"],
            "count": a["count"],
        })
    write_csv(out / "gazetteer_terms.csv", csv_rows, ["entity_id", "surface", "canonical", "label", "count"])
    if conflict_rows:
        write_csv(out / "skipped_conflicts.csv", conflict_rows, ["surface_id", "surface", "canonical", "label", "labels_seen", "count"])

    for label, terms in sorted(by_label.items()):
        terms = sorted(terms, key=lambda x: (-len(x), x))
        with open(out / label_to_filename(label), "w", encoding="utf-8") as f:
            for t in terms:
                f.write(t + "\n")

    summary = {
        "reviewed_csv": str(reviewed_csv),
        "accepted_surface_count": len(aliases),
        "canonical_entity_count": len(canonical_rows),
        "skipped_row_count": skipped,
        "skipped_conflict_count": skipped_conflicts,
        "skipped_low_count": skipped_low_count,
        "include_conflicts": include_conflicts,
        "min_count": min_count,
        "by_label_surface_count": {k: len(v) for k, v in sorted(by_label.items())},
    }
    write_json(out / "gazetteer_summary.json", summary)
    return summary
