from __future__ import annotations
from pathlib import Path
from .io_utils import clean_row, ensure_dir, label_summary, read_jsonl, write_csv, write_json, write_jsonl

def collect_from_entities_root(entities_root, auto_clean=True):
    root = Path(entities_root)
    rows = []
    for pkg_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = pkg_dir / "sentence_entities.jsonl"
        if not f.exists(): continue
        for row in read_jsonl(f):
            text = row.get("text") or ""
            if not text: continue
            out = {
                "sentence_id": row.get("sentence_id"), "passage_id": row.get("passage_id"),
                "source_unit_id": row.get("source_unit_id"), "package_id": row.get("package_id"),
                "document_id": row.get("document_id"), "document_number": row.get("document_number"),
                "text": text, "entities": row.get("entities") or [], "quality": row.get("quality"),
                "model": row.get("model"), "review_status": row.get("review_status"),
            }
            rows.append(clean_row(out, auto_clean=auto_clean))
    return rows

def prepare_dataset(entities_root, output_dir, auto_clean=True):
    output_dir = ensure_dir(output_dir)
    rows = collect_from_entities_root(entities_root, auto_clean=auto_clean)
    out_file = output_dir / "sentence_entities_trainable_flat.jsonl"
    write_jsonl(out_file, rows)
    counts = label_summary(rows)
    write_csv(output_dir / "label_summary.csv", [{"label": k, "count": v} for k, v in counts.items()], ["label", "count"])
    summary = {
        "input_entities_root": str(entities_root), "row_count": len(rows),
        "row_with_entity_count": sum(1 for r in rows if r.get("entities")),
        "entity_count": sum(len(r.get("entities") or []) for r in rows),
        "by_label": counts, "train_file": str(out_file), "auto_clean": auto_clean,
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary
