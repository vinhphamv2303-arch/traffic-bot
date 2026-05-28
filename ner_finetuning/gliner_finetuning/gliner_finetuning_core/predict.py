
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .common import LABELS, ensure_dir, iter_sentence_entity_files, read_jsonl, stable_id, write_json, write_jsonl


LABELS_FOR_GLINER = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]


def predict_all(
    sentences_or_entities_root: str | Path,
    model_dir: str,
    output_dir: str | Path,
    threshold: float = 0.35,
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Run GLiNER on sentence rows with entity annotations.

    It only uses the text fields, so the input root can contain
    */sentences_with_entities.jsonl. Legacy names such as
    */sentence_entities.jsonl and */sentences_with_entity_links.jsonl
    are still supported by iter_sentence_entity_files().
    """
    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("Install GLiNER first: pip install gliner") from e

    out_root = ensure_dir(output_dir)
    model = GLiNER.from_pretrained(model_dir)
    model = model.to(device)

    all_mentions = []
    summary = {
        "sentence_count": 0,
        "sentence_with_entity_count": 0,
        "entity_count": 0,
        "threshold": threshold,
        "model_dir": model_dir,
        "device": device,
        "packages": {},
    }

    for f in iter_sentence_entity_files(sentences_or_entities_root):
        pkg = f.parent.name
        pkg_out = ensure_dir(out_root / pkg)
        rows_out = []
        mentions = []

        batch_rows = []
        def flush():
            nonlocal batch_rows, rows_out, mentions
            if not batch_rows:
                return
            texts = [r.get("text") or "" for r in batch_rows]
            # Some GLiNER versions support batch_predict_entities; fallback to loop.
            try:
                preds_batch = model.batch_predict_entities(texts, LABELS_FOR_GLINER, threshold=threshold)
            except Exception:
                preds_batch = [model.predict_entities(t, LABELS_FOR_GLINER, threshold=threshold) for t in texts]

            for r, preds in zip(batch_rows, preds_batch):
                ents = []
                for p in preds:
                    e = {
                        "text": p.get("text"),
                        "label": p.get("label"),
                        "start": p.get("start"),
                        "end": p.get("end"),
                        "confidence": float(p.get("score", 0.0)),
                        "scope": "direct",
                        "source": "gliner_v2",
                        "graph_weight": float(p.get("score", 0.0)),
                    }
                    ents.append(e)
                    mentions.append({
                        "mention_id": stable_id(r.get("sentence_id") or "", e["label"] or "", e["text"] or "", str(e["start"]), prefix="ment"),
                        "sentence_id": r.get("sentence_id"),
                        "passage_id": r.get("passage_id"),
                        "source_unit_id": r.get("source_unit_id"),
                        "package_id": r.get("package_id"),
                        "document_id": r.get("document_id"),
                        "document_number": r.get("document_number"),
                        "document_title": r.get("document_title"),
                        "path_text": r.get("path_text"),
                        **e,
                    })
                rows_out.append({**r, "entities": ents, "entity_count": len(ents)})
            batch_rows = []

        for r in read_jsonl(f):
            batch_rows.append(r)
            if len(batch_rows) >= batch_size:
                flush()
        flush()

        write_jsonl(pkg_out / "sentences_with_entities.jsonl", rows_out)
        write_jsonl(pkg_out / "entity_mentions.jsonl", mentions)

        pkg_summary = {
            "sentence_count": len(rows_out),
            "sentence_with_entity_count": sum(1 for r in rows_out if r.get("entity_count", 0) > 0),
            "entity_count": len(mentions),
        }
        write_json(pkg_out / "entity_summary.json", pkg_summary)

        summary["packages"][pkg] = pkg_summary
        summary["sentence_count"] += pkg_summary["sentence_count"]
        summary["sentence_with_entity_count"] += pkg_summary["sentence_with_entity_count"]
        summary["entity_count"] += pkg_summary["entity_count"]
        all_mentions.extend(mentions)

    write_jsonl(out_root / "all_entity_mentions.jsonl", all_mentions)
    write_json(out_root / "entity_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run trained GLiNER over all sentence files.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    summary = predict_all(
        sentences_or_entities_root=args.input_root,
        model_dir=args.model_dir,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("GLiNER prediction completed")
    print(summary)


if __name__ == "__main__":
    main()
