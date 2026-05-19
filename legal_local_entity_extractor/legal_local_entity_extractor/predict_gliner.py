\
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import LABELS, ensure_dir, find_sentence_package_dirs, read_jsonl, stable_id, write_json, write_jsonl


class GLiNERLocalPredictor:
    def __init__(self, model_dir: str | Path, labels: List[str] | None = None, threshold: float = 0.35):
        try:
            from gliner import GLiNER
        except Exception as e:
            raise RuntimeError("Install GLiNER first: pip install gliner") from e
        self.model = GLiNER.from_pretrained(str(model_dir))
        self.labels = labels or LABELS
        self.threshold = threshold

    def predict_text(self, text: str) -> List[Dict[str, Any]]:
        preds = self.model.predict_entities(text, self.labels, threshold=self.threshold)
        out = []
        for p in preds:
            out.append({
                "text": p.get("text"),
                "label": p.get("label"),
                "start": p.get("start"),
                "end": p.get("end"),
                "confidence": float(p.get("score", 0.0)),
                "scope": "direct",
                "source": "gliner_local",
                "graph_weight": float(p.get("score", 0.0)),
            })
        return out


def predict_all_gliner(
    sentences_root: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    threshold: float = 0.35,
):
    output_dir = ensure_dir(output_dir)
    predictor = GLiNERLocalPredictor(model_dir, threshold=threshold)

    all_mentions = []
    summary = {"sentence_count": 0, "sentence_with_entity_count": 0, "entity_count": 0, "packages": {}}

    for pkg_dir in find_sentence_package_dirs(sentences_root):
        out_pkg = ensure_dir(output_dir / pkg_dir.name)
        rows = []
        mentions = []

        for r in read_jsonl(pkg_dir / "sentences.jsonl"):
            ents = predictor.predict_text(r.get("text") or "")
            row = {**r, "entities": ents, "entity_count": len(ents)}
            rows.append(row)
            for e in ents:
                m = {
                    "mention_id": stable_id(r.get("sentence_id") or "", e.get("label") or "", e.get("text") or "", str(e.get("start")), prefix="ment"),
                    "sentence_id": r.get("sentence_id"),
                    "passage_id": r.get("passage_id"),
                    "source_unit_id": r.get("source_unit_id"),
                    "package_id": r.get("package_id"),
                    "document_id": r.get("document_id"),
                    "document_number": r.get("document_number"),
                    "path_text": r.get("path_text"),
                    **e,
                }
                mentions.append(m)

        write_jsonl(out_pkg / "sentence_entities.jsonl", rows)
        write_jsonl(out_pkg / "entity_mentions.jsonl", mentions)
        pkg_summary = {
            "sentence_count": len(rows),
            "sentence_with_entity_count": sum(1 for r in rows if r["entity_count"]),
            "entity_count": len(mentions),
        }
        write_json(out_pkg / "entity_summary.json", pkg_summary)

        summary["packages"][pkg_dir.name] = pkg_summary
        summary["sentence_count"] += pkg_summary["sentence_count"]
        summary["sentence_with_entity_count"] += pkg_summary["sentence_with_entity_count"]
        summary["entity_count"] += pkg_summary["entity_count"]
        all_mentions.extend(mentions)

    write_jsonl(output_dir / "all_entity_mentions.jsonl", all_mentions)
    write_json(output_dir / "entity_summary.json", summary)
    return summary
