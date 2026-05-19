\
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import (
    ensure_dir,
    find_all_exact,
    find_sentence_package_dirs,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
)
from .gazetteer import GazetteerMatcher


class GazetteerLocalExtractor:
    """
    CPU-only deterministic extractor.
    Useful as a high-precision fallback or baseline before trained model.
    """
    def __init__(self, gazetteer_root: str | Path):
        self.matcher = GazetteerMatcher.from_root(gazetteer_root)

    def extract(self, text: str, context: str = "") -> Dict[str, Any]:
        direct = []
        inherited = []

        for a in self.matcher.aliases:
            surface = a.get("surface") or ""
            for start, end in find_all_exact(text, surface):
                direct.append({
                    "text": text[start:end],
                    "label": a.get("label"),
                    "start": start,
                    "end": end,
                    "canonical": a.get("canonical"),
                    "entity_id": a.get("entity_id"),
                    "scope": "direct",
                    "source": "gazetteer_local",
                    "confidence": 1.0,
                    "graph_weight": float(a.get("graph_weight", 1.0)),
                })
            if context:
                for start, end in find_all_exact(context, surface):
                    inherited.append({
                        "text": context[start:end],
                        "label": a.get("label"),
                        "start": start,
                        "end": end,
                        "canonical": a.get("canonical"),
                        "entity_id": a.get("entity_id"),
                        "scope": "inherited",
                        "source": "gazetteer_local_inherited",
                        "confidence": 0.75,
                        "graph_weight": min(0.45, float(a.get("graph_weight", 1.0))),
                    })

        return {"entities": direct, "inherited_entities": inherited}


def predict_all_gazetteer(sentences_root: str | Path, gazetteer_root: str | Path, output_dir: str | Path):
    output_dir = ensure_dir(output_dir)
    extractor = GazetteerLocalExtractor(gazetteer_root)

    all_mentions = []
    summary = {
        "sentence_count": 0,
        "sentence_with_entity_count": 0,
        "entity_count": 0,
        "inherited_entity_count": 0,
        "packages": {},
    }

    for pkg_dir in find_sentence_package_dirs(sentences_root):
        pkg_out = ensure_dir(output_dir / pkg_dir.name)
        rows = []
        mentions = []
        for r in read_jsonl(pkg_dir / "sentences.jsonl"):
            text = r.get("text") or ""
            context = r.get("path_text") or ""
            pred = extractor.extract(text, context)
            out = {**r, **pred, "entity_count": len(pred["entities"]), "inherited_entity_count": len(pred["inherited_entities"])}
            rows.append(out)
            for e in pred["entities"] + pred["inherited_entities"]:
                m = {
                    "mention_id": stable_id(r.get("sentence_id") or "", e.get("label") or "", e.get("text") or "", e.get("scope") or "", str(e.get("start")), prefix="ment"),
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

        write_jsonl(pkg_out / "sentence_entities.jsonl", rows)
        write_jsonl(pkg_out / "entity_mentions.jsonl", mentions)

        pkg_summary = {
            "sentence_count": len(rows),
            "sentence_with_entity_count": sum(1 for r in rows if r["entity_count"] or r["inherited_entity_count"]),
            "entity_count": sum(r["entity_count"] for r in rows),
            "inherited_entity_count": sum(r["inherited_entity_count"] for r in rows),
        }
        write_json(pkg_out / "entity_summary.json", pkg_summary)
        summary["packages"][pkg_dir.name] = pkg_summary
        summary["sentence_count"] += pkg_summary["sentence_count"]
        summary["sentence_with_entity_count"] += pkg_summary["sentence_with_entity_count"]
        summary["entity_count"] += pkg_summary["entity_count"]
        summary["inherited_entity_count"] += pkg_summary["inherited_entity_count"]
        all_mentions.extend(mentions)

    write_jsonl(output_dir / "all_entity_mentions.jsonl", all_mentions)
    write_json(output_dir / "entity_summary.json", summary)
    return summary
