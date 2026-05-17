from pathlib import Path
import json
import time
from typing import Dict, Any, List

from .config import LLMNERConfig
from .providers import make_provider
from .prompt import build_system_prompt, build_user_prompt
from .schema import ALLOWED_LABELS, LEGACY_LABEL_ALIASES
from .utils import (
    append_jsonl,
    chunks,
    collapse_ws,
    dedupe_entities,
    ensure_dir,
    extract_json_object,
    find_offsets,
    find_sentence_package_dirs,
    is_reference_like_entity,
    coerce_semantic_label,
    md5_text,
    read_jsonl,
    write_json,
    write_jsonl,
)

class LegalLLMNER:
    def __init__(self, config: LLMNERConfig):
        self.config = config
        self.package_dirs = find_sentence_package_dirs(config.sentences_root)
        self.provider = make_provider(config)
        self.system_prompt = build_system_prompt()

    def run_all(self):
        root = ensure_dir(self.config.output_root)
        summary = {
            "package_count": 0,
            "total_sentences": 0,
            "total_entity_mentions": 0,
            "packages": {},
            "model": self.config.model,
            "provider": self.config.provider,
        }
        all_mentions = []
        remaining = self.config.limit

        for package_dir in self.package_dirs:
            if remaining is not None and remaining <= 0:
                break
            package_id = package_dir.name
            result = self.run_package(package_dir, sentence_limit=remaining)
            summary["packages"][package_id] = result["summary"]
            summary["package_count"] += 1
            summary["total_sentences"] += result["summary"]["sentence_count"]
            summary["total_entity_mentions"] += result["summary"]["entity_mention_count"]
            all_mentions.extend(result["mentions"])
            if remaining is not None:
                remaining -= result["selected_sentence_count"]

        write_jsonl(root / "all_entity_mentions.jsonl", all_mentions)
        write_json(root / "entity_summary.json", summary)
        return summary

    def run_package(self, package_dir: Path, sentence_limit: int | None = None):
        package_id = package_dir.name
        out_dir = ensure_dir(self.config.output_root / package_id)
        sentence_entities_path = out_dir / "sentence_entities.jsonl"
        entity_mentions_path = out_dir / "entity_mentions.jsonl"
        annotation_path = out_dir / "annotation_silver.jsonl"

        if not self.config.resume:
            for path in [sentence_entities_path, entity_mentions_path, annotation_path]:
                if path.exists():
                    path.unlink()

        processed = set()
        if self.config.resume and sentence_entities_path.exists():
            for row in read_jsonl(sentence_entities_path):
                if row.get("sentence_id"):
                    processed.add(row["sentence_id"])

        sentences = list(read_jsonl(package_dir / "sentences.jsonl"))
        if sentence_limit:
            sentences = sentences[:sentence_limit]

        todo = [s for s in sentences if s.get("sentence_id") not in processed]
        mentions_all = []
        summary_by_label = {}

        for batch in chunks(todo, self.config.batch_size):
            rows, mentions = self.extract_batch(batch)
            append_jsonl(sentence_entities_path, rows)
            append_jsonl(entity_mentions_path, mentions)
            append_jsonl(annotation_path, [self.to_annotation_row(r) for r in rows])
            mentions_all.extend(mentions)
            for m in mentions:
                label = m.get("label") or "UNKNOWN"
                summary_by_label[label] = summary_by_label.get(label, 0) + 1

        # Re-read complete outputs so resume summary is correct.
        all_sentence_rows = list(read_jsonl(sentence_entities_path)) if sentence_entities_path.exists() else []
        all_mentions = list(read_jsonl(entity_mentions_path)) if entity_mentions_path.exists() else []
        summary_by_label = {}
        for m in all_mentions:
            label = m.get("label") or "UNKNOWN"
            summary_by_label[label] = summary_by_label.get(label, 0) + 1

        summary = {
            "package_id": package_id,
            "sentence_count": len(all_sentence_rows),
            "entity_mention_count": len(all_mentions),
            "by_label": summary_by_label,
            "model": self.config.model,
            "provider": self.config.provider,
        }
        write_json(out_dir / "entity_summary.json", summary)
        return {"summary": summary, "mentions": all_mentions, "selected_sentence_count": len(sentences)}

    def extract_batch(self, sentences: List[Dict[str, Any]]):
        items = []
        sentence_by_id = {}
        for s in sentences:
            sid = s.get("sentence_id")
            text = s.get("text") or ""
            input_text = s.get(self.config.input_field) or s.get(self.config.fallback_input_field) or text
            context_text = input_text if input_text and input_text != text else (s.get("context_text") or "")
            items.append({
                "id": sid,
                "text": text,
                "context": context_text,
                "input": input_text,
            })
            sentence_by_id[sid] = s

        # In prompt, use sentence text + context separately. Entity text must be from sentence text.
        prompt_items = [{"id": x["id"], "text": x["text"], "context": x["context"]} for x in items]
        user_prompt = build_user_prompt(json.dumps(prompt_items, ensure_ascii=False, indent=2))
        raw_output = self.provider.generate(self.system_prompt, user_prompt)
        parsed = extract_json_object(raw_output)
        results = parsed.get("results", [])

        result_by_id = {r.get("id"): r for r in results}
        sentence_rows = []
        mention_rows = []

        for s in sentences:
            sid = s.get("sentence_id")
            text = s.get("text") or ""
            res = result_by_id.get(sid, {"entities": []})
            raw_entities = res.get("entities") or []
            entities = []

            for ent in raw_entities:
                normalized = self.normalize_entity(ent, s)
                if normalized is None:
                    continue
                entities.append(normalized)

            entities = dedupe_entities(entities)
            for normalized in entities:
                mention_rows.append({
                    **normalized,
                    "sentence_id": sid,
                    "passage_id": s.get("passage_id"),
                    "source_unit_id": s.get("source_unit_id"),
                    "package_id": s.get("package_id"),
                    "document_id": s.get("document_id"),
                    "document_number": s.get("document_number"),
                    "source_type": s.get("source_type"),
                    "attachment_id": s.get("attachment_id"),
                    "unit_type": s.get("unit_type"),
                    "path_text": s.get("path_text"),
                    "review_status": self.config.review_status,
                    "source": "llm",
                    "model": self.config.model,
                    "provider": self.config.provider,
                })

            sentence_rows.append({
                "sentence_id": sid,
                "passage_id": s.get("passage_id"),
                "source_unit_id": s.get("source_unit_id"),
                "package_id": s.get("package_id"),
                "document_id": s.get("document_id"),
                "document_number": s.get("document_number"),
                "text": text,
                "entities": entities,
                "entity_count": len(entities),
                "review_status": self.config.review_status,
                "source": "llm",
                "model": self.config.model,
                "provider": self.config.provider,
                "prompt_hash": md5_text(self.system_prompt + user_prompt),
                "created_at_unix": int(time.time()),
            })

        return sentence_rows, mention_rows

    def normalize_entity(self, ent: Dict[str, Any], sentence: Dict[str, Any]):
        text = collapse_ws(ent.get("text") or "")
        raw_label = collapse_ws(ent.get("label") or "")
        label = LEGACY_LABEL_ALIASES.get(raw_label, raw_label)
        if label not in ALLOWED_LABELS:
            return None
        if not text:
            return None
        label = coerce_semantic_label(text, label)
        if label is None:
            return None
        if self.config.block_reference_like_entities and is_reference_like_entity(text, label):
            return None

        sent_text = sentence.get("text") or ""
        start, end = find_offsets(sent_text, text)
        if start is None or end is None:
            # Keep entity but mark unaligned; useful for review, but avoid if text is totally absent.
            if text.lower() not in sent_text.lower():
                return None
        else:
            text = sent_text[start:end]

        try:
            confidence = float(ent.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        if confidence < self.config.min_confidence:
            return None
        confidence = max(0.0, min(1.0, confidence))

        normalized = {
            "entity_id": "ent_" + md5_text(f"{sentence.get('sentence_id')}|{label}|{text}|{start}|{end}")[:16],
            "text": text,
            "label": label,
            "start": start,
            "end": end,
            "confidence": confidence,
            "alignment_status": "aligned" if start is not None else "unaligned",
        }
        if raw_label and raw_label != label:
            normalized["raw_label"] = raw_label
        return normalized

    def to_annotation_row(self, sentence_row: Dict[str, Any]):
        return {
            "sentence_id": sentence_row.get("sentence_id"),
            "text": sentence_row.get("text"),
            "entities": [
                {
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "label": e.get("label"),
                    "text": e.get("text"),
                    "confidence": e.get("confidence"),
                }
                for e in sentence_row.get("entities", [])
            ],
            "review_status": "needs_review",
            "silver_source": {
                "model": sentence_row.get("model"),
                "provider": sentence_row.get("provider"),
            },
        }
