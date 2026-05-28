\
from .config import SentenceSplitterConfig
from .rules import legal_sentence_split
from .utils import (
    build_sentence_text_for_ner,
    collapse_ws,
    ensure_dir,
    find_passage_package_dirs,
    make_context_text,
    md5_text,
    read_jsonl,
    safe_int,
    write_json,
    write_jsonl,
)

class LegalSentenceSplitter:
    def __init__(self, config: SentenceSplitterConfig):
        self.config = config
        self.package_dirs = find_passage_package_dirs(config.passages_root)

    def split_all(self):
        root = ensure_dir(self.config.output_root)
        summary = {
            "package_count": len(self.package_dirs),
            "total_sentences": 0,
            "packages": {},
        }
        all_sentences = []

        for package_dir in self.package_dirs:
            package_id = package_dir.name
            sentences = self.split_package(package_dir)
            out_dir = ensure_dir(root / package_id)
            write_jsonl(out_dir / "sentences.jsonl", sentences)
            pkg_summary = self.summarize(package_id, sentences)
            write_json(out_dir / "sentence_summary.json", pkg_summary)
            summary["packages"][package_id] = pkg_summary
            summary["total_sentences"] += pkg_summary["sentence_count"]
            all_sentences.extend(sentences)

        write_jsonl(root / "all_sentences.jsonl", all_sentences)
        write_json(root / "sentence_summary.json", summary)
        return summary

    def split_package(self, package_dir):
        path = package_dir / "passages.jsonl"
        if not path.exists():
            return []
        rows = []
        for passage in read_jsonl(path):
            rows.extend(self.split_passage(passage))
        rows.sort(key=lambda x: (
            x.get("package_id") or "",
            safe_int(x.get("passage_order")),
            safe_int(x.get("sentence_order")),
            x.get("sentence_id") or "",
        ))
        return rows

    def split_passage(self, passage):
        passage_id = passage.get("passage_id")
        if not passage_id:
            return []

        unit_type = passage.get("unit_type") or ""
        text = collapse_ws(passage.get(self.config.split_source_field) or "")
        if not text:
            text = collapse_ws(passage.get("content") or "")
        if not text:
            return []

        if unit_type in self.config.keep_whole_unit_types:
            sentence_texts = [text]
            sentence_type = f"{unit_type}_sentence"
        else:
            sentence_texts = legal_sentence_split(text)
            sentence_type = "legal_sentence"

        context_text = make_context_text(passage)
        rows = []
        for idx, sent in enumerate(sentence_texts, start=1):
            sent = collapse_ws(sent)
            if len(sent) < self.config.min_sentence_chars and unit_type not in self.config.keep_whole_unit_types:
                continue
            sentence_id = f"{passage_id}.s{idx:03d}"
            sentence_text_for_ner = (
                build_sentence_text_for_ner(context_text, sent)
                if self.config.include_context_for_ner
                else sent
            )
            rows.append({
                "sentence_id": sentence_id,
                "passage_id": passage_id,
                "source_unit_id": passage.get("source_unit_id"),
                "package_id": passage.get("package_id"),
                "document_id": passage.get("document_id"),
                "document_number": passage.get("document_number"),
                "document_title": passage.get("document_title"),
                "source_type": passage.get("source_type"),
                "attachment_id": passage.get("attachment_id"),
                "attachment_type": passage.get("attachment_type"),
                "unit_type": unit_type,
                "passage_kind": passage.get("passage_kind"),
                "passage_role": passage.get("passage_role"),
                "sentence_type": sentence_type,
                "sentence_order": idx,
                "passage_order": passage.get("order"),
                "text": sent,
                "context_text": context_text,
                "sentence_text_for_ner": sentence_text_for_ner,
                "path_text": passage.get("path_text"),
                "effective_from": passage.get("effective_from"),
                "ceased_from": passage.get("ceased_from"),
                "effectivity_status": passage.get("effectivity_status"),
                "has_amendment_action": passage.get("has_amendment_action") or False,
                "amendment_actions": passage.get("amendment_actions") or [],
                "outgoing_refs": passage.get("outgoing_refs") or [],
                "incoming_refs": passage.get("incoming_refs") or [],
                "reference_expansion_policies": passage.get("reference_expansion_policies") or [],
                "source_file": passage.get("source_file"),
                "text_hash": md5_text(sent),
            })
        return rows

    @staticmethod
    def summarize(package_id, sentences):
        by_type = {}
        by_unit_type = {}
        for s in sentences:
            st = s.get("sentence_type") or "unknown"
            ut = s.get("unit_type") or "unknown"
            by_type[st] = by_type.get(st, 0) + 1
            by_unit_type[ut] = by_unit_type.get(ut, 0) + 1
        return {
            "package_id": package_id,
            "sentence_count": len(sentences),
            "by_sentence_type": by_type,
            "by_unit_type": by_unit_type,
        }
