
from pathlib import Path
from .utils import read_jsonl

class ReferenceStore:
    def __init__(self, resolved_refs_root):
        self.root = Path(resolved_refs_root) if resolved_refs_root else None

    def load_package(self, package_id):
        outgoing = {}
        incoming = {}
        if not self.root:
            return outgoing, incoming
        p = self.root / package_id / "resolved_references.jsonl"
        if not p.exists():
            return outgoing, incoming
        for row in read_jsonl(p):
            src = row.get("source_unit_id")
            tgt = row.get("selected_target_id")
            if src:
                outgoing.setdefault(src, []).append(row)
            if tgt:
                incoming.setdefault(tgt, []).append(row)
        return outgoing, incoming

def expansion_policy(ref):
    target_type = ref.get("selected_target_type") or ""
    label = (ref.get("selected_target_label") or "").lower()
    raw = (ref.get("raw") or "").lower()
    if target_type in {"unit", "dieu", "khoan", "diem"}:
        return "inline_if_short"
    if target_type in {"attachment", "attachment_container", "appendix_group"}:
        return "search_within_target"
    if "phụ lục" in label or "qcvn" in label or "quy chuẩn" in label:
        return "search_within_target"
    if "phụ lục" in raw or "qcvn" in raw or "quy chuẩn" in raw:
        return "search_within_target"
    if target_type == "document":
        return "search_within_target"
    return "candidate_only"

def compact_ref(ref, amendment_actions=None):
    actions = sorted({a.get("action_hint") for a in (amendment_actions or []) if a.get("action_hint")})
    return {
        "resolution_id": ref.get("resolution_id"),
        "source_unit_id": ref.get("source_unit_id"),
        "source_document_id": ref.get("source_document_id"),
        "raw": ref.get("raw"),
        "mention_type": ref.get("mention_type"),
        "status": ref.get("status"),
        "target_id": ref.get("selected_target_id"),
        "target_type": ref.get("selected_target_type"),
        "target_label": ref.get("selected_target_label"),
        "confidence": ref.get("confidence"),
        "resolver": ref.get("resolver"),
        "expansion_policy": expansion_policy(ref),
        "relation_type": "amendment" if actions else "reference",
        "amendment_actions": actions,
    }
