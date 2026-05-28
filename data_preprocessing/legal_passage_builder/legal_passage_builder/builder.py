
import json
from .config import PassageBuilderConfig
from .effectivity import EffectivityStore
from .references import ReferenceStore, compact_ref
from .utils import collapse_ws, ensure_dir, find_package_dirs, md5_text, normalize_unit_id, read_json, read_jsonl, write_json, write_jsonl

ATOMIC_TYPES = {
    "diem", "khoan", "text", "table_row",
    "appendix_item_decimal", "appendix_point", "appendix_bullet", "appendix_paragraph",
    "form_field", "form_text", "technical_requirement", "qcvn_requirement",
}
CONTAINER_TYPES = {
    "phan", "chuong", "muc", "tieu_muc", "dieu",
    "attachment_summary", "appendix_section_alpha", "appendix_section_roman",
    "appendix_table", "form_summary", "form_section", "form_table", "embedded_form_title",
}

class PassageBuilder:
    def __init__(self, config: PassageBuilderConfig):
        self.config = config
        self.package_dirs = find_package_dirs(config.parsed_root)
        self.effectivity = EffectivityStore(config.effectivity_root)
        self.references = ReferenceStore(config.resolved_refs_root)

    def build_all(self):
        root = ensure_dir(self.config.output_root)
        summary = {"package_count": len(self.package_dirs), "total_passages": 0, "total_atomic": 0, "total_container": 0, "packages": {}}
        all_passages = []
        for package_dir in self.package_dirs:
            package_id = package_dir.name
            passages = self.build_package(package_dir)
            out_dir = ensure_dir(root / package_id)
            write_jsonl(out_dir / "passages.jsonl", passages)
            pkg_summary = self._summary(package_id, passages)
            write_json(out_dir / "passage_summary.json", pkg_summary)
            summary["packages"][package_id] = pkg_summary
            summary["total_passages"] += pkg_summary["passage_count"]
            summary["total_atomic"] += pkg_summary["atomic_count"]
            summary["total_container"] += pkg_summary["container_count"]
            all_passages.extend(passages)
        write_jsonl(root / "all_passages.jsonl", all_passages)
        write_json(root / "passage_summary.json", summary)
        return summary

    def build_package(self, package_dir):
        package_id = package_dir.name
        inv = self._load_inventory(package_dir)
        units_path = package_dir / "all_units.jsonl"
        if not units_path.exists():
            return []
        outgoing, incoming = self.references.load_package(package_id)
        amendment_actions = self._load_amendment_actions(package_dir)
        units = list(read_jsonl(units_path))
        child_count = self._infer_child_count(units)
        passages = []
        for unit in units:
            uid = normalize_unit_id(unit)
            if not uid:
                continue
            unit_type = unit.get("unit_type") or unit.get("type") or "unknown"
            kind = self._passage_kind(unit, child_count.get(uid, 0))
            if kind == "container" and not self.config.include_container_passages:
                continue
            unit_actions = amendment_actions.get(uid, [])
            out_refs = [compact_ref(r, unit_actions) for r in outgoing.get(uid, [])]
            in_refs = [compact_ref(r, amendment_actions.get(r.get("source_unit_id"), [])) for r in incoming.get(uid, [])]
            dates = self.effectivity.dates_for_unit(unit)
            passages.append(self._make_passage(package_id, inv, unit, uid, unit_type, kind, out_refs, in_refs, dates, unit_actions))
        passages.sort(key=lambda p: (str(p.get("package_id")), int(p.get("order") or 0), p.get("passage_id") or ""))
        return passages

    def _load_inventory(self, package_dir):
        p = package_dir / "package_inventory.json"
        return read_json(p) if p.exists() else {"package_id": package_dir.name}

    def _load_amendment_actions(self, package_dir):
        actions = {}
        for path in package_dir.glob("**/amendment_mentions.jsonl"):
            for row in read_jsonl(path):
                uid = row.get("source_unit_id")
                if not uid:
                    continue
                actions.setdefault(uid, []).append({
                    "mention_id": row.get("mention_id"),
                    "action_hint": row.get("action_hint"),
                    "raw": row.get("raw"),
                    "span": row.get("span"),
                    "operation_status": (row.get("operation") or {}).get("status"),
                })
        return actions

    def _infer_child_count(self, units):
        count = {}
        ids = {normalize_unit_id(u) for u in units if normalize_unit_id(u)}
        for u in units:
            uid = normalize_unit_id(u)
            parent = u.get("parent_id")
            if parent:
                count[parent] = count.get(parent, 0) + 1
                continue
            if uid:
                parts = uid.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    pid = ".".join(parts[:i])
                    if pid in ids and pid != uid:
                        count[pid] = count.get(pid, 0) + 1
                        break
        return count

    def _passage_kind(self, unit, child_count):
        t = unit.get("unit_type") or unit.get("type") or ""
        if t in ATOMIC_TYPES:
            return "atomic"
        if t in CONTAINER_TYPES:
            if t == "dieu" and child_count == 0:
                return "atomic"
            return "container"
        return "container" if child_count > 0 else "atomic"

    def _make_passage(self, package_id, inv, unit, uid, unit_type, kind, out_refs, in_refs, dates, amendment_actions=None):
        main = inv.get("main_document") or {}
        doc_number = unit.get("document_number") or main.get("document_number")
        doc_title = unit.get("document_title") or main.get("document_title")
        doc_id = unit.get("document_id") or main.get("document_id")
        content = collapse_ws(unit.get("content") or "")
        path_text = collapse_ws(unit.get("path_text") or "")
        passage_text = self._passage_text(doc_number, doc_title, unit, path_text, content, dates, out_refs, amendment_actions or [])
        policies = sorted(set([r.get("expansion_policy") for r in out_refs if r.get("expansion_policy")]))

        return {
            "passage_id": f"{uid}.passage",
            "source_unit_id": uid,
            "package_id": package_id,
            "document_id": doc_id,
            "document_number": doc_number,
            "document_title": doc_title,
            "source_type": unit.get("source_type") or ("attachment" if unit.get("attachment_id") else "main_document"),
            "attachment_id": unit.get("attachment_id"),
            "attachment_type": unit.get("attachment_type"),
            "unit_type": unit_type,
            "passage_kind": kind,
            "passage_role": "source_law",
            "path_text": path_text,
            "content": content,
            "passage_text": passage_text,
            "effective_from": dates.get("effective_from"),
            "ceased_from": dates.get("ceased_from"),
            "effective_to": dates.get("effective_to"),
            "effectivity_status": dates.get("effectivity_status") or "unknown",
            "unit_effectivity_override": dates.get("unit_effectivity_override"),
            "has_amendment_action": bool(amendment_actions),
            "amendment_actions": amendment_actions or [],
            "outgoing_refs": out_refs,
            "incoming_refs": in_refs,
            "has_long_reference": any(p == "search_within_target" for p in policies),
            "reference_expansion_policies": policies,
            "order": unit.get("order"),
            "structured_fields": unit.get("structured_fields") or {},
            "source_file": unit.get("source_file"),
            "content_hash": md5_text(passage_text),
        }

    def _passage_text(self, doc_number, doc_title, unit, path_text, content, dates, out_refs, amendment_actions):
        lines = []
        if doc_number:
            lines.append(f"Văn bản: {doc_number}")
        if doc_title:
            lines.append(f"Tên văn bản: {doc_title}")
        if unit.get("attachment_id"):
            lines.append(f"Tài liệu đính kèm: {unit.get('attachment_id')}")
            if unit.get("attachment_type"):
                lines.append(f"Loại tài liệu đính kèm: {unit.get('attachment_type')}")
        if dates.get("effective_from"):
            lines.append(f"Hiệu lực từ: {dates.get('effective_from')}")
        if dates.get("ceased_from"):
            lines.append(f"Hết hiệu lực từ: {dates.get('ceased_from')}")
        elif dates.get("effective_to"):
            lines.append(f"Hiệu lực đến: {dates.get('effective_to')}")
        if path_text:
            lines.append(f"Đường dẫn pháp lý: {path_text}")
        if amendment_actions:
            actions = sorted({a.get("action_hint") for a in amendment_actions if a.get("action_hint")})
            if actions:
                lines.append(f"Thao tác sửa đổi/bổ sung: {', '.join(actions)}")
        if out_refs and self.config.include_source_reference_text:
            ref_lines = []
            for r in out_refs[: self.config.max_ref_summary]:
                if r.get("target_id"):
                    ref_lines.append(f"- {r.get('raw')} -> {r.get('target_label') or r.get('target_id')} ({r.get('expansion_policy')})")
            if ref_lines:
                lines.append("Tham chiếu đã giải:")
                lines.extend(ref_lines)
        if content:
            lines.append("Nội dung:")
            lines.append(content)
        structured = unit.get("structured_fields") or {}
        if isinstance(structured, dict):
            compact = {k: v for k, v in structured.items() if k not in {"cells", "header"}}
            if compact:
                lines.append(f"Dữ liệu cấu trúc: {json.dumps(compact, ensure_ascii=False)}")
        return "\n".join([str(x) for x in lines if str(x).strip()])

    @staticmethod
    def _summary(package_id, passages):
        by_type = {}
        by_kind = {}
        for p in passages:
            by_type[p.get("unit_type") or "unknown"] = by_type.get(p.get("unit_type") or "unknown", 0) + 1
            by_kind[p.get("passage_kind") or "unknown"] = by_kind.get(p.get("passage_kind") or "unknown", 0) + 1
        return {
            "package_id": package_id,
            "passage_count": len(passages),
            "atomic_count": by_kind.get("atomic", 0),
            "container_count": by_kind.get("container", 0),
            "by_unit_type": by_type,
            "by_passage_kind": by_kind,
        }
