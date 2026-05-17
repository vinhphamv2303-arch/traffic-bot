\
import re
from collections import defaultdict

from .utils import canonical_key, doc_number_key, normalize_numeric_label, point_key, read_json, read_jsonl, term_overlap_score

class LegalInventory:
    """
    Inventory from package parser output.

    Important fix:
    - Article index stores exact article node plus descendants for fallback.
    - Clause index stores only exact khoan nodes, not diem children.
    - Point index stores only exact diem nodes.
    This prevents "khoản 1 Điều 4" returning Khoản 1, Điểm a, Điểm b with same score.
    """
    def __init__(self):
        self.packages = {}
        self.documents_by_id = {}
        self.documents_by_number = {}
        self.documents = []

        self.attachments_by_id = {}
        self.attachments_by_package = defaultdict(list)
        self.attachments_by_label = defaultdict(list)
        self.forms_by_package = defaultdict(list)
        self.forms_by_label = defaultdict(list)
        self.forms_by_number = defaultdict(list)

        self.units_by_id = {}
        self.article_exact = defaultdict(list)
        self.article_desc = defaultdict(list)
        self.clause_exact = defaultdict(list)
        self.point_exact = defaultdict(list)

    @classmethod
    def from_packages(cls, package_dirs):
        inv = cls()
        for d in package_dirs:
            inv.add_package(d)
        return inv

    def add_package(self, package_dir):
        pkg = read_json(package_dir / "package_inventory.json")
        package_id = pkg.get("package_id") or package_dir.name
        self.packages[package_id] = {"package_id": package_id, "package_dir": str(package_dir), **pkg}

        main = pkg.get("main_document") or {}
        if main:
            doc_id = main.get("document_id")
            doc_no = main.get("document_number")
            rec = {
                "target_id": doc_id,
                "target_type": "document",
                "package_id": package_id,
                "document_id": doc_id,
                "document_number": doc_no,
                "label": doc_no,
                "title": main.get("document_title"),
                "source_file": main.get("source_file"),
            }
            if doc_id:
                self.documents_by_id[doc_id] = rec
                self.documents.append(rec)
            k = doc_number_key(doc_no or "")
            if k:
                self.documents_by_number[k] = rec

        for att in pkg.get("attachments", []) or []:
            att_id = att.get("attachment_id")
            label = att.get("label") or ""
            title = att.get("title") or ""
            kind = att.get("attachment_kind") or att.get("attachment_type")
            rec = {
                "target_id": att_id,
                "target_type": "attachment",
                "package_id": package_id,
                "document_id": att.get("document_id") or main.get("document_id"),
                "document_number": att.get("document_number") or main.get("document_number"),
                "attachment_id": att_id,
                "attachment_type": kind,
                "label": label,
                "title": title,
                "source_file": att.get("source_file"),
                "parsed_dir": att.get("parsed_dir"),
                "parent_appendix_label": att.get("parent_appendix_label"),
                "parent_appendix_labels": att.get("parent_appendix_labels") or [],
                "parent_appendix_inferences": att.get("parent_appendix_inferences") or [],
            }
            if att_id:
                self.attachments_by_id[att_id] = rec
            self.attachments_by_package[package_id].append(rec)

            for key in {
                canonical_key(label),
                canonical_key(label.split()[-1] if label else ""),
                canonical_key(title),
            }:
                if key:
                    self.attachments_by_label[(package_id, key)].append(rec)

            # Form-like attachments: index by normalized number and by title/label.
            if kind in {"form", "appendix_form"}:
                self.forms_by_package[package_id].append(rec)
                num_key = normalize_numeric_label(label or title)
                if num_key:
                    self.forms_by_number[(package_id, num_key)].append(rec)
                for key in {canonical_key(label), canonical_key(title)}:
                    if key:
                        self.forms_by_label[(package_id, key)].append(rec)

        for group in pkg.get("inferred_appendix_groups", []) or []:
            label = group.get("label") or ""
            title = group.get("title") or ""
            group_id = group.get("attachment_id") or f"{package_id}.{canonical_key(label)}"
            rec = {
                "target_id": group_id,
                "target_type": "appendix_group",
                "package_id": package_id,
                "document_id": group.get("document_id") or main.get("document_id"),
                "document_number": group.get("document_number") or main.get("document_number"),
                "attachment_id": group_id,
                "attachment_type": group.get("attachment_kind") or "inferred_appendix_group",
                "label": label,
                "title": title,
                "source_file": group.get("source_file"),
                "parsed_dir": group.get("parsed_dir"),
                "child_attachment_ids": group.get("child_attachment_ids") or [],
                "inference_method": group.get("inference_method"),
                "inference_confidence": group.get("inference_confidence"),
            }
            if group_id:
                self.attachments_by_id[group_id] = rec
            self.attachments_by_package[package_id].append(rec)
            for key in {canonical_key(label), canonical_key(label.split()[-1] if label else ""), canonical_key(title)}:
                if key:
                    self.attachments_by_label[(package_id, key)].append(rec)

        units_path = package_dir / "all_units.jsonl"
        if units_path.exists():
            for u in read_jsonl(units_path):
                self.add_unit(package_id, u)

    def add_unit(self, package_id, unit):
        uid = unit.get("unit_id") or unit.get("id")
        if not uid:
            return
        unit = {**unit, "package_id": package_id, "target_id": uid, "target_type": "unit"}
        self.units_by_id[uid] = unit

        doc_id = unit.get("document_id")
        if not doc_id:
            return

        typ = unit.get("type") or unit.get("unit_type")
        art = self._article_no(unit)
        cl = self._clause_no(unit)
        pt = self._point_no(unit)

        if art:
            if typ == "dieu":
                self.article_exact[(doc_id, canonical_key(art))].append(unit)
            self.article_desc[(doc_id, canonical_key(art))].append(unit)

        if art and cl and typ == "khoan":
            self.clause_exact[(doc_id, canonical_key(art), canonical_key(cl))].append(unit)

        if art and cl and pt and typ == "diem":
            self.point_exact[(doc_id, canonical_key(art), canonical_key(cl), point_key(pt))].append(unit)

    def _article_no(self, u):
        typ = u.get("type") or u.get("unit_type")
        if typ == "dieu":
            return str(u.get("raw_no") or u.get("no") or "")
        m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", u.get("path_text") or "", re.IGNORECASE)
        return m.group(1) if m else None

    def _clause_no(self, u):
        typ = u.get("type") or u.get("unit_type")
        if typ == "khoan":
            return str(u.get("raw_no") or u.get("no") or "")
        m = re.search(r"Khoản\s+(\d+)", u.get("path_text") or "", re.IGNORECASE)
        return m.group(1) if m else None

    def _point_no(self, u):
        typ = u.get("type") or u.get("unit_type")
        if typ == "diem":
            return str(u.get("raw_no") or u.get("no") or "")
        m = re.search(r"Điểm\s+([a-zđ])", u.get("path_text") or "", re.IGNORECASE)
        return m.group(1) if m else None

    def main_document(self, package_id):
        main = (self.packages.get(package_id) or {}).get("main_document") or {}
        return self.documents_by_id.get(main.get("document_id"))

    def find_attachment_by_id(self, attachment_id):
        return self.attachments_by_id.get(attachment_id or "")

    def find_doc_by_number(self, raw):
        k = doc_number_key(raw or "")
        return self.documents_by_number.get(k) if k else None

    def find_doc_by_title_hint(self, title_hint, min_score=0.82, min_gap=0.08):
        hint_key = canonical_key(title_hint or "")
        if not hint_key:
            return []

        scored = []
        for doc in self.documents:
            title = doc.get("title") or ""
            title_key = canonical_key(title)
            if not title_key:
                continue

            score = term_overlap_score(title_hint, title)
            if title_key in hint_key:
                score = max(score, 0.98)
            if score >= min_score:
                scored.append((doc, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return []
        if len(scored) >= 2 and scored[1][1] >= scored[0][1] - min_gap:
            return []
        return scored

    def find_attachment(self, package_id, label):
        keys = {canonical_key(label), canonical_key(label.split()[-1] if label else "")}
        out, seen = [], set()
        for k in keys:
            for x in self.attachments_by_label.get((package_id, k), []):
                tid = x.get("target_id")
                if tid not in seen:
                    seen.add(tid)
                    out.append(x)
        return out
