import re
from pathlib import Path

from .config import ResolverConfig
from .inventory import LegalInventory
from .selectors import parse_selector
from .utils import (
    canonical_key,
    ensure_dir,
    find_package_dirs,
    md5_text,
    normalize_document_number,
    normalize_numeric_label,
    point_key,
    read_jsonl,
    short_context,
    term_overlap_score,
    write_json,
    write_jsonl,
)

class ReferenceResolver:
    def __init__(self, config: ResolverConfig):
        self.config = config
        self.package_dirs = find_package_dirs(config.parsed_root)
        inventory_dirs = self.package_dirs
        parsed_root = Path(config.parsed_root)
        if (parsed_root / "package_inventory.json").exists() and parsed_root.parent.exists():
            sibling_dirs = find_package_dirs(parsed_root.parent)
            if len(sibling_dirs) > len(self.package_dirs):
                inventory_dirs = sibling_dirs
        self.inventory = LegalInventory.from_packages(inventory_dirs)

    def resolve_all(self):
        root = ensure_dir(self.config.output_root)
        summary = {
            "package_count": len(self.package_dirs),
            "total_mentions": 0,
            "total_resolved": 0,
            "total_ambiguous": 0,
            "total_unresolved": 0,
            "total_non_reference": 0,
            "packages": {},
        }
        all_rows = []
        for d in self.package_dirs:
            rows = self.resolve_package(d)
            out = ensure_dir(root / d.name)
            write_jsonl(out / "resolved_references.jsonl", rows)
            sm = self.summarize(rows)
            write_json(out / "reference_resolution_summary.json", sm)
            summary["packages"][d.name] = sm
            summary["total_mentions"] += sm["total"]
            summary["total_resolved"] += sm["resolved"]
            summary["total_ambiguous"] += sm["ambiguous"]
            summary["total_unresolved"] += sm["unresolved"]
            summary["total_non_reference"] += sm.get("non_reference", 0)
            all_rows.extend(rows)
        write_jsonl(root / "all_resolved_references.jsonl", all_rows)
        write_json(root / "reference_resolution_summary.json", summary)
        return summary

    def resolve_package(self, package_dir):
        p = Path(package_dir)
        mpath = p / "all_ref_mentions.jsonl"
        if not mpath.exists():
            return []
        return [self.resolve_mention(p.name, m) for m in read_jsonl(mpath)]

    def resolve_mention(self, package_id, m):
        raw = m.get("raw") or ""
        mt = m.get("mention_type") or m.get("type") or "unknown"
        source_text = m.get("source_text") or ""
        sel = parse_selector(
            raw,
            source_text,
            mention_type=mt,
            span=m.get("span"),
            source_path_text=m.get("source_path_text"),
        )

        input_issue = self.input_inconsistency(m)
        non_reference_reason = None
        cand = []
        if not input_issue:
            non_reference_reason = self.non_reference_reason(m, mt, raw, sel)

        if input_issue:
            status, top = "input_inconsistent", None
        elif non_reference_reason:
            status, top = "non_reference", None
        else:
            if mt == "legal_document":
                cand += self.resolve_document(raw, sel)
            elif mt == "appendix":
                cand += self.resolve_appendix(package_id, m, sel)
            elif mt == "form":
                cand += self.resolve_form(package_id, m, sel)
            elif mt in {"article", "clause", "point"}:
                cand += self.resolve_unit(package_id, m, sel)
            else:
                cand += self.resolve_generic(package_id, m, sel)

            cand = self.dedupe(cand)
            cand.sort(key=lambda x: x.get("score", 0), reverse=True)
            cand = cand[: self.config.max_candidates]
            status, top = self.decide(cand)

        return {
            "resolution_id": "refres_" + md5_text(str((m.get("source_unit_id"), raw, m.get("span"), top.get("target_id") if top else None)))[:16],
            "package_id": package_id,
            "source_unit_id": m.get("source_unit_id"),
            "source_document_id": m.get("document_id"),
            "source_path_text": m.get("source_path_text"),
            "mention_id": m.get("mention_id"),
            "mention_type": mt,
            "raw": raw,
            "source_context": short_context(source_text, raw, m.get("span")),
            "selector": sel,
            "status": status,
            "selected_target_id": top.get("target_id") if top else None,
            "selected_target_type": top.get("target_type") if top else None,
            "selected_target_label": (top.get("label") or top.get("title")) if top else None,
            "selected_score": top.get("score") if top else None,
            "resolver": top.get("resolver") if top else "rule",
            "confidence": top.get("score") if top else 0.0,
            "input_issue": input_issue,
            "non_reference_reason": non_reference_reason,
            "candidates": cand,
        }

    def resolve_document(self, raw, sel):
        doc_no = sel.get("document_number") or normalize_document_number(raw)
        if doc_no:
            hit = self.inventory.find_doc_by_number(doc_no)
            if hit:
                return [self.candidate(hit, 0.99, "exact_document_number")]
            return [{
                "target_id": None,
                "target_type": "document",
                "target_document_number": doc_no,
                "label": doc_no,
                "score": 0.80,
                "resolver": "exact_document_number_missing",
                "status_hint": "unresolved_missing_document",
            }]

        title_hint = sel.get("document_title_hint")
        if title_hint:
            hits = self.inventory.find_doc_by_title_hint(title_hint)
            if hits:
                doc, score = hits[0]
                return [self.candidate(doc, min(0.97, score), "document_title_hint")]
            return [{
                "target_id": None,
                "target_type": "document",
                "target_document_title": title_hint,
                "label": title_hint,
                "score": 0.78,
                "resolver": "document_title_hint_missing",
                "status_hint": "unresolved_missing_document",
            }]

        return []

    def resolve_appendix(self, package_id, m, sel):
        label = sel.get("appendix_label") or f"Phụ lục {m.get('label') or ''}".strip()
        hits = [self.candidate(x, 0.98, "same_package_appendix_label") for x in self.inventory.find_attachment(package_id, label)]
        if hits:
            return hits
        if label:
            main_doc = self.inventory.main_document(package_id) or {}
            return [{
                "target_id": None,
                "target_type": "attachment",
                "package_id": package_id,
                "document_id": main_doc.get("document_id"),
                "document_number": main_doc.get("document_number"),
                "label": label,
                "score": 0.80,
                "resolver": "same_package_appendix_label_missing",
                "status_hint": "unresolved_missing_attachment",
            }]
        return []

    def resolve_form(self, package_id, m, sel):
        """
        Fixes:
        - Mẫu số 1 matches Mẫu số 01 via numeric normalization.
        - If context contains "Phụ lục Y", prefer forms in/near that appendix/title.
        - Uses title/label scoring to reduce ambiguity across many Mẫu số 01.
        """
        if sel.get("scope_hint") == "this_attachment":
            current_attachment = self.current_attachment(m)
            current_type = (current_attachment or {}).get("attachment_type")
            if current_attachment and current_type not in {"form", "appendix_form"}:
                return [self.candidate(current_attachment, 0.98, "current_attachment_form_reference")]

        label = sel.get("form_label") or f"Mẫu số {m.get('label') or ''}".strip()
        num_key = sel.get("form_number_norm") or normalize_numeric_label(label)
        appendix_label = sel.get("appendix_label")

        hits = []
        if num_key:
            hits.extend(self.inventory.forms_by_number.get((package_id, num_key), []))

        key = canonical_key(label)
        hits.extend(self.inventory.forms_by_label.get((package_id, key), []))

        # Fallback: all form-like attachments whose label/title contains the normalized form number.
        if not hits and num_key:
            for att in self.inventory.forms_by_package.get(package_id, []):
                text_key = canonical_key(f"{att.get('label') or ''} {att.get('title') or ''}")
                if num_key and num_key in text_key:
                    hits.append(att)

        context_text = " ".join([
            m.get("source_path_text") or "",
            m.get("source_text") or "",
            sel.get("selector_raw") or "",
        ])

        unique_hit_ids = {h.get("target_id") for h in hits if h.get("target_id")}
        single_number_hit = bool(num_key) and len(unique_hit_ids) == 1

        out = []
        app_key = canonical_key(appendix_label or "")
        parent_filtered = False
        if app_key:
            exact_parent_hits = [h for h in hits if app_key in self.parent_appendix_keys(h)]
            if exact_parent_hits:
                hits = exact_parent_hits
                parent_filtered = True

        for h in hits:
            score = 0.72
            label_key = canonical_key(h.get("label") or "")
            title_key = canonical_key(h.get("title") or "")
            parsed_dir_key = canonical_key(h.get("parsed_dir") or "")
            source_file_key = canonical_key(h.get("source_file") or "")

            # Exact normalized numeric match.
            h_num = normalize_numeric_label(f"{h.get('label') or ''} {h.get('title') or ''}")
            if num_key and h_num == num_key:
                score += 0.12
                if single_number_hit:
                    score = max(score, 0.93)

            # Appendix context boost.
            if parent_filtered:
                score = max(score, 0.96)
            if app_key and (app_key in title_key or app_key in label_key or app_key in parsed_dir_key or app_key in source_file_key):
                score += 0.10

            # Direct title/label contains label key.
            if key and (key in label_key or key in title_key):
                score += 0.03

            candidate_text = " ".join([
                h.get("title") or "",
                self.basename_text(h.get("source_file")),
                self.basename_text(h.get("parsed_dir")),
            ])
            lexical = term_overlap_score(context_text, candidate_text)
            if lexical >= 0.85:
                score += 0.11 + 0.08 * min(1.0, (lexical - 0.85) / 0.15)
            elif lexical >= 0.55:
                score += 0.05 * ((lexical - 0.55) / 0.30)

            resolver_name = "same_package_form_parent_appendix" if parent_filtered else "same_package_form_label_contextual"
            out.append(self.candidate(h, min(score, 0.99), resolver_name))

        return out

    def resolve_unit(self, package_id, m, sel):
        doc_id = None
        resolver_scope = "same_document"
        doc_confidence = 1.0
        force_same_document = self.relative_unit_scope(sel)
        if not force_same_document and sel.get("document_number"):
            doc = self.inventory.find_doc_by_number(sel["document_number"])
            if doc:
                doc_id = doc.get("document_id")
                resolver_scope = "document_number"
            else:
                return [{
                    "target_id": None,
                    "target_type": "document",
                    "target_document_number": sel["document_number"],
                    "label": sel["document_number"],
                    "score": 0.80,
                    "resolver": "unit_selector_document_number_missing",
                    "status_hint": "unresolved_missing_document",
                }]
        elif not force_same_document and sel.get("document_title_hint"):
            hits = self.inventory.find_doc_by_title_hint(sel["document_title_hint"])
            if hits:
                doc, doc_confidence = hits[0]
                doc_id = doc.get("document_id")
                resolver_scope = "document_title"
            else:
                return [{
                    "target_id": None,
                    "target_type": "document",
                    "target_document_title": sel["document_title_hint"],
                    "label": sel["document_title_hint"],
                    "score": 0.78,
                    "resolver": "unit_selector_document_title_missing",
                    "status_hint": "unresolved_missing_document",
                }]
        if not doc_id:
            doc = self.inventory.main_document(package_id)
            doc_id = doc.get("document_id") if doc else m.get("document_id")
        if not doc_id:
            return []

        a, c, p = sel.get("article"), sel.get("clause"), sel.get("point")

        if a and c and p:
            return [
                self.candidate(x, 0.98 * doc_confidence, f"{resolver_scope}_point_selector")
                for x in self.inventory.point_exact.get((doc_id, canonical_key(a), canonical_key(c), point_key(p)), [])
            ]

        if a and c:
            return [
                self.candidate(x, 0.97 * doc_confidence, f"{resolver_scope}_clause_selector_exact")
                for x in self.inventory.clause_exact.get((doc_id, canonical_key(a), canonical_key(c)), [])
            ]

        if a:
            exact = [
                self.candidate(x, 0.96 * doc_confidence, f"{resolver_scope}_article_selector_exact")
                for x in self.inventory.article_exact.get((doc_id, canonical_key(a)), [])
            ]
            if exact:
                return exact
            # Fallback descendants lower confidence.
            return [
                self.candidate(x, 0.72 * doc_confidence, f"{resolver_scope}_article_selector_descendant")
                for x in self.inventory.article_desc.get((doc_id, canonical_key(a)), [])
            ]

        return []

    @staticmethod
    def relative_unit_scope(sel):
        return (sel.get("scope_hint") or "") in {
            "this_unit_or_article",
            "this_article",
            "this_unit_or_clause",
            "this_clause",
        }

    def resolve_generic(self, package_id, m, sel):
        out = []
        if sel.get("document_number"):
            out += self.resolve_document(m.get("raw") or "", sel)
        elif sel.get("document_title_hint"):
            out += self.resolve_document(m.get("raw") or "", sel)
        if sel.get("appendix_label"):
            out += self.resolve_appendix(package_id, m, sel)
        if sel.get("form_label"):
            out += self.resolve_form(package_id, m, sel)
        if sel.get("article") or sel.get("clause") or sel.get("point"):
            out += self.resolve_unit(package_id, m, sel)
        return out

    @staticmethod
    def basename_text(path_text):
        if not path_text:
            return ""
        text = str(path_text).replace("\\", "/").rstrip("/")
        return text.rsplit("/", 1)[-1]

    def current_attachment(self, m):
        att_id = m.get("attachment_id")
        if att_id:
            hit = self.inventory.find_attachment_by_id(att_id)
            if hit:
                return hit

        source_unit_id = m.get("source_unit_id") or ""
        if not source_unit_id:
            return None
        best = None
        for candidate_id, att in self.inventory.attachments_by_id.items():
            if source_unit_id.startswith(candidate_id + "."):
                if best is None or len(candidate_id) > len(best.get("attachment_id") or ""):
                    best = att
        return best

    def non_reference_reason(self, m, mention_type, raw, sel):
        if mention_type not in {"appendix", "form"}:
            return None

        current = self.current_attachment(m)
        if not current:
            return None

        if mention_type == "appendix":
            if not self._same_appendix_identity(raw, sel, current):
                return None
        elif mention_type == "form":
            if not self._same_form_identity(raw, sel, current):
                return None

        if self._source_is_attachment_self_label(m, raw, current):
            return "attachment_self_label"
        return None

    @staticmethod
    def _source_is_attachment_self_label(m, raw, current):
        source_unit_id = (m.get("source_unit_id") or "").lower()
        unit_type = (m.get("source_unit_type") or m.get("unit_type") or "").lower()
        if source_unit_id.endswith(".summary"):
            return True
        if ".form_title_" in source_unit_id:
            return True
        if unit_type in {"attachment_summary", "form_summary", "embedded_form_title"}:
            return True

        source_text = m.get("source_text") or ""
        text_key = canonical_key(source_text)
        raw_key = canonical_key(raw)
        label_key = canonical_key(current.get("label") or "")
        title_key = canonical_key(current.get("title") or "")

        if not text_key or not raw_key:
            return False

        short_title_like = len(source_text) <= 240 and text_key.startswith(raw_key)
        if short_title_like and label_key and text_key.startswith(label_key):
            return True
        if short_title_like and title_key and title_key in text_key:
            return True

        # Form/table headers can repeat their own label before "issued with"
        # boilerplate. These are identifiers, not outbound references.
        if text_key.startswith(raw_key) and "banhanhkemtheo" in text_key[:220]:
            return True

        return False

    @staticmethod
    def _same_appendix_identity(raw, sel, current):
        wanted = ReferenceResolver._appendix_identity_key(sel.get("appendix_label") or raw)
        current_key = ReferenceResolver._appendix_identity_key(current.get("label") or "")
        return bool(wanted and current_key and wanted == current_key)

    @staticmethod
    def _same_form_identity(raw, sel, current):
        wanted_keys = {
            ReferenceResolver._form_identity_key(raw),
            ReferenceResolver._form_identity_key(sel.get("form_label") or ""),
        }
        wanted_keys.discard("")
        current_keys = {
            ReferenceResolver._form_identity_key(current.get("label") or ""),
            ReferenceResolver._form_identity_key(current.get("title") or ""),
        }
        current_keys.discard("")
        if wanted_keys & current_keys:
            return True
        for wanted in wanted_keys:
            if wanted.isdigit():
                continue
            if len(wanted) >= 2 and any(current.startswith(wanted) for current in current_keys):
                return True

        wanted_num = (
            ReferenceResolver._simple_numeric_form_key(sel.get("form_label") or "")
            or ReferenceResolver._simple_numeric_form_key(raw)
        )
        current_num = ReferenceResolver._simple_numeric_form_key(current.get("label") or "")
        return bool(wanted_num and current_num and wanted_num == current_num)

    @staticmethod
    def _appendix_identity_key(text):
        key = canonical_key(text)
        return key[6:] if key.startswith("phuluc") else key

    @staticmethod
    def _form_identity_key(text):
        key = canonical_key(text)
        if key.startswith("mauso"):
            return key[5:]
        if key.startswith("mau"):
            return key[3:]
        return key

    @staticmethod
    def _simple_numeric_form_key(text):
        key = ReferenceResolver._form_identity_key(text)
        if key.isdigit():
            return str(int(key))
        return None

    @staticmethod
    def parent_appendix_keys(att):
        labels = []
        if att.get("parent_appendix_label"):
            labels.append(att.get("parent_appendix_label"))
        labels.extend(att.get("parent_appendix_labels") or [])
        return {canonical_key(x) for x in labels if x}

    @staticmethod
    def input_inconsistency(m):
        raw = m.get("raw") or ""
        source_text = m.get("source_text") or ""
        if not raw or not source_text:
            return None
        if raw in source_text:
            return None
        raw_list = m.get("raw_list") or ""
        if raw_list and raw_list in source_text and ReferenceResolver._expanded_list_item_in_raw_list(m, raw, raw_list):
            return None
        raw_key = canonical_key(raw)
        source_key = canonical_key(source_text)
        if raw_key and raw_key in source_key:
            return None
        if (m.get("mention_type") or m.get("type")) in {"article", "clause", "point", "appendix", "form"}:
            return "raw_not_found_in_source_text"
        return None

    @staticmethod
    def _expanded_list_item_in_raw_list(m, raw, raw_list):
        label = str(m.get("label") or "").strip()
        values = {label} if label else set()
        values.update(re.findall(r"\d+[a-zA-Z]?|[a-zđĐ]\d?", raw or "", flags=re.UNICODE))
        for value in values:
            if value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", raw_list, flags=re.IGNORECASE | re.UNICODE):
                return True
        return False

    @staticmethod
    def candidate(t, score, resolver):
        return {
            "target_id": t.get("target_id") or t.get("unit_id") or t.get("id"),
            "target_type": t.get("target_type") or t.get("type") or t.get("unit_type"),
            "package_id": t.get("package_id"),
            "document_id": t.get("document_id"),
            "document_number": t.get("document_number"),
            "attachment_id": t.get("attachment_id"),
            "label": t.get("label") or t.get("path_text") or t.get("document_number"),
            "title": t.get("title") or t.get("document_title"),
            "path_text": t.get("path_text"),
            "parent_appendix_label": t.get("parent_appendix_label"),
            "parent_appendix_labels": t.get("parent_appendix_labels"),
            "child_attachment_ids": t.get("child_attachment_ids"),
            "score": round(float(score), 4),
            "resolver": resolver,
        }

    @staticmethod
    def dedupe(candidates):
        best = {}
        for c in candidates:
            key = c.get("target_id") or (c.get("target_type"), c.get("target_document_number"), c.get("label"))
            if key not in best or c.get("score", 0) > best[key].get("score", 0):
                best[key] = c
        return list(best.values())

    def decide(self, candidates):
        if not candidates:
            return "unresolved", None
        top = candidates[0]
        score = top.get("score", 0)
        if top.get("status_hint") in {"unresolved_missing_document", "unresolved_missing_attachment"}:
            return top.get("status_hint"), top
        if score >= self.config.resolved_threshold:
            if len(candidates) >= 2 and candidates[1].get("score", 0) >= score - 0.03:
                return "ambiguous", top
            return "resolved", top
        if score >= self.config.ambiguous_threshold:
            return "ambiguous", top
        return "unresolved", top

    @staticmethod
    def summarize(rows):
        out = {
            "total": len(rows),
            "resolved": 0,
            "ambiguous": 0,
            "unresolved": 0,
            "non_reference": 0,
            "by_status": {},
            "by_mention_type": {},
        }
        for r in rows:
            st = r.get("status") or "unknown"
            out["by_status"][st] = out["by_status"].get(st, 0) + 1
            if st == "resolved":
                out["resolved"] += 1
            elif st == "ambiguous":
                out["ambiguous"] += 1
            elif st == "non_reference":
                out["non_reference"] += 1
            else:
                out["unresolved"] += 1
            mt = r.get("mention_type") or "unknown"
            out["by_mention_type"][mt] = out["by_mention_type"].get(mt, 0) + 1
        return out
