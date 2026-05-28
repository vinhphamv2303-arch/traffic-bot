
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import EffectivityConfig, EffectivityEvent
from .patterns import (
    DATE_PATTERN,
    EFFECTIVE_FROM_PATTERNS,
    LEGAL_DOCUMENT_REF_PATTERN,
    RELATIVE_SIGNING_EFFECTIVE_FROM_PATTERNS,
    REPEAL_DOCUMENT_PATTERN,
    REPEAL_KEYWORD_PATTERN,
    REPLACEMENT_KEYWORD_PATTERN,
    SIGNING_EFFECTIVE_FROM_PATTERNS,
    UNIT_EFFECTIVE_INDIRECT_PATTERNS,
    UNIT_EFFECTIVE_FROM_PATTERNS,
    UNIT_SELECTOR_PATTERN,
)
from .utils import collapse_ws, ensure_dir, extract_all_vietnamese_dates, md5_text, normalize_document_number, normalize_id, parse_vietnamese_date_to_iso, read_jsonl, units_file_output_name, write_jsonl

class EffectivityExtractor:
    def __init__(self, config: Optional[EffectivityConfig] = None):
        self.config = config or EffectivityConfig()

    def extract_from_units_file(self, units_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        units_path = Path(units_path)
        units = list(read_jsonl(units_path))
        events = self.extract_from_units(units)
        if output_dir is None:
            output_dir = self.config.output_base_dir / units_file_output_name(units_path)
        output_dir = ensure_dir(output_dir)
        write_jsonl(output_dir / "effectivity_events.jsonl", events)
        summary = {"source_units": str(units_path), "event_count": len(events), "events_by_type": {}}
        for ev in events:
            summary["events_by_type"][ev["event_type"]] = summary["events_by_type"].get(ev["event_type"], 0) + 1
        with open(output_dir / "effectivity_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return events

    def write_effectivity_index_csv(
        self,
        units_files: List[Union[str, Path]],
        events: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        rows: Dict[str, Dict[str, Any]] = {}

        for units_file in units_files:
            units_file = Path(units_file)
            first_unit = next(read_jsonl(units_file), None)
            if first_unit is None:
                doc_id = units_file.parent.name
                doc_number = None
            else:
                doc_id = first_unit.get("document_id") or units_file.parent.name
                doc_number = first_unit.get("document_number")
            key = self._document_key(doc_number, doc_id)
            rows[key] = {
                "document_id": doc_id,
                "document_number": doc_number or "",
                "effective_from": None,
                "effective_to": None,
                "effective_from_event_id": None,
                "effective_to_event_id": None,
                "effective_to_source_document_number": None,
                "in_corpus": "true",
                "_from_confidence": -1.0,
                "_to_confidence": -1.0,
            }

        for ev in events:
            if ev.get("event_type") == "effective_from" and ev.get("date"):
                if ev.get("target_scope") == "this_document_unit":
                    continue
                doc_number = ev.get("target_document_number") or ev.get("source_document_number")
                doc_id = ev.get("source_document_id")
                key = self._document_key(doc_number, doc_id)
                row = rows.setdefault(key, self._external_row(doc_number, doc_id))
                conf = float(ev.get("confidence") or 0)
                if self._prefer_date(row.get("effective_from"), row.get("_from_confidence", -1), ev["date"], conf):
                    row["effective_from"] = ev["date"]
                    row["effective_from_event_id"] = ev.get("event_id")
                    row["_from_confidence"] = conf

            elif ev.get("event_type") == "repeal_document" and ev.get("target_document_number") and ev.get("date"):
                doc_number = ev.get("target_document_number")
                key = self._document_key(doc_number, None)
                row = rows.setdefault(key, self._external_row(doc_number, None))
                conf = float(ev.get("confidence") or 0)
                if self._prefer_date(row.get("effective_to"), row.get("_to_confidence", -1), ev["date"], conf):
                    row["effective_to"] = ev["date"]
                    row["effective_to_event_id"] = ev.get("event_id")
                    row["effective_to_source_document_number"] = ev.get("source_document_number")
                    row["_to_confidence"] = conf

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fieldnames = [
            "document_id",
            "document_number",
            "effective_from",
            "effective_to",
            "effective_from_event_id",
            "effective_to_event_id",
            "effective_to_source_document_number",
            "in_corpus",
        ]
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows.values(), key=lambda r: (r["in_corpus"] != "true", r["document_id"] or "")):
                out = {k: (row.get(k) if row.get(k) not in {None, ""} else "null") for k in fieldnames}
                writer.writerow(out)
        return output_path

    def write_unit_overrides_csv(
        self,
        events: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        fieldnames = [
            "document_id",
            "document_number",
            "target_selector_raw",
            "target_article",
            "target_clause",
            "target_point",
            "target_appendix",
            "effective_from",
            "event_id",
            "source_unit_id",
            "source_path_text",
            "confidence",
            "raw_text",
        ]
        rows = []
        seen = set()
        for ev in events:
            if ev.get("event_type") != "effective_from":
                continue
            if ev.get("target_scope") != "this_document_unit" or not ev.get("date"):
                continue
            selector = ev.get("target_unit_selector") or {}
            row = {
                "document_id": ev.get("source_document_id"),
                "document_number": ev.get("source_document_number"),
                "target_selector_raw": ev.get("target_selector_raw"),
                "target_article": selector.get("article"),
                "target_clause": selector.get("clause"),
                "target_point": selector.get("point"),
                "target_appendix": selector.get("appendix"),
                "effective_from": ev.get("date"),
                "event_id": ev.get("event_id"),
                "source_unit_id": ev.get("source_unit_id"),
                "source_path_text": ev.get("source_path_text"),
                "confidence": ev.get("confidence"),
                "raw_text": ev.get("raw_text"),
            }
            key = tuple(row.get(k) for k in fieldnames)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (
                r.get("document_id") or "",
                r.get("target_article") or "",
                r.get("target_clause") or "",
                r.get("target_point") or "",
                r.get("target_appendix") or "",
                r.get("effective_from") or "",
            )):
                out = {k: (row.get(k) if row.get(k) not in {None, ""} else "null") for k in fieldnames}
                writer.writerow(out)
        return output_path

    def write_unresolved_effectivity_csv(
        self,
        events: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        fieldnames = [
            "document_id",
            "document_number",
            "target_selector_raw",
            "target_article",
            "target_clause",
            "target_point",
            "target_appendix",
            "date_inference",
            "event_id",
            "source_unit_id",
            "source_path_text",
            "confidence",
            "raw_text",
            "notes",
        ]
        rows = []
        seen = set()
        for ev in events:
            if ev.get("event_type") != "effective_from":
                continue
            if ev.get("target_scope") != "this_document_unit" or ev.get("date"):
                continue
            selector = ev.get("target_unit_selector") or {}
            row = {
                "document_id": ev.get("source_document_id"),
                "document_number": ev.get("source_document_number"),
                "target_selector_raw": ev.get("target_selector_raw"),
                "target_article": selector.get("article"),
                "target_clause": selector.get("clause"),
                "target_point": selector.get("point"),
                "target_appendix": selector.get("appendix"),
                "date_inference": ev.get("date_inference"),
                "event_id": ev.get("event_id"),
                "source_unit_id": ev.get("source_unit_id"),
                "source_path_text": ev.get("source_path_text"),
                "confidence": ev.get("confidence"),
                "raw_text": ev.get("raw_text"),
                "notes": ev.get("notes"),
            }
            key = tuple(row.get(k) for k in fieldnames)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (
                r.get("document_id") or "",
                r.get("target_article") or "",
                r.get("target_clause") or "",
                r.get("target_point") or "",
            )):
                out = {k: (row.get(k) if row.get(k) not in {None, ""} else "null") for k in fieldnames}
                writer.writerow(out)
        return output_path

    @staticmethod
    def _document_key(document_number: Optional[str], document_id: Optional[str]) -> str:
        normalized = normalize_document_number(document_number or "") if document_number else None
        return normalized or (document_id or "unknown")

    @staticmethod
    def _external_row(document_number: Optional[str], document_id: Optional[str]) -> Dict[str, Any]:
        normalized = normalize_document_number(document_number or "") if document_number else None
        doc_id = document_id or (normalize_id(normalized) if normalized else "unknown")
        return {
            "document_id": doc_id,
            "document_number": normalized or (document_number or ""),
            "effective_from": None,
            "effective_to": None,
            "effective_from_event_id": None,
            "effective_to_event_id": None,
            "effective_to_source_document_number": None,
            "in_corpus": "false",
            "_from_confidence": -1.0,
            "_to_confidence": -1.0,
        }

    @staticmethod
    def _prefer_date(old_date: Optional[str], old_confidence: float, new_date: str, new_confidence: float) -> bool:
        if not old_date:
            return True
        if new_confidence != old_confidence:
            return new_confidence > old_confidence
        return new_date < old_date

    def extract_from_units(self, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = self._select_candidate_units(units)
        raw_events: List[EffectivityEvent] = []
        for unit in candidates:
            raw_events.extend(self._extract_unit_effective_from(unit))
            raw_events.extend(self._extract_effective_from(unit))
            raw_events.extend(self._extract_repeals(unit))
        if self.config.infer_repeal_date_from_document_effective_date:
            raw_events = self._infer_repeal_dates(raw_events)
        # Prefer the more specific effective_from event when the same unit/date
        # is matched twice by broad and specific patterns.
        raw_events = self._dedupe_effective_from(raw_events)
        raw_events = self._dedupe_repeal_documents(raw_events)

        rows, seen = [], set()
        for ev in raw_events:
            if ev.confidence < self.config.min_confidence:
                continue
            row = ev.to_dict()
            key = (
                row["event_type"],
                row["source_unit_id"],
                row["target_scope"],
                row["target_document_number"],
                str(row["target_unit_selector"]),
                row["date"],
                row["raw_text"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        rows.sort(key=lambda r: (r.get("source_unit_id") or "", r.get("event_type") or "", r.get("event_id") or ""))
        return rows

    @staticmethod
    def _dedupe_effective_from(events: List[EffectivityEvent]) -> List[EffectivityEvent]:
        grouped = {}
        others = []
        for ev in events:
            if ev.event_type != "effective_from":
                others.append(ev)
                continue
            selector = json.dumps(ev.target_unit_selector, ensure_ascii=False, sort_keys=True) if ev.target_unit_selector else ""
            key = (
                ev.source_unit_id,
                ev.date,
                ev.target_scope,
                ev.target_document_number,
                ev.target_selector_raw,
                selector,
            )
            grouped.setdefault(key, []).append(ev)

        kept = []
        for group in grouped.values():
            # Prefer explicit "this_document", then higher confidence, then longer raw_text.
            group.sort(
                key=lambda ev: (
                    1 if ev.target_scope == "this_document" else 0,
                    ev.confidence,
                    len(ev.raw_text or ""),
                ),
                reverse=True,
            )
            kept.append(group[0])

        specific_by_unit_date = {
            (ev.source_unit_id, ev.date)
            for ev in kept
            if ev.target_scope in {"this_document", "this_document_unit"} and ev.date
        }
        deduped = [
            ev
            for ev in kept
            if not (
                ev.target_scope == "unknown"
                and ev.date
                and (ev.source_unit_id, ev.date) in specific_by_unit_date
            )
        ]
        return deduped + others

    @staticmethod
    def _dedupe_repeal_documents(events: List[EffectivityEvent]) -> List[EffectivityEvent]:
        grouped: Dict[tuple, List[EffectivityEvent]] = {}
        others = []
        for ev in events:
            if ev.event_type != "repeal_document" or not ev.target_document_number:
                others.append(ev)
                continue
            key = (ev.source_document_id, ev.target_document_number, ev.date)
            grouped.setdefault(key, []).append(ev)

        kept = []
        for group in grouped.values():
            group.sort(
                key=lambda ev: (
                    1 if re.search(r"\btrừ\b", ev.raw_text or "", re.I | re.U) else 0,
                    ev.confidence,
                    len(ev.raw_text or ""),
                ),
                reverse=True,
            )
            kept.append(group[0])
        return kept + others

    def _select_candidate_units(self, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.config.prefer_final_provisions:
            return units
        selected = []
        for unit in units:
            combined = f"{unit.get('path_text') or ''} {unit.get('content') or ''}".lower()
            if any(k in combined for k in ["hiệu lực", "điều khoản thi hành", "tổ chức thực hiện", "bãi bỏ", "thay thế", "hết hiệu lực"]):
                selected.append(unit)
        return selected or units

    def _event_id(self, unit: Dict[str, Any], event_type: str, raw_text: str, target: str = "") -> str:
        h = md5_text(f"{unit.get('id') or ''}|{event_type}|{target}|{raw_text}")[:16]
        return f"{unit.get('document_id', 'unknown')}.effectivity.{h}"

    @staticmethod
    def _source_path_for_match(unit: Dict[str, Any], text: str = "", position: Optional[int] = None) -> Optional[str]:
        path_text = unit.get("path_text")
        if position is None or not text:
            return path_text
        prefix = text[:position]
        headings = list(re.finditer(r"(Điều\s+\d+[a-zA-Z]?\.\s+[^.\n]{0,140})", prefix, re.I | re.U))
        if not headings:
            return path_text
        embedded = collapse_ws(headings[-1].group(1))
        if embedded and embedded.lower() not in (path_text or "").lower():
            return f"{path_text} > [embedded] {embedded}"
        return path_text

    @staticmethod
    def _clause_bounds(text: str, position: int) -> tuple[int, int]:
        starts = [text.rfind(sep, 0, position) for sep in [".", ";", "\n"]]
        start = max(starts) + 1
        ends = [idx for idx in [text.find(sep, position) for sep in [".", ";", "\n"]] if idx != -1]
        end = min(ends) if ends else len(text)
        return start, end

    @staticmethod
    def _is_reference_context(text: str, start: int) -> bool:
        prefix = text[max(0, start - 360):start].lower()
        local_start = max(prefix.rfind(","), prefix.rfind(";"), prefix.rfind("."))
        local_prefix = prefix[local_start + 1:]
        paren_start = prefix.rfind("(")
        paren_end = prefix.rfind(")")
        if paren_start > paren_end:
            paren_prefix = prefix[paren_start + 1:]
            if re.search(r"đã\s+được\s+sửa\s+đổi|được\s+sửa\s+đổi|sửa\s+đổi,\s*bổ\s+sung", paren_prefix, re.I | re.U):
                return True
        semicolon_start = prefix.rfind(";")
        segment_prefix = prefix[semicolon_start + 1:]
        amendment_markers = list(re.finditer(
            r"đã\s+được\s+sửa\s+đổi|được\s+sửa\s+đổi|sửa\s+đổi,\s*bổ\s+sung|bổ\s+sung\s+một\s+số\s+điều\s+của|sửa\s+đổi\s+một\s+số\s+điều\s+của",
            segment_prefix,
            re.I | re.U,
        ))
        if amendment_markers:
            last_amendment = amendment_markers[-1].start()
            governing_markers = [
                m.start()
                for m in re.finditer(
                    r"bãi\s+bỏ|hết\s+hiệu\s+lực|chấm\s+dứt\s+hiệu\s+lực|ngưng\s+hiệu\s+lực|thay\s+thế",
                    segment_prefix,
                    re.I | re.U,
                )
            ]
            if not governing_markers or last_amendment > max(governing_markers):
                return True
        reference_markers = [
            r"hướng\s+dẫn(?:\s+thực\s+hiện|\s+một\s+số\s+điều\s+của)?",
            r"quy\s+định\s+chi\s+tiết",
            r"quy\s+định\s+về",
            r"theo\s+quy\s+định\s+tại",
            r"được\s+cấp\s+theo",
            r"đã\s+được\s+sửa\s+đổi(?:,\s*bổ\s+sung)?\s+(?:tại|theo|bởi)?",
            r"được\s+sửa\s+đổi",
            r"sửa\s+đổi,\s*bổ\s+sung",
            r"bổ\s+sung\s+(?:một\s+số\s+)?(?:điều|khoản|điểm)\s+(?:của|tại)",
            r"sửa\s+đổi\s+(?:một\s+số\s+)?(?:điều|khoản|điểm)\s+(?:của|tại)",
        ]
        return any(re.search(marker + r".{0,120}$", local_prefix, re.I | re.U) for marker in reference_markers)

    @staticmethod
    def _article_list_selector_before(text: str) -> Optional[str]:
        m = re.search(r"((?:các\s+)?Điều\s*:\s*[\d,\svà]+)$", text, re.I | re.U)
        return collapse_ws(m.group(1)) if m else None

    @staticmethod
    def _has_unit_repeal_selector_before(text: str, position: int) -> bool:
        start, _ = EffectivityExtractor._clause_bounds(text, position)
        prefix = text[start:position]
        if not REPEAL_KEYWORD_PATTERN.search(prefix):
            return False
        return bool(UNIT_SELECTOR_PATTERN.search(prefix) or re.search(r"Phụ\s+lục\s+(?:[IVXLCDM]+|\d+|[A-ZĐ]+)", prefix, re.I | re.U))

    @staticmethod
    def _is_listed_repeal_context(unit: Dict[str, Any]) -> bool:
        path_text = (unit.get("path_text") or "").lower()
        legal_list_noun = (
            r"văn\s+bản|thông\s+tư|nghị\s+định|quyết\s+định|nghị\s+quyết|"
            r"luật|bộ\s+luật|điều|khoản|điểm|quy\s+định"
        )
        patterns = [
            r"\bbãi\s+bỏ\s+(?:các\s+)?(?:" + legal_list_noun + r").{0,120}(?:sau|sau\s+đây|cụ\s+thể)",
            r"(?:các\s+)?(?:" + legal_list_noun + r").{0,80}(?:sau|sau\s+đây).{0,80}(?:hết\s+hiệu\s+lực|bãi\s+bỏ)",
            r"(?:các\s+)?(?:" + legal_list_noun + r").{0,80}(?:hết\s+hiệu\s+lực).{0,80}(?:sau|sau\s+đây)",
        ]
        return any(re.search(pattern, path_text, re.I | re.U) for pattern in patterns)

    @staticmethod
    def _is_listed_replacement_context(unit: Dict[str, Any]) -> bool:
        path_text = (unit.get("path_text") or "").lower()
        legal_list_noun = (
            r"văn\s+bản|thông\s+tư|nghị\s+định|quyết\s+định|nghị\s+quyết|"
            r"luật|bộ\s+luật|điều|khoản|điểm|quy\s+định"
        )
        patterns = [
            r"\bthay\s+thế\s+(?:các\s+)?(?:" + legal_list_noun + r").{0,120}(?:sau|sau\s+đây|tại)",
            r"\bcó\s+hiệu\s+lực.{0,120}\bthay\s+thế\s+(?:các\s+)?(?:" + legal_list_noun + r")",
        ]
        return any(re.search(pattern, path_text, re.I | re.U) for pattern in patterns)

    @staticmethod
    def _is_word_level_replacement_context(text: str, position: int) -> bool:
        prefix = text[max(0, position - 220):position].lower()
        keyword_pos = prefix.rfind("thay thế")
        if keyword_pos == -1:
            return False
        local = prefix[keyword_pos:]
        if re.search(r"thay\s+thế\s+(?:các\s+)?(?:văn\s+bản|thông\s+tư|nghị\s+định|luật|bộ\s+luật|quy\s+định|điều|khoản|điểm)\b", local, re.I | re.U):
            return False
        return bool(re.search(
            r"thay\s+thế.{0,120}(?:một\s+số\s+)?(?:từ|cụm\s+từ|dấu\s+chấm|dấu\s+phẩy|dấu\s+chấm\s+phẩy|phụ\s+lục|mẫu|biểu\s+mẫu)\b",
            local,
            re.I | re.U,
        ))

    @staticmethod
    def _inherited_target_document_number(unit: Dict[str, Any]) -> Optional[str]:
        path_text = unit.get("path_text") or ""
        refs = list(LEGAL_DOCUMENT_REF_PATTERN.finditer(path_text))
        if not refs:
            return None
        cutoff = len(path_text)
        amendment = re.search(
            r"đã\s+được\s+sửa\s+đổi|được\s+sửa\s+đổi|sửa\s+đổi,\s*bổ\s+sung",
            path_text,
            re.I | re.U,
        )
        if amendment:
            cutoff = amendment.start()
        for ref in refs:
            if ref.start() <= cutoff:
                doc_number = normalize_document_number(ref.group("doc_number"))
                if doc_number and doc_number != normalize_document_number(unit.get("document_number") or ""):
                    return doc_number
        doc_number = normalize_document_number(refs[0].group("doc_number"))
        return doc_number if doc_number != normalize_document_number(unit.get("document_number") or "") else None

    @staticmethod
    def _selector_items_for_unit_match(text: str, match: re.Match) -> List[Dict[str, Any]]:
        selector = {"article": match.group("article"), "clause": match.group("clause"), "point": match.group("point")}
        items = [{"raw": collapse_ws(match.group(0)), "selector": selector}]

        clause_start, clause_end = EffectivityExtractor._clause_bounds(text, match.start())
        prefix = text[clause_start:match.start()]
        suffix = text[match.end():clause_end]

        if selector.get("point") and selector.get("clause") and selector.get("article"):
            points = re.findall(r"điểm\s+([a-zđ])", prefix, re.I | re.U)
            standalone_point_tail = r"(?:\s*,\s*(?!điểm\b)[a-zđ](?![a-zà-ỹđ])|\s+và\s+(?!điểm\b)[a-zđ](?![a-zà-ỹđ]))*"
            for first, tail in re.findall(r"điểm\s+([a-zđ])(" + standalone_point_tail + r")", prefix, re.I | re.U):
                points.append(first)
                points.extend(re.findall(r"(?:,|\bvà)\s*(?!điểm\b)([a-zđ])(?![a-zà-ỹđ])", tail, re.I | re.U))
            for point in points:
                sibling = dict(selector)
                sibling["point"] = point
                items.append({
                    "raw": f"điểm {point} khoản {selector['clause']} Điều {selector['article']}",
                    "selector": sibling,
                })
        elif selector.get("clause") and selector.get("article"):
            for clause in re.findall(r"khoản\s+(\d+[a-zA-Z]?)", prefix, re.I | re.U):
                sibling = dict(selector)
                sibling["clause"] = clause
                items.append({
                    "raw": f"khoản {clause} Điều {selector['article']}",
                    "selector": sibling,
                })
        elif selector.get("article"):
            tail = re.match(r"\s*((?:,\s*\d+[a-zA-Z]?|\s+và\s+\d+[a-zA-Z]?)+)", suffix, re.I | re.U)
            if tail:
                for article in re.findall(r"(?:,|\bvà)\s*(\d+[a-zA-Z]?)", tail.group(1), re.I | re.U):
                    sibling = dict(selector)
                    sibling["article"] = article
                    items.append({"raw": f"Điều {article}", "selector": sibling})

        deduped = []
        seen = set()
        for item in items:
            key = json.dumps(item["selector"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _has_governing_repeal_or_replacement(text: str, position: int) -> bool:
        start = max(text.rfind(".", 0, position), text.rfind("\n", 0, position)) + 1
        prefix = text[start:position]
        return bool(REPEAL_KEYWORD_PATTERN.search(prefix) or REPLACEMENT_KEYWORD_PATTERN.search(prefix))

    @staticmethod
    def _target_segment_before_doc(text: str, doc_match: re.Match, previous_doc_end: Optional[int]) -> str:
        start = max(text.rfind(".", 0, doc_match.start()), text.rfind("\n", 0, doc_match.start())) + 1
        if previous_doc_end is not None and previous_doc_end > start:
            start = previous_doc_end
            semicolon = text.rfind(";", start, doc_match.start())
            if semicolon != -1:
                start = semicolon + 1
        return text[start:doc_match.start()]

    @staticmethod
    def _unit_selectors_before_doc(text: str) -> List[Dict[str, Any]]:
        selectors: List[Dict[str, Any]] = []
        for match in UNIT_SELECTOR_PATTERN.finditer(text):
            selector = {
                "article": match.group("article"),
                "clause": match.group("clause"),
                "point": match.group("point"),
            }
            raw_selector = collapse_ws(match.group(0))
            if selector.get("clause") and selector.get("article"):
                prefix = text[max(0, match.start() - 90):match.start()]
                for clause in re.findall(r"khoản\s+(\d+[a-zA-Z]?)\s*(?=,|\s+và)", prefix, re.I | re.U):
                    sibling = dict(selector)
                    sibling["clause"] = clause
                    selectors.append({
                        "raw": f"khoản {clause} Điều {selector['article']}",
                        "selector": sibling,
                    })
            if selector.get("point") and selector.get("clause") and selector.get("article"):
                prefix = text[max(0, match.start() - 90):match.start()]
                for point in re.findall(r"điểm\s+([a-zđ])\s*(?=,|\s+và)", prefix, re.I | re.U):
                    sibling = dict(selector)
                    sibling["point"] = point
                    selectors.append({
                        "raw": f"điểm {point} khoản {selector['clause']} Điều {selector['article']}",
                        "selector": sibling,
                    })
            selectors.append({"raw": raw_selector, "selector": selector})

        deduped = []
        seen = set()
        for item in selectors:
            key = json.dumps(item["selector"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _should_skip_repeal_unspecified(unit: Dict[str, Any], text: str) -> bool:
        lower = f"{text} {unit.get('path_text') or ''}".lower()
        text_lower = text.lower()
        legal_list_noun = (
            r"văn\s+bản|thông\s+tư|nghị\s+định|quyết\s+định|nghị\s+quyết|"
            r"luật|bộ\s+luật|điều|khoản|điểm|quy\s+định"
        )
        if re.search(
            r"^\s*(?:kể\s+từ\s+ngày.{0,80},\s*)?"
            r"(?:(?:nghị\s+định|thông\s+tư|luật|bộ\s+luật|văn\s+bản)\s+này\s+)?"
            r"(?:bãi\s+bỏ\s+(?:các\s+)?(?:" + legal_list_noun + r")|"
            r"(?:các\s+)?(?:" + legal_list_noun + r").{0,80}(?:sau|sau\s+đây).{0,80}hết\s+hiệu\s+lực)",
            text_lower,
            re.I | re.U,
        ) and not LEGAL_DOCUMENT_REF_PATTERN.search(text):
            return True
        if re.search(
            r"(?:sửa\s+đổi,\s*bổ\s+sung(?:\s+và)?\s+bãi\s+bỏ|bãi\s+bỏ)\s+một\s+số\s+(?:điều|khoản|điểm)",
            text_lower,
            re.I | re.U,
        ):
            return True
        if re.search(r"\bbãi\s+bỏ\b", text_lower, re.I | re.U):
            return False
        non_legal_markers = [
            "giấy chứng nhận",
            "chứng chỉ",
            "tem kiểm định",
            "giấy phép",
            "phù hiệu",
            "thông báo kết quả",
            "văn bản thông báo",
            "hồ sơ",
            "xe ",
        ]
        if any(marker in lower for marker in non_legal_markers):
            return True
        if EffectivityExtractor._is_final_or_transition_context(unit):
            return False
        return False

    @staticmethod
    def _selector_from_match(unit: Dict[str, Any], match: re.Match) -> Dict[str, Any]:
        groups = match.groupdict()
        selector = {}
        for key in ("article", "clause", "point", "appendix"):
            value = groups.get(key)
            if value:
                selector[key] = value
        if selector.get("article", "").lower() == "này":
            current_article = EffectivityExtractor._current_article(unit)
            if current_article:
                selector["article"] = current_article
                selector["article_reference"] = "this_article"
        return selector

    def _selectors_for_effective_match(
        self,
        unit: Dict[str, Any],
        text: str,
        match: re.Match,
    ) -> List[Dict[str, Any]]:
        groups = match.groupdict()
        fallback_raw = collapse_ws(groups.get("selector") or "")
        fallback_selector = self._selector_from_match(unit, match)

        if groups.get("appendix"):
            return [{"raw": fallback_raw, "selector": fallback_selector}]

        try:
            date_start = match.start("date")
        except IndexError:
            date_start = match.end()
        delimiter_positions = [
            text.rfind(";", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        ]
        context_start = max(delimiter_positions) + 1
        context = text[context_start:date_start]

        selectors: List[Dict[str, Any]] = []
        for selector_match in UNIT_SELECTOR_PATTERN.finditer(context):
            selector = self._selector_from_match(unit, selector_match)
            raw_selector = collapse_ws(selector_match.group(0))
            if selector.get("point") and selector.get("clause") and selector.get("article"):
                prefix = context[max(0, selector_match.start() - 100):selector_match.start()]
                sibling_points = re.findall(r"điểm\s+([a-zđ])\s*(?=,|\s+và)", prefix, re.I | re.U)
                for point in sibling_points:
                    sibling_selector = dict(selector)
                    sibling_selector["point"] = point
                    selectors.append({
                        "raw": f"điểm {point} khoản {selector['clause']} Điều {selector['article']}",
                        "selector": sibling_selector,
                    })
            selectors.append({"raw": raw_selector, "selector": selector})

        for appendix_match in re.finditer(r"Phụ\s+lục\s+(?P<appendix>[IVXLCDM]+|\d+|[A-ZĐ]+)\b", context, re.I | re.U):
            selectors.append({
                "raw": collapse_ws(appendix_match.group(0)),
                "selector": {"appendix": appendix_match.group("appendix")},
            })

        if not selectors and fallback_selector:
            selectors.append({"raw": fallback_raw, "selector": fallback_selector})

        deduped = []
        seen = set()
        for item in selectors:
            key = json.dumps(item["selector"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _effective_match_context(text: str, match: re.Match) -> str:
        try:
            date_end = match.end("date")
        except IndexError:
            date_end = match.end()
        delimiter_positions = [
            text.rfind(";", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        ]
        context_start = max(delimiter_positions) + 1
        return collapse_ws(text[context_start:date_end])

    @staticmethod
    def _current_article(unit: Dict[str, Any]) -> Optional[str]:
        unit_id = unit.get("id") or ""
        m = re.search(r"(?:^|[._-])dieu_(?P<article>\d+[a-zA-Z]?)", unit_id, re.I)
        if m:
            return m.group("article")
        path_text = unit.get("path_text") or ""
        m = re.search(r"Điều\s+(?P<article>\d+[a-zA-Z]?)\b", path_text, re.I)
        return m.group("article") if m else None

    @staticmethod
    def _has_unit_selector_before(text: str, position: int) -> bool:
        window = text[max(0, position - 140):position]
        if UNIT_SELECTOR_PATTERN.search(window):
            return True
        return bool(re.search(r"Phụ\s+lục\s+(?:[IVXLCDM]+|\d+|[A-ZĐ]+)\b", window, re.I | re.U))

    @staticmethod
    def _is_final_or_transition_context(unit: Dict[str, Any]) -> bool:
        path_text = (unit.get("path_text") or "").lower()
        markers = [
            "hiệu lực",
            "điều khoản thi hành",
            "tổ chức thực hiện",
            "quy định chuyển tiếp",
            "điều khoản chuyển tiếp",
        ]
        return any(marker in path_text for marker in markers)

    def _extract_unit_effective_from(self, unit: Dict[str, Any]) -> List[EffectivityEvent]:
        if not self._is_final_or_transition_context(unit):
            return []
        text = collapse_ws(unit.get("content") or "")
        if not text:
            return []
        events = []
        for pat in UNIT_EFFECTIVE_FROM_PATTERNS:
            for m in pat.finditer(text):
                raw = collapse_ws(m.group(0))
                date_raw = m.groupdict().get("date")
                date_iso = parse_vietnamese_date_to_iso(date_raw or raw)
                if not date_iso:
                    continue
                selectors = self._selectors_for_effective_match(unit, text, m)
                event_raw = self._effective_match_context(text, m)
                raw_lower = raw.lower()
                conf = 0.88
                if "có hiệu lực thi hành" in raw_lower:
                    conf += 0.04
                if "kể từ" in raw_lower or "kể ngày" in raw_lower or "từ ngày" in raw_lower:
                    conf += 0.03
                for selector_item in selectors:
                    selector_raw = selector_item["raw"]
                    selector = selector_item["selector"]
                    selector_conf = conf + (0.02 if selector.get("article") or selector.get("appendix") else 0)
                    events.append(EffectivityEvent(
                        event_id=self._event_id(unit, "effective_from", event_raw, selector_raw),
                        event_type="effective_from",
                        source_document_id=unit.get("document_id"),
                        source_document_number=unit.get("document_number"),
                        source_unit_id=unit.get("id"),
                        source_path_text=self._source_path_for_match(unit, text, m.start()),
                        target_scope="this_document_unit",
                        target_selector_raw=selector_raw,
                        target_document_number=unit.get("document_number"),
                        target_unit_selector=selector,
                        date=date_iso,
                        date_role="effective_from",
                        raw_text=event_raw,
                        status="candidate",
                        resolver="rule_unit_effective_from",
                        confidence=min(selector_conf, 0.97),
                    ))
        for pat in UNIT_EFFECTIVE_INDIRECT_PATTERNS:
            for m in pat.finditer(text):
                raw = self._effective_match_context(text, m)
                selectors = self._selectors_for_effective_match(unit, text, m)
                for selector_item in selectors:
                    selector_raw = selector_item["raw"]
                    selector = selector_item["selector"]
                    events.append(EffectivityEvent(
                        event_id=self._event_id(unit, "effective_from", raw, selector_raw),
                        event_type="effective_from",
                        source_document_id=unit.get("document_id"),
                        source_document_number=unit.get("document_number"),
                        source_unit_id=unit.get("id"),
                        source_path_text=self._source_path_for_match(unit, text, m.start()),
                        target_scope="this_document_unit",
                        target_selector_raw=selector_raw,
                        target_document_number=unit.get("document_number"),
                        target_unit_selector=selector,
                        date=None,
                        date_role="effective_from",
                        date_inference="indirect_effective_date",
                        raw_text=raw,
                        status="candidate_unresolved_date",
                        resolver="rule_unit_effective_from_indirect",
                        confidence=0.74,
                        notes="The provision has an indirect effective date and needs a resolver to map the referenced rule to a concrete date.",
                    ))
        return events

    def _extract_effective_from(self, unit: Dict[str, Any]) -> List[EffectivityEvent]:
        text = collapse_ws(unit.get("content") or "")
        if not text:
            return []
        events = []

        def append_event(
            match: re.Match,
            raw: str,
            date_iso: str,
            resolver: str,
            confidence: float,
            date_inference: Optional[str] = None,
        ) -> None:
            target_raw = match.groupdict().get("target") if "target" in match.groupdict() else None
            if target_raw:
                prefix = text[max(0, match.start("target") - 8):match.start("target")].lower()
                if re.search(r"của\s+$", prefix, re.I | re.U):
                    return
            elif self._has_unit_selector_before(text, match.start()):
                return
            target_scope = "this_document" if target_raw and "này" in target_raw.lower() else "unknown"
            events.append(EffectivityEvent(
                event_id=self._event_id(unit, "effective_from", raw, target_raw or ""),
                event_type="effective_from",
                source_document_id=unit.get("document_id"),
                source_document_number=unit.get("document_number"),
                source_unit_id=unit.get("id"),
                source_path_text=self._source_path_for_match(unit, text, match.start()),
                target_scope=target_scope,
                target_selector_raw=target_raw or "current_document_or_unspecified",
                target_document_number=unit.get("document_number") if target_scope == "this_document" else None,
                target_unit_selector=None,
                date=date_iso,
                date_role="effective_from",
                date_inference=date_inference,
                raw_text=raw,
                status="candidate",
                resolver=resolver,
                confidence=min(confidence, 0.99),
            ))

        for pat in EFFECTIVE_FROM_PATTERNS:
            for m in pat.finditer(text):
                raw = collapse_ws(m.group(0))
                date_raw = m.groupdict().get("date")
                date_iso = parse_vietnamese_date_to_iso(date_raw or raw)
                if not date_iso:
                    continue
                raw_lower = raw.lower()
                conf = 0.90 + (0.05 if "hiệu lực thi hành" in raw_lower else 0) + (0.03 if ("kể từ" in raw_lower or "kể ngày" in raw_lower or "từ ngày" in raw_lower) else 0)
                append_event(m, raw, date_iso, "rule", conf)

        issue_date_iso = self._issue_date_iso(unit)
        if issue_date_iso:
            for pat in RELATIVE_SIGNING_EFFECTIVE_FROM_PATTERNS:
                for m in pat.finditer(text):
                    raw = collapse_ws(m.group(0))
                    days = int(m.group("days"))
                    date_iso = self._add_days(issue_date_iso, days)
                    if not date_iso:
                        continue
                    append_event(
                        m,
                        raw,
                        date_iso,
                        "rule_relative_to_signing_date",
                        0.88,
                        f"issue_date_plus_{days}_days",
                    )
            for pat in SIGNING_EFFECTIVE_FROM_PATTERNS:
                for m in pat.finditer(text):
                    raw = collapse_ws(m.group(0))
                    append_event(
                        m,
                        raw,
                        issue_date_iso,
                        "rule_signing_date",
                        0.87,
                        "same_as_issue_date",
                    )
        return events

    @staticmethod
    def _issue_date_iso(unit: Dict[str, Any]) -> Optional[str]:
        return parse_vietnamese_date_to_iso(unit.get("issue_date") or "")

    @staticmethod
    def _add_days(date_iso: str, days: int) -> Optional[str]:
        try:
            from datetime import date, timedelta
            y, m, d = [int(x) for x in date_iso.split("-")]
            return (date(y, m, d) + timedelta(days=days)).isoformat()
        except Exception:
            return None

    def _extract_repeals(self, unit: Dict[str, Any]) -> List[EffectivityEvent]:
        text = collapse_ws(unit.get("content") or "")
        listed_repeal_context = self._is_listed_repeal_context(unit)
        listed_replacement_context = self._is_listed_replacement_context(unit)
        has_replacement_keyword = bool(REPLACEMENT_KEYWORD_PATTERN.search(text))
        if not text or (
            not REPEAL_KEYWORD_PATTERN.search(text)
            and not has_replacement_keyword
            and not listed_repeal_context
            and not listed_replacement_context
        ):
            return []
        events = []
        emitted_document_targets = set()
        emitted_unit_targets = set()

        for kw in REPEAL_KEYWORD_PATTERN.finditer(text):
            keyword = kw.group(0).lower()
            if "bãi bỏ" in keyword:
                continue
            clause_start, clause_end = self._clause_bounds(text, kw.start())
            clause = text[clause_start:clause_end]
            before_keyword = clause[:kw.start() - clause_start]
            refs = list(LEGAL_DOCUMENT_REF_PATTERN.finditer(before_keyword))
            if not refs:
                continue
            cutoff = len(before_keyword)
            descriptive = re.search(
                r"đã\s+được\s+sửa\s+đổi|được\s+sửa\s+đổi|quy\s+định\s+chi\s+tiết|hướng\s+dẫn\s+thực\s+hiện|quy\s+định\s+về",
                before_keyword,
                re.I | re.U,
            )
            if descriptive:
                cutoff = descriptive.start()
            subject_refs = []
            for ref in refs:
                if ref.start() >= cutoff:
                    continue
                if self._is_reference_context(text, clause_start + ref.start()):
                    continue
                subject_refs.append(ref)
            for ref in subject_refs:
                doc_number = normalize_document_number(ref.group("doc_number"))
                if not doc_number or doc_number in emitted_document_targets:
                    continue
                emitted_document_targets.add(doc_number)
                raw = collapse_ws(ref.group(0))
                date_iso = self._find_date_near(text, clause_start + ref.start(), kw.end())
                events.append(EffectivityEvent(
                    event_id=self._event_id(unit, "repeal_document", clause, doc_number),
                    event_type="repeal_document",
                    source_document_id=unit.get("document_id"),
                    source_document_number=unit.get("document_number"),
                    source_unit_id=unit.get("id"),
                    source_path_text=self._source_path_for_match(unit, text, clause_start + ref.start()),
                    target_scope="external_document",
                    target_selector_raw=raw,
                    target_document_number=doc_number,
                    target_unit_selector=None,
                    date=date_iso,
                    date_role="ceased_from" if date_iso else None,
                    date_inference=None if date_iso else "missing_explicit_date",
                    raw_text=collapse_ws(clause),
                    status="candidate",
                    resolver="rule_subject_before_repeal_keyword",
                    confidence=0.86 if date_iso else 0.80,
                ))

        document_refs = list(LEGAL_DOCUMENT_REF_PATTERN.finditer(text))
        previous_doc_end: Optional[int] = None
        for m in document_refs:
            clause_start, clause_end = self._clause_bounds(text, m.start())
            clause = text[clause_start:clause_end]
            before_ref = clause[:m.start() - clause_start]
            segment_before_ref = self._target_segment_before_doc(text, m, previous_doc_end)
            previous_doc_end = m.end()
            has_direct_repeal = bool(REPEAL_KEYWORD_PATTERN.search(before_ref))
            has_direct_replacement = bool(REPLACEMENT_KEYWORD_PATTERN.search(before_ref))
            has_governing_context = (
                has_direct_repeal
                or has_direct_replacement
                or listed_repeal_context
                or listed_replacement_context
                or self._has_governing_repeal_or_replacement(text, m.start())
            )
            if not has_governing_context:
                continue
            if has_direct_replacement or listed_replacement_context or REPLACEMENT_KEYWORD_PATTERN.search(text[:m.start()]):
                if self._is_word_level_replacement_context(text, m.start()):
                    continue
            if re.search(r"bãi\s+bỏ\s+một\s+số\s+điều", clause, re.I | re.U):
                continue
            if self._is_reference_context(text, m.start()):
                continue
            doc_number = normalize_document_number(m.group("doc_number"))
            if not doc_number:
                continue

            unit_selectors = self._unit_selectors_before_doc(segment_before_ref)
            article_list_selector = self._article_list_selector_before(segment_before_ref)
            is_partial_repeal = bool(unit_selectors or article_list_selector) and has_governing_context
            date_iso = self._find_date_near(text, m.start(), m.end())
            if is_partial_repeal:
                if not unit_selectors and article_list_selector:
                    unit_key = (doc_number, article_list_selector, date_iso, collapse_ws(clause))
                    if unit_key not in emitted_unit_targets:
                        emitted_unit_targets.add(unit_key)
                        events.append(EffectivityEvent(
                            event_id=self._event_id(unit, "repeal_unit", clause, f"{doc_number}|{article_list_selector}"),
                            event_type="repeal_unit",
                            source_document_id=unit.get("document_id"),
                            source_document_number=unit.get("document_number"),
                            source_unit_id=unit.get("id"),
                            source_path_text=self._source_path_for_match(unit, text, m.start()),
                            target_scope="external_document_unit",
                            target_selector_raw=article_list_selector,
                            target_document_number=doc_number,
                            target_unit_selector={"raw": article_list_selector},
                            date=date_iso,
                            date_role="ceased_from" if date_iso else None,
                            date_inference=None if date_iso else "missing_explicit_date",
                            raw_text=collapse_ws(clause),
                            status="candidate",
                            resolver="rule_external_unit_repeal",
                            confidence=0.76 if date_iso else 0.70,
                        ))
                    continue
                for selector_item in unit_selectors:
                    selector = selector_item["selector"]
                    raw_selector = selector_item["raw"]
                    unit_key = (doc_number, raw_selector, date_iso, collapse_ws(clause))
                    if unit_key in emitted_unit_targets:
                        continue
                    emitted_unit_targets.add(unit_key)
                    events.append(EffectivityEvent(
                        event_id=self._event_id(unit, "repeal_unit", clause, f"{doc_number}|{raw_selector}"),
                        event_type="repeal_unit",
                        source_document_id=unit.get("document_id"),
                        source_document_number=unit.get("document_number"),
                        source_unit_id=unit.get("id"),
                        source_path_text=self._source_path_for_match(unit, text, m.start()),
                        target_scope="external_document_unit",
                        target_selector_raw=raw_selector,
                        target_document_number=doc_number,
                        target_unit_selector=selector,
                        date=date_iso,
                        date_role="ceased_from" if date_iso else None,
                        date_inference=None if date_iso else "missing_explicit_date",
                        raw_text=collapse_ws(clause),
                        status="candidate",
                        resolver="rule_external_unit_repeal",
                        confidence=0.78 if date_iso else 0.72,
                    ))
                continue

            if doc_number in emitted_document_targets:
                continue
            emitted_document_targets.add(doc_number)
            raw = collapse_ws(m.group(0))
            events.append(EffectivityEvent(
                event_id=self._event_id(unit, "repeal_document", clause, doc_number),
                event_type="repeal_document",
                source_document_id=unit.get("document_id"),
                source_document_number=unit.get("document_number"),
                source_unit_id=unit.get("id"),
                source_path_text=self._source_path_for_match(unit, text, m.start()),
                target_scope="external_document",
                target_selector_raw=raw,
                target_document_number=doc_number,
                target_unit_selector=None,
                date=date_iso,
                date_role="ceased_from" if date_iso else None,
                date_inference=None if date_iso else "missing_explicit_date",
                raw_text=collapse_ws(clause),
                status="candidate",
                resolver="rule_direct_or_listed_document_repeal",
                confidence=0.86 if date_iso else 0.78,
            ))

        for m in UNIT_SELECTOR_PATTERN.finditer(text):
            clause_start, clause_end = self._clause_bounds(text, m.start())
            clause = text[clause_start:clause_end]
            before_selector = clause[:m.start() - clause_start]
            if not re.search(r"\bbãi\s+bỏ\b", before_selector, re.I | re.U):
                continue
            if LEGAL_DOCUMENT_REF_PATTERN.search(clause[m.end() - clause_start:]):
                continue
            date_iso = self._find_date_near(text, m.start(), m.end())
            inherited_doc_number = self._inherited_target_document_number(unit)
            for selector_item in self._selector_items_for_unit_match(text, m):
                raw_selector = selector_item["raw"]
                selector = selector_item["selector"]
                unit_key = (inherited_doc_number, raw_selector, date_iso, collapse_ws(clause))
                if unit_key in emitted_unit_targets:
                    continue
                emitted_unit_targets.add(unit_key)
                events.append(EffectivityEvent(
                    event_id=self._event_id(unit, "repeal_unit", clause, f"{inherited_doc_number or ''}|{raw_selector}"),
                    event_type="repeal_unit",
                    source_document_id=unit.get("document_id"),
                    source_document_number=unit.get("document_number"),
                    source_unit_id=unit.get("id"),
                    source_path_text=self._source_path_for_match(unit, text, m.start()),
                    target_scope="external_document_unit" if inherited_doc_number else "unknown_document_unit",
                    target_selector_raw=raw_selector,
                    target_document_number=inherited_doc_number,
                    target_unit_selector=selector,
                    date=date_iso,
                    date_role="ceased_from" if date_iso else None,
                    date_inference=None if date_iso else "missing_explicit_date",
                    raw_text=collapse_ws(clause),
                    status="candidate",
                    resolver="rule_inherited_external_unit_repeal" if inherited_doc_number else "rule+needs_target_resolution",
                    confidence=0.78 if (date_iso and inherited_doc_number) else (0.72 if date_iso else 0.64),
                ))

        for m in re.finditer(
            r"(?:nội\s+dung\s+tại\s+)?(?P<section>(?:Mục|mục)\s+(?:[IVXLCDM]+|\d+)(?:\s+Chương\s+[IVXLCDM]+)?|Phụ\s+lục\s+(?:[IVXLCDM]+|\d+|[A-ZĐ]+))",
            text,
            re.I | re.U,
        ):
            clause_start, clause_end = self._clause_bounds(text, m.start())
            clause = text[clause_start:clause_end]
            before_selector = clause[:m.start() - clause_start]
            if not re.search(r"\bbãi\s+bỏ\b", before_selector, re.I | re.U):
                continue
            raw_selector = collapse_ws(m.group("section"))
            date_iso = self._find_date_near(text, m.start(), m.end())
            inherited_doc_number = self._inherited_target_document_number(unit)
            unit_key = (inherited_doc_number, raw_selector, date_iso, collapse_ws(clause))
            if unit_key in emitted_unit_targets:
                continue
            emitted_unit_targets.add(unit_key)
            events.append(EffectivityEvent(
                event_id=self._event_id(unit, "repeal_unit", clause, f"{inherited_doc_number or ''}|{raw_selector}"),
                event_type="repeal_unit",
                source_document_id=unit.get("document_id"),
                source_document_number=unit.get("document_number"),
                source_unit_id=unit.get("id"),
                source_path_text=self._source_path_for_match(unit, text, m.start()),
                target_scope="external_document_unit" if inherited_doc_number else "unknown_document_unit",
                target_selector_raw=raw_selector,
                target_document_number=inherited_doc_number,
                target_unit_selector={"raw": raw_selector},
                date=date_iso,
                date_role="ceased_from" if date_iso else None,
                date_inference=None if date_iso else "missing_explicit_date",
                raw_text=collapse_ws(clause),
                status="candidate",
                resolver="rule_inherited_external_section_repeal" if inherited_doc_number else "rule_section_repeal_needs_target_resolution",
                confidence=0.76 if (date_iso and inherited_doc_number) else (0.70 if date_iso else 0.62),
            ))

        if not events:
            if not REPEAL_KEYWORD_PATTERN.search(text) and not listed_repeal_context:
                return []
            if self._should_skip_repeal_unspecified(unit, text):
                return []
            date_iso = self._find_date_near(text, 0, len(text))
            events.append(EffectivityEvent(
                event_id=self._event_id(unit, "repeal_unspecified", text, ""),
                event_type="repeal_unspecified",
                source_document_id=unit.get("document_id"),
                source_document_number=unit.get("document_number"),
                source_unit_id=unit.get("id"),
                source_path_text=self._source_path_for_match(unit, text, 0),
                target_scope="unknown",
                target_selector_raw=None,
                target_document_number=None,
                target_unit_selector=None,
                date=date_iso,
                date_role="ceased_from" if date_iso else None,
                date_inference=None if date_iso else "missing_explicit_date",
                raw_text=text,
                status="candidate_needs_llm_parse",
                resolver="rule_detect_only",
                confidence=0.45,
                notes="Repeal keyword found, but no concrete target was captured by rules.",
            ))
        return events

    @staticmethod
    def _find_date_near(text: str, start: int, end: int) -> Optional[str]:
        w_start, w_end = max(0, start-160), min(len(text), end+220)
        window = text[w_start:w_end]
        explicit = re.search(
            r"(?:kể\s+từ|kể|từ)\s+(?P<date>" + DATE_PATTERN + r")",
            window,
            re.I,
        )
        if explicit:
            dates = extract_all_vietnamese_dates(explicit.group("date"))
            return dates[0]["date"] if dates else None
        dates = []
        if not dates:
            return None
        after = [d for d in dates if d["span"][0] >= (start - w_start)]
        return (after[0] if after else dates[0])["date"]

    def _infer_repeal_dates(self, events: List[EffectivityEvent]) -> List[EffectivityEvent]:
        effective_by_doc = {}
        for ev in events:
            if (
                ev.event_type == "effective_from"
                and ev.date
                and ev.source_document_id
                and ev.target_scope in {"this_document", "unknown"}
            ):
                effective_by_doc[ev.source_document_id] = ev.date
        for ev in events:
            if ev.event_type.startswith("repeal") and not ev.date and ev.source_document_id:
                inferred = effective_by_doc.get(ev.source_document_id)
                if inferred:
                    ev.date = inferred
                    ev.date_role = "ceased_from"
                    ev.date_inference = "same_as_source_document_effective_from"
                    ev.confidence = max(ev.confidence, 0.76)
        return events
