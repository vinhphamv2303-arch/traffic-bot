from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
from ..common.utils import collapse_ws, md5_text

ROMAN = r"[IVXLCDM]+"
APPENDIX_LABEL = r"(?:[IVXLCDM]+[A-Z]?|\d+[A-Z]?|[A-Z]+)"
LEGAL_DOC_TYPES = r"Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|Quy\s+chuẩn|QCVN"
REFERENCE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("article", re.compile(r"\b(?:[ĐđD]iều)\s+(?P<label>\d+[a-zA-Z]?|này)\b", re.UNICODE)),
    ("clause", re.compile(r"\b[Kk]hoản\s+(?P<label>\d+|này)\b", re.UNICODE)),
    ("point", re.compile(r"\b[ĐđD]iểm\s+(?P<label>[a-zđ]\d*|này)\b", re.UNICODE)),
    ("chapter", re.compile(r"\b[Cc]hương\s+(?P<label>" + ROMAN + r"|\d+|này)\b", re.UNICODE)),
    ("section", re.compile(r"\b[Mm]ục\s+(?P<label>" + ROMAN + r"|\d+|này)\b", re.UNICODE)),
    ("appendix", re.compile(r"\b[Pp]hụ\s+[Ll]ục\s+(?P<label>" + APPENDIX_LABEL + r")\b", re.UNICODE)),
    ("form", re.compile(r"\b[Mm]ẫu\s+(?:số\s+)?(?P<label>\d{1,3}[A-Za-zĐđ]?(?:[._\-/]\d+)*)\b", re.UNICODE)),
    (
        "legal_document",
        re.compile(
            r"\b(?P<doc_type>" + LEGAL_DOC_TYPES + r")"
            r"(?:\s+(?!số\b|này\b|nay\b|thay\b|hết\b|có\b|co\b|được\b|duoc\b)\S+){0,12}"
            r"\s+số\s+(?P<label>[0-9]+/[0-9A-ZĐđ\-_/\.]+)",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
]
LIST_REFERENCE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("article", re.compile(r"\b(?:các\s+)?[ĐđD]iều\s+(?P<labels>\d+[a-zA-Z]?(?:\s*(?:,|\bvà\b|\bhoặc\b)\s*(?:[ĐđD]iều\s+)?\d+[a-zA-Z]?)+)", re.IGNORECASE | re.UNICODE)),
    ("clause", re.compile(r"\b(?:các\s+)?[Kk]hoản\s+(?P<labels>\d+(?:\s*(?:,|\bvà\b|\bhoặc\b)\s*(?:[Kk]hoản\s+)?\d+)+)", re.IGNORECASE | re.UNICODE)),
    ("point", re.compile(r"\b(?:các\s+)?[ĐđD]iểm\s+(?P<labels>[a-zđ]\d*(?:\s*(?:,|\bvà\b|\bhoặc\b)\s*(?:[ĐđD]iểm\s+)?[a-zđ]\d*)+)", re.IGNORECASE | re.UNICODE)),
]
AMENDMENT_PATTERN = re.compile(r"\b(?P<action>sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|hủy\s+bỏ|bỏ\s+cụm\s+từ|đính\s+chính)\b", re.IGNORECASE | re.UNICODE)
AMENDMENT_TARGET_PATTERN = re.compile(
    r"\b("
    r"(?:một\s+số|các)\s+điều|"
    r"(?:một\s+số|các)?\s*(?:điểm|khoản|điều)(?:\s*,\s*(?:điểm|khoản|điều))+|"
    r"điều\s+(?:\d+[a-zA-Z]?|này)|"
    r"khoản\s+(?:\d+|này)|"
    r"điểm\s+(?:[a-zđ]|này)|"
    r"chương\s+(?:" + ROMAN + r"|\d+|này)|"
    r"mục\s+(?:" + ROMAN + r"|\d+|này)|"
    r"phụ\s+lục\s+" + APPENDIX_LABEL + r"|"
    r"mẫu\s+số\s+\d{1,3}[A-Za-zĐđ]?(?:[._\-/]\d+)*|"
    r"(?:thông\s+tư|nghị\s+định|luật|bộ\s+luật|quyết\s+định|nghị\s+quyết)\s+số\s+[0-9]+/[0-9A-ZĐđ\-_/\.]+|"
    r"(?:các|một\s+số)\s+(?:văn\s+bản|thông\s+tư|nghị\s+định|luật|bộ\s+luật|quyết\s+định|nghị\s+quyết)(?:\s+sau\s+đây|\s+sau)?|"
    r"cụm\s+từ"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

def infer_reference_scope(raw: str) -> str:
    s = raw.lower()
    if any(x in s for x in ["điều này", "khoản này", "điểm này", "chương này", "mục này"]): return "this_document"
    if any(x in s for x in ["thông tư này", "nghị định này", "luật này", "văn bản này"]): return "this_document"
    if "ban hành kèm theo" in s: return "same_package"
    return "unknown"

def has_legal_amendment_target(text: str, start: int, end: int) -> bool:
    action = (text[start:end] or "").lower()
    before = (text[max(0, start - 40):start] or "").lower()
    if action == "bổ sung" and re.search(
        r"(hình\s+phạt|xử\s+phạt|biện\s+pháp|hồ\s+sơ|tài\s+liệu|thông\s+tin|dịch\s+vụ)\s+$",
        before,
        re.IGNORECASE | re.UNICODE,
    ):
        return False
    action_context = text[start:min(len(text), end + 100)]
    return bool(AMENDMENT_TARGET_PATTERN.search(action_context))

def make_mention_id(source_text: str, raw: str, start: int, end: int, mention_type: str) -> str:
    return md5_text(f"{mention_type}|{start}|{end}|{raw}|{source_text[:200]}")[:16]

def split_reference_labels(labels_text: str, ref_type: str = "") -> List[str]:
    parts = re.split(r"\s*(?:,|\bvà\b|\bhoặc\b)\s*", labels_text or "", flags=re.IGNORECASE)
    out = []
    prefix_patterns = {
        "article": r"^(?:[ĐđD]iều)\s+",
        "clause": r"^(?:[Kk]hoản)\s+",
        "point": r"^(?:[ĐđD]iểm)\s+",
    }
    for part in parts:
        label = collapse_ws(part)
        if not label:
            continue
        pat = prefix_patterns.get(ref_type)
        if pat:
            label = collapse_ws(re.sub(pat, "", label, flags=re.IGNORECASE | re.UNICODE))
        if label:
            out.append(label)
    return out

def list_ref_raw(ref_type: str, label: str) -> str:
    prefixes = {"article": "Điều", "clause": "khoản", "point": "điểm"}
    return f"{prefixes.get(ref_type, ref_type)} {label}"

def extract_ref_mentions(text: str) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    if not text: return mentions
    seen=set()
    list_spans = []
    for ref_type, pattern in LIST_REFERENCE_PATTERNS:
        for m in pattern.finditer(text):
            raw_list = collapse_ws(m.group(0))
            labels = split_reference_labels(m.group("labels"), ref_type)
            label_set = {x.lower() for x in labels}
            list_spans.append((ref_type, m.start(), m.end(), label_set))
            for label in labels:
                raw = list_ref_raw(ref_type, label)
                key = (ref_type, label.lower(), m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                mentions.append({
                    "mention_id": make_mention_id(text, raw, m.start(), m.end(), ref_type),
                    "mention_type": ref_type, "label": label, "doc_type_hint": None,
                    "raw": raw, "raw_list": raw_list, "scope_hint": infer_reference_scope(raw_list),
                    "span": [m.start(), m.end()],
                    "needs_resolution": True,
                    "resolution": {"status": "unresolved", "target_id": None, "target_type": None, "resolver": None, "confidence": None},
                })
    for ref_type, pattern in REFERENCE_PATTERNS:
        for m in pattern.finditer(text):
            raw=collapse_ws(m.group(0)); label=collapse_ws(m.groupdict().get("label", ""))
            doc_type=collapse_ws(m.groupdict().get("doc_type", "")) if "doc_type" in m.groupdict() else ""
            if any(
                ref_type == span_type and span_start <= m.start() and m.end() <= span_end and label.lower() in labels
                for span_type, span_start, span_end, labels in list_spans
            ):
                continue
            key=(ref_type, label.lower(), raw.lower(), m.start(), m.end())
            if key in seen: continue
            seen.add(key)
            mentions.append({
                "mention_id": make_mention_id(text, raw, m.start(), m.end(), ref_type),
                "mention_type": ref_type, "label": label, "doc_type_hint": doc_type or None,
                "raw": raw, "scope_hint": infer_reference_scope(raw), "span": [m.start(), m.end()],
                "needs_resolution": True,
                "resolution": {"status": "unresolved", "target_id": None, "target_type": None, "resolver": None, "confidence": None},
            })
    mentions.sort(key=lambda x: x["span"][0]); return mentions

def extract_amendment_mentions(text: str) -> List[Dict[str, Any]]:
    items=[]
    if not text: return items
    for m in AMENDMENT_PATTERN.finditer(text):
        if not has_legal_amendment_target(text, m.start(), m.end()):
            continue
        raw=collapse_ws(m.group(0)); action=collapse_ws(m.group("action")).lower()
        items.append({
            "mention_id": make_mention_id(text, raw, m.start(), m.end(), f"amendment:{action}"),
            "action_hint": action, "raw": raw, "span": [m.start(), m.end()], "needs_llm_parse": True,
            "operation": {"status": "unparsed", "operation_type": None, "target_id": None, "target_selector": None, "new_text": None, "processor": None, "confidence": None},
        })
    return items

def flags_for_text(text: str) -> Dict[str, Any]:
    refs=extract_ref_mentions(text); amends=extract_amendment_mentions(text)
    return {"has_ref_or_amend": bool(refs or amends), "has_reference_mention": bool(refs), "has_amendment_mention": bool(amends), "needs_reference_resolution": bool(refs), "needs_amendment_processing": bool(amends)}
