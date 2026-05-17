from __future__ import annotations
import re

DOC_NUMBER_PATTERN = (
    r"(?:"
    r"\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ]+[0-9]*(?:\s*[-–—]\s*[A-ZĐ0-9]+)*"
    r"|"
    r"\d+\s*/\s*[A-ZĐ]+[0-9]*(?:\s*[-–—]\s*[A-ZĐ0-9]+)*"
    r")"
)
LEGAL_DOC_TYPES = r"Thông\s+tư\s+liên\s+tịch|Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết"
THIS_DOCUMENT_TYPES = r"Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|Văn\s+bản"
DATE_PATTERN = r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
DOC_REF_TITLE_WORD = r"(?!(?:số|này|bãi|bỏ|hết|chấm|ngưng|hiệu|lực|và|theo|tại|được|do|bởi)\b)[A-Za-zÀ-ỹĐđ]+"
UNIT_SELECTOR_RAW_PATTERN = (
    r"(?:(?:điểm)\s+(?P<point>[a-zđ])\s+)?"
    r"(?:(?:khoản)\s+(?P<clause>\d+[a-zA-Z]?)\s+)?"
    r"(?:Điều)\s+(?P<article>\d+[a-zA-Z]?|này)"
)
APPENDIX_SELECTOR_RAW_PATTERN = r"Phụ\s+lục\s+(?P<appendix>[IVXLCDM]+|\d+|[A-ZĐ]+)"
LEGAL_DOCUMENT_REF_PATTERN = re.compile(
    r"(?P<doc_type>" + LEGAL_DOC_TYPES + r")"
    r"(?:\s+" + DOC_REF_TITLE_WORD + r"){0,8}"
    r"\s+số\s+"
    r"(?P<doc_number>" + DOC_NUMBER_PATTERN + r")"
    r"(?=\s*(?:ngày|,|;|\.|\)|>|và|sửa|được|đã|hết|của|$))",
    re.I | re.U,
)

EFFECTIVE_FROM_PATTERNS = [
    re.compile(
        r"(?P<target>(?:" + THIS_DOCUMENT_TYPES + r")\s+này)"
        r"\s+có\s+hiệu\s+lực(?:\s+thi\s+hành)?\s+(?:kể\s+từ\s+|kể\s+|từ\s+)"
        r"(?P<date>" + DATE_PATTERN + r")",
        re.I | re.U,
    ),
    re.compile(
        r"có\s+hiệu\s+lực(?:\s+thi\s+hành)?\s+(?:kể\s+từ\s+|kể\s+|từ\s+)"
        r"(?P<date>" + DATE_PATTERN + r")",
        re.I | re.U,
    ),
]

UNIT_EFFECTIVE_FROM_PATTERNS = [
    re.compile(
        r"(?:quy\s+định\s+(?:tại\s+)?|tại\s+|trừ\s+)?"
        r"(?P<selector>" + UNIT_SELECTOR_RAW_PATTERN + r")"
        r"(?:\s+(?:của\s+)?(?:" + THIS_DOCUMENT_TYPES + r")\s+này)?"
        r"\s+có\s+hiệu\s+lực(?:\s+thi\s+hành)?\s+(?:kể\s+từ\s+|kể\s+|từ\s+)"
        r"(?P<date>" + DATE_PATTERN + r")",
        re.I | re.U,
    ),
    re.compile(
        r"(?:áp\s+dụng|được\s+áp\s+dụng)\s+(?:đối\s+với\s+)?"
        r"(?P<selector>" + UNIT_SELECTOR_RAW_PATTERN + r")"
        r"(?:\s+(?:của\s+)?(?:" + THIS_DOCUMENT_TYPES + r")\s+này)?"
        r"\s+(?:kể\s+từ\s+|kể\s+|từ\s+)"
        r"(?P<date>" + DATE_PATTERN + r")",
        re.I | re.U,
    ),
    re.compile(
        r"(?P<selector>" + APPENDIX_SELECTOR_RAW_PATTERN + r")"
        r".{0,120}?(?:được\s+)?áp\s+dụng\s+(?:kể\s+từ\s+|kể\s+|từ\s+)"
        r"(?P<date>" + DATE_PATTERN + r")",
        re.I | re.U,
    ),
]

UNIT_EFFECTIVE_INDIRECT_PATTERNS = [
    re.compile(
        r"(?:quy\s+định\s+(?:tại\s+)?|tại\s+|trừ\s+)?"
        r"(?P<selector>" + UNIT_SELECTOR_RAW_PATTERN + r")"
        r"(?:\s+(?:của\s+)?(?:" + THIS_DOCUMENT_TYPES + r")\s+này)?"
        r"\s+có\s+hiệu\s+lực(?:\s+thi\s+hành)?\s+"
        r"(?P<condition>theo\s+quy\s+định.{0,180}?)(?=;|\.|$)",
        re.I | re.U,
    ),
]

REPEAL_KEYWORD_PATTERN = re.compile(
    r"\b(bãi\s+bỏ|hết\s+hiệu\s+lực|chấm\s+dứt\s+hiệu\s+lực|ngưng\s+hiệu\s+lực)\b",
    re.I | re.U,
)

REPLACEMENT_KEYWORD_PATTERN = re.compile(
    r"\b(thay\s+thế)\b",
    re.I | re.U,
)

REPEAL_DOCUMENT_PATTERN = re.compile(
    r"(?:bãi\s+bỏ|hết\s+hiệu\s+lực|chấm\s+dứt\s+hiệu\s+lực).{0,180}?"
    r"(?P<doc_type>" + LEGAL_DOC_TYPES + r")"
    r"(?:\s+" + DOC_REF_TITLE_WORD + r"){0,8}"
    r"\s+số\s+"
    r"(?P<doc_number>" + DOC_NUMBER_PATTERN + r")"
    r"(?=\s*(?:ngày|,|;|\.|\)|>|và|sửa|được|đã|hết|của|$))",
    re.I | re.U,
)

UNIT_SELECTOR_PATTERN = re.compile(
    UNIT_SELECTOR_RAW_PATTERN,
    re.I | re.U,
)
