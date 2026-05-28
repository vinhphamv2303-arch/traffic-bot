import re

from .utils import collapse_ws, extract_article_clause_point_from_path, normalize_document_number, normalize_numeric_label

MENTION_TYPE_PATTERNS = {
    "appendix": re.compile(r"[Pp]hụ\s+[Ll]ục\s+([IVXLCDM]+[A-Z]?|\d+[A-Z]?|[A-Z]+)", re.UNICODE),
    "form": re.compile(r"[Mm]ẫu\s+(?:s[^\s]*\s+)?([0-9][0-9A-Za-zĐđ_.\-/]*)", re.UNICODE),
    "point": re.compile(r"[Đđ]iểm\s+([a-zđ]|này)\b", re.UNICODE),
    "clause": re.compile(r"[Kk]hoản\s+(\d+|này)\b", re.UNICODE),
    "article": re.compile(r"[Đđ]iều\s+(\d+[a-zA-Z]?|này)\b", re.UNICODE),
    "legal_document": re.compile(
        r"(?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết)\s+số\s+"
        r"([0-9]+/[0-9]{4}/[A-ZĐ]+(?:[-–—]?[A-ZĐ0-9]+)*)",
        re.IGNORECASE | re.UNICODE,
    ),
}

RELATIVE_UNIT_SCOPES = {
    "this_unit_or_article",
    "this_article",
    "this_unit_or_clause",
    "this_clause",
}

def parse_selector(raw, source_text, mention_type=None, span=None, source_path_text=None):
    """
    Parse selector anchored to the actual mention/span.

    Fixes:
    - If context contains Điều 13 and Điều 14, mention Điều 14 resolves to 14.
    - If context contains Mẫu số 01/02/03, mention Mẫu số 03 resolves to 03.
    - Relative refs: Điều này, khoản này, điểm a, điểm b khoản này use source_path_text.
    """
    text = source_text or raw or ""
    raw = raw or ""
    mention_type = mention_type or "unknown"
    m_start, m_end = _locate_mention(text, raw, span, mention_type)
    ctx_start = max(0, m_start - 160) if m_start is not None else 0
    ctx = _context_around(text, m_start, m_end, window=160)
    anchor_start = m_start - ctx_start if m_start is not None else None
    anchor_end = m_end - ctx_start if m_end is not None else None

    sel = {
        "article": None,
        "clause": None,
        "point": None,
        "appendix_label": None,
        "form_label": None,
        "form_number_norm": None,
        "document_number": None,
        "document_title_hint": None,
        "scope_hint": None,
        "selector_raw": collapse_ws(ctx),
        "anchor_span": [m_start, m_end] if m_start is not None and m_end is not None else None,
    }

    src_art, src_clause, src_point = extract_article_clause_point_from_path(source_path_text or "")

    # First parse exact mention text if possible.
    mention_text = raw or (text[m_start:m_end] if m_start is not None and m_end is not None else "")
    _parse_direct_mention(sel, mention_text, mention_type)

    # Then parse compound selector around the anchor.
    _parse_compound_near_anchor(sel, ctx, mention_text, mention_type, anchor_start, anchor_end)

    if mention_type in {"clause", "point"} and not sel.get("article"):
        nearest_article, article_dist = _nearest_selector_value(ctx, "article", anchor_start, anchor_end)
        if nearest_article and article_dist <= 100:
            if nearest_article.lower() == "này":
                sel["article"] = src_art
                sel["scope_hint"] = sel.get("scope_hint") or "this_article"
            else:
                sel["article"] = nearest_article

    if mention_type == "point" and not sel.get("clause"):
        nearest_clause, clause_dist = _nearest_selector_value(ctx, "clause", anchor_start, anchor_end)
        if nearest_clause and clause_dist <= 80:
            if nearest_clause.lower() == "này":
                sel["article"] = src_art
                sel["clause"] = src_clause
                sel["scope_hint"] = sel.get("scope_hint") or "this_clause"
            else:
                sel["clause"] = nearest_clause

    # Relative references.
    low_mention = mention_text.lower()
    low_ctx = ctx.lower()
    if "điều này" in low_mention or (mention_type == "article" and low_mention.strip().endswith("này")):
        sel["article"] = src_art
        sel["scope_hint"] = "this_unit_or_article"
    if mention_type in {"clause", "point"} and "điều này" in low_ctx and not sel.get("article"):
        sel["article"] = src_art
        sel["scope_hint"] = sel.get("scope_hint") or "this_article"
    if "khoản này" in low_mention or (mention_type == "clause" and low_mention.strip().endswith("này")):
        sel["article"] = src_art
        sel["clause"] = src_clause
        sel["scope_hint"] = "this_unit_or_clause"
    if mention_type == "point" and "khoản này" in low_ctx and not sel.get("clause"):
        sel["article"] = src_art
        sel["clause"] = src_clause
        sel["scope_hint"] = sel.get("scope_hint") or "this_clause"

    # "điểm a" inside current clause.
    if mention_type == "point" and sel.get("point") and not sel.get("clause") and not sel.get("article"):
        sel["article"] = src_art
        sel["clause"] = src_clause

    # "điểm b khoản này"
    if mention_type == "point" and sel.get("point") and "khoản này" in low_ctx:
        sel["article"] = src_art
        sel["clause"] = src_clause
        sel["scope_hint"] = "this_clause"

    # "khoản 2" without Điều, relative to current article.
    if mention_type == "clause" and sel.get("clause") and not sel.get("article"):
        sel["article"] = src_art
        sel["scope_hint"] = sel.get("scope_hint") or "this_article"

    # "Điều 14" without explicit document -> current document normally.
    if mention_type == "article" and sel.get("article") and not sel.get("document_number"):
        sel["scope_hint"] = sel.get("scope_hint") or "this_document"

    doc_no, title_hint = None, None
    if not (mention_type == "legal_document" and sel.get("document_number")):
        doc_no, title_hint = _extract_document_hint_for_anchor(ctx, mention_type, anchor_start, anchor_end, sel)
        if not doc_no and not title_hint:
            doc_no, title_hint = _extract_amendment_target_document_hint(ctx, source_path_text or "", mention_type)
        if doc_no:
            sel["document_number"] = doc_no
        elif title_hint:
            sel["document_title_hint"] = title_hint

    low = ctx.lower()
    if any(x in low for x in ["thông tư này", "nghị định này", "luật này", "văn bản này"]):
        sel["scope_hint"] = "this_document"
    elif "phụ lục này" in low:
        sel["scope_hint"] = "this_attachment"
    elif "ban hành kèm theo" in low:
        sel["scope_hint"] = "same_package"

    _apply_selector_granularity(sel, mention_type)
    return sel

def _locate_mention(text, raw, span, mention_type):
    if span and isinstance(span, list) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
            if 0 <= start <= end <= len(text):
                return start, end
        except Exception:
            pass

    raw = raw or ""
    if raw:
        i = text.find(raw)
        if i >= 0:
            return i, i + len(raw)

    # Fallback by type: use first matching mention of that type only when raw unavailable.
    pat = MENTION_TYPE_PATTERNS.get(mention_type)
    if pat:
        m = pat.search(text or "")
        if m:
            return m.start(), m.end()

    return None, None

def _context_around(text, start, end, window=160):
    if start is None or end is None:
        return (text or "")[:2 * window]
    return (text or "")[max(0, start - window): min(len(text or ""), end + window)]

def _parse_direct_mention(sel, mention_text, mention_type):
    t = mention_text or ""

    if mention_type == "appendix":
        m = MENTION_TYPE_PATTERNS["appendix"].search(t)
        if m:
            sel["appendix_label"] = f"Phụ lục {m.group(1).upper()}"

    elif mention_type == "form":
        m = MENTION_TYPE_PATTERNS["form"].search(t)
        if m:
            raw_num = m.group(1)
            sel["form_label"] = f"Mẫu số {raw_num}"
            sel["form_number_norm"] = normalize_numeric_label(raw_num)

    elif mention_type == "article":
        m = MENTION_TYPE_PATTERNS["article"].search(t)
        if m and m.group(1).lower() != "này":
            sel["article"] = m.group(1)

    elif mention_type == "clause":
        m = MENTION_TYPE_PATTERNS["clause"].search(t)
        if m and m.group(1).lower() != "này":
            sel["clause"] = m.group(1)

    elif mention_type == "point":
        m = MENTION_TYPE_PATTERNS["point"].search(t)
        if m and m.group(1).lower() != "này":
            sel["point"] = m.group(1)

    elif mention_type == "legal_document":
        doc_no = normalize_document_number(t)
        if doc_no:
            sel["document_number"] = doc_no

def _parse_compound_near_anchor(sel, ctx, mention_text, mention_type, anchor_start=None, anchor_end=None):
    # We work inside ctx. Use mention_text occurrence in ctx as local anchor.
    if anchor_start is None or anchor_end is None:
        local_anchor = ctx.find(mention_text) if mention_text else -1
        if local_anchor < 0:
            local_anchor = len(ctx) // 2
        anchor_start = local_anchor
        anchor_end = local_anchor + len(mention_text or "")

    # Parse closest compound selector around the mention, not the first in ctx.
    # Only fill selector levels that are valid for the current mention type:
    # article -> article, clause -> article/clause, point -> article/clause/point.
    if mention_type in {"article", "clause", "point"}:
        patterns = [
            re.compile(r"[Đđ]iểm\s+(?P<point>[a-zđ])\s+[Kk]hoản\s+(?P<clause>\d+|này)\s+[Đđ]iều\s+(?P<article>\d+[a-zA-Z]?|này)", re.UNICODE),
            re.compile(r"[Kk]hoản\s+(?P<clause>\d+|này)\s+[Đđ]iều\s+(?P<article>\d+[a-zA-Z]?|này)", re.UNICODE),
            re.compile(r"[Đđ]iểm\s+(?P<point>[a-zđ])\s+[Kk]hoản\s+(?P<clause>\d+|này)", re.UNICODE),
            re.compile(r"[Đđ]iều\s+(?P<article>\d+[a-zA-Z]?|này)", re.UNICODE),
        ]

        best = None
        best_dist = 10**9
        for pat in patterns:
            for m in pat.finditer(ctx):
                dist = _span_distance(m.start(), m.end(), anchor_start, anchor_end)
                if dist < best_dist:
                    best = m
                    best_dist = dist
            if best and best_dist == 0:
                break

        if best:
            gd = best.groupdict()
            if gd.get("article") and gd["article"].lower() != "này":
                sel["article"] = sel.get("article") or gd["article"]
            if mention_type in {"clause", "point"} and gd.get("clause") and gd["clause"].lower() != "này":
                sel["clause"] = sel.get("clause") or gd["clause"]
            if mention_type == "point" and gd.get("point") and gd["point"].lower() != "này":
                sel["point"] = sel.get("point") or gd["point"]

    # Closest form/appendix mention, not first in context.
    for typ, key, fmt in [
        ("form", "form_label", lambda x: f"Mẫu số {x}"),
        ("appendix", "appendix_label", lambda x: f"Phụ lục {x.upper()}"),
    ]:
        pat = MENTION_TYPE_PATTERNS[typ]
        best2 = None
        best2_dist = 10**9
        for m in pat.finditer(ctx):
            dist = _span_distance(m.start(), m.end(), anchor_start, anchor_end)
            if dist < best2_dist:
                best2 = m
                best2_dist = dist
        if best2 and (key not in sel or not sel.get(key)):
            raw_val = best2.group(1)
            sel[key] = fmt(raw_val)
            if typ == "form":
                sel["form_number_norm"] = normalize_numeric_label(raw_val)

def _span_distance(start, end, anchor_start, anchor_end):
    if anchor_start is None or anchor_end is None:
        return 0
    if start <= anchor_end and end >= anchor_start:
        return 0
    return min(abs(start - anchor_end), abs(anchor_start - end))

def _nearest_selector_value(ctx, selector_type, anchor_start, anchor_end):
    pat = MENTION_TYPE_PATTERNS.get(selector_type)
    if not pat:
        return None, 10**9
    best_value = None
    best_dist = 10**9
    for m in pat.finditer(ctx or ""):
        dist = _span_distance(m.start(), m.end(), anchor_start, anchor_end)
        if dist < best_dist:
            best_value = m.group(1)
            best_dist = dist
    return best_value, best_dist

def _extract_document_hint_for_anchor(ctx, mention_type, anchor_start, anchor_end, sel):
    """
    Extract a document hint only when it is attached to the current mention.

    The old resolver scanned the whole context window, so in a sentence such as
    "... điểm c, điểm d ... khoản này ... khoản 3 Điều 8 Luật Đường bộ" the
    external title at the end was incorrectly assigned to every earlier point.
    """
    ctx = ctx or ""

    if mention_type == "legal_document":
        doc_no = normalize_document_number(ctx)
        return (doc_no, None) if doc_no else (None, _extract_document_title_hint(ctx))

    if mention_type in {"article", "clause", "point"}:
        if _has_relative_unit_scope(sel):
            return None, None
        segment = _forward_document_segment(ctx, anchor_start, anchor_end)
    else:
        segment = ctx

    doc_no = normalize_document_number(segment)
    if doc_no:
        return doc_no, None
    return None, _extract_document_title_hint(segment)

def _extract_amendment_target_document_hint(ctx, source_path_text, mention_type):
    if mention_type not in {"article", "clause", "point"}:
        return None, None
    if not _looks_like_amendment_context(ctx):
        return None, None

    for segment in _amendment_path_segments(source_path_text):
        doc_no = normalize_document_number(segment)
        if doc_no:
            return doc_no, None
        title_hint = _extract_document_title_hint(segment)
        if title_hint:
            return None, title_hint
    return None, None

def _looks_like_amendment_context(text):
    low = (text or "").lower()
    if not re.search(r"\b(sửa\s+đổi|bổ\s+sung|bãi\s+bỏ|thay\s+thế)\b", low, flags=re.UNICODE):
        return False
    return bool(re.search(r"\b(như\s+sau|vào\s+sau|một\s+số\s+điều|điểm|khoản|điều)\b", low, flags=re.UNICODE))

def _amendment_path_segments(source_path_text):
    segments = []
    for part in (source_path_text or "").split(">"):
        part = collapse_ws(part)
        if _looks_like_amendment_context(part):
            segments.append(part)
    return segments

def _has_relative_unit_scope(sel):
    return (sel.get("scope_hint") or "") in RELATIVE_UNIT_SCOPES

def _forward_document_segment(ctx, anchor_start, anchor_end, max_chars=260):
    if anchor_start is None or anchor_end is None:
        return (ctx or "")[:max_chars]

    segment = (ctx or "")[anchor_start: min(len(ctx or ""), anchor_end + max_chars)]
    stop = re.search(r"[;\n]", segment)
    if stop:
        return segment[:stop.start()]
    return segment

def _extract_document_title_hint(ctx):
    pat = re.compile(
        r"\b(Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết)\s+(?!số\b)([^.;:\n]+)",
        re.UNICODE,
    )
    for m in pat.finditer(ctx or ""):
        raw = collapse_ws(m.group(0))
        raw = _trim_document_title_hint(raw)
        tail = collapse_ws(m.group(2)).lower()
        if not tail or tail in {"n"} or tail.startswith(("này", "nay", "nà", "na")):
            continue
        return raw.rstrip(" ,")
    return None

def _trim_document_title_hint(raw):
    stop_patterns = [
        r",\s+(?=Bộ\s+luật\b|Luật\b|Nghị\s+định\b|Thông\s+tư\b|Quyết\s+định\b|Nghị\s+quyết\b)",
        r",\s+(?=Điều\b|Mục\b|Chương\b|Phần\b)",
        r"\s+và\s+(?=Điều\b|khoản\b|điểm\b|thành\s+phần\b|hồ\s+sơ\b|theo\b|Mẫu\b|Phụ\s+lục\b|Thông\s+tư\s+này\b)",
        r"\s+tại\s+(?=điểm\b|khoản\b|Điều\b)",
        r"\s+đối\s+với\b",
        r"\s+theo\s+quy\s+định\b",
        r"\s+thì\b",
        r"\s+để\b",
        r"\s+được\s+(?=sửa\s+đổi|bổ\s+sung|quy\s+định|áp\s+dụng|phân\s+loại|thực\s+hiện|xác\s+định)",
        r",\s+được\s+(?=sửa\s+đổi|bổ\s+sung)",
        r",\s+trong\s+đó\b",
        r",\s+nếu\b",
        r",\s+định\s+giá\b",
        r",\s+chia\s+sẻ\b",
        r"\s+và\s+các\s+quy\s+định\b",
        r"\s+và\s+quy\s+định\s+tại\b",
        r"\s+và\s+căn\s+cứ\b",
    ]
    for pat in stop_patterns:
        raw = re.split(pat, raw, maxsplit=1, flags=re.IGNORECASE | re.UNICODE)[0]
    return collapse_ws(raw)

def _apply_selector_granularity(sel, mention_type):
    if mention_type == "article":
        sel["clause"] = None
        sel["point"] = None
    elif mention_type == "clause":
        sel["point"] = None
