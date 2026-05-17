\
import hashlib
import json
import re
import unicodedata
from pathlib import Path

def collapse_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()

def strip_accents(text, keep_dd=False):
    if keep_dd:
        text = (text or "").replace("đ", "dd").replace("Đ", "dd")
    else:
        text = (text or "").replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

def canonical_key(text):
    text = strip_accents(text or "", keep_dd=False).lower()
    return re.sub(r"[^a-z0-9]+", "", text)

_CONTENT_STOPWORDS = {
    "ban",
    "cac",
    "cho",
    "co",
    "cua",
    "duoc",
    "hanh",
    "hoac",
    "kem",
    "khong",
    "la",
    "mau",
    "mot",
    "nay",
    "ngay",
    "nam",
    "nhung",
    "phai",
    "phu",
    "quy",
    "so",
    "tai",
    "theo",
    "thang",
    "trong",
    "va",
    "ve",
    "voi",
    "doi",
    "dinh",
    "luc",
}

def normalized_terms(text):
    text = strip_accents(text or "", keep_dd=False).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    terms = []
    for term in text.split():
        if len(term) < 2 or term.isdigit() or term in _CONTENT_STOPWORDS:
            continue
        terms.append(term)
    return terms

def term_overlap_score(query_text, candidate_text):
    query_terms = set(normalized_terms(query_text))
    candidate_terms = set(normalized_terms(candidate_text))
    if not query_terms or not candidate_terms:
        return 0.0

    query_key = canonical_key(query_text)
    candidate_key = canonical_key(candidate_text)
    if candidate_key and candidate_key in query_key:
        return 1.0

    overlap = query_terms & candidate_terms
    if not overlap:
        return 0.0
    candidate_coverage = len(overlap) / len(candidate_terms)
    query_coverage = len(overlap) / len(query_terms)
    return 0.75 * candidate_coverage + 0.25 * query_coverage

def point_key(text):
    text = strip_accents(text or "", keep_dd=True).lower()
    return re.sub(r"[^a-z0-9]+", "", text)

def slugify(text):
    text = strip_accents(text or "", keep_dd=True).lower()
    text = text.replace("/", "_").replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\.]", "", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"

def md5_text(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_document_number(text):
    """
    Strictly normalize legal document numbers.

    Fixes greedy bug:
      "35/2024/TT-BGTVT ngày..." must become "35/2024/TT-BGTVT",
      not "35/2024/TT-BGTVTNG".
    """
    if not text:
        return None

    raw = collapse_ws(text).replace("\u00a0", " ")
    raw = re.sub(r"\s*[-–—]\s*", "-", raw)
    raw = raw.upper().replace("Đ", "D")
    # Token-bound capture:
    # - agency suffix accepts hyphen-separated uppercase tokens.
    # - supports NĐ-CP / ND-CP / TT-BGTVT / QĐ-TTG even when OCR inserts spaces.
    m = re.search(
        r"(?P<num>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*"
        r"(?P<kind>[A-Z]{1,12}\d{0,2})"
        r"(?:-(?P<agency>[A-Z0-9]{1,16}(?:-[A-Z0-9]{1,16})*))?"
        r"(?=$|[\s,.;:)\\]])",
        raw,
        re.UNICODE,
    )
    if not m:
        # Fallback but still non-greedy and uppercase-token bounded.
        m = re.search(
            r"(?P<num>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*"
            r"(?P<kind>[A-Z]{1,12}\d{0,2})(?:-(?P<agency>[A-Z0-9-]{1,40}))?",
            raw,
            re.UNICODE,
        )
    if not m:
        return None

    num = m.group("num")
    year = m.group("year")
    kind = (m.group("kind") or "").upper()
    agency = (m.group("agency") or "").upper()
    agency = re.sub(r"[^A-Z0-9]+", "", agency)
    if kind == "QH" and agency.isdigit():
        return f"{num}/{year}/QH{agency}"
    return f"{num}/{year}/{kind}-{agency}" if agency else f"{num}/{year}/{kind}"

def doc_number_key(text):
    n = normalize_document_number(text)
    return canonical_key(n) if n else None

def find_package_dirs(parsed_root):
    p = Path(parsed_root)
    if (p / "package_inventory.json").exists():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_dir() and (x / "package_inventory.json").exists()])
    return []

def short_context(text, raw, span=None, window=160):
    text = text or ""
    raw = raw or ""

    if span and isinstance(span, list) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
            return collapse_ws(text[max(0, start - window): min(len(text), end + window)])
        except Exception:
            pass

    idx = text.find(raw)
    if idx < 0:
        return collapse_ws(text[: 2 * window])
    return collapse_ws(text[max(0, idx - window): min(len(text), idx + len(raw) + window)])

def normalize_numeric_label(label):
    """
    Normalize form labels:
      "1" == "01"
      "Mẫu số 1" == "Mẫu số 01"
    Keeps non-numeric labels as canonical text.
    """
    if label is None:
        return ""
    s = str(label).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return str(int(m.group(1)))
    return canonical_key(s)

def pad_numeric_label(label, width=2):
    m = re.search(r"(\d+)", str(label or ""))
    if not m:
        return str(label or "")
    return m.group(1).zfill(width)

def extract_article_clause_point_from_path(path_text):
    path = path_text or ""
    art = cl = pt = None
    m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path, re.IGNORECASE)
    if m:
        art = m.group(1)
    m = re.search(r"Khoản\s+(\d+)", path, re.IGNORECASE)
    if m:
        cl = m.group(1)
    m = re.search(r"Điểm\s+([a-zđ])", path, re.IGNORECASE)
    if m:
        pt = m.group(1)
    return art, cl, pt
