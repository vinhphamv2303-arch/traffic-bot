
import csv
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

def read_csv_dicts(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def find_package_dirs(parsed_root):
    p = Path(parsed_root)
    if (p / "package_inventory.json").exists():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_dir() and (x / "package_inventory.json").exists()])
    return []

def normalize_unit_id(row):
    return str(row.get("unit_id") or row.get("id") or row.get("target_id") or "").strip()

def none_if_null(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "nat"}:
        return None
    return s

def extract_article_clause_point_from_path(path_text):
    path = path_text or ""
    art = cl = pt = None
    m = re.search(r"Điều\s+(\d+[a-zA-Z]?)", path, flags=re.IGNORECASE)
    if m:
        art = m.group(1)
    m = re.search(r"Khoản\s+(\d+)", path, flags=re.IGNORECASE)
    if m:
        cl = m.group(1)
    m = re.search(r"Điểm\s+([a-zđ])", path, flags=re.IGNORECASE)
    if m:
        pt = m.group(1)
    return art, cl, pt

def unit_selector_key(unit):
    doc_id = str(unit.get("document_id") or "")
    art, cl, pt = extract_article_clause_point_from_path(unit.get("path_text") or "")
    typ = unit.get("type") or unit.get("unit_type")
    no = str(unit.get("raw_no") or unit.get("no") or "")
    if typ == "dieu" and no:
        art = no
    elif typ == "khoan" and no:
        cl = no
    elif typ == "diem" and no:
        pt = no
    return (doc_id, canonical_key(art or ""), canonical_key(cl or ""), canonical_key(pt or ""))
