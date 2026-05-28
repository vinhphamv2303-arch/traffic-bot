
import csv, json, re, hashlib
from pathlib import Path

VALID_LABELS = {
    "BEHAVIOR", "VEHICLE", "ACTOR", "INFRASTRUCTURE",
    "DOCUMENT", "VEHICLE_CONDITION_OR_EQUIPMENT", "CONDITION",
}

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def collapse_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()

def normalize_surface(text):
    text = collapse_ws(text).lower()
    text = re.sub(r"^[\"'“”‘’\(\[\{]+|[\"'“”‘’\)\]\};:,.]+$", "", text)
    return collapse_ws(text)

def stable_id(*parts, prefix="id"):
    raw = "|".join([p or "" for p in parts])
    return prefix + "_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def label_to_filename(label):
    return (label or "").lower() + ".txt"

def is_word_char(ch):
    return ch.isalnum() or ch == "_"

def boundary_ok(text, start, end):
    return (start == 0 or not is_word_char(text[start-1])) and (end >= len(text) or not is_word_char(text[end]))

def find_sentence_package_dirs(root):
    p = Path(root)
    if (p / "sentences.jsonl").exists():
        return [p]
    return sorted([x for x in p.iterdir() if x.is_dir() and (x / "sentences.jsonl").exists()])
