
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

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

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def collapse_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()

def strip_accents(text):
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

def normalize_surface(text):
    text = collapse_ws(text).lower()
    text = re.sub(r"^[\"'“”‘’\(\[\{]+|[\"'“”‘’\)\]\};:,.]+$", "", text)
    return collapse_ws(text)

def canonical_key(text):
    text = normalize_surface(text)
    text = strip_accents(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return collapse_ws(text)

def stable_id(*parts, prefix="sf"):
    raw = "|".join([p or "" for p in parts])
    return f"{prefix}_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def is_reference_like(text):
    t = normalize_surface(text)
    patterns = [
        r"^điều\s+\d+[a-z]?$",
        r"^khoản\s+\d+$",
        r"^điểm\s+[a-zđ]$",
        r"^phụ\s+lục\s+([ivxlcdm]+|\d+|[a-z])$",
        r"^mẫu\s+(số\s+)?[0-9a-z_.\-\/]+$",
        r"^chương\s+([ivxlcdm]+|\d+)$",
        r"^mục\s+([ivxlcdm]+|\d+)$",
        r"^qcvn\s+.+$",
        r"^\d+/\d{4}/[a-zđ]+.*$",
        r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+(này|số\s+.+)$",
    ]
    return any(re.match(p, t, flags=re.I | re.U) for p in patterns)

def is_too_generic(text, label):
    t = normalize_surface(text)
    generic_by_label = {
        "BEHAVIOR": {"vi phạm", "vi phạm quy định", "thực hiện", "tham gia giao thông", "bị xử phạt", "xử phạt"},
        "VEHICLE": {"xe", "phương tiện", "loại xe", "phương tiện giao thông"},
        "ACTOR": {"người", "cá nhân", "tổ chức", "đối tượng", "người điều khiển phương tiện", "cơ quan có thẩm quyền"},
        "INFRASTRUCTURE": {"đường", "nơi", "khu vực", "vị trí", "địa điểm", "hệ thống", "thiết bị"},
        "DOCUMENT": {"văn bản", "quy định", "hồ sơ", "giấy tờ"},
        "VEHICLE_CONDITION_OR_EQUIPMENT": {"không đạt chuẩn", "không đúng quy định", "không bảo đảm", "thiết bị", "bộ phận"},
        "CONDITION": {"điều kiện", "yêu cầu", "tiêu chuẩn", "quy định", "phù hợp", "được phép", "không được"},
    }
    if t in generic_by_label.get(label, set()):
        return True
    if len(t) <= 2:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    return False

def default_status(surface, label, count):
    if is_reference_like(surface):
        return "reject", "reference_like"
    if is_too_generic(surface, label):
        return "reject", "too_generic"
    if count >= 3:
        return "accept", "auto_accept_count_ge_3"
    return "review", "low_frequency"
