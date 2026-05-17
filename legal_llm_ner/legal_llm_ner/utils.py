import hashlib
import json
import re
from pathlib import Path

def collapse_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()

def md5_text(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

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

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def find_sentence_package_dirs(sentences_root):
    p = Path(sentences_root)
    if (p / "sentences.jsonl").exists():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_dir() and (x / "sentences.jsonl").exists()])
    return []

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_json_object(text):
    """
    Extract JSON object from LLM output, tolerating markdown fences.
    """
    text = text or ""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end+1])
    raise ValueError("No valid JSON object found in model output")

def find_offsets(text, span_text):
    if not text or not span_text:
        return None, None
    idx = text.find(span_text)
    if idx >= 0:
        return idx, idx + len(span_text)

    idx = text.lower().find(span_text.lower())
    if idx >= 0:
        return idx, idx + len(span_text)

    # Whitespace-insensitive fallback with offsets mapped back to original text.
    norm_span = collapse_ws(span_text).lower()
    norm_text, index_map = normalize_with_index_map(text)
    idx = norm_text.lower().find(norm_span)
    if idx >= 0 and index_map:
        start = index_map[idx]
        last_norm_idx = idx + len(norm_span) - 1
        if 0 <= last_norm_idx < len(index_map):
            end = index_map[last_norm_idx] + 1
            return start, end

    return None, None

def normalize_with_index_map(text):
    chars = []
    index_map = []
    previous_was_space = False
    for idx, ch in enumerate(text or ""):
        if ch.isspace():
            if chars and not previous_was_space:
                chars.append(" ")
                index_map.append(idx)
            previous_was_space = True
        else:
            chars.append(ch)
            index_map.append(idx)
            previous_was_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map

def is_reference_like_entity(text, label=None):
    """
    Block structural legal references from semantic NER.
    These are handled by reference resolver, not entity graph.
    """
    t = collapse_ws(text).lower()
    if not t:
        return True
    if re.match(r"^(thông tư|nghị định|luật|bộ luật|quyết định|nghị quyết)(\s|$)", t, flags=re.I):
        return True
    if re.search(r"\bqcvn\b", t, flags=re.I):
        return True
    if label == "DOCUMENT_OR_PERMIT" and re.match(r"^(qcvn\s*\d|quy\s+chuẩn\s+kỹ\s+thuật\s+quốc\s+gia)", t, flags=re.I):
        return True
    patterns = [
        r"^điều\s+\d+[a-z]?$",
        r"^khoản\s+\d+$",
        r"^điểm\s+[a-zđ]$",
        r"^khoản\s+\d+\s+điều\s+\d+[a-z]?$",
        r"^điểm\s+[a-zđ]\s+khoản\s+\d+$",
        r"^điểm\s+[a-zđ]\s+khoản\s+\d+\s+điều\s+\d+[a-z]?$",
        r"^phụ\s+lục\s+([ivxlcdm]+|\d+|[a-z])$",
        r"^mẫu\s+(số\s+)?[0-9a-z_.-]+$",
        r"^chương\s+([ivxlcdm]+|\d+)$",
        r"^mục\s+([ivxlcdm]+|\d+)$",
        r"^\d+/\d{4}/[a-zđ]+(?:\s*[-–]\s*[a-zđ0-9]+)*$",
        r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+(này|số\s+.+)$",
        r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+của\s+.+$",
        r"^(thông tư|nghị định|luật|quyết định|nghị quyết)\s+.+$",
    ]
    return any(re.match(p, t, flags=re.I) for p in patterns)

def coerce_semantic_label(text, label):
    """
    Correct frequent label confusions that are deterministic from the span text.
    Returns None when the span is too generic to keep.
    """
    t = collapse_ws(text).lower()
    if not t:
        return None

    generic_spans = {
        "trật tự, an toàn giao thông",
        "trật tự an toàn giao thông",
        "hành vi",
        "hoạt động",
        "hậu quả",
        "nội dung",
    }
    if t in generic_spans:
        return None

    if label == "AUTHORITY" and any(x in t for x in [
        "tổ chức hành nghề",
        "nhà cung cấp dịch vụ",
        "đơn vị kinh doanh",
        "chủ xe",
        "chủ phương tiện",
    ]):
        return "REGULATED_SUBJECT"

    if any(x in t for x in ["cơ sở đăng kiểm", "trung tâm sát hạch", "bến xe", "trạm dừng nghỉ"]):
        return "FACILITY_OR_INFRASTRUCTURE"

    if any(x in t for x in ["cổng dịch vụ công", "ứng dụng", "hệ thống", "phần mềm", "máy chủ", "thiết bị"]):
        if label in {"AUTHORITY", "FACILITY_OR_INFRASTRUCTURE", "PROCEDURE", "TECHNICAL_REQUIREMENT"}:
            return "EQUIPMENT_OR_SYSTEM"

    if label == "PLAN_OR_PROJECT" and any(x in t for x in ["quy chuẩn", "tiêu chuẩn"]):
        return "TECHNICAL_REQUIREMENT"

    if label == "CONSEQUENCE_OR_HARM" and "hành vi" in t and not any(
        x in t for x in ["hậu quả", "thiệt hại", "tai nạn", "tổn hại", "thương tích"]
    ):
        return "VIOLATION_OR_BEHAVIOR"

    return label

def dedupe_entities(entities):
    best = {}
    for e in entities:
        key = (e.get("label"), e.get("text"), e.get("start"), e.get("end"))
        if key not in best or (e.get("confidence") or 0) > (best[key].get("confidence") or 0):
            best[key] = e
    return list(best.values())
