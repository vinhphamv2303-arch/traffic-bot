\
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

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_passage_package_dirs(passages_root):
    p = Path(passages_root)
    if (p / "passages.jsonl").exists():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_dir() and (x / "passages.jsonl").exists()])
    return []

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def make_context_text(passage):
    lines = []
    if passage.get("document_number"):
        lines.append(f"Văn bản: {passage.get('document_number')}")
    if passage.get("document_title"):
        lines.append(f"Tên văn bản: {passage.get('document_title')}")
    if passage.get("attachment_id"):
        lines.append(f"Tài liệu đính kèm: {passage.get('attachment_id')}")
    if passage.get("path_text"):
        lines.append(f"Đường dẫn pháp lý: {passage.get('path_text')}")
    if passage.get("effective_from"):
        lines.append(f"Hiệu lực từ: {passage.get('effective_from')}")
    if passage.get("ceased_from"):
        lines.append(f"Hết hiệu lực từ: {passage.get('ceased_from')}")
    if passage.get("amendment_actions"):
        actions = sorted({x.get("action_hint") for x in passage.get("amendment_actions") if x.get("action_hint")})
        if actions:
            lines.append(f"Thao tác sửa đổi/bổ sung: {', '.join(actions)}")
    return "\n".join(lines)

def build_sentence_text_for_ner(context_text, sentence):
    if not context_text:
        return sentence
    return f"{context_text}\nCâu: {sentence}".strip()
