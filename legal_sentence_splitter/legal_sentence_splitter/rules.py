\
import re
from .utils import collapse_ws

ABBR_PATTERNS = [
    r"\bTT-BCA\b", r"\bTT-BXD\b", r"\bTT-BGTVT\b", r"\bTT-BYT\b", r"\bTT-BQP\b",
    r"\bNĐ-CP\b", r"\bND-CP\b", r"\bQH\d+\b", r"\bQCVN\b", r"\bTCVN\b",
    r"\bTP\.\s*[A-ZÀ-ỸĐ]", r"\bTS\.\s*[A-ZÀ-ỸĐ]", r"\bThS\.\s*[A-ZÀ-ỸĐ]",
    r"\bPGS\.\s*TS\.", r"\bGS\.\s*TS\.",
]

def legal_sentence_split(text):
    text = normalize_text(text)
    if not text:
        return []

    lines = [x.strip() for x in re.split(r"[\r\n]+", text) if x.strip()]
    out = []
    if len(lines) > 1:
        for line in lines:
            out.extend(split_one_line(line))
    else:
        out.extend(split_one_line(text))
    return cleanup(out)

def normalize_text(text):
    text = (text or "").replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text.strip()

def split_one_line(line):
    line = line.strip()
    if not line:
        return []

    protected, mapping = protect_periods(line)

    parts = []
    start = 0
    i = 0
    while i < len(protected):
        ch = protected[i]
        if ch in ".!?" and is_boundary(protected, i):
            parts.append(protected[start:i+1])
            start = i + 1
        i += 1
    if start < len(protected):
        parts.append(protected[start:])

    final = []
    for part in parts:
        restored = restore(part, mapping)
        final.extend(split_semicolon(restored))
    return final

def protect_periods(text):
    mapping = {}
    out = text

    # Protect numeric dots: 5.000.000, 1.2.3
    out = re.sub(r"(?<=\d)\.(?=\d)", "§NUMDOT§", out)

    # Protect abbreviations with dots.
    for pat in ABBR_PATTERNS:
        def repl(m, pat=pat):
            token = f"§ABBR{len(mapping)}§"
            mapping[token] = m.group(0)
            return token
        out = re.sub(pat, repl, out, flags=re.IGNORECASE | re.UNICODE)

    return out, mapping

def restore(text, mapping):
    text = text.replace("§NUMDOT§", ".")
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def is_boundary(text, idx):
    ch = text[idx]
    if ch not in ".!?":
        return False

    if idx + 1 < len(text) and text[idx + 1] == ".":
        return False
    if idx > 0 and text[idx - 1] == ".":
        return False
    if idx > 0 and idx + 1 < len(text) and text[idx - 1].isdigit() and text[idx + 1].isdigit():
        return False

    # Avoid splitting immediately after enumerators: "1. Nội dung", "1.2. Nội dung".
    if is_numeric_enumerator_dot(text, idx):
        return False

    j = idx + 1
    while j < len(text) and text[j] in ' "\')]}':
        j += 1
    if j >= len(text):
        return True

    nxt = text[j]
    if nxt.isupper() or nxt.isdigit() or nxt in "-–•":
        return True

    rest = text[j:j+50].lower()
    if rest.startswith((" theo ", " trường hợp ", " đối với ", " người ", " cơ quan ", " tổ chức ")):
        return True

    return False

def is_numeric_enumerator_dot(text, idx):
    prefix = text[:idx + 1]
    match = re.search(r"(?P<token>\d+[a-zA-Z]?(?:§NUMDOT§\d+[a-zA-Z]?)*\.)$", prefix)
    if not match:
        return False

    token_start = match.start("token")
    before = text[:token_start].rstrip()
    if not before:
        allowed_position = True
    elif before[-1] in "\"'“”([{:;.!?":
        allowed_position = True
    elif re.search(r"(Điều|Khoản|Mục|Chương|Phần|Bảng|Mẫu)\s*$", before, flags=re.IGNORECASE | re.UNICODE):
        allowed_position = True
    else:
        allowed_position = False
    if not allowed_position:
        return False

    j = idx + 1
    while j < len(text) and text[j].isspace():
        j += 1
    return j < len(text)

def split_semicolon(text):
    text = text.strip()
    if ";" not in text:
        return [text]
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) <= 1:
        return [text]
    meaningful = sum(1 for p in parts if len(p) >= 12)
    if meaningful >= 2:
        return [p + (";" if i < len(parts)-1 else "") for i, p in enumerate(parts)]
    return [text]

def cleanup(items):
    out = []
    seen = set()
    for item in items:
        s = collapse_ws(restore(item.strip(), {})).replace("§NUMDOT§", ".")
        if not s:
            continue
        if not re.search(r"[A-Za-z0-9À-Ỹà-ỹĐđ]", s, flags=re.UNICODE):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*\.?", s):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
