from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from .schema import REFERENCE_LIKE_PATTERNS

def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()
def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)
def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
def find_sentence_package_dirs(root: str | Path) -> List[Path]:
    p=Path(root)
    if (p/"sentences.jsonl").exists(): return [p]
    return sorted([x for x in p.iterdir() if x.is_dir() and (x/"sentences.jsonl").exists()]) if p.is_dir() else []
def chunks(items: List[Any], n: int):
    for i in range(0,len(items),n): yield items[i:i+n]
def extract_json_object(text: str) -> Dict[str, Any]:
    text=(text or "").strip()
    if text.startswith("```"):
        text=re.sub(r"^```(?:json)?", "", text, flags=re.I).strip(); text=re.sub(r"```$", "", text).strip()
    try: return json.loads(text)
    except Exception: pass
    s=text.find("{"); e=text.rfind("}")
    if s>=0 and e>s: return json.loads(text[s:e+1])
    raise ValueError("No valid JSON object found")
def find_offsets(text: str, span: str) -> Tuple[Optional[int], Optional[int]]:
    text=text or ""; span=collapse_ws(span)
    if not span: return None, None
    i=text.find(span)
    if i>=0: return i, i+len(span)
    i=text.lower().find(span.lower())
    if i>=0: return i, i+len(span)
    norm_text, index_map = normalize_with_index_map(text)
    norm_span = collapse_ws(span).lower()
    i = norm_text.lower().find(norm_span)
    if i >= 0 and index_map:
        end_i = i + len(norm_span) - 1
        if 0 <= end_i < len(index_map):
            return index_map[i], index_map[end_i] + 1
    return None, None

def normalize_with_index_map(text: str) -> Tuple[str, List[int]]:
    chars=[]; index_map=[]; previous_space=False
    for idx,ch in enumerate(text or ""):
        if ch.isspace():
            if chars and not previous_space:
                chars.append(" "); index_map.append(idx)
            previous_space=True
        else:
            chars.append(ch); index_map.append(idx); previous_space=False
    if chars and chars[-1]==" ":
        chars.pop(); index_map.pop()
    return "".join(chars), index_map
def is_reference_like(text: str) -> bool:
    t=collapse_ws(text).lower()
    return (not t) or any(re.match(p,t,flags=re.I|re.U) for p in REFERENCE_LIKE_PATTERNS)
def is_too_generic(text: str, label: str) -> bool:
    t=collapse_ws(text).lower()
    generic={
      "BEHAVIOR":{"vi phạm","vi phạm quy định","thực hiện","thực hiện theo quy định","tham gia giao thông","bị xử phạt"},
      "VEHICLE":{"xe","phương tiện","loại xe","phương tiện giao thông"},
      "ACTOR":{"người","cá nhân","tổ chức","đối tượng","người điều khiển phương tiện","cơ quan có thẩm quyền"},
      "INFRASTRUCTURE":{"đường","nơi","khu vực","vị trí","địa điểm","hệ thống","thiết bị"},
      "DOCUMENT":{"văn bản","quy định","hồ sơ","giấy tờ"},
      "VEHICLE_CONDITION_OR_EQUIPMENT":{"không đạt chuẩn","không đúng quy định","không bảo đảm","thiết bị","bộ phận"},
      "CONDITION":{"điều kiện","yêu cầu","tiêu chuẩn","quy định","phù hợp","được phép","không được"},
    }
    return (not t) or t in generic.get(label,set()) or (len(t)<=2 and label!="VEHICLE")
def dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best={}
    for e in entities:
        key=(e.get("label"),e.get("text"),e.get("start"),e.get("end")); c=e.get("confidence") or 0
        if key not in best or c>(best[key].get("confidence") or 0): best[key]=e
    return list(best.values())
def summarize_mentions(mentions: List[Dict[str, Any]]) -> Dict[str,int]:
    out={}
    for m in mentions:
        lab=m.get("label") or "UNKNOWN"; out[lab]=out.get(lab,0)+1
    return dict(sorted(out.items()))
