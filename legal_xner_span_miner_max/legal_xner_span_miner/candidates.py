from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import (
    LABELS,
    collapse_ws,
    ensure_dir,
    find_sentence_package_dirs,
    is_reference_like,
    is_too_generic,
    log,
    normalize_surface,
    normalize_key,
    read_jsonl,
    stable_id,
    token_count,
    write_json,
    write_jsonl,
)


DOMAIN_KEYWORDS_BY_LABEL = {
    "BEHAVIOR": [
        "không", "vượt", "đi ngược", "chở quá", "quá tốc độ", "nồng độ cồn", "ma túy",
        "không chấp hành", "không đội", "không cài", "điều khiển", "dừng xe", "đỗ xe",
        "sử dụng", "đua xe", "lạng lách", "đánh võng"
    ],
    "VEHICLE": ["xe", "mô tô", "gắn máy", "ô tô", "rơ moóc", "sơ mi rơ moóc", "máy kéo", "xe máy chuyên dùng"],
    "ACTOR": ["người điều khiển", "người lái", "chủ xe", "chủ phương tiện", "cảnh sát", "sát hạch viên", "thí sinh", "cơ sở đào tạo"],
    "INFRASTRUCTURE": ["đường", "làn", "biển báo", "đèn tín hiệu", "vạch kẻ", "cao tốc", "trạm", "cầu", "hầm", "bến xe"],
    "DOCUMENT": ["giấy phép", "giấy đăng ký", "chứng nhận", "chứng chỉ", "biên bản", "hồ sơ", "phù hiệu", "sổ"],
    "VEHICLE_CONDITION_OR_EQUIPMENT": ["gương", "phanh", "đèn", "biển số", "thiết bị", "kết cấu", "lốp", "bánh", "khí thải", "niên hạn"],
    "CONDITION": ["đủ tuổi", "không có", "chưa đủ", "phù hợp", "còn hiệu lực", "hết hiệu lực"],
}


STOP_PREFIXES = {
    "và", "hoặc", "thì", "là", "của", "cho", "với", "tại", "theo", "trong",
    "khi", "nếu", "để", "về", "các", "những", "một", "này", "đó"
}

STOP_SUFFIXES = {
    "và", "hoặc", "thì", "là", "của", "cho", "với", "tại", "theo", "trong",
    "khi", "nếu", "để", "về", "các", "những", "một", "này", "đó", "sau", "trước"
}


def split_tokens_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    # Vietnamese legal text is whitespace-separated enough for span candidate mining.
    out = []
    edge_punct = ".,;:()[]{}“”\"'<>"
    for m in re.finditer(r"\S+", text):
        token = m.group(0)
        # trim punctuation around token but preserve offsets for internal punctuation.
        start, end = m.start(), m.end()
        while start < end and text[start] in edge_punct:
            start += 1
        while end > start and text[end - 1] in edge_punct:
            end -= 1
        if start < end:
            out.append((text[start:end], start, end))
    return out


def guess_labels(surface: str) -> List[str]:
    s = normalize_surface(surface)
    labels = []
    for label, kws in DOMAIN_KEYWORDS_BY_LABEL.items():
        if any(k in s for k in kws):
            labels.append(label)

    # Strong heuristics to avoid weird multi-labeling.
    if "giấy phép lái xe" in s or "giấy đăng ký" in s or "chứng nhận" in s:
        if "DOCUMENT" not in labels:
            labels.append("DOCUMENT")
    if s.startswith("không có giấy phép lái xe") or "chưa đủ tuổi" in s:
        if "CONDITION" not in labels:
            labels.append("CONDITION")
    if s.startswith("không ") and any(k in s for k in ["mũ", "chấp hành", "cài quai", "gương", "giấy phép"]):
        if "BEHAVIOR" not in labels and "gương" not in s and "giấy phép" not in s:
            labels.append("BEHAVIOR")

    return labels[:3]


def candidate_is_valid(surface: str, max_tokens: int = 14) -> bool:
    s = normalize_surface(surface)
    if not s or is_reference_like(s) or is_too_generic(s):
        return False
    n = token_count(s)
    if n < 2 or n > max_tokens:
        return False
    parts = s.split()
    if parts[0] in STOP_PREFIXES or parts[-1] in STOP_SUFFIXES:
        return False
    # Avoid money/range/pure numeric.
    if re.fullmatch(r"[\d\s.,/%\-]+", s):
        return False
    # Avoid sentence fragments ending with punctuation-like legal connectors.
    if s.endswith(("sau đây", "như sau", "quy định")):
        return False
    return True


def generate_candidates_for_text(text: str, max_ngram: int = 14) -> List[Dict[str, Any]]:
    toks = split_tokens_with_offsets(text)
    out = []
    seen = set()

    for i in range(len(toks)):
        for j in range(i + 2, min(len(toks), i + max_ngram) + 1):
            surface = text[toks[i][1]:toks[j - 1][2]]
            surface_norm = normalize_surface(surface)
            if not candidate_is_valid(surface_norm, max_tokens=max_ngram):
                continue
            labels = guess_labels(surface_norm)
            if not labels:
                continue
            key = (normalize_key(surface_norm), toks[i][1], toks[j - 1][2])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "surface": surface_norm,
                "start": toks[i][1],
                "end": toks[j - 1][2],
                "labels": labels,
            })
    return out


def collect_span_candidates(
    sentences_root: str | Path,
    output_dir: str | Path,
    max_sentences: int | None = None,
    max_ngram: int = 14,
    include_path_text: bool = True,
    min_surface_count: int = 1,
    progress_every: int = 5000,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    all_candidates = []
    surface_stats = {}

    sentence_count = 0
    for pkg_dir in find_sentence_package_dirs(sentences_root):
        log(f"[xner:candidates] scanning package {pkg_dir.name}")
        for row in read_jsonl(pkg_dir / "sentences.jsonl"):
            if max_sentences and sentence_count >= max_sentences:
                break
            sentence_count += 1
            if progress_every and sentence_count % progress_every == 0:
                log(
                    "[xner:candidates] "
                    f"sentences={sentence_count} unique_candidates={len(surface_stats)}"
                )

            text_items = [("text", row.get("text") or "")]
            if include_path_text:
                text_items.append(("path_text", row.get("path_text") or ""))

            for source_field, text in text_items:
                if not text:
                    continue
                mining_texts = [text]
                if source_field == "path_text":
                    mining_texts = [collapse_ws(part) for part in text.split(">") if collapse_ws(part)]

                for mining_text in mining_texts:
                    for c in generate_candidates_for_text(mining_text, max_ngram=max_ngram):
                        for label in c["labels"]:
                            key = (normalize_key(c["surface"]), label)
                            if key not in surface_stats:
                                surface_stats[key] = {
                                    "candidate_id": stable_id(label, normalize_key(c["surface"]), prefix="cand"),
                                    "surface": c["surface"],
                                    "normalized_key": normalize_key(c["surface"]),
                                    "label": label,
                                    "count": 0,
                                    "examples": [],
                                }
                            st = surface_stats[key]
                            st["count"] += 1
                            if len(st["examples"]) < 5:
                                st["examples"].append({
                                    "sentence_id": row.get("sentence_id"),
                                    "passage_id": row.get("passage_id"),
                                    "package_id": row.get("package_id"),
                                    "document_number": row.get("document_number"),
                                    "source_field": source_field,
                                    "start": c["start"],
                                    "end": c["end"],
                                    "text": mining_text,
                                    "path_text": row.get("path_text"),
                                })

        if max_sentences and sentence_count >= max_sentences:
            break

    rows = [v for v in surface_stats.values() if v["count"] >= min_surface_count]
    rows.sort(key=lambda x: (-x["count"], x["label"], x["surface"]))
    write_jsonl(output_dir / "span_candidates.jsonl", rows)

    summary = {
        "sentence_count_scanned": sentence_count,
        "candidate_surface_count": len(rows),
        "min_surface_count": min_surface_count,
        "include_path_text": include_path_text,
        "output": str(output_dir / "span_candidates.jsonl"),
    }
    write_json(output_dir / "candidate_summary.json", summary)
    log(
        "[xner:candidates] completed "
        f"sentences={sentence_count} candidates={len(rows)}"
    )
    return summary
