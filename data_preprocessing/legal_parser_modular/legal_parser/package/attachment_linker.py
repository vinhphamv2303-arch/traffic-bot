from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..attachments.common import collapse_ws, read_jsonl, slugify, strip_vietnamese_accents, write_json


FORM_RE = re.compile(r"\b[Mm](?:ẫu|au)\s+(?:(?:số|so)\s+)?(?P<num>[0-9A-Za-zĐđ_.-]+)", re.UNICODE)
APPENDIX_RE = re.compile(r"\b[Pp](?:hụ|hu)\s+[Ll](?:ục|uc)\s+(?P<label>[IVXLCDM]+|\d+|[A-Z]+)", re.UNICODE)

STOPWORDS = {
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
    "phai",
    "phu",
    "quy",
    "so",
    "tai",
    "theo",
    "trong",
    "va",
    "ve",
    "voi",
    "doi",
    "dinh",
    "luc",
}


def infer_appendix_form_links(
    *,
    package_out_dir: Path,
    inventory: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Infer parent appendix metadata for standalone "Mẫu số ..." files.

    Many source datasets store each form as a separate file named only
    "Mẫu số 01/02/03", while the parent appendix is only visible in the
    main legal text, e.g. "Mẫu số 01, Mẫu số 02 Phụ lục II".  This step
    extracts those main-body references and records auditable metadata in
    package_inventory.json and each attachment.json.
    """
    package_out_dir = Path(package_out_dir)
    attachments = inventory.get("attachments") or []
    form_attachments = [a for a in attachments if _is_form_attachment(a)]
    if not form_attachments:
        inventory["appendix_form_linking"] = _summary(0, 0, 0)
        inventory["inferred_appendix_groups"] = []
        return inventory

    contexts = _load_unique_main_contexts(package_out_dir / "main" / "ref_mentions.jsonl")
    evidences = []
    for ctx in contexts:
        evidences.extend(_extract_evidences(ctx))

    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ev in evidences:
        by_key.setdefault((ev["appendix_label"], ev["form_number_norm"]), []).append(ev)

    forms_by_number: Dict[str, List[Dict[str, Any]]] = {}
    for att in form_attachments:
        num = _normalize_form_number(f"{att.get('label') or ''} {att.get('title') or ''}")
        if num:
            forms_by_number.setdefault(num, []).append(att)

    linked = 0
    unresolved = 0
    decisions = []
    for (appendix_label, form_number), group_evidence in sorted(by_key.items()):
        candidates = forms_by_number.get(form_number, [])
        if not candidates:
            unresolved += 1
            decisions.append({
                "appendix_label": appendix_label,
                "form_number_norm": form_number,
                "status": "missing_form_attachment",
                "evidence_count": len(group_evidence),
            })
            continue

        ranked = _rank_candidates(package_out_dir, candidates, group_evidence)
        if not ranked:
            unresolved += 1
            continue

        top = ranked[0]
        next_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        confident = top["score"] >= 0.68 and (len(ranked) == 1 or top["score"] >= next_score + 0.04)

        decision = {
            "appendix_label": appendix_label,
            "form_number_norm": form_number,
            "status": "linked" if confident else "ambiguous",
            "selected_attachment_id": top["attachment"].get("attachment_id") if confident else None,
            "selected_title": top["attachment"].get("title") if confident else None,
            "confidence": round(top["score"], 4),
            "evidence_count": len(group_evidence),
            "evidence": _compact_evidence(group_evidence),
            "alternatives": [
                {
                    "attachment_id": item["attachment"].get("attachment_id"),
                    "title": item["attachment"].get("title"),
                    "score": round(item["score"], 4),
                }
                for item in ranked[:5]
            ],
        }
        decisions.append(decision)

        if not confident:
            unresolved += 1
            continue

        linked += 1
        _apply_parent_link(top["attachment"], decision)

    groups = _build_inferred_appendix_groups(inventory, attachments)
    inventory["inferred_appendix_groups"] = groups
    inventory["appendix_form_linking"] = _summary(len(by_key), linked, unresolved, decisions)

    _write_attachment_metadata(package_out_dir, attachments)

    if logger:
        logger.info(
            "Appendix form linker | inferred_pairs=%s | linked=%s | unresolved=%s | groups=%s",
            len(by_key),
            linked,
            unresolved,
            len(groups),
        )

    return inventory


def _is_form_attachment(att: Dict[str, Any]) -> bool:
    kind = att.get("attachment_kind") or att.get("attachment_type")
    return kind in {"form", "appendix_form"}


def _load_unique_main_contexts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        source_unit_id = row.get("source_unit_id")
        text = row.get("source_text") or ""
        if not source_unit_id or not text:
            continue
        out.setdefault(source_unit_id, {
            "source_unit_id": source_unit_id,
            "source_path_text": row.get("source_path_text") or "",
            "source_text": text,
        })
    return list(out.values())


def _extract_evidences(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = ctx.get("source_text") or ""
    forms = [
        {
            "kind": "form",
            "start": m.start(),
            "end": m.end(),
            "raw": collapse_ws(m.group(0)),
            "form_number_norm": _normalize_form_number(m.group("num")),
        }
        for m in FORM_RE.finditer(text)
        if _normalize_form_number(m.group("num"))
    ]
    appendices = [
        {
            "kind": "appendix",
            "start": m.start(),
            "end": m.end(),
            "raw": collapse_ws(m.group(0)),
            "appendix_label": f"Phụ lục {m.group('label').upper()}",
        }
        for m in APPENDIX_RE.finditer(text)
    ]
    if not forms or not appendices:
        return []

    evidences: List[Dict[str, Any]] = []
    previous_appendix_end = 0
    for app in appendices:
        nearby_forms = [
            f for f in forms
            if previous_appendix_end <= f["start"] < app["start"] and app["start"] - f["start"] <= 220
        ]
        if not nearby_forms:
            next_appendix_start = _next_appendix_start(appendices, app["end"], len(text))
            nearby_forms = [
                f for f in forms
                if app["end"] <= f["start"] < next_appendix_start and f["start"] - app["end"] <= 120
            ]

        for form in nearby_forms:
            start = max(0, min(form["start"], app["start"]) - 180)
            end = min(len(text), max(form["end"], app["end"]) + 180)
            evidences.append({
                "appendix_label": app["appendix_label"],
                "form_number_norm": form["form_number_norm"],
                "raw_form": form["raw"],
                "raw_appendix": app["raw"],
                "source_unit_id": ctx.get("source_unit_id"),
                "source_path_text": ctx.get("source_path_text"),
                "source_text": text,
                "evidence_text": collapse_ws(text[start:end]),
            })
        previous_appendix_end = app["end"]

    return evidences


def _next_appendix_start(appendices: List[Dict[str, Any]], after: int, default: int) -> int:
    for app in appendices:
        if app["start"] >= after:
            return app["start"]
    return default


def _rank_candidates(package_out_dir: Path, candidates: List[Dict[str, Any]], evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query = _evidence_query_text(evidences)
    ranked = []
    for att in candidates:
        title_text = " ".join([
            att.get("title") or "",
            Path(att.get("source_file") or "").stem,
            str(att.get("parsed_dir") or ""),
        ])
        body_text = _attachment_body_text(package_out_dir, att)
        title_score = _term_overlap_score(query, title_text)
        body_score = _term_overlap_score(query, f"{title_text} {body_text}")

        score = 0.45 + 0.42 * title_score + 0.13 * body_score
        if _canonical_key(att.get("title") or "") and _canonical_key(att.get("title") or "") in _canonical_key(query):
            score = max(score, 0.94)
        if len(candidates) == 1:
            score = max(score, 0.92)
        ranked.append({"attachment": att, "score": min(score, 0.99)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def _evidence_query_text(evidences: List[Dict[str, Any]]) -> str:
    parts = []
    for ev in evidences[:12]:
        parts.append(ev.get("source_path_text") or "")
        parts.append(ev.get("evidence_text") or "")
    return " ".join(parts)


def _attachment_body_text(package_out_dir: Path, att: Dict[str, Any], *, max_chars: int = 5000) -> str:
    rel = att.get("parsed_dir")
    if not rel:
        return ""
    units_path = package_out_dir / rel / "units.jsonl"
    if not units_path.exists():
        return ""
    parts = []
    total = 0
    for row in read_jsonl(units_path):
        value = row.get("content") or ""
        if not value:
            continue
        parts.append(value)
        total += len(value)
        if total >= max_chars:
            break
    return " ".join(parts)


def _apply_parent_link(att: Dict[str, Any], decision: Dict[str, Any]) -> None:
    label = decision["appendix_label"]
    labels = list(att.get("parent_appendix_labels") or [])
    if label not in labels:
        labels.append(label)
    labels.sort(key=_appendix_sort_key)
    att["parent_appendix_labels"] = labels
    if len(labels) == 1:
        att["parent_appendix_label"] = labels[0]
    else:
        att["parent_appendix_label"] = None

    inferences = list(att.get("parent_appendix_inferences") or [])
    inferences = [
        x for x in inferences
        if not (
            x.get("appendix_label") == label
            and x.get("form_number_norm") == decision.get("form_number_norm")
        )
    ]
    inferences.append({
        "appendix_label": label,
        "form_number_norm": decision.get("form_number_norm"),
        "method": "main_body_reference_context",
        "confidence": decision.get("confidence"),
        "evidence_count": decision.get("evidence_count"),
        "evidence": decision.get("evidence"),
        "alternatives": decision.get("alternatives"),
    })
    att["parent_appendix_inferences"] = sorted(inferences, key=lambda x: _appendix_sort_key(x.get("appendix_label") or ""))


def _build_inferred_appendix_groups(inventory: Dict[str, Any], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    main = inventory.get("main_document") or {}
    package_id = inventory.get("package_id")
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for att in attachments:
        for label in att.get("parent_appendix_labels") or []:
            by_label.setdefault(label, []).append(att)

    groups = []
    for label, children in sorted(by_label.items(), key=lambda x: _appendix_sort_key(x[0])):
        confidences = [
            inf.get("confidence")
            for att in children
            for inf in att.get("parent_appendix_inferences") or []
            if inf.get("appendix_label") == label and isinstance(inf.get("confidence"), (int, float))
        ]
        group_id = f"{main.get('document_id') or slugify(package_id or 'package')}.{slugify(label)}"
        groups.append({
            "package_id": package_id,
            "document_id": main.get("document_id"),
            "document_number": main.get("document_number"),
            "attachment_id": group_id,
            "attachment_slug": slugify(label),
            "attachment_kind": "inferred_appendix_group",
            "label": label,
            "title": f"{label} (inferred form group)",
            "source_file": None,
            "parser": "appendix_form_linker",
            "parsed_dir": None,
            "child_attachment_ids": [att.get("attachment_id") for att in children],
            "child_count": len(children),
            "inference_method": "main_body_reference_context",
            "inference_confidence": round(min(confidences), 4) if confidences else None,
        })
    return groups


def _write_attachment_metadata(package_out_dir: Path, attachments: List[Dict[str, Any]]) -> None:
    for att in attachments:
        rel = att.get("parsed_dir")
        if not rel:
            continue
        path = package_out_dir / rel / "attachment.json"
        if path.exists():
            write_json(path, att)


def _summary(total_pairs: int, linked: int, unresolved: int, decisions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "method": "main_body_reference_context",
        "total_pairs": total_pairs,
        "linked_pairs": linked,
        "unresolved_pairs": unresolved,
        "decisions": decisions or [],
    }


def _compact_evidence(evidences: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    out = []
    for ev in evidences[:limit]:
        out.append({
            "source_unit_id": ev.get("source_unit_id"),
            "source_path_text": ev.get("source_path_text"),
            "text": ev.get("evidence_text"),
        })
    return out


def _normalize_form_number(value: Any) -> str:
    m = re.search(r"\d+", str(value or ""))
    return str(int(m.group(0))) if m else ""


def _appendix_sort_key(label: str) -> Tuple[int, str]:
    token = (label or "").split()[-1] if label else ""
    if token.isdigit():
        return int(token), token
    roman = _roman_to_int(token)
    if roman:
        return roman, token
    return 10_000, token


def _roman_to_int(token: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed((token or "").upper()):
        val = values.get(ch, 0)
        if not val:
            return 0
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _term_overlap_score(query_text: str, candidate_text: str) -> float:
    query_terms = set(_normalized_terms(query_text))
    candidate_terms = set(_normalized_terms(candidate_text))
    if not query_terms or not candidate_terms:
        return 0.0
    candidate_key = _canonical_key(candidate_text)
    query_key = _canonical_key(query_text)
    if candidate_key and candidate_key in query_key:
        return 1.0
    overlap = query_terms & candidate_terms
    if not overlap:
        return 0.0
    candidate_coverage = len(overlap) / len(candidate_terms)
    query_coverage = len(overlap) / len(query_terms)
    return 0.75 * candidate_coverage + 0.25 * query_coverage


def _normalized_terms(text: str) -> List[str]:
    text = strip_vietnamese_accents(text or "", keep_dd=False).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    terms = []
    for term in text.split():
        if len(term) < 2 or term.isdigit() or term in STOPWORDS:
            continue
        terms.append(term)
    return terms


def _canonical_key(text: str) -> str:
    text = strip_vietnamese_accents(text or "", keep_dd=False).lower()
    return re.sub(r"[^a-z0-9]+", "", text)
