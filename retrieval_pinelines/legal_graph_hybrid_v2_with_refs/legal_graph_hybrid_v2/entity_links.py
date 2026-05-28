
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
from .common import *

def _row_key(row: dict[str, Any]) -> str:
    return row.get("sentence_id") or row.get("source_unit_id") or row.get("passage_id") or ""

def load_sentence_rows(root: str|Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for f in iter_sentence_entity_files(root):
        for r in read_jsonl(f):
            k = _row_key(r)
            if k: rows[k] = r
    return rows

def normalize_entity(e: dict[str, Any], source: str, row: dict[str, Any], inherited: bool=False) -> dict[str, Any]|None:
    label = e.get("label")
    if label not in LABELS: return None
    text = e.get("text") or e.get("surface") or e.get("canonical")
    canonical = e.get("canonical") or text
    if not text: return None
    try:
        start, end = int(e.get("start", -1)), int(e.get("end", -1))
    except Exception:
        start, end = -1, -1
    if not inherited and (start < 0 or end <= start): return None
    try: conf = float(e.get("confidence", e.get("score", 1.0)) or 1.0)
    except Exception: conf = 1.0
    scope = "inherited" if inherited else "direct"
    w = base_weight(label, text, source, confidence=conf, scope=scope)
    eid = entity_id(label, canonical, text)
    sid = row.get("sentence_id") or row.get("source_unit_id") or stable_id(row.get("passage_id"), row.get("text"), prefix="sent")
    mid = stable_id(sid, eid, source, str(start), str(end), text, prefix="ment")
    return {
        "mention_id": mid, "entity_id": eid, "label": label, "text": text, "canonical": canonical,
        "norm": normalize_text(canonical or text), "start": start, "end": end, "confidence": conf,
        "source": source, "scope": scope, "graph_weight": w,
        "sentence_id": sid, "passage_id": row.get("passage_id"), "source_unit_id": row.get("source_unit_id"),
        "package_id": row.get("package_id"), "document_id": row.get("document_id"),
        "document_number": row.get("document_number"), "path_text": row.get("path_text"),
        "sentence_text": row.get("text"),
    }

def merge_direct_mentions(gaz: list[dict], gli: list[dict]) -> list[dict]:
    candidates, used_gli = [], set()
    for g in gaz:
        best_i, best_score = None, -1
        for i,p in enumerate(gli):
            if i in used_gli: continue
            if g["label"] == p["label"] and span_overlap(g,p):
                ov = min(int(g["end"]),int(p["end"])) - max(int(g["start"]),int(p["start"]))
                score = ov + 0.01*max(span_len(g), span_len(p))
                if score > best_score: best_i, best_score = i, score
        if best_i is not None:
            p = gli[best_i]; used_gli.add(best_i)
            chosen = g if span_len(g) >= span_len(p) else p
            m = dict(chosen)
            m["source"] = "hybrid_agree"
            m["confidence"] = max(float(g.get("confidence",1.0)), float(p.get("confidence",0.0)))
            m["graph_weight"] = base_weight(m["label"], m["text"], "hybrid_agree", m["confidence"], "direct")
            m["mention_id"] = stable_id(m["sentence_id"], m["entity_id"], "hybrid_agree", str(m["start"]), str(m["end"]), m["text"], prefix="ment")
            candidates.append(m)
        else:
            candidates.append(g)
    for i,p in enumerate(gli):
        if i not in used_gli: candidates.append(p)
    def priority(m):
        agree = 2.0 if m.get("source")=="hybrid_agree" else 0.0
        sb = {"gazetteer":1.0, "gliner":0.7, "hybrid_agree":1.2}.get(m.get("source"),0.5)
        return agree + sb + float(m.get("graph_weight",0)) + min(span_len(m),80)*0.001
    selected = []
    for c in sorted(candidates, key=lambda x:(-priority(x), int(x.get("start",-1)), -span_len(x))):
        if any(span_overlap(c,s) for s in selected): continue
        selected.append(c)
    return sorted(selected, key=lambda x:(x.get("sentence_id") or "", int(x.get("start",-1)), int(x.get("end",-1)), x.get("label","")))

def build_hybrid_entity_links(gazetteer_entities_root: str|Path, gliner_entities_root: str|Path, output_dir: str|Path, include_inherited: bool=True) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    gaz_rows, gli_rows = load_sentence_rows(gazetteer_entities_root), load_sentence_rows(gliner_entities_root)
    keys = sorted(set(gaz_rows) | set(gli_rows))
    all_mentions, entity_rows, sentence_rows = [], {}, []
    by_label, by_source, by_scope = Counter(), Counter(), Counter()
    for k in keys:
        row = gaz_rows.get(k) or gli_rows.get(k)
        gaz_direct = [m for e in (gaz_rows.get(k,{}).get("entities") or []) if (m:=normalize_entity(e,"gazetteer",gaz_rows[k],False))]
        gli_direct = [m for e in (gli_rows.get(k,{}).get("entities") or []) if (m:=normalize_entity(e,"gliner",gli_rows[k],False))]
        direct = merge_direct_mentions(gaz_direct, gli_direct)
        inherited = []
        if include_inherited and k in gaz_rows:
            inherited = [m for e in (gaz_rows[k].get("inherited_entities") or []) if (m:=normalize_entity(e,"gazetteer",gaz_rows[k],True))]
        mentions = direct + inherited
        for m in mentions:
            all_mentions.append(m)
            by_label[m["label"]] += 1; by_source[m["source"]] += 1; by_scope[m["scope"]] += 1
            entity_rows[m["entity_id"]] = {"entity_id":m["entity_id"],"label":m["label"],"canonical":m["canonical"],"norm":m["norm"]}
        sentence_rows.append({
            "sentence_id": row.get("sentence_id") or row.get("source_unit_id"), "passage_id": row.get("passage_id"),
            "source_unit_id": row.get("source_unit_id"), "package_id": row.get("package_id"), "document_id": row.get("document_id"),
            "document_number": row.get("document_number"), "path_text": row.get("path_text"), "text": row.get("text"),
            "direct_entity_count": len(direct), "inherited_entity_count": len(inherited),
            "entity_ids": sorted(set(m["entity_id"] for m in mentions)),
        })
    write_jsonl(output_dir/"entity_mentions.jsonl", all_mentions)
    write_jsonl(output_dir/"entities.jsonl", entity_rows.values())
    write_jsonl(output_dir/"sentences.jsonl", sentence_rows)
    summary = {
        "gazetteer_entities_root": str(gazetteer_entities_root), "gliner_entities_root": str(gliner_entities_root),
        "output_dir": str(output_dir), "sentence_count": len(sentence_rows), "entity_count": len(entity_rows),
        "mention_count": len(all_mentions), "direct_mention_count": sum(1 for m in all_mentions if m["scope"]=="direct"),
        "inherited_mention_count": sum(1 for m in all_mentions if m["scope"]=="inherited"),
        "by_label": dict(sorted(by_label.items())), "by_source": dict(sorted(by_source.items())),
        "by_scope": dict(sorted(by_scope.items())), "include_inherited": include_inherited,
    }
    write_json(output_dir/"entity_links_summary.json", summary)
    return summary
