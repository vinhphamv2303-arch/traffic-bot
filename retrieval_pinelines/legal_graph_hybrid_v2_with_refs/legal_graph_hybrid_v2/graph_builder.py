
from __future__ import annotations
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any
from .common import ensure_dir, read_jsonl, write_json, write_jsonl, stable_id

REFERENCE_BASE_WEIGHT = {
    "point": 1.00,
    "clause": 0.90,
    "article": 0.80,
    "form": 0.75,
    "appendix": 0.70,
    "legal_document": 0.50,
    "document": 0.50,
    "unknown": 0.50,
}

def node(node_id, node_type, **attrs): return {"id":node_id, "type":node_type, **attrs}
def edge(src,dst,edge_type,weight=1.0,**attrs): return {"source":src,"target":dst,"type":edge_type,"weight":round(float(weight),6),**attrs}

def _unit_to_node_id(unit_id: str|None) -> str|None:
    if not unit_id: return None
    return f"passage:{unit_id}"

def _ref_weight(r: dict[str, Any]) -> float:
    mtype = str(r.get("mention_type") or r.get("reference_type") or "unknown")
    base = REFERENCE_BASE_WEIGHT.get(mtype, 0.60)
    try:
        conf = float(r.get("confidence", r.get("selected_score", 1.0)) or 1.0)
    except Exception:
        conf = 1.0
    conf = max(0.0, min(conf, 1.0))
    return round(base * conf, 6)

def load_reference_edges(references_file: str|Path|None, known_passage_ids: set[str], create_missing_target_nodes: bool=True):
    if not references_file:
        return [], []
    p = Path(references_file)
    if not p.exists():
        raise FileNotFoundError(f"references_file does not exist: {p}")

    ref_edges, missing_nodes = [], {}
    seen = set()
    total = resolved = skipped = target_missing = 0

    for r in read_jsonl(p):
        total += 1
        status = r.get("status")
        if status and status != "resolved":
            skipped += 1
            continue

        src = r.get("source_unit_id") or r.get("source_id") or r.get("source_passage_id")
        tgt = r.get("selected_target_id") or r.get("target_unit_id") or r.get("target_id") or r.get("target_passage_id")
        if not src or not tgt:
            skipped += 1
            continue

        src_node = _unit_to_node_id(src)
        tgt_node = _unit_to_node_id(tgt)
        if not src_node or not tgt_node:
            skipped += 1
            continue

        if tgt not in known_passage_ids:
            target_missing += 1
            if create_missing_target_nodes:
                missing_nodes[tgt_node] = node(
                    tgt_node, "passage",
                    passage_id=tgt,
                    is_reference_only=True,
                    selected_target_type=r.get("selected_target_type"),
                )
            else:
                skipped += 1
                continue

        w = _ref_weight(r)
        key = (src_node, tgt_node, r.get("mention_type"), r.get("raw"))
        if key in seen:
            continue
        seen.add(key)
        ref_edges.append(edge(
            src_node,
            tgt_node,
            "REFERENCES",
            weight=w,
            raw=r.get("raw"),
            mention_type=r.get("mention_type"),
            selected_target_type=r.get("selected_target_type"),
            selected_score=r.get("selected_score"),
            confidence=r.get("confidence"),
            resolver_status=r.get("status"),
        ))
        resolved += 1

    stats = {
        "reference_file": str(p),
        "reference_rows": total,
        "reference_edges": len(ref_edges),
        "reference_rows_resolved_or_usable": resolved,
        "reference_rows_skipped": skipped,
        "reference_targets_missing_from_passages": target_missing,
        "reference_missing_target_nodes_created": len(missing_nodes),
    }
    return ref_edges, list(missing_nodes.values()), stats

def build_graph(
    entity_links_dir: str|Path,
    output_dir: str|Path,
    references_file: str|Path|None=None,
    min_cooccur_weight: float=0.05,
    max_entity_degree_for_cooccur: int=2500,
    create_missing_reference_targets: bool=True,
) -> dict[str, Any]:
    entity_links_dir, output_dir = Path(entity_links_dir), ensure_dir(output_dir)
    mentions = list(read_jsonl(entity_links_dir/"entity_mentions.jsonl"))
    entities = list(read_jsonl(entity_links_dir/"entities.jsonl"))
    sentences = list(read_jsonl(entity_links_dir/"sentences.jsonl"))

    passage_info, pew, pesrc, p_sent_count, doc_passages = {}, defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(set)), Counter(), defaultdict(set)
    for s in sentences:
        pid = s.get("passage_id") or s.get("source_unit_id") or s.get("sentence_id")
        if not pid: continue
        passage_info.setdefault(pid, {"passage_id":pid,"package_id":s.get("package_id"),"document_id":s.get("document_id"),
            "document_number":s.get("document_number"),"path_text":s.get("path_text"),"text_sample":s.get("text")})
        p_sent_count[pid] += 1
        doc = s.get("document_number") or s.get("document_id") or s.get("package_id")
        if doc: doc_passages[doc].add(pid)

    for m in mentions:
        pid, eid = m.get("passage_id") or m.get("source_unit_id") or m.get("sentence_id"), m.get("entity_id")
        if not pid or not eid: continue
        w = float(m.get("graph_weight",0) or 0)
        pew[pid][eid] += w
        pesrc[pid][eid].add(m.get("source","unknown"))

    nodes, edges = [], []
    for doc,pids in sorted(doc_passages.items()):
        nodes.append(node(f"doc:{doc}","document",document_number=doc,passage_count=len(pids)))

    for pid,info in sorted(passage_info.items()):
        nodes.append(node(f"passage:{pid}","passage",passage_id=pid,package_id=info.get("package_id"),document_id=info.get("document_id"),
            document_number=info.get("document_number"),path_text=info.get("path_text"),text_sample=info.get("text_sample"),
            sentence_count=p_sent_count[pid],entity_count=len(pew.get(pid,{})),is_reference_only=False))
        doc = info.get("document_number") or info.get("document_id") or info.get("package_id")
        if doc: edges.append(edge(f"doc:{doc}", f"passage:{pid}", "CONTAINS", 1.0))

    known_passage_ids = set(passage_info.keys())

    ent_deg = Counter()
    for pid,ew in pew.items():
        for eid in ew: ent_deg[eid] += 1

    for e in entities:
        eid=e["entity_id"]
        nodes.append(node(f"entity:{eid}","entity",entity_id=eid,label=e.get("label"),canonical=e.get("canonical"),norm=e.get("norm"),passage_degree=ent_deg.get(eid,0)))

    for pid,ew in pew.items():
        for eid,raw_w in ew.items():
            if raw_w <= 0: continue
            edges.append(edge(f"passage:{pid}", f"entity:{eid}", "HAS_ENTITY", math.log1p(raw_w), raw_weight=round(float(raw_w),6), sources=sorted(pesrc[pid][eid])))

    co = defaultdict(float)
    for pid,ew in pew.items():
        valid = [(eid,w) for eid,w in ew.items() if ent_deg[eid] <= max_entity_degree_for_cooccur and w > 0]
        if len(valid) > 40: valid = sorted(valid, key=lambda x:-x[1])[:40]
        for (e1,w1),(e2,w2) in combinations(valid,2):
            a,b = sorted([e1,e2]); co[(a,b)] += min(w1,w2)

    for (e1,e2),w in co.items():
        if w >= min_cooccur_weight:
            edges.append(edge(f"entity:{e1}", f"entity:{e2}", "COOCCURS", math.log1p(w), raw_weight=round(float(w),6)))

    ref_stats = {}
    if references_file:
        ref_edges, missing_ref_nodes, ref_stats = load_reference_edges(
            references_file,
            known_passage_ids,
            create_missing_target_nodes=create_missing_reference_targets,
        )
        # Avoid duplicate nodes by id.
        existing_node_ids = {n["id"] for n in nodes}
        for n in missing_ref_nodes:
            if n["id"] not in existing_node_ids:
                nodes.append(n)
                existing_node_ids.add(n["id"])
        edges.extend(ref_edges)

    write_jsonl(output_dir/"nodes.jsonl", nodes)
    write_jsonl(output_dir/"edges.jsonl", edges)

    summary = {
        "entity_links_dir": str(entity_links_dir),
        "references_file": str(references_file) if references_file else None,
        "output_dir": str(output_dir),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "passage_node_count": sum(1 for n in nodes if n["type"]=="passage"),
        "entity_node_count": sum(1 for n in nodes if n["type"]=="entity"),
        "document_node_count": sum(1 for n in nodes if n["type"]=="document"),
        "reference_only_passage_node_count": sum(1 for n in nodes if n.get("is_reference_only")),
        "has_entity_edge_count": sum(1 for e in edges if e["type"]=="HAS_ENTITY"),
        "cooccur_edge_count": sum(1 for e in edges if e["type"]=="COOCCURS"),
        "contains_edge_count": sum(1 for e in edges if e["type"]=="CONTAINS"),
        "references_edge_count": sum(1 for e in edges if e["type"]=="REFERENCES"),
        "min_cooccur_weight": min_cooccur_weight,
        "max_entity_degree_for_cooccur": max_entity_degree_for_cooccur,
        **ref_stats,
    }
    write_json(output_dir/"graph_summary.json", summary)
    return summary
