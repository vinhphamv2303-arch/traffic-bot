from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .loaders import load_canonical_entities, load_entity_links, load_passages
from .utils import compact_text, ensure_dir, stable_id, write_json, write_jsonl


def passage_node_id(passage_id: str) -> str:
    return f"passage::{passage_id}"


def entity_node_id(entity_id: str) -> str:
    return f"entity::{entity_id}"


def document_node_id(document_id: str | None, document_number: str | None = None) -> str:
    base = document_id or document_number or "unknown_document"
    return f"document::{base}"


def normalize_entity_id(link: Dict[str, Any]) -> str:
    if link.get("entity_id"):
        return str(link["entity_id"])
    return stable_id(link.get("label") or "", link.get("canonical") or link.get("surface") or "", prefix="ent")


REFERENCE_TARGET_FIELDS = (
    "target_passage_id",
    "resolved_target_passage_id",
    "target_id",
    "target_unit_id",
    "resolved_target_unit_id",
)


def _float_or_default(value: Any, default: float = 1.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _reference_target_value(ref: Dict[str, Any]) -> Any:
    return next((ref.get(field) for field in REFERENCE_TARGET_FIELDS if ref.get(field)), None)


def _resolve_reference_target(
    ref: Dict[str, Any],
    passage_id_set: set[str],
    unit_to_passage_id: Dict[str, str],
    document_node_ids: set[str],
) -> Tuple[str | None, str | None, str | None, str | None]:
    """Return target node info, or a skipped reason when no graph node exists."""
    target = _reference_target_value(ref)
    if not target:
        return None, None, None, "missing_target_id"

    target = str(target)
    target_type = ref.get("target_type")

    if target_type == "document":
        doc_node = document_node_id(target)
        if doc_node in document_node_ids:
            return doc_node, target, "DOCUMENT", None
        return None, target, "DOCUMENT", "missing_document_node"

    passage_candidates = [
        unit_to_passage_id.get(target),
        target,
        f"{target}.passage",
        f"{target}.summary.passage",
    ]
    for candidate in passage_candidates:
        if candidate and candidate in passage_id_set:
            return passage_node_id(candidate), candidate, "PASSAGE", None

    return None, target, "PASSAGE", "missing_passage_node"


def extract_reference_edges_from_passage(
    p: Dict[str, Any],
    passage_id_set: set[str],
    unit_to_passage_id: Dict[str, str],
    document_node_ids: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Best-effort support for existing passage outgoing_refs.

    The passage builder README says passages may contain outgoing_refs/incoming_refs and
    reference_expansion_policies. This function accepts common shapes and only creates
    reference edges when a target passage id is directly available.
    """
    out = []
    skipped = []
    source_pid = p.get("passage_id")
    if not source_pid:
        return out, skipped

    refs = p.get("outgoing_refs") or []
    if not isinstance(refs, list):
        return out, skipped

    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue

        status = ref.get("status")
        raw = ref.get("mention_text") or ref.get("raw") or ref.get("raw_text")
        original_target = _reference_target_value(ref)
        if status != "resolved":
            skipped.append({
                "source_passage_id": source_pid,
                "source_unit_id": p.get("source_unit_id"),
                "raw": raw,
                "status": status,
                "target_id": original_target,
                "target_type": ref.get("target_type"),
                "reason": "status_not_resolved",
            })
            continue

        target_node, graph_target_id, target_node_type, reason = _resolve_reference_target(
            ref=ref,
            passage_id_set=passage_id_set,
            unit_to_passage_id=unit_to_passage_id,
            document_node_ids=document_node_ids,
        )
        if reason:
            skipped.append({
                "source_passage_id": source_pid,
                "source_unit_id": p.get("source_unit_id"),
                "raw": raw,
                "status": status,
                "target_id": original_target,
                "target_type": ref.get("target_type"),
                "target_label": ref.get("target_label"),
                "reason": reason,
            })
            continue

        edge_type = "PASSAGE_REFERS_TO_DOCUMENT" if target_node_type == "DOCUMENT" else "PASSAGE_REFERS_TO_PASSAGE"
        out.append({
            "edge_id": stable_id(source_pid, target_node, str(ref.get("resolution_id") or i), prefix="edge_ref"),
            "source": passage_node_id(source_pid),
            "target": target_node,
            "source_id": source_pid,
            "target_id": graph_target_id,
            "edge_type": edge_type,
            "weight": _float_or_default(ref.get("score") or ref.get("confidence"), 1.0),
            "metadata": {
                "resolution_id": ref.get("resolution_id"),
                "relation_type": ref.get("relation_type"),
                "mention_type": ref.get("mention_type") or ref.get("ref_type"),
                "mention_text": raw,
                "resolver_status": status,
                "target_type": ref.get("target_type"),
                "target_label": ref.get("target_label"),
                "original_target_id": original_target,
                "expansion_policy": ref.get("expansion_policy"),
            },
        })
    return out, skipped


def build_passage_nodes(passages: List[Dict[str, Any]]) -> tuple[list[dict], dict[str, dict], dict[str, set]]:
    nodes = []
    passage_by_id = {}
    doc_to_passages = defaultdict(set)

    for p in passages:
        pid = p.get("passage_id")
        if not pid:
            continue
        passage_by_id[pid] = p

        document_id = p.get("document_id")
        document_number = p.get("document_number")
        doc_id = document_node_id(document_id, document_number)
        doc_to_passages[doc_id].add(pid)

        nodes.append({
            "id": passage_node_id(pid),
            "node_type": "PASSAGE",
            "passage_id": pid,
            "source_unit_id": p.get("source_unit_id"),
            "package_id": p.get("package_id"),
            "document_id": document_id,
            "document_number": document_number,
            "passage_kind": p.get("passage_kind"),
            "unit_type": p.get("unit_type"),
            "path_text": p.get("path_text"),
            "effective_from": p.get("effective_from"),
            "ceased_from": p.get("ceased_from"),
            "reference_expansion_policies": p.get("reference_expansion_policies") or [],
            "text_preview": compact_text(p.get("content") or p.get("passage_text") or "", 700),
            "passage_text": p.get("passage_text"),
        })

    return nodes, passage_by_id, doc_to_passages


def build_document_nodes_and_edges(passages: List[Dict[str, Any]], doc_to_passages: Dict[str, set]) -> tuple[list[dict], list[dict]]:
    doc_meta: Dict[str, Dict[str, Any]] = {}

    for p in passages:
        doc_node = document_node_id(p.get("document_id"), p.get("document_number"))
        if doc_node not in doc_meta:
            doc_meta[doc_node] = {
                "id": doc_node,
                "node_type": "DOCUMENT",
                "document_id": p.get("document_id"),
                "document_number": p.get("document_number"),
                "package_id": p.get("package_id"),
                "document_title": p.get("document_title"),
                "effective_from": p.get("effective_from"),
                "ceased_from": p.get("ceased_from"),
            }

    edges = []
    for doc_node, pids in doc_to_passages.items():
        for pid in sorted(pids):
            edges.append({
                "edge_id": stable_id(doc_node, pid, prefix="edge_doc_contains"),
                "source": doc_node,
                "target": passage_node_id(pid),
                "source_id": doc_node.replace("document::", "", 1),
                "target_id": pid,
                "edge_type": "DOCUMENT_CONTAINS_PASSAGE",
                "weight": 1.0,
                "metadata": {},
            })

    return list(doc_meta.values()), edges


def build_entity_nodes(canonical_entities: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}

    for e in canonical_entities:
        eid = e.get("entity_id")
        if not eid:
            continue
        nodes[eid] = {
            "id": entity_node_id(eid),
            "node_type": "ENTITY",
            "entity_id": eid,
            "canonical": e.get("canonical"),
            "label": e.get("label"),
            "aliases": e.get("aliases") or [],
            "count": e.get("count"),
            "is_generic_hub": bool(e.get("is_generic_hub", False)),
            "min_graph_weight": e.get("min_graph_weight", 1.0),
        }

    # Fallback: entity links may include entities not present in canonical_entities.
    for l in links:
        eid = normalize_entity_id(l)
        if eid not in nodes:
            nodes[eid] = {
                "id": entity_node_id(eid),
                "node_type": "ENTITY",
                "entity_id": eid,
                "canonical": l.get("canonical") or l.get("surface"),
                "label": l.get("label"),
                "aliases": [l.get("surface")] if l.get("surface") else [],
                "count": None,
                "is_generic_hub": (l.get("match_mode") == "downweight"),
                "min_graph_weight": l.get("graph_weight", 1.0),
            }

    return nodes


def aggregate_passage_entity_edges(links: List[Dict[str, Any]], passage_by_id: Dict[str, Dict[str, Any]], strong_only: bool = False) -> tuple[list[dict], dict[str, list], dict[str, list]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    passage_to_entities = defaultdict(list)
    entity_to_passages = defaultdict(list)

    for l in links:
        pid = l.get("passage_id")
        if not pid or pid not in passage_by_id:
            continue

        mode = l.get("match_mode") or "keep"
        graph_weight = float(l.get("graph_weight", 1.0))
        if strong_only and mode != "keep":
            continue

        eid = normalize_entity_id(l)
        key = (pid, eid)

        if key not in groups:
            groups[key] = {
                "passage_id": pid,
                "entity_id": eid,
                "canonical": l.get("canonical") or l.get("surface"),
                "label": l.get("label"),
                "mention_count": 0,
                "keep_mention_count": 0,
                "downweight_mention_count": 0,
                "max_graph_weight": 0.0,
                "total_graph_weight": 0.0,
                "examples": [],
            }

        g = groups[key]
        g["mention_count"] += 1
        if mode == "keep":
            g["keep_mention_count"] += 1
        elif mode == "downweight":
            g["downweight_mention_count"] += 1
        g["max_graph_weight"] = max(g["max_graph_weight"], graph_weight)
        g["total_graph_weight"] += graph_weight

        if len(g["examples"]) < 5:
            g["examples"].append({
                "sentence_id": l.get("sentence_id"),
                "surface": l.get("surface"),
                "start": l.get("start"),
                "end": l.get("end"),
                "match_mode": mode,
                "graph_weight": graph_weight,
            })

    edges = []
    for (pid, eid), g in groups.items():
        edge_mode = "strong" if g["keep_mention_count"] > 0 else "weak"
        # Keep edge is strong. Weak edge remains weak even if repeated many times.
        edge_weight = 1.0 if edge_mode == "strong" else min(0.5, g["max_graph_weight"] or 0.25)

        edge = {
            "edge_id": stable_id(pid, eid, prefix="edge_mention"),
            "source": passage_node_id(pid),
            "target": entity_node_id(eid),
            "source_id": pid,
            "target_id": eid,
            "edge_type": "PASSAGE_MENTIONS_ENTITY",
            "weight": edge_weight,
            "metadata": {
                "canonical": g["canonical"],
                "label": g["label"],
                "edge_mode": edge_mode,
                "mention_count": g["mention_count"],
                "keep_mention_count": g["keep_mention_count"],
                "downweight_mention_count": g["downweight_mention_count"],
                "total_graph_weight": round(g["total_graph_weight"], 4),
                "examples": g["examples"],
            },
        }
        edges.append(edge)

        passage_to_entities[pid].append({
            "entity_id": eid,
            "canonical": g["canonical"],
            "label": g["label"],
            "weight": edge_weight,
            "edge_mode": edge_mode,
            "mention_count": g["mention_count"],
        })
        entity_to_passages[eid].append({
            "passage_id": pid,
            "weight": edge_weight,
            "edge_mode": edge_mode,
            "mention_count": g["mention_count"],
        })

    for pid in passage_to_entities:
        passage_to_entities[pid].sort(key=lambda x: (-x["weight"], x["label"] or "", x["canonical"] or ""))
    for eid in entity_to_passages:
        entity_to_passages[eid].sort(key=lambda x: (-x["weight"], -x["mention_count"], x["passage_id"]))

    return edges, passage_to_entities, entity_to_passages


def build_legal_graph(
    passages_root: str | Path,
    entity_links_root: str | Path,
    gazetteer_root: str | Path,
    output_dir: str | Path,
    include_reference_edges: bool = True,
    strong_only: bool = False,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    passages = load_passages(passages_root)
    links = load_entity_links(entity_links_root)
    canonical_entities = load_canonical_entities(gazetteer_root)

    passage_nodes, passage_by_id, doc_to_passages = build_passage_nodes(passages)
    document_nodes, document_edges = build_document_nodes_and_edges(passages, doc_to_passages)
    entity_nodes_by_id = build_entity_nodes(canonical_entities, links)
    unit_to_passage_id = {
        p["source_unit_id"]: p["passage_id"]
        for p in passages
        if p.get("source_unit_id") and p.get("passage_id")
    }

    mention_edges, passage_to_entities, entity_to_passages = aggregate_passage_entity_edges(
        links=links,
        passage_by_id=passage_by_id,
        strong_only=strong_only,
    )

    reference_edges: List[Dict[str, Any]] = []
    skipped_reference_edges: List[Dict[str, Any]] = []
    if include_reference_edges:
        pids = set(passage_by_id.keys())
        document_node_ids = {n["id"] for n in document_nodes}
        for p in passages:
            extracted, skipped = extract_reference_edges_from_passage(
                p=p,
                passage_id_set=pids,
                unit_to_passage_id=unit_to_passage_id,
                document_node_ids=document_node_ids,
            )
            reference_edges.extend(extracted)
            skipped_reference_edges.extend(skipped)

    nodes = [*document_nodes, *passage_nodes, *entity_nodes_by_id.values()]
    edges = [*document_edges, *mention_edges, *reference_edges]

    write_jsonl(output_dir / "nodes.jsonl", nodes)
    write_jsonl(output_dir / "edges.jsonl", edges)
    write_jsonl(output_dir / "passage_nodes.jsonl", passage_nodes)
    write_jsonl(output_dir / "entity_nodes.jsonl", entity_nodes_by_id.values())
    write_jsonl(output_dir / "document_nodes.jsonl", document_nodes)
    write_jsonl(output_dir / "mention_edges.jsonl", mention_edges)
    write_jsonl(output_dir / "document_edges.jsonl", document_edges)
    write_jsonl(output_dir / "reference_edges.jsonl", reference_edges)
    write_jsonl(output_dir / "skipped_reference_edges.jsonl", skipped_reference_edges)

    write_json(output_dir / "passage_to_entities.json", dict(passage_to_entities))
    write_json(output_dir / "entity_to_passages.json", dict(entity_to_passages))

    by_edge_type = defaultdict(int)
    by_node_type = defaultdict(int)
    by_entity_label = defaultdict(int)
    by_edge_mode = defaultdict(int)
    skipped_reference_by_reason = defaultdict(int)

    for n in nodes:
        by_node_type[n.get("node_type") or "UNKNOWN"] += 1
        if n.get("node_type") == "ENTITY":
            by_entity_label[n.get("label") or "UNKNOWN"] += 1

    for e in edges:
        by_edge_type[e.get("edge_type") or "UNKNOWN"] += 1
        if e.get("edge_type") == "PASSAGE_MENTIONS_ENTITY":
            by_edge_mode[e.get("metadata", {}).get("edge_mode") or "UNKNOWN"] += 1

    for row in skipped_reference_edges:
        skipped_reference_by_reason[row.get("reason") or "UNKNOWN"] += 1

    summary = {
        "passages_root": str(passages_root),
        "entity_links_root": str(entity_links_root),
        "gazetteer_root": str(gazetteer_root),
        "output_dir": str(output_dir),
        "input": {
            "passage_count": len(passages),
            "entity_link_count": len(links),
            "canonical_entity_count": len(canonical_entities),
        },
        "graph": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "by_node_type": dict(sorted(by_node_type.items())),
            "by_edge_type": dict(sorted(by_edge_type.items())),
            "by_entity_label": dict(sorted(by_entity_label.items())),
            "by_mention_edge_mode": dict(sorted(by_edge_mode.items())),
        },
        "references": {
            "input_outgoing_ref_count": len(reference_edges) + len(skipped_reference_edges),
            "reference_edge_count": len(reference_edges),
            "skipped_reference_count": len(skipped_reference_edges),
            "skipped_by_reason": dict(sorted(skipped_reference_by_reason.items())),
        },
        "options": {
            "include_reference_edges": include_reference_edges,
            "strong_only": strong_only,
        },
        "outputs": {
            "nodes": str(output_dir / "nodes.jsonl"),
            "edges": str(output_dir / "edges.jsonl"),
            "passage_to_entities": str(output_dir / "passage_to_entities.json"),
            "entity_to_passages": str(output_dir / "entity_to_passages.json"),
            "skipped_reference_edges": str(output_dir / "skipped_reference_edges.jsonl"),
        },
    }
    write_json(output_dir / "graph_summary.json", summary)
    return summary
