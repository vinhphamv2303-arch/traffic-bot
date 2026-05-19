
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_node_prefix(node_id: str | None, prefix: str) -> str | None:
    if not node_id:
        return None
    if node_id.startswith(prefix):
        return node_id[len(prefix):]
    return node_id


def convert_graph(graph_root: str | Path, output: str | Path):
    graph_root = Path(graph_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    nodes = list(read_jsonl(graph_root / "nodes.jsonl"))
    edges = list(read_jsonl(graph_root / "edges.jsonl"))

    passage_nodes = []
    entity_nodes = []
    document_nodes = []

    passage_ids = set()
    entity_ids = set()

    for n in nodes:
        ntype = n.get("type")
        if ntype == "passage":
            pid = n.get("passage_id") or strip_node_prefix(n.get("id"), "passage:")
            if not pid:
                continue
            passage_ids.add(pid)
            text = n.get("passage_text") or n.get("text") or n.get("text_sample") or n.get("text_preview") or ""
            passage_nodes.append({
                "id": f"passage::{pid}",
                "passage_id": pid,
                "source_unit_id": n.get("source_unit_id") or pid,
                "document_id": n.get("document_id"),
                "document_number": n.get("document_number"),
                "package_id": n.get("package_id"),
                "path_text": n.get("path_text"),
                "unit_type": n.get("unit_type"),
                "passage_kind": n.get("passage_kind"),
                "effective_from": n.get("effective_from"),
                "ceased_from": n.get("ceased_from"),
                "passage_text": text,
                "text_preview": text[:1000],
                "is_reference_only": bool(n.get("is_reference_only", False)),
            })
        elif ntype == "entity":
            eid = n.get("entity_id") or strip_node_prefix(n.get("id"), "entity:")
            if not eid:
                continue
            entity_ids.add(eid)
            degree = int(n.get("passage_degree") or 0)
            # Retriever cũ dùng min_graph_weight làm hub_penalty cho semantic entity activation.
            # Degree càng cao thì min_graph_weight càng thấp.
            if degree > 5000:
                min_graph_weight = 0.15
                is_generic_hub = True
            elif degree > 2000:
                min_graph_weight = 0.25
                is_generic_hub = True
            elif degree > 800:
                min_graph_weight = 0.45
                is_generic_hub = True
            else:
                min_graph_weight = 1.0
                is_generic_hub = False

            entity_nodes.append({
                "id": f"entity::{eid}",
                "entity_id": eid,
                "label": n.get("label"),
                "canonical": n.get("canonical"),
                "aliases": n.get("aliases") or [],
                "is_generic_hub": is_generic_hub,
                "min_graph_weight": min_graph_weight,
                "passage_degree": degree,
            })
        elif ntype == "document":
            document_nodes.append(n)

    mention_edges = []
    reference_edges = []
    skipped_edges = Counter()

    for e in edges:
        etype = e.get("type")
        src = e.get("source")
        tgt = e.get("target")
        if etype == "HAS_ENTITY":
            pid = strip_node_prefix(src, "passage:")
            eid = strip_node_prefix(tgt, "entity:")
            if not pid or not eid:
                skipped_edges["bad_has_entity"] += 1
                continue
            mention_edges.append({
                "edge_id": f"mention::{pid}::{eid}",
                "edge_type": "PASSAGE_MENTIONS_ENTITY",
                "source_id": pid,
                "target_id": eid,
                "weight": float(e.get("weight", e.get("raw_weight", 1.0)) or 1.0),
                "metadata": {
                    "edge_mode": ",".join(e.get("sources") or []),
                    "sources": e.get("sources") or [],
                    "raw_weight": e.get("raw_weight"),
                },
            })
        elif etype == "REFERENCES":
            src_pid = strip_node_prefix(src, "passage:")
            tgt_pid = strip_node_prefix(tgt, "passage:")
            if not src_pid or not tgt_pid:
                skipped_edges["bad_reference"] += 1
                continue
            reference_edges.append({
                "edge_id": f"ref::{src_pid}::{tgt_pid}::{len(reference_edges)}",
                "edge_type": "PASSAGE_REFERS_TO_PASSAGE",
                "source_id": src_pid,
                "target_id": tgt_pid,
                "weight": float(e.get("weight", 1.0) or 1.0),
                "metadata": {
                    "raw": e.get("raw"),
                    "mention_type": e.get("mention_type"),
                    "selected_target_type": e.get("selected_target_type"),
                    "selected_score": e.get("selected_score"),
                    "confidence": e.get("confidence"),
                },
            })
        else:
            skipped_edges[etype or "unknown"] += 1

    write_jsonl(output / "passage_nodes.jsonl", passage_nodes)
    write_jsonl(output / "entity_nodes.jsonl", entity_nodes)
    write_jsonl(output / "mention_edges.jsonl", mention_edges)
    write_jsonl(output / "reference_edges.jsonl", reference_edges)

    summary = {
        "source_graph_root": str(graph_root),
        "output": str(output),
        "passage_nodes": len(passage_nodes),
        "entity_nodes": len(entity_nodes),
        "document_nodes_seen": len(document_nodes),
        "mention_edges": len(mention_edges),
        "reference_edges": len(reference_edges),
        "skipped_edges": dict(skipped_edges),
        "note": "Compatibility graph for legal_linearrag_retriever. Converts nodes.jsonl/edges.jsonl to passage_nodes/entity_nodes/mention_edges/reference_edges.",
    }
    write_json(output / "compat_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Convert legal_graph_hybrid_v2 nodes/edges to legal_linearrag_retriever-compatible graph format.")
    ap.add_argument("--graph-root", required=True, help="Input graph folder containing nodes.jsonl and edges.jsonl")
    ap.add_argument("--output", required=True, help="Output compatible graph folder")
    args = ap.parse_args()
    summary = convert_graph(args.graph_root, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
