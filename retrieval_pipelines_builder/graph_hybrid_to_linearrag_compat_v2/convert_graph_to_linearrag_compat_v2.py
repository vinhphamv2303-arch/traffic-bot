
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


def has_passage_text(node: dict[str, Any]) -> bool:
    text = node.get("passage_text") or node.get("text") or node.get("text_sample") or node.get("text_preview") or ""
    return bool(str(text).strip())


def add_id_aliases(pid: str, mapping: dict[str, str], *, prefer: bool = False):
    """
    Map many resolver/unit-id variants to the real passage_id used by retriever.
    Important because parsed units often use:
      abc.dieu_1.khoan_2.diem_a
    while passage nodes use:
      abc.dieu_1.khoan_2.diem_a.passage
    """
    if not pid:
        return
    if prefer:
        mapping[pid] = pid
    else:
        mapping.setdefault(pid, pid)

    if pid.endswith(".passage"):
        core = pid[: -len(".passage")]
        if prefer:
            mapping[core] = pid
            mapping[core + ".text_1"] = pid
        else:
            mapping.setdefault(core, pid)
            mapping.setdefault(core + ".text_1", pid)
    else:
        mapping.setdefault(pid + ".passage", pid)

    # Text nodes often appear as .text_1, .text_2; map base to the first seen.
    if ".text_" in pid:
        base = pid.split(".text_", 1)[0]
        if prefer:
            mapping[base] = pid
            mapping[base + ".passage"] = pid
        else:
            mapping.setdefault(base, pid)
            mapping.setdefault(base + ".passage", pid)

    # Table/appendix rows may have .table_1.passage etc.; keep a softer base alias.
    for suffix in [".table_1.passage", ".table_1", ".appendix_item_decimal_1.passage"]:
        if pid.endswith(suffix):
            if prefer:
                mapping[pid[: -len(suffix)]] = pid
            else:
                mapping.setdefault(pid[: -len(suffix)], pid)


def resolve_passage_id(raw_id: str | None, id_map: dict[str, str]) -> str | None:
    if not raw_id:
        return None
    if raw_id in id_map:
        return id_map[raw_id]
    if raw_id.endswith(".passage") and raw_id[:-8] in id_map:
        return id_map[raw_id[:-8]]
    if (raw_id + ".passage") in id_map:
        return id_map[raw_id + ".passage"]
    if ".text_" in raw_id:
        base = raw_id.split(".text_", 1)[0]
        if base in id_map:
            return id_map[base]
        if (base + ".passage") in id_map:
            return id_map[base + ".passage"]
    return raw_id


def convert_graph(graph_root: str | Path, output: str | Path):
    graph_root = Path(graph_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    nodes = list(read_jsonl(graph_root / "nodes.jsonl"))
    edges = list(read_jsonl(graph_root / "edges.jsonl"))

    # First pass: collect passage ids and aliases. Build aliases in two phases:
    # content-bearing passage nodes first, then placeholders. This avoids unit ids
    # such as "abc.dieu_1.khoan_2" overriding their real
    # "abc.dieu_1.khoan_2.passage" content node.
    id_map: dict[str, str] = {}
    raw_passage_nodes = []
    for n in nodes:
        if n.get("type") == "passage":
            pid = n.get("passage_id") or strip_node_prefix(n.get("id"), "passage:")
            if pid:
                raw_passage_nodes.append((pid, n))

    for pid, n in raw_passage_nodes:
        if has_passage_text(n):
            add_id_aliases(pid, id_map, prefer=True)
    for pid, n in raw_passage_nodes:
        add_id_aliases(pid, id_map, prefer=False)

    passage_nodes = []
    entity_nodes = []
    document_nodes = []
    entity_ids = set()

    for pid, n in raw_passage_nodes:
        text = n.get("passage_text") or n.get("text") or n.get("text_sample") or n.get("text_preview") or ""
        passage_nodes.append({
            "id": f"passage::{pid}",
            "passage_id": pid,
            "source_unit_id": n.get("source_unit_id") or pid,
            "document_id": n.get("document_id"),
            "document_number": n.get("document_number"),
            "document_title": n.get("document_title"),
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

    for n in nodes:
        ntype = n.get("type")
        if ntype == "entity":
            eid = n.get("entity_id") or strip_node_prefix(n.get("id"), "entity:")
            if not eid:
                continue
            entity_ids.add(eid)
            degree = int(n.get("passage_degree") or 0)
            if degree > 5000:
                min_graph_weight = 0.10
                is_generic_hub = True
            elif degree > 2000:
                min_graph_weight = 0.20
                is_generic_hub = True
            elif degree > 800:
                min_graph_weight = 0.35
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
    remapped_reference_source = 0
    remapped_reference_target = 0
    unresolved_reference_target = 0

    for e in edges:
        etype = e.get("type")
        src = e.get("source")
        tgt = e.get("target")

        if etype == "HAS_ENTITY":
            raw_pid = strip_node_prefix(src, "passage:")
            pid = resolve_passage_id(raw_pid, id_map)
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
            raw_src = strip_node_prefix(src, "passage:")
            raw_tgt = strip_node_prefix(tgt, "passage:")
            src_pid = resolve_passage_id(raw_src, id_map)
            tgt_pid = resolve_passage_id(raw_tgt, id_map)

            if raw_src != src_pid:
                remapped_reference_source += 1
            if raw_tgt != tgt_pid:
                remapped_reference_target += 1

            if not src_pid or not tgt_pid:
                skipped_edges["bad_reference"] += 1
                continue

            # If target still not in passage_nodes, keep it, but count it.
            if tgt_pid not in {pid for pid, _ in raw_passage_nodes}:
                unresolved_reference_target += 1

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
                    "raw_source": raw_src,
                    "raw_target": raw_tgt,
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
        "id_alias_count": len(id_map),
        "remapped_reference_source": remapped_reference_source,
        "remapped_reference_target": remapped_reference_target,
        "unresolved_reference_target_after_remap": unresolved_reference_target,
        "note": "Compatibility graph for legal_linearrag_retriever. v2 remaps legal-unit ids to .passage ids so reference expansion can work.",
    }
    write_json(output / "compat_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Convert legal_graph_hybrid_v2 nodes/edges to legal_linearrag_retriever-compatible graph format. v2 fixes reference id mapping.")
    ap.add_argument("--graph-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    summary = convert_graph(args.graph_root, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
