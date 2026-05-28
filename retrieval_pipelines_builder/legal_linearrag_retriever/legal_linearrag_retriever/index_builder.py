from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .bm25 import BM25Index
from .utils import ensure_dir, read_jsonl, save_pickle, write_json, write_jsonl


def _node_raw_id(node_id: str, prefix: str) -> str:
    return node_id.replace(prefix, "", 1) if node_id.startswith(prefix) else node_id


def _add_passage_aliases(pid: str, mapping: Dict[str, str], *, prefer: bool = False) -> None:
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

    if ".text_" in pid:
        base = pid.split(".text_", 1)[0]
        if prefer:
            mapping[base] = pid
            mapping[base + ".passage"] = pid
        else:
            mapping.setdefault(base, pid)
            mapping.setdefault(base + ".passage", pid)

    for suffix in [".table_1.passage", ".table_1", ".appendix_item_decimal_1.passage"]:
        if pid.endswith(suffix):
            base = pid[: -len(suffix)]
            if prefer:
                mapping[base] = pid
            else:
                mapping.setdefault(base, pid)


def _resolve_passage_id(raw_id: str | None, mapping: Dict[str, str]) -> str | None:
    if not raw_id:
        return None
    if raw_id in mapping:
        return mapping[raw_id]
    if raw_id.endswith(".passage") and raw_id[:-8] in mapping:
        return mapping[raw_id[:-8]]
    if (raw_id + ".passage") in mapping:
        return mapping[raw_id + ".passage"]
    if ".text_" in raw_id:
        base = raw_id.split(".text_", 1)[0]
        if base in mapping:
            return mapping[base]
        if (base + ".passage") in mapping:
            return mapping[base + ".passage"]
    return None


def build_passage_id_map(passages: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in passages:
        if (p.get("passage_text") or "").strip():
            _add_passage_aliases(p.get("passage_id") or "", mapping, prefer=True)
    for p in passages:
        _add_passage_aliases(p.get("passage_id") or "", mapping, prefer=False)
    return mapping


def load_graph(graph_root: str | Path):
    graph_root = Path(graph_root)
    passage_nodes = list(read_jsonl(graph_root / "passage_nodes.jsonl"))
    entity_nodes = list(read_jsonl(graph_root / "entity_nodes.jsonl"))
    mention_edges = list(read_jsonl(graph_root / "mention_edges.jsonl"))
    reference_edges = list(read_jsonl(graph_root / "reference_edges.jsonl")) if (graph_root / "reference_edges.jsonl").exists() else []
    return passage_nodes, entity_nodes, mention_edges, reference_edges


def build_passage_records(passage_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for p in passage_nodes:
        pid = p.get("passage_id") or _node_raw_id(p.get("id") or "", "passage::")
        text = p.get("passage_text") or p.get("text_preview") or ""
        if not str(text).strip():
            continue
        document_title = p.get("document_title") or ""
        index_text = "\n".join([
            f"Van ban: {p.get('document_number') or p.get('document_id') or ''}",
            f"Ten van ban: {document_title}",
            f"Duong dan: {p.get('path_text') or ''}",
            f"Noi dung: {text}",
        ])
        records.append({
            "passage_id": pid,
            "document_id": p.get("document_id"),
            "document_number": p.get("document_number"),
            "document_title": p.get("document_title"),
            "package_id": p.get("package_id"),
            "source_unit_id": p.get("source_unit_id"),
            "path_text": p.get("path_text"),
            "unit_type": p.get("unit_type"),
            "passage_kind": p.get("passage_kind"),
            "effective_from": p.get("effective_from"),
            "ceased_from": p.get("ceased_from"),
            "passage_text": text,
            "index_text": index_text,
        })
    return records


def build_entity_records(entity_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for e in entity_nodes:
        eid = e.get("entity_id") or _node_raw_id(e.get("id") or "", "entity::")
        aliases = e.get("aliases") or []
        text = " ; ".join([e.get("canonical") or "", e.get("label") or "", *aliases])
        records.append({
            "entity_id": eid,
            "canonical": e.get("canonical"),
            "label": e.get("label"),
            "aliases": aliases,
            "is_generic_hub": e.get("is_generic_hub", False),
            "min_graph_weight": e.get("min_graph_weight", 1.0),
            "index_text": text,
        })
    return records


def build_graph_maps(
    mention_edges: List[Dict[str, Any]],
    reference_edges: List[Dict[str, Any]],
    passage_id_map: Dict[str, str],
    valid_passage_ids: set[str],
):
    entity_to_passages = defaultdict(list)
    passage_to_entities = defaultdict(list)
    passage_neighbors = defaultdict(list)
    stats = Counter()

    for edge in mention_edges:
        if edge.get("edge_type") != "PASSAGE_MENTIONS_ENTITY":
            continue
        raw_pid = edge.get("source_id")
        pid = _resolve_passage_id(raw_pid, passage_id_map)
        eid = edge.get("target_id")
        if not pid or pid not in valid_passage_ids or not eid:
            stats["skipped_mention_edge_bad_passage"] += 1
            continue
        if raw_pid != pid:
            stats["remapped_mention_source"] += 1
        meta = edge.get("metadata") or {}
        item = {
            "passage_id": pid,
            "entity_id": eid,
            "weight": float(edge.get("weight", 1.0)),
            "edge_mode": meta.get("edge_mode"),
            "label": meta.get("label"),
            "canonical": meta.get("canonical"),
            "mention_count": meta.get("mention_count", 1),
        }
        entity_to_passages[eid].append(item)
        passage_to_entities[pid].append(item)

    for edge in reference_edges:
        if edge.get("edge_type") != "PASSAGE_REFERS_TO_PASSAGE":
            continue
        raw_src = edge.get("source_id")
        raw_tgt = edge.get("target_id")
        src = _resolve_passage_id(raw_src, passage_id_map)
        tgt = _resolve_passage_id(raw_tgt, passage_id_map)
        if src and tgt and src in valid_passage_ids and tgt in valid_passage_ids:
            if raw_src != src:
                stats["remapped_reference_source"] += 1
            if raw_tgt != tgt:
                stats["remapped_reference_target"] += 1
            passage_neighbors[src].append({
                "passage_id": tgt,
                "weight": float(edge.get("weight", 1.0)),
                "edge_type": edge.get("edge_type"),
            })
        else:
            stats["skipped_reference_edge_bad_passage"] += 1

    return dict(entity_to_passages), dict(passage_to_entities), dict(passage_neighbors), dict(stats)


def build_embeddings(texts: List[str], model_name: str, batch_size: int = 64):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for dense embeddings. Install: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def build_index(
    graph_root: str | Path,
    gazetteer_root: str | Path,
    output_dir: str | Path,
    embedding_model: str = "BAAI/bge-m3",
    embedding_batch_size: int = 64,
    skip_embeddings: bool = False,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    passage_nodes, entity_nodes, mention_edges, reference_edges = load_graph(graph_root)
    passages = build_passage_records(passage_nodes)
    entities = build_entity_records(entity_nodes)
    passage_id_map = build_passage_id_map(passages)
    valid_passage_ids = {p["passage_id"] for p in passages}
    entity_to_passages, passage_to_entities, passage_neighbors, graph_map_stats = build_graph_maps(
        mention_edges,
        reference_edges,
        passage_id_map,
        valid_passage_ids,
    )

    write_jsonl(output_dir / "passages.jsonl", passages)
    write_jsonl(output_dir / "entities.jsonl", entities)
    write_json(output_dir / "entity_to_passages.json", entity_to_passages)
    write_json(output_dir / "passage_to_entities.json", passage_to_entities)
    write_json(output_dir / "passage_neighbors.json", passage_neighbors)

    bm25 = BM25Index.from_texts([p["index_text"] for p in passages], [p["passage_id"] for p in passages])
    save_pickle(output_dir / "bm25.pkl", bm25)

    if not skip_embeddings:
        passage_emb = build_embeddings([p["index_text"] for p in passages], embedding_model, embedding_batch_size)
        entity_emb = build_embeddings([e["index_text"] for e in entities], embedding_model, embedding_batch_size)
        np.save(output_dir / "passage_embeddings.npy", passage_emb)
        np.save(output_dir / "entity_embeddings.npy", entity_emb)
    else:
        embedding_model = None

    meta = {
        "graph_root": str(graph_root),
        "gazetteer_root": str(gazetteer_root),
        "output_dir": str(output_dir),
        "passage_count": len(passages),
        "source_passage_node_count": len(passage_nodes),
        "skipped_empty_passage_count": len(passage_nodes) - len(passages),
        "entity_count": len(entities),
        "mention_edge_count": len(mention_edges),
        "reference_edge_count": len(reference_edges),
        "graph_map_stats": graph_map_stats,
        "embedding_model": embedding_model,
        "skip_embeddings": skip_embeddings,
        "files": {
            "passages": str(output_dir / "passages.jsonl"),
            "entities": str(output_dir / "entities.jsonl"),
            "bm25": str(output_dir / "bm25.pkl"),
            "passage_embeddings": str(output_dir / "passage_embeddings.npy"),
            "entity_embeddings": str(output_dir / "entity_embeddings.npy"),
        },
    }
    write_json(output_dir / "index_summary.json", meta)
    return meta
