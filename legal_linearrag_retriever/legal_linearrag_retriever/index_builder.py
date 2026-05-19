from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .bm25 import BM25Index
from .utils import ensure_dir, read_jsonl, save_pickle, write_json, write_jsonl


def _node_raw_id(node_id: str, prefix: str) -> str:
    return node_id.replace(prefix, "", 1) if node_id.startswith(prefix) else node_id


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
        index_text = "\n".join([
            f"Van ban: {p.get('document_number') or p.get('document_id') or ''}",
            f"Duong dan: {p.get('path_text') or ''}",
            f"Noi dung: {text}",
        ])
        records.append({
            "passage_id": pid,
            "document_id": p.get("document_id"),
            "document_number": p.get("document_number"),
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


def build_graph_maps(mention_edges: List[Dict[str, Any]], reference_edges: List[Dict[str, Any]]):
    entity_to_passages = defaultdict(list)
    passage_to_entities = defaultdict(list)
    passage_neighbors = defaultdict(list)

    for edge in mention_edges:
        if edge.get("edge_type") != "PASSAGE_MENTIONS_ENTITY":
            continue
        pid = edge.get("source_id")
        eid = edge.get("target_id")
        if not pid or not eid:
            continue
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
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if src and tgt:
            passage_neighbors[src].append({
                "passage_id": tgt,
                "weight": float(edge.get("weight", 1.0)),
                "edge_type": edge.get("edge_type"),
            })

    return dict(entity_to_passages), dict(passage_to_entities), dict(passage_neighbors)


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
    entity_to_passages, passage_to_entities, passage_neighbors = build_graph_maps(mention_edges, reference_edges)

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
        "entity_count": len(entities),
        "mention_edge_count": len(mention_edges),
        "reference_edge_count": len(reference_edges),
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
