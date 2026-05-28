from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .gazetteer import GazetteerMatcher
from .utils import cosine_matrix, load_pickle, minmax_normalize, read_json, read_jsonl, topk_dict


class LinearRAGRetriever:
    """
    LinearRAG-style hybrid retriever:
      1) Local semantic bridging: activate entities by exact query anchors + dense query/entity similarity.
      2) Global passage importance aggregation: combine dense passage retrieval, BM25, entity-passage graph, and optional reference expansion.
    """

    def __init__(
        self,
        index_dir: str | Path,
        gazetteer_root: str | Path,
        embedding_model: str | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.gazetteer_root = Path(gazetteer_root)

        self.passages = list(read_jsonl(self.index_dir / "passages.jsonl"))
        self.entities = list(read_jsonl(self.index_dir / "entities.jsonl"))
        self.passage_by_id = {p["passage_id"]: p for p in self.passages}
        self.entity_by_id = {e["entity_id"]: e for e in self.entities}

        self.entity_to_passages = read_json(self.index_dir / "entity_to_passages.json")
        self.passage_to_entities = read_json(self.index_dir / "passage_to_entities.json")
        self.passage_neighbors = read_json(self.index_dir / "passage_neighbors.json")
        self.bm25 = load_pickle(self.index_dir / "bm25.pkl")
        self.gazetteer = GazetteerMatcher.from_gazetteer_root(gazetteer_root)

        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.passage_embeddings = None
        self.entity_embeddings = None

        pe = self.index_dir / "passage_embeddings.npy"
        ee = self.index_dir / "entity_embeddings.npy"
        if pe.exists() and ee.exists():
            self.passage_embeddings = np.load(pe)
            self.entity_embeddings = np.load(ee)
            if len(self.passage_embeddings) != len(self.passages):
                raise ValueError(
                    f"passage_embeddings row count {len(self.passage_embeddings)} "
                    f"does not match passages {len(self.passages)}"
                )
            if len(self.entity_embeddings) != len(self.entities):
                raise ValueError(
                    f"entity_embeddings row count {len(self.entity_embeddings)} "
                    f"does not match entities {len(self.entities)}"
                )
            if embedding_model:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model)

    @classmethod
    def from_index(cls, index_dir: str | Path, gazetteer_root: str | Path):
        summary = read_json(Path(index_dir) / "index_summary.json")
        return cls(index_dir=index_dir, gazetteer_root=gazetteer_root, embedding_model=summary.get("embedding_model"))

    def _has_retrievable_text(self, passage_id: str | None) -> bool:
        if not passage_id:
            return False
        passage = self.passage_by_id.get(passage_id)
        return bool(passage and (passage.get("passage_text") or "").strip())

    def _encode_query(self, query: str):
        if self.embedding_model is None:
            return None
        return self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")

    def activate_entities(
        self,
        query: str,
        query_vec=None,
        exact_weight: float = 1.0,
        semantic_entity_top_k: int = 20,
        semantic_entity_min_score: float = 0.45,
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        activated: Dict[str, float] = {}
        evidence = []

        exact_matches = self.gazetteer.match(query)
        for m in exact_matches:
            eid = m.get("entity_id")
            if not eid:
                continue
            score = exact_weight * float(m.get("graph_weight", 1.0))
            activated[eid] = max(activated.get(eid, 0.0), score)
            evidence.append({**m, "activation_type": "exact"})

        # Local semantic bridging: query -> nearby entity nodes.
        if query_vec is not None and self.entity_embeddings is not None and len(self.entities) > 0:
            sims = cosine_matrix(query_vec, self.entity_embeddings)
            idxs = np.argsort(-sims)[:semantic_entity_top_k]
            for idx in idxs:
                sim = float(sims[idx])
                if sim < semantic_entity_min_score:
                    continue
                e = self.entities[int(idx)]
                eid = e["entity_id"]
                # avoid generic hubs dominating
                hub_penalty = float(e.get("min_graph_weight") or 1.0)
                score = sim * hub_penalty
                activated[eid] = max(activated.get(eid, 0.0), score)
                evidence.append({
                    "entity_id": eid,
                    "canonical": e.get("canonical"),
                    "label": e.get("label"),
                    "score": score,
                    "raw_similarity": sim,
                    "activation_type": "semantic",
                    "is_generic_hub": e.get("is_generic_hub", False),
                })

        activated = topk_dict(activated, semantic_entity_top_k)
        return activated, evidence

    def dense_passage_scores(self, query_vec, top_k: int = 200) -> Dict[str, float]:
        if query_vec is None or self.passage_embeddings is None:
            return {}
        sims = cosine_matrix(query_vec, self.passage_embeddings)
        idxs = np.argsort(-sims)[:top_k]
        return {self.passages[int(i)]["passage_id"]: float(sims[int(i)]) for i in idxs}

    def bm25_scores(self, query: str, top_k: int = 200) -> Dict[str, float]:
        return self.bm25.search(query, top_k=top_k)

    def graph_scores(
        self,
        activated_entities: Dict[str, float],
        top_k_per_entity: int = 200,
    ) -> Dict[str, float]:
        scores = defaultdict(float)
        for eid, e_score in activated_entities.items():
            links = self.entity_to_passages.get(eid) or []
            links = sorted(links, key=lambda x: (float(x.get("weight", 0)), int(x.get("mention_count", 1))), reverse=True)[:top_k_per_entity]
            for l in links:
                pid = l.get("passage_id")
                if not self._has_retrievable_text(pid):
                    continue
                edge_w = float(l.get("weight", 1.0))
                mention_boost = min(1.5, 1.0 + 0.1 * max(int(l.get("mention_count", 1)) - 1, 0))
                scores[pid] += e_score * edge_w * mention_boost
        return dict(scores)

    def reference_expand_scores(
        self,
        base_scores: Dict[str, float],
        max_seed_passages: int = 50,
        ref_decay: float = 0.35,
    ) -> Dict[str, float]:
        expanded = defaultdict(float)
        for pid, score in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max_seed_passages]:
            if not self._has_retrievable_text(pid):
                continue
            for nb in self.passage_neighbors.get(pid) or []:
                target = nb.get("passage_id")
                if not self._has_retrievable_text(target):
                    continue
                expanded[target] += score * ref_decay * float(nb.get("weight", 1.0))
        return dict(expanded)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 300,
        semantic_entity_top_k: int = 20,
        semantic_entity_min_score: float = 0.45,
        weights: Dict[str, float] | None = None,
        use_reference_expansion: bool = True,
        reference_seed_weights: Dict[str, float] | None = None,
        graph_only_penalty: float = 0.65,
    ) -> Dict[str, Any]:
        weights = weights or {
            "dense": 0.35,
            "bm25": 0.25,
            "graph": 0.35,
            "reference": 0.05,
        }

        reference_seed_weights = reference_seed_weights or {"bm25": 0.7, "graph": 0.3}
        component_weights = {
            "dense": float(weights.get("dense", 0.0)),
            "bm25": float(weights.get("bm25", 0.0)),
            "graph": float(weights.get("graph", 0.0)),
            "reference": float(weights.get("reference", 0.0)),
        }
        reference_seed_weights = {
            "dense": float(reference_seed_weights.get("dense", 0.0)),
            "bm25": float(reference_seed_weights.get("bm25", 0.0)),
            "graph": float(reference_seed_weights.get("graph", 0.0)),
        }

        uses_reference = use_reference_expansion and component_weights["reference"] > 0.0
        needs_dense = component_weights["dense"] > 0.0 or (uses_reference and reference_seed_weights["dense"] > 0.0)
        needs_bm25 = component_weights["bm25"] > 0.0 or (uses_reference and reference_seed_weights["bm25"] > 0.0)
        needs_graph = component_weights["graph"] > 0.0 or (uses_reference and reference_seed_weights["graph"] > 0.0)
        needs_query_vec = needs_dense or needs_graph

        query_vec = self._encode_query(query) if needs_query_vec else None
        if needs_dense and query_vec is None:
            raise RuntimeError(
                "Dense retrieval was requested, but this index has no loaded embedding model. "
                "Check index_summary.json and make sure the dense index was built with embeddings."
            )

        if needs_graph:
            activated_entities, entity_evidence = self.activate_entities(
                query=query,
                query_vec=query_vec,
                semantic_entity_top_k=semantic_entity_top_k,
                semantic_entity_min_score=semantic_entity_min_score,
            )
        else:
            activated_entities, entity_evidence = {}, []

        dense = self.dense_passage_scores(query_vec, top_k=candidate_k) if needs_dense else {}
        bm25 = self.bm25_scores(query, top_k=candidate_k) if needs_bm25 else {}
        graph = self.graph_scores(activated_entities, top_k_per_entity=candidate_k) if needs_graph else {}

        reference_seed = defaultdict(float)
        for pid, score in dense.items():
            reference_seed[pid] += reference_seed_weights["dense"] * float(score)
        for pid, score in bm25.items():
            reference_seed[pid] += reference_seed_weights["bm25"] * float(score)
        for pid, score in graph.items():
            reference_seed[pid] += reference_seed_weights["graph"] * float(score)

        ref = self.reference_expand_scores(
            reference_seed,
            max_seed_passages=100,
            ref_decay=1.0,
        ) if uses_reference else {}

        dense_n = minmax_normalize(dense)
        bm25_n = minmax_normalize(bm25)
        graph_n = minmax_normalize(graph)
        ref_n = minmax_normalize(ref)

        all_pids = {
            pid
            for pid in (set(dense_n) | set(bm25_n) | set(graph_n) | set(ref_n))
            if self._has_retrievable_text(pid)
        }
        final_scores = {}
        adjusted_graph_n = {}
        for pid in all_pids:
            graph_score = graph_n.get(pid, 0.0)
            if graph_score > 0 and dense_n.get(pid, 0.0) == 0 and bm25_n.get(pid, 0.0) == 0:
                graph_score *= graph_only_penalty
            adjusted_graph_n[pid] = graph_score
            final_score = (
                component_weights["dense"] * dense_n.get(pid, 0.0)
                + component_weights["bm25"] * bm25_n.get(pid, 0.0)
                + component_weights["graph"] * graph_score
                + component_weights["reference"] * ref_n.get(pid, 0.0)
            )
            if final_score > 0.0:
                final_scores[pid] = final_score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for pid, score in ranked:
            p = self.passage_by_id.get(pid, {})
            results.append({
                "passage_id": pid,
                "score": round(float(score), 6),
                "score_components": {
                    "dense": round(dense_n.get(pid, 0.0), 6),
                    "bm25": round(bm25_n.get(pid, 0.0), 6),
                    "graph": round(adjusted_graph_n.get(pid, 0.0), 6),
                    "reference": round(ref_n.get(pid, 0.0), 6),
                },
                "document_number": p.get("document_number"),
                "document_id": p.get("document_id"),
                "document_title": p.get("document_title"),
                "package_id": p.get("package_id"),
                "path_text": p.get("path_text"),
                "passage_kind": p.get("passage_kind"),
                "unit_type": p.get("unit_type"),
                "text": p.get("passage_text") or "",
                "entities": self.passage_to_entities.get(pid, [])[:20],
            })

        return {
            "query": query,
            "activated_entities": [
                {
                    "entity_id": eid,
                    "score": round(float(score), 6),
                    "canonical": (self.entity_by_id.get(eid) or {}).get("canonical"),
                    "label": (self.entity_by_id.get(eid) or {}).get("label"),
                }
                for eid, score in activated_entities.items()
            ],
            "entity_evidence": entity_evidence[:50],
            "weights": component_weights,
            "graph_only_penalty": graph_only_penalty,
            "results": results,
            "debug": {
                "used_components": {
                    "dense": needs_dense,
                    "bm25": needs_bm25,
                    "graph": needs_graph,
                    "reference": uses_reference,
                },
                "reference_seed_weights": reference_seed_weights,
                "dense_candidates": len(dense),
                "bm25_candidates": len(bm25),
                "graph_candidates": len(graph),
                "reference_candidates": len(ref),
                "final_candidates": len(final_scores),
            },
        }

        query_vec = self._encode_query(query)

        activated_entities, entity_evidence = self.activate_entities(
            query=query,
            query_vec=query_vec,
            semantic_entity_top_k=semantic_entity_top_k,
            semantic_entity_min_score=semantic_entity_min_score,
        )

        dense = self.dense_passage_scores(query_vec, top_k=candidate_k)
        bm25 = self.bm25_scores(query, top_k=candidate_k)
        graph = self.graph_scores(activated_entities, top_k_per_entity=candidate_k)

        # Reference expansion nên xuất phát từ cả lexical candidates và graph candidates.
        # Nhiều passage có dạng "theo quy định tại Điều..." được BM25 bắt rất tốt,
        # nhưng nếu chỉ expand từ graph thì passage đích khó được kéo lên.
        reference_seed_weights = reference_seed_weights or {"bm25": 0.7, "graph": 0.3}
        reference_seed = defaultdict(float)

        for pid, s in dense.items():
            reference_seed[pid] += float(reference_seed_weights.get("dense", 0.0)) * float(s)

        for pid, s in bm25.items():
            reference_seed[pid] += float(reference_seed_weights.get("bm25", 0.0)) * float(s)

        for pid, s in graph.items():
            reference_seed[pid] += float(reference_seed_weights.get("graph", 0.0)) * float(s)

        ref = self.reference_expand_scores(reference_seed, max_seed_passages=100,
                                           ref_decay=1.0) if use_reference_expansion else {}

        dense_n = minmax_normalize(dense)
        bm25_n = minmax_normalize(bm25)
        graph_n = minmax_normalize(graph)
        ref_n = minmax_normalize(ref)

        all_pids = {
            pid
            for pid in (set(dense_n) | set(bm25_n) | set(graph_n) | set(ref_n))
            if self._has_retrievable_text(pid)
        }
        final_scores = {}
        adjusted_graph_n = {}
        for pid in all_pids:
            graph_score = graph_n.get(pid, 0.0)
            if graph_score > 0 and dense_n.get(pid, 0.0) == 0 and bm25_n.get(pid, 0.0) == 0:
                graph_score *= graph_only_penalty
            adjusted_graph_n[pid] = graph_score
            final_scores[pid] = (
                weights.get("dense", 0.0) * dense_n.get(pid, 0.0)
                + weights.get("bm25", 0.0) * bm25_n.get(pid, 0.0)
                + weights.get("graph", 0.0) * graph_score
                + weights.get("reference", 0.0) * ref_n.get(pid, 0.0)
            )

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for pid, score in ranked:
            p = self.passage_by_id.get(pid, {})
            results.append({
                "passage_id": pid,
                "score": round(float(score), 6),
                "score_components": {
                    "dense": round(dense_n.get(pid, 0.0), 6),
                    "bm25": round(bm25_n.get(pid, 0.0), 6),
                    "graph": round(adjusted_graph_n.get(pid, 0.0), 6),
                    "reference": round(ref_n.get(pid, 0.0), 6),
                },
                "document_number": p.get("document_number"),
                "document_id": p.get("document_id"),
                "document_title": p.get("document_title"),
                "package_id": p.get("package_id"),
                "path_text": p.get("path_text"),
                "passage_kind": p.get("passage_kind"),
                "unit_type": p.get("unit_type"),
                "text": p.get("passage_text") or "",
                "entities": self.passage_to_entities.get(pid, [])[:20],
            })

        return {
            "query": query,
            "activated_entities": [
                {
                    "entity_id": eid,
                    "score": round(float(score), 6),
                    "canonical": (self.entity_by_id.get(eid) or {}).get("canonical"),
                    "label": (self.entity_by_id.get(eid) or {}).get("label"),
                }
                for eid, score in activated_entities.items()
            ],
            "entity_evidence": entity_evidence[:50],
            "weights": weights,
            "graph_only_penalty": graph_only_penalty,
            "results": results,
            "debug": {
                "dense_candidates": len(dense),
                "bm25_candidates": len(bm25),
                "graph_candidates": len(graph),
                "reference_candidates": len(ref),
                "final_candidates": len(final_scores),
            },
        }
