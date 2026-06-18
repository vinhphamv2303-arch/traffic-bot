from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .gazetteer import GazetteerMatcher
from .utils import collapse_ws, cosine_matrix, load_pickle, minmax_normalize, normalize_for_match, read_json, read_jsonl, topk_dict


DEFAULT_QUERY_GLINER_MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "ner_finetuning"
    / "data"
    / "models"
    / "gliner_traffic_ner"
    / "final_model"
)

QUERY_GLINER_LABELS = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]

QUERY_GLINER_LINK_LABELS = set(QUERY_GLINER_LABELS)
SUBJECT_ANCHOR_LABELS = {"ACTOR", "VEHICLE"}
PRECISE_ANCHOR_TYPES = {"exact", "query_lexical_anchor", "query_gliner"}

QUERY_ENTITY_SYNONYMS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("VEHICLE", "o to"): [("VEHICLE", "xe o to")],
    ("VEHICLE", "oto"): [("VEHICLE", "xe o to")],
    ("VEHICLE", "xe o to"): [("VEHICLE", "xe o to")],
    ("ACTOR", "nguoi dieu khien o to"): [("ACTOR", "nguoi dieu khien xe o to"), ("VEHICLE", "xe o to")],
    ("ACTOR", "nguoi lai o to"): [("ACTOR", "nguoi dieu khien xe o to"), ("VEHICLE", "xe o to")],
    ("VEHICLE", "xe may"): [("VEHICLE", "xe mo to"), ("VEHICLE", "xe gan may")],
    ("VEHICLE", "mo to"): [("VEHICLE", "xe mo to")],
    ("VEHICLE", "xe mo to"): [("VEHICLE", "xe mo to")],
    ("VEHICLE", "xe gan may"): [("VEHICLE", "xe gan may")],
    ("ACTOR", "nguoi dieu khien xe may"): [("VEHICLE", "xe mo to"), ("VEHICLE", "xe gan may")],
    ("ACTOR", "nguoi lai xe may"): [("VEHICLE", "xe mo to"), ("VEHICLE", "xe gan may")],
    ("BEHAVIOR", "vuot den do"): [("BEHAVIOR", "khong chap hanh hieu lenh cua den tin hieu giao thong")],
    ("BEHAVIOR", "vuot den tin hieu"): [("BEHAVIOR", "khong chap hanh hieu lenh cua den tin hieu giao thong")],
    ("BEHAVIOR", "khong chap hanh hieu lenh cua den tin hieu giao thong"): [
        ("BEHAVIOR", "khong chap hanh hieu lenh cua den tin hieu giao thong")
    ],
}


def _contains_normalized_phrase(text_norm: str, phrase_norm: str) -> bool:
    if not text_norm or not phrase_norm:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


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
        self.entity_text_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for entity in self.entities:
            entity_id = entity.get("entity_id")
            label = str(entity.get("label") or "").upper()
            if not entity_id or not label:
                continue
            texts = [entity.get("canonical") or ""]
            texts.extend(entity.get("aliases") or [])
            for text in texts:
                norm = normalize_for_match(text)
                if norm:
                    self.entity_text_lookup[(label, norm)].append(entity_id)

        self.entity_to_passages = read_json(self.index_dir / "entity_to_passages.json")
        self.passage_to_entities = read_json(self.index_dir / "passage_to_entities.json")
        self.passage_neighbors = read_json(self.index_dir / "passage_neighbors.json")
        self.bm25 = load_pickle(self.index_dir / "bm25.pkl")
        self.gazetteer = GazetteerMatcher.from_gazetteer_root(gazetteer_root)

        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.passage_embeddings = None
        self.entity_embeddings = None
        self.query_gliner_model = None
        self.query_gliner_model_dir: Path | None = None

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

    def _load_query_gliner_model(self, model_dir: str | Path | None = None):
        model_path = Path(model_dir) if model_dir else DEFAULT_QUERY_GLINER_MODEL_DIR
        if self.query_gliner_model is not None and self.query_gliner_model_dir == model_path:
            return self.query_gliner_model
        if not model_path.exists():
            raise FileNotFoundError(f"Query GLiNER model not found: {model_path}")
        from gliner import GLiNER

        self.query_gliner_model = GLiNER.from_pretrained(str(model_path))
        self.query_gliner_model_dir = model_path
        return self.query_gliner_model

    def _entity_ids_for_text(self, label: str, text_norm: str) -> List[str]:
        return list(dict.fromkeys(self.entity_text_lookup.get((label.upper(), text_norm), [])))

    def _query_anchor_graph_weight(self, entity_id: str, entity: Dict[str, Any]) -> float:
        base = float(entity.get("min_graph_weight") or 1.0)
        degree = len(self.entity_to_passages.get(entity_id) or [])
        if degree > 50:
            base = min(base, (50.0 / float(degree)) ** 0.5)
        return max(0.05, min(1.0, base))

    def _prefer_surface_exact_entity_ids(self, entity_ids: List[str], surface: str) -> List[str]:
        surface_key = collapse_ws((surface or "").casefold())
        if not surface_key:
            return entity_ids
        exact_ids: List[str] = []
        for entity_id in entity_ids:
            entity = self.entity_by_id.get(entity_id) or {}
            texts = [entity.get("canonical") or ""]
            texts.extend(entity.get("aliases") or [])
            for text in texts:
                if collapse_ws(str(text or "").casefold()) == surface_key:
                    exact_ids.append(entity_id)
                    break
        return exact_ids or entity_ids

    def _subject_anchor_entity_ids(self, evidence: List[Dict[str, Any]]) -> set[str]:
        subject_ids: set[str] = set()
        for item in evidence:
            entity_id = item.get("entity_id")
            label = str(item.get("label") or "").upper()
            activation_type = item.get("activation_type")
            if entity_id and label in SUBJECT_ANCHOR_LABELS and activation_type in PRECISE_ANCHOR_TYPES:
                subject_ids.add(str(entity_id))
        return subject_ids

    def _passage_entity_ids(self, passage_id: str) -> set[str]:
        entity_ids: set[str] = set()
        for item in self.passage_to_entities.get(passage_id) or []:
            if isinstance(item, dict):
                entity_id = item.get("entity_id")
            else:
                entity_id = item
            if entity_id:
                entity_ids.add(str(entity_id))
        return entity_ids

    def _passage_matches_entity_anchor(self, passage_id: str, anchor_entity_ids: set[str]) -> bool:
        if not anchor_entity_ids:
            return True
        if self._passage_entity_ids(passage_id) & anchor_entity_ids:
            return True

        passage = self.passage_by_id.get(passage_id) or {}
        haystack = normalize_for_match(
            " ".join(
                [
                    str(passage.get("document_title") or ""),
                    str(passage.get("path_text") or ""),
                    str(passage.get("passage_text") or ""),
                ]
            )
        )
        if not haystack:
            return False

        for entity_id in anchor_entity_ids:
            entity = self.entity_by_id.get(entity_id) or {}
            texts = [entity.get("canonical") or ""]
            texts.extend(entity.get("aliases") or [])
            for text in texts:
                text_norm = normalize_for_match(str(text or ""))
                if text_norm and _contains_normalized_phrase(haystack, text_norm):
                    return True
        return False

    def _link_query_mention_to_entities(
        self,
        surface: str,
        label: str,
        max_entities_per_mention: int = 3,
    ) -> tuple[List[str], str]:
        label = (label or "").upper()
        surface_norm = normalize_for_match(surface)
        if not label or not surface_norm:
            return [], "empty"

        linked: List[str] = []
        synonym_specs = QUERY_ENTITY_SYNONYMS.get((label, surface_norm), [])
        if synonym_specs:
            for target_label, target_norm in synonym_specs:
                linked.extend(self._entity_ids_for_text(target_label, target_norm))
            return list(dict.fromkeys(linked))[:max_entities_per_mention], "synonym"

        linked.extend(self._entity_ids_for_text(label, surface_norm))
        linked = self._prefer_surface_exact_entity_ids(list(dict.fromkeys(linked)), surface)
        return list(dict.fromkeys(linked))[:max_entities_per_mention], "exact_entity_text"

    def query_lexical_entity_matches(self, query: str) -> List[Dict[str, Any]]:
        query_norm = normalize_for_match(query)
        matches: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for (source_label, surface_norm), synonym_specs in sorted(
            QUERY_ENTITY_SYNONYMS.items(),
            key=lambda item: len(item[0][1]),
            reverse=True,
        ):
            if not _contains_normalized_phrase(query_norm, surface_norm):
                continue
            for target_label, target_norm in synonym_specs:
                for entity_id in self._entity_ids_for_text(target_label, target_norm):
                    if entity_id in seen:
                        continue
                    seen.add(entity_id)
                    entity = self.entity_by_id.get(entity_id) or {}
                    graph_weight = self._query_anchor_graph_weight(entity_id, entity)
                    matches.append(
                        {
                            "entity_id": entity_id,
                            "surface": surface_norm,
                            "canonical": entity.get("canonical"),
                            "label": entity.get("label"),
                            "source_label": source_label,
                            "score": 0.95 * graph_weight,
                            "confidence": 0.95,
                            "graph_weight": graph_weight,
                            "linked_by": "query_lexical_anchor",
                            "activation_type": "query_lexical_anchor",
                            "is_generic_hub": entity.get("is_generic_hub", False),
                        }
                    )
        return matches

    def query_gliner_entity_matches(
        self,
        query: str,
        model_dir: str | Path | None = None,
        threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        model = self._load_query_gliner_model(model_dir)
        try:
            predictions = model.predict_entities(query, QUERY_GLINER_LABELS, threshold=threshold)
        except Exception:
            predictions = model.batch_predict_entities([query], QUERY_GLINER_LABELS, threshold=threshold)[0]

        matches: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, int | None, int | None]] = set()
        for pred in predictions or []:
            label = str(pred.get("label") or "").upper()
            if label not in QUERY_GLINER_LINK_LABELS:
                continue
            surface = str(pred.get("text") or "").strip()
            if not surface:
                continue
            confidence = float(pred.get("score", pred.get("confidence", 1.0)) or 0.0)
            if confidence < threshold:
                continue
            entity_ids, linked_by = self._link_query_mention_to_entities(surface, label)
            for entity_id in entity_ids:
                entity = self.entity_by_id.get(entity_id) or {}
                key = (entity_id, normalize_for_match(surface), pred.get("start"), pred.get("end"))
                if key in seen:
                    continue
                seen.add(key)
                graph_weight = self._query_anchor_graph_weight(entity_id, entity)
                matches.append(
                    {
                        "entity_id": entity_id,
                        "surface": surface,
                        "canonical": entity.get("canonical"),
                        "label": entity.get("label"),
                        "gliner_label": label,
                        "score": confidence * graph_weight,
                        "confidence": confidence,
                        "graph_weight": graph_weight,
                        "start": pred.get("start"),
                        "end": pred.get("end"),
                        "linked_by": linked_by,
                        "activation_type": "query_gliner",
                        "is_generic_hub": entity.get("is_generic_hub", False),
                    }
                )
        return matches

    def activate_entities(
        self,
        query: str,
        query_vec=None,
        exact_weight: float = 1.0,
        semantic_entity_top_k: int = 20,
        semantic_entity_min_score: float = 0.60,
        use_query_gliner: bool = False,
        query_gliner_model_dir: str | Path | None = None,
        query_gliner_threshold: float = 0.85,
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        activated: Dict[str, float] = {}
        evidence = []

        exact_matches = self.gazetteer.match(query)
        seen_exact_evidence: set[tuple[str, str]] = set()
        for m in exact_matches:
            eid = m.get("entity_id")
            if not eid or eid not in self.entity_by_id:
                continue
            entity = self.entity_by_id.get(eid) or {}
            graph_weight = float(m.get("graph_weight", 1.0))
            graph_weight = min(graph_weight, self._query_anchor_graph_weight(eid, entity))
            score = exact_weight * graph_weight
            activated[eid] = max(activated.get(eid, 0.0), score)
            evidence_key = (eid, normalize_for_match(str(m.get("surface") or "")))
            if evidence_key not in seen_exact_evidence:
                seen_exact_evidence.add(evidence_key)
                evidence.append({**m, "score": score, "graph_weight": graph_weight, "activation_type": "exact"})

        for m in self.query_lexical_entity_matches(query):
            eid = m.get("entity_id")
            if not eid:
                continue
            score = float(m.get("score", 0.0))
            activated[eid] = max(activated.get(eid, 0.0), score)
            evidence.append(m)

        if use_query_gliner:
            for m in self.query_gliner_entity_matches(
                query,
                model_dir=query_gliner_model_dir,
                threshold=query_gliner_threshold,
            ):
                eid = m.get("entity_id")
                if not eid:
                    continue
                score = float(m.get("score", 0.0))
                activated[eid] = max(activated.get(eid, 0.0), score)
                evidence.append(m)

        has_precise_subject_anchor = any(
            e.get("activation_type") in {"exact", "query_lexical_anchor", "query_gliner"}
            and str(e.get("label") or "").upper() in {"VEHICLE", "ACTOR"}
            for e in evidence
        )
        has_precise_behavior_anchor = any(
            e.get("activation_type") in {"exact", "query_lexical_anchor", "query_gliner"}
            and str(e.get("label") or "").upper() == "BEHAVIOR"
            for e in evidence
        )

        # Local semantic bridging: query -> nearby entity nodes.
        if (
            query_vec is not None
            and self.entity_embeddings is not None
            and len(self.entities) > 0
            and not (has_precise_subject_anchor and has_precise_behavior_anchor)
        ):
            sims = cosine_matrix(query_vec, self.entity_embeddings)
            idxs = np.argsort(-sims)[:semantic_entity_top_k]
            for idx in idxs:
                sim = float(sims[idx])
                if sim < semantic_entity_min_score:
                    continue
                e = self.entities[int(idx)]
                eid = e["entity_id"]
                label = str(e.get("label") or "").upper()
                if has_precise_subject_anchor and label in {"VEHICLE", "ACTOR", "DOCUMENT"}:
                    continue
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
        subject_anchor_entity_ids: set[str] | None = None,
        subject_mismatch_penalty: float = 0.20,
    ) -> Dict[str, float]:
        scores = defaultdict(float)
        subject_anchor_entity_ids = subject_anchor_entity_ids or set()
        for eid, e_score in activated_entities.items():
            links = self.entity_to_passages.get(eid) or []
            links = sorted(links, key=lambda x: (float(x.get("weight", 0)), int(x.get("mention_count", 1))), reverse=True)[:top_k_per_entity]
            for l in links:
                pid = l.get("passage_id")
                if not self._has_retrievable_text(pid):
                    continue
                edge_w = float(l.get("weight", 1.0))
                mention_boost = min(1.5, 1.0 + 0.1 * max(int(l.get("mention_count", 1)) - 1, 0))
                contribution = e_score * edge_w * mention_boost
                if (
                    subject_anchor_entity_ids
                    and eid not in subject_anchor_entity_ids
                    and not self._passage_matches_entity_anchor(pid, subject_anchor_entity_ids)
                ):
                    contribution *= subject_mismatch_penalty
                scores[pid] += contribution
        return dict(scores)

    def reference_expand_scores(
        self,
        base_scores: Dict[str, float],
        max_seed_passages: int = 50,
        ref_decay: float = 0.35,
    ) -> Dict[str, float]:
        expanded: Dict[str, float] = {}
        for pid, score in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max_seed_passages]:
            if not self._has_retrievable_text(pid):
                continue
            for nb in self.passage_neighbors.get(pid) or []:
                target = nb.get("passage_id")
                if not self._has_retrievable_text(target):
                    continue
                edge_weight = max(0.0, min(1.0, float(nb.get("weight", 1.0))))
                propagated_score = float(score) * ref_decay * edge_weight
                if propagated_score > expanded.get(target, 0.0):
                    expanded[target] = propagated_score
        return expanded

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 300,
        semantic_entity_top_k: int = 20,
        semantic_entity_min_score: float = 0.60,
        weights: Dict[str, float] | None = None,
        use_reference_expansion: bool = True,
        reference_seed_weights: Dict[str, float] | None = None,
        reference_max_seed_passages: int = 30,
        graph_only_penalty: float = 0.65,
        use_query_gliner: bool = False,
        query_gliner_model_dir: str | Path | None = None,
        query_gliner_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        weights = weights or {
            "dense": 0.15,
            "bm25": 0.25,
            "graph": 0.15,
            "reference": 0.30,
        }

        component_weights = {
            "dense": float(weights.get("dense", 0.0)),
            "bm25": float(weights.get("bm25", 0.0)),
            "graph": float(weights.get("graph", 0.0)),
            "reference": float(weights.get("reference", 0.0)),
        }
        if reference_seed_weights is None:
            reference_seed_weights = {
                "dense": component_weights["dense"],
                "bm25": component_weights["bm25"],
                "graph": component_weights["graph"],
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
                use_query_gliner=use_query_gliner,
                query_gliner_model_dir=query_gliner_model_dir,
                query_gliner_threshold=query_gliner_threshold,
            )
        else:
            activated_entities, entity_evidence = {}, []
        subject_anchor_entity_ids = self._subject_anchor_entity_ids(entity_evidence)
        subject_mismatch_penalty = 0.20

        dense = self.dense_passage_scores(query_vec, top_k=candidate_k) if needs_dense else {}
        bm25 = self.bm25_scores(query, top_k=candidate_k) if needs_bm25 else {}
        graph = (
            self.graph_scores(
                activated_entities,
                top_k_per_entity=candidate_k,
                subject_anchor_entity_ids=subject_anchor_entity_ids,
                subject_mismatch_penalty=subject_mismatch_penalty,
            )
            if needs_graph
            else {}
        )

        dense_n = minmax_normalize(dense)
        bm25_n = minmax_normalize(bm25)
        graph_n = minmax_normalize(graph)

        direct_pids = {
            pid
            for pid in (set(dense_n) | set(bm25_n) | set(graph_n))
            if self._has_retrievable_text(pid)
        }
        adjusted_graph_n = {}
        direct_scores = {}
        reference_seed = {}
        subject_match_factors = {}
        for pid in direct_pids:
            subject_match_factor = (
                1.0
                if not subject_anchor_entity_ids or self._passage_matches_entity_anchor(pid, subject_anchor_entity_ids)
                else subject_mismatch_penalty
            )
            subject_match_factors[pid] = subject_match_factor
            graph_score = graph_n.get(pid, 0.0)
            if graph_score > 0 and dense_n.get(pid, 0.0) == 0 and bm25_n.get(pid, 0.0) == 0:
                graph_score *= graph_only_penalty
            adjusted_graph_n[pid] = graph_score

            direct_score = (
                component_weights["dense"] * dense_n.get(pid, 0.0)
                + component_weights["bm25"] * bm25_n.get(pid, 0.0)
                + component_weights["graph"] * graph_score
            )
            direct_score *= subject_match_factor
            if direct_score > 0.0:
                direct_scores[pid] = direct_score

            direct_seed_score = (
                reference_seed_weights["dense"] * dense_n.get(pid, 0.0)
                + reference_seed_weights["bm25"] * bm25_n.get(pid, 0.0)
                + reference_seed_weights["graph"] * graph_score
            )
            direct_seed_score *= subject_match_factor
            if direct_seed_score > 0.0:
                reference_seed[pid] = direct_seed_score

        ref = self.reference_expand_scores(
            reference_seed,
            max_seed_passages=reference_max_seed_passages,
            ref_decay=1.0,
        ) if uses_reference else {}

        all_pids = {
            pid
            for pid in (set(direct_scores) | set(ref))
            if self._has_retrievable_text(pid)
        }
        final_scores = {}
        for pid in all_pids:
            subject_match_factor = subject_match_factors.get(pid)
            if subject_match_factor is None:
                subject_match_factor = (
                    1.0
                    if not subject_anchor_entity_ids or self._passage_matches_entity_anchor(pid, subject_anchor_entity_ids)
                    else subject_mismatch_penalty
                )
                subject_match_factors[pid] = subject_match_factor
            final_score = (
                direct_scores.get(pid, 0.0)
                + component_weights["reference"] * ref.get(pid, 0.0) * subject_match_factor
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
                    "reference": round(ref.get(pid, 0.0), 6),
                    "direct": round(direct_scores.get(pid, 0.0), 6),
                    "subject_match": round(subject_match_factors.get(pid, 1.0), 6),
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
                "reference_seed_mode": "weighted_direct_component_score",
                "reference_score_mode": "propagated_direct_score_without_independent_normalization",
                "reference_max_seed_passages": reference_max_seed_passages,
                "subject_anchor_entity_ids": sorted(subject_anchor_entity_ids),
                "subject_mismatch_penalty": subject_mismatch_penalty,
                "query_gliner": {
                    "enabled": use_query_gliner,
                    "threshold": query_gliner_threshold,
                    "model_dir": str(Path(query_gliner_model_dir) if query_gliner_model_dir else DEFAULT_QUERY_GLINER_MODEL_DIR),
                },
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
