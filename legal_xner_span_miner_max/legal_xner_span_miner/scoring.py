from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import math

from .common import ensure_dir, normalize_key, read_jsonl, safe_float, write_csv, write_json, write_jsonl


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def try_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers first: pip install sentence-transformers") from e


def cosine(a, b) -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def build_seed_text(seed: Dict[str, Any]) -> str:
    return f"{seed.get('label')}: {seed.get('surface')} ; {seed.get('canonical') or seed.get('surface')}"


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    # Use examples so score is contextual, not just string-level.
    ex_texts = []
    for ex in candidate.get("examples") or []:
        t = ex.get("text") or ""
        if len(t) > 500:
            t = t[:500]
        ex_texts.append(t)
    ctx = "\n".join(ex_texts[:3])
    return f"{candidate.get('label')}: {candidate.get('surface')}\nContext:\n{ctx}"


def score_candidates_embedding(
    seeds_path: str | Path,
    candidates_path: str | Path,
    output_dir: str | Path,
    embedding_model: str = "BAAI/bge-m3",
    batch_size: int = 256,
    top_k_seeds_per_candidate: int = 10,
    min_score: float = 0.35,
    device: str | None = "cuda",
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    seeds = load_rows(seeds_path)
    candidates = load_rows(candidates_path)

    if not seeds:
        raise ValueError(f"No seeds found in {seeds_path}")
    if not candidates:
        write_jsonl(output_dir / "mined_candidates.jsonl", [])
        write_csv(output_dir / "mined_candidates.csv", [], [
            "surface", "label", "canonical", "count", "score", "status", "best_seed", "example_text", "example_path"
        ])
        summary = {
            "seed_count": len(seeds),
            "candidate_count": 0,
            "mined_count": 0,
            "embedding_model": embedding_model,
            "min_score": min_score,
            "device": device,
            "top_k_seeds_per_candidate": top_k_seeds_per_candidate,
            "outputs": {
                "mined_candidates_jsonl": str(output_dir / "mined_candidates.jsonl"),
                "mined_candidates_csv": str(output_dir / "mined_candidates.csv"),
            },
        }
        write_json(output_dir / "mining_summary.json", summary)
        return summary

    # Group seeds by label.
    seeds_by_label = defaultdict(list)
    for s in seeds:
        seeds_by_label[s["label"]].append(s)

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    SentenceTransformer = try_import_sentence_transformer()
    model = SentenceTransformer(embedding_model, device=device)

    # Embed all seeds.
    seed_texts = [build_seed_text(s) for s in seeds]
    seed_emb = model.encode(seed_texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)

    seed_index_by_label = defaultdict(list)
    for i, s in enumerate(seeds):
        seed_index_by_label[s["label"]].append(i)

    cand_texts = [build_candidate_text(c) for c in candidates]
    cand_emb = model.encode(cand_texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

    mined = []
    for ci, cand in enumerate(candidates):
        label = cand.get("label")
        seed_idxs = seed_index_by_label.get(label) or []
        if not seed_idxs:
            continue

        scored = []
        for si in seed_idxs:
            sim = cosine(cand_emb[ci], seed_emb[si])
            scored.append((sim, seeds[si]))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seed = scored[0]
        avg_top = sum(x[0] for x in scored[:top_k_seeds_per_candidate]) / max(1, min(top_k_seeds_per_candidate, len(scored)))

        # Combine semantic score with frequency saturation.
        count = int(cand.get("count") or 1)
        freq_bonus = min(0.08, math.log1p(count) / 50.0)
        final_score = 0.75 * best_score + 0.25 * avg_top + freq_bonus

        if final_score < min_score:
            continue

        status = "accept_candidate" if final_score >= 0.72 and count >= 2 else "review"
        mined.append({
            "candidate_id": cand.get("candidate_id"),
            "surface": cand.get("surface"),
            "normalized_key": cand.get("normalized_key"),
            "label": label,
            "canonical": cand.get("surface"),
            "count": count,
            "score": round(float(final_score), 6),
            "best_seed": best_seed.get("surface"),
            "best_seed_source": best_seed.get("source"),
            "best_seed_score": round(float(best_score), 6),
            "avg_top_seed_score": round(float(avg_top), 6),
            "status": status,
            "examples": cand.get("examples") or [],
        })

    mined.sort(key=lambda x: (-x["score"], -x["count"], x["label"], x["surface"]))
    write_jsonl(output_dir / "mined_candidates.jsonl", mined)

    csv_rows = []
    for r in mined:
        ex = (r.get("examples") or [{}])[0]
        csv_rows.append({
            "surface": r.get("surface"),
            "label": r.get("label"),
            "canonical": r.get("canonical"),
            "count": r.get("count"),
            "score": r.get("score"),
            "status": r.get("status"),
            "best_seed": r.get("best_seed"),
            "example_text": ex.get("text"),
            "example_path": ex.get("path_text"),
        })
    write_csv(output_dir / "mined_candidates.csv", csv_rows, [
        "surface", "label", "canonical", "count", "score", "status", "best_seed", "example_text", "example_path"
    ])

    summary = {
        "seed_count": len(seeds),
        "candidate_count": len(candidates),
        "mined_count": len(mined),
        "embedding_model": embedding_model,
        "min_score": min_score,
        "device": device,
        "top_k_seeds_per_candidate": top_k_seeds_per_candidate,
        "outputs": {
            "mined_candidates_jsonl": str(output_dir / "mined_candidates.jsonl"),
            "mined_candidates_csv": str(output_dir / "mined_candidates.csv"),
        }
    }
    write_json(output_dir / "mining_summary.json", summary)
    return summary
