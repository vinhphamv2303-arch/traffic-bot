from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .candidates import collect_span_candidates
from .scoring import score_candidates_embedding
from .seeds import build_seeds


def run_xner_mining(
    sentences_root: str | Path,
    gazetteer_root: str | Path,
    output_dir: str | Path,
    manual_seed_file: str | Path | None = None,
    max_sentences: int | None = None,
    max_ngram: int = 14,
    min_surface_count: int = 1,
    embedding_model: str = "BAAI/bge-m3",
    batch_size: int = 256,
    min_score: float = 0.35,
    device: str | None = "cuda",
):
    output_dir = Path(output_dir)
    seed_summary = build_seeds(
        gazetteer_root=gazetteer_root,
        output_dir=output_dir,
        manual_seed_file=manual_seed_file,
    )
    cand_summary = collect_span_candidates(
        sentences_root=sentences_root,
        output_dir=output_dir,
        max_sentences=max_sentences,
        max_ngram=max_ngram,
        include_path_text=True,
        min_surface_count=min_surface_count,
    )
    mining_summary = score_candidates_embedding(
        seeds_path=output_dir / "seeds.jsonl",
        candidates_path=output_dir / "span_candidates.jsonl",
        output_dir=output_dir,
        embedding_model=embedding_model,
        batch_size=batch_size,
        min_score=min_score,
        top_k_seeds_per_candidate=10,
        device=device,
    )
    return {
        "seed_summary": seed_summary,
        "candidate_summary": cand_summary,
        "mining_summary": mining_summary,
    }
