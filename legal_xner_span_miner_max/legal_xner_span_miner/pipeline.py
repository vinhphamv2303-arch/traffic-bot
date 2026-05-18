from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .candidates import collect_span_candidates
from .common import log
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
    skip_scoring: bool = False,
):
    output_dir = Path(output_dir)
    log("[xner] start pipeline")
    log(f"[xner] output_dir={output_dir}")
    log("[xner] step 1/3: build seeds")
    seed_summary = build_seeds(
        gazetteer_root=gazetteer_root,
        output_dir=output_dir,
        manual_seed_file=manual_seed_file,
    )
    log(f"[xner] seeds done: {seed_summary}")

    log("[xner] step 2/3: generate span candidates")
    cand_summary = collect_span_candidates(
        sentences_root=sentences_root,
        output_dir=output_dir,
        max_sentences=max_sentences,
        max_ngram=max_ngram,
        include_path_text=True,
        min_surface_count=min_surface_count,
    )
    log(f"[xner] candidates done: {cand_summary}")

    if skip_scoring:
        mining_summary = {
            "skipped": True,
            "reason": "skip_scoring",
            "next_command": (
                "python score_candidates.py "
                f"--seeds \"{output_dir / 'seeds.jsonl'}\" "
                f"--candidates \"{output_dir / 'span_candidates.jsonl'}\" "
                f"--output \"{output_dir}\""
            ),
        }
        log("[xner] step 3/3 skipped: score candidates")
    else:
        log("[xner] step 3/3: score candidates with embedding model")
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
        log(f"[xner] scoring done: {mining_summary}")
    log("[xner] pipeline completed")
    return {
        "seed_summary": seed_summary,
        "candidate_summary": cand_summary,
        "mining_summary": mining_summary,
    }
