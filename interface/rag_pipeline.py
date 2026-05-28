from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_generation.answerer import answer_one  # noqa: E402


RETRIEVER_SCRIPT = ROOT / "retrieval_pipelines_builder" / "legal_linearrag_retriever" / "retrieve.py"
GAZETTEER_ROOT = ROOT / "ner_finetuning" / "data" / "preprocessed" / "expanded_gazetteer"


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    key: str
    display_name: str
    index_dir: Path
    weights: dict[str, float]
    use_reference_expansion: bool
    description: str


PIPELINES: dict[str, RetrievalPipelineConfig] = {
    "hybrid_cpu": RetrievalPipelineConfig(
        key="hybrid_cpu",
        display_name="Hybrid CPU: BM25 + Graph + Reference",
        index_dir=ROOT / "data" / "retrieval" / "index_bm25_graph",
        weights={"dense": 0.0, "bm25": 0.25, "graph": 0.15, "reference": 0.60},
        use_reference_expansion=True,
        description="Nhanh nhất để demo local vì không cần load embedding model.",
    ),
    "hybrid_bge_m3": RetrievalPipelineConfig(
        key="hybrid_bge_m3",
        display_name="Hybrid BGE-M3: Dense + BM25 + Graph + Reference",
        index_dir=ROOT / "data" / "retrieval" / "index_bge_m3_hybrid",
        weights={"dense": 0.25, "bm25": 0.25, "graph": 0.20, "reference": 0.30},
        use_reference_expansion=True,
        description="Pipeline đầy đủ nhất, dùng BGE-M3 cho dense retrieval.",
    ),
    "hybrid_minilm": RetrievalPipelineConfig(
        key="hybrid_minilm",
        display_name="Hybrid MiniLM: Dense + BM25 + Graph + Reference",
        index_dir=ROOT / "data" / "retrieval" / "index_minilm_hybrid",
        weights={"dense": 0.20, "bm25": 0.30, "graph": 0.20, "reference": 0.30},
        use_reference_expansion=True,
        description="Pipeline dense nhẹ hơn BGE-M3, phù hợp CPU hơn.",
    ),
    "bm25": RetrievalPipelineConfig(
        key="bm25",
        display_name="Naive BM25",
        index_dir=ROOT / "data" / "retrieval" / "index_bm25_graph",
        weights={"dense": 0.0, "bm25": 1.0, "graph": 0.0, "reference": 0.0},
        use_reference_expansion=False,
        description="Baseline từ khóa, không dùng graph/reference.",
    ),
    "dense_bge_m3": RetrievalPipelineConfig(
        key="dense_bge_m3",
        display_name="Naive Dense BGE-M3",
        index_dir=ROOT / "data" / "retrieval" / "index_bge_m3_hybrid",
        weights={"dense": 1.0, "bm25": 0.0, "graph": 0.0, "reference": 0.0},
        use_reference_expansion=False,
        description="Baseline dense, không dùng graph/reference.",
    ),
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "local": "Qwen/Qwen2.5-7B-Instruct",
}


def validate_runtime_paths(config: RetrievalPipelineConfig) -> None:
    missing = []
    if not RETRIEVER_SCRIPT.exists():
        missing.append(str(RETRIEVER_SCRIPT))
    if not config.index_dir.exists():
        missing.append(str(config.index_dir))
    if not GAZETTEER_ROOT.exists():
        missing.append(str(GAZETTEER_ROOT))
    if missing:
        raise FileNotFoundError("Thiếu artifact để chạy demo:\n" + "\n".join(missing))


def run_demo_answer(
    question: str,
    pipeline_key: str = "hybrid_cpu",
    mode: str = "openai",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    top_k: int = 5,
    candidate_k: int = 300,
    max_context_passages: int = 5,
    max_chars_per_passage: int = 1800,
    answer_mode: str = "extractive_multi_agent",
    enable_query_router: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    config = PIPELINES[pipeline_key]
    validate_runtime_paths(config)
    weights = config.weights
    selected_model = model_name or DEFAULT_MODELS[mode]

    result = answer_one(
        query=question,
        model_name=selected_model,
        mode=mode,
        retriever_script=RETRIEVER_SCRIPT,
        index_dir=config.index_dir,
        gazetteer_root=GAZETTEER_ROOT,
        top_k=top_k,
        max_context_passages=max_context_passages,
        candidate_k=candidate_k,
        dense_weight=weights["dense"],
        bm25_weight=weights["bm25"],
        graph_weight=weights["graph"],
        reference_weight=weights["reference"],
        use_reference_expansion=config.use_reference_expansion,
        answer_mode=answer_mode,
        enable_query_rewrite=enable_query_router,
        api_key=api_key,
        base_url=base_url,
        max_chars_per_passage=max_chars_per_passage,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    result["pipeline"] = pipeline_key
    result["pipeline_display_name"] = config.display_name
    result["index_dir"] = str(config.index_dir)
    return result
