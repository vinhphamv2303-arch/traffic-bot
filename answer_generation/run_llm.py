from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_generation.answerer import (  # noqa: E402
    answer_one,
    apply_rule_based_query_rewrite,
    build_prompt,
    format_context,
    needs_retrieval_postprocess,
    postprocess_retrieval_for_query,
    repair_mojibake_text,
)


PIPELINES: dict[str, dict[str, Any]] = {
    "naive_bm25": {
        "index_dir": ROOT / "data/retrieval/index_bm25_graph",
        "weights": {"dense": 0.0, "bm25": 1.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "naive_dense": {
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "weights": {"dense": 1.0, "bm25": 0.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "bge_m3": {
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "weights": {"dense": 0.25, "bm25": 0.25, "graph": 0.20, "reference": 0.30},
        "use_reference_expansion": True,
    },
    "minilm": {
        "index_dir": ROOT / "data/retrieval/index_minilm_hybrid",
        "weights": {"dense": 0.20, "bm25": 0.30, "graph": 0.20, "reference": 0.30},
        "use_reference_expansion": True,
    },
}

DEFAULT_MODELS = {
    "local": "Qwen/Qwen2.5-7B-Instruct",
    "openai": "gpt-4o-mini",
    "openrouter": "qwen/qwen-2.5-7b-instruct",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run one RAG answer with local Hugging Face model or OpenAI/OpenRouter API.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["local", "openai", "openrouter"], default="openai")
    parser.add_argument("--model", default=None, help="HF model id for local mode, OpenAI model id, or OpenRouter model id.")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="bge_m3")
    parser.add_argument("--retriever-script", type=Path, default=ROOT / "retrieval_pipelines_builder/legal_linearrag_retriever/retrieve.py")
    parser.add_argument("--gazetteer-root", type=Path, default=ROOT / "ner_finetuning/data/preprocessed/expanded_gazetteer")
    parser.add_argument("--index-dir", type=Path, default=None, help="Override pipeline index directory.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--semantic-entity-top-k", type=int, default=20)
    parser.add_argument("--semantic-entity-min-score", type=float, default=0.45)
    parser.add_argument("--max-context-passages", type=int, default=5)
    parser.add_argument("--max-chars-per-passage", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--answer-mode", choices=["direct", "extractive_multi_agent"], default="extractive_multi_agent")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--no-query-router", action="store_true", help="Skip LLM-based query routing and rewrite.")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print retrieval context and prompt size without calling the LLM.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pipeline = PIPELINES[args.pipeline]
    weights = pipeline["weights"]
    model_name = args.model or DEFAULT_MODELS[args.mode]
    index_dir = args.index_dir or pipeline["index_dir"]

    if args.dry_run:
        from answer_generation.answerer import run_retriever

        retrieval_query = apply_rule_based_query_rewrite(args.query)
        retrieval_top_k = max(args.top_k, 40) if needs_retrieval_postprocess(args.query, retrieval_query) else args.top_k
        retrieval = run_retriever(
            retriever_script=args.retriever_script,
            index_dir=index_dir,
            gazetteer_root=args.gazetteer_root,
            query=retrieval_query,
            top_k=retrieval_top_k,
            candidate_k=args.candidate_k,
            semantic_entity_top_k=args.semantic_entity_top_k,
            semantic_entity_min_score=args.semantic_entity_min_score,
            dense_weight=weights["dense"],
            bm25_weight=weights["bm25"],
            graph_weight=weights["graph"],
            reference_weight=weights["reference"],
            use_reference_expansion=bool(pipeline.get("use_reference_expansion", True)),
        )
        retrieval = postprocess_retrieval_for_query(retrieval, args.query, retrieval_query, top_k=args.top_k)
        context = format_context(
            retrieval.get("results", []),
            max_passages=args.max_context_passages,
            max_chars_per_passage=args.max_chars_per_passage,
        )
        messages = build_prompt(args.query, context, answer_mode=args.answer_mode)
        result = {
            "query": repair_mojibake_text(args.query),
            "rewritten_query": repair_mojibake_text(retrieval_query),
            "route": "traffic_law",
            "query_router_skipped": True,
            "mode": args.mode,
            "model": model_name,
            "pipeline": args.pipeline,
            "index_dir": str(index_dir),
            "prompt_chars": sum(len(m["content"]) for m in messages),
            "context_used": context,
            "retrieval": retrieval,
        }
    else:
        result = answer_one(
            query=args.query,
            model_name=model_name,
            mode=args.mode,
            retriever_script=args.retriever_script,
            index_dir=index_dir,
            gazetteer_root=args.gazetteer_root,
            top_k=args.top_k,
            max_context_passages=args.max_context_passages,
            candidate_k=args.candidate_k,
            dense_weight=weights["dense"],
            bm25_weight=weights["bm25"],
            graph_weight=weights["graph"],
            reference_weight=weights["reference"],
            use_reference_expansion=bool(pipeline.get("use_reference_expansion", True)),
            semantic_entity_top_k=args.semantic_entity_top_k,
            semantic_entity_min_score=args.semantic_entity_min_score,
            load_4bit=args.load_4bit,
            dtype=args.dtype,
            device_map=args.device_map,
            answer_mode=args.answer_mode,
            enable_query_rewrite=not args.no_query_router,
            api_key=args.api_key,
            base_url=args.base_url,
            max_chars_per_passage=args.max_chars_per_passage,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        result["pipeline"] = args.pipeline
        result["index_dir"] = str(index_dir)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
