from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval_pipelines_builder.legal_linearrag_retriever.legal_linearrag_retriever import LinearRAGRetriever  # noqa: E402


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "naive_bm25": {
        "index_dir": ROOT / "data/retrieval/index_bm25_graph",
        "output_name": "traffic_rag_retrieval_naive_bm25_top5.json",
        "weights": {"dense": 0.0, "bm25": 1.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "naive_dense": {
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "output_name": "traffic_rag_retrieval_naive_dense_top5.json",
        "weights": {"dense": 1.0, "bm25": 0.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "no_embedding": {
        "index_dir": ROOT / "data/retrieval/index_bm25_graph",
        "output_name": "traffic_rag_retrieval_no_embedding_top5.json",
        "weights": {"dense": 0.0, "bm25": 0.25, "graph": 0.15, "reference": 0.60},
    },
    "bge_m3": {
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "output_name": "traffic_rag_retrieval_bge_m3_top5.json",
        "weights": {"dense": 0.25, "bm25": 0.25, "graph": 0.20, "reference": 0.30},
    },
    "bge_m3_no_graph": {
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "output_name": "traffic_rag_retrieval_bge_m3_no_graph_top5.json",
        "weights": {"dense": 0.70, "bm25": 0.0, "graph": 0.0, "reference": 0.30},
        "reference_seed_weights": {"dense": 1.0, "bm25": 0.0, "graph": 0.0},
    },
    "minilm": {
        "index_dir": ROOT / "data/retrieval/index_minilm_hybrid",
        "output_name": "traffic_rag_retrieval_minilm_top5.json",
        "weights": {"dense": 0.20, "bm25": 0.30, "graph": 0.20, "reference": 0.30},
    },
    "minilm_no_graph": {
        "index_dir": ROOT / "data/retrieval/index_minilm_hybrid",
        "output_name": "traffic_rag_retrieval_minilm_no_graph_top5.json",
        "weights": {"dense": 0.70, "bm25": 0.0, "graph": 0.0, "reference": 0.30},
        "reference_seed_weights": {"dense": 1.0, "bm25": 0.0, "graph": 0.0},
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id"):
                rows[row["id"]] = row
    return rows


def compact_result(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "passage_id": row.get("passage_id"),
        "score": row.get("score"),
        "score_components": row.get("score_components"),
        "document_number": row.get("document_number"),
        "document_id": row.get("document_id"),
        "package_id": row.get("package_id"),
        "path_text": row.get("path_text"),
        "text": row.get("text") or "",
    }


def benchmark_item_base(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "seq_id": item.get("seq_id"),
        "benchmark_group": item.get("benchmark_group"),
        "is_multi_reference": item.get("is_multi_reference"),
        "question": item.get("question"),
        "gold_doc_numbers": item.get("gold_doc_numbers") or [],
        "gold_citation_spans": item.get("gold_citation_spans") or [],
        "gold_reference_text": item.get("reference_text"),
        "answer": item.get("answer"),
    }


def validate_index(config: dict[str, Any]) -> dict[str, Any]:
    index_dir = Path(config["index_dir"])
    summary_path = index_dir / "index_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing index summary: {summary_path}")
    summary = read_json(summary_path)
    required = [
        "passages.jsonl",
        "entities.jsonl",
        "bm25.pkl",
        "entity_to_passages.json",
        "passage_to_entities.json",
        "passage_neighbors.json",
    ]
    if not summary.get("skip_embeddings"):
        required.extend(["passage_embeddings.npy", "entity_embeddings.npy"])
    missing = [name for name in required if not (index_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing index files in {index_dir}: {missing}")

    needs_dense = float((config.get("weights") or {}).get("dense", 0.0)) > 0.0
    if needs_dense and (summary.get("skip_embeddings") or not summary.get("embedding_model")):
        raise ValueError(
            f"Model requires dense retrieval but index is not dense-ready: {index_dir}. "
            f"index_summary skip_embeddings={summary.get('skip_embeddings')}, "
            f"embedding_model={summary.get('embedding_model')!r}"
        )
    return summary


def run_model(
    model_key: str,
    config: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    benchmark_path: Path,
    output_dir: Path,
    gazetteer_root: Path,
    top_k: int,
    candidate_k: int,
    semantic_entity_top_k: int,
    semantic_entity_min_score: float,
    resume: bool,
    overwrite: bool,
) -> Path:
    index_summary = validate_index(config)
    output_path = output_dir / config["output_name"]
    partial_path = output_path.with_suffix(".partial.jsonl")

    if overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    completed = load_partial(partial_path) if resume else {}
    if output_path.exists() and not overwrite and not completed:
        existing = read_json(output_path)
        done_count = len(existing.get("items") or [])
        if done_count >= len(benchmark_rows):
            print(f"[{model_key}] skip existing complete output: {output_path}")
            return output_path

    print(f"[{model_key}] loading index: {config['index_dir']}")
    t_load = time.time()
    retriever = LinearRAGRetriever.from_index(config["index_dir"], gazetteer_root)
    load_seconds = round(time.time() - t_load, 3)
    print(f"[{model_key}] loaded in {load_seconds}s")

    for idx, item in enumerate(benchmark_rows, 1):
        item_id = item.get("id")
        if resume and item_id in completed:
            print(f"[{model_key}] [{idx}/{len(benchmark_rows)}] skip {item_id}")
            continue

        question = item.get("question") or ""
        rec = benchmark_item_base(item)
        rec["model_key"] = model_key
        rec["index_dir"] = str(config["index_dir"])
        rec["weights"] = config["weights"]
        rec["use_reference_expansion"] = bool(config.get("use_reference_expansion", True))
        rec["reference_seed_weights"] = config.get("reference_seed_weights")
        rec["top_k"] = top_k

        t0 = time.time()
        try:
            retrieved = retriever.retrieve(
                query=question,
                top_k=top_k,
                candidate_k=candidate_k,
                semantic_entity_top_k=semantic_entity_top_k,
                semantic_entity_min_score=semantic_entity_min_score,
                weights=config["weights"],
                use_reference_expansion=bool(config.get("use_reference_expansion", True)),
                reference_seed_weights=config.get("reference_seed_weights"),
            )
            rec["activated_entities"] = retrieved.get("activated_entities") or []
            rec["top_results"] = [
                compact_result(row, rank)
                for rank, row in enumerate(retrieved.get("results") or [], 1)
            ]
            rec["debug"] = retrieved.get("debug") or {}
            rec["error"] = None
        except Exception as exc:
            rec["activated_entities"] = []
            rec["top_results"] = []
            rec["debug"] = {}
            rec["error"] = {"type": type(exc).__name__, "message": str(exc)}
        rec["elapsed_seconds"] = round(time.time() - t0, 3)
        append_jsonl(partial_path, rec)
        print(f"[{model_key}] [{idx}/{len(benchmark_rows)}] {item_id} top={len(rec['top_results'])} err={rec['error'] is not None}")

    partial_rows = load_partial(partial_path)
    ordered_items = [partial_rows[item["id"]] for item in benchmark_rows if item.get("id") in partial_rows]
    errors = [row for row in ordered_items if row.get("error")]
    output = {
        "metadata": {
            "model_key": model_key,
            "index_dir": str(config["index_dir"]),
            "index_summary": {
                "passage_count": index_summary.get("passage_count"),
                "entity_count": index_summary.get("entity_count"),
                "embedding_model": index_summary.get("embedding_model"),
                "skip_embeddings": index_summary.get("skip_embeddings"),
            },
            "gazetteer_root": str(gazetteer_root),
            "benchmark": str(benchmark_path),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "semantic_entity_top_k": semantic_entity_top_k,
            "semantic_entity_min_score": semantic_entity_min_score,
            "weights": config["weights"],
            "use_reference_expansion": bool(config.get("use_reference_expansion", True)),
            "reference_seed_weights": config.get("reference_seed_weights"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "load_seconds": load_seconds,
            "question_count": len(benchmark_rows),
            "completed_count": len(ordered_items),
            "error_count": len(errors),
        },
        "items": ordered_items,
    }
    write_json(output_path, output)
    if not errors and len(ordered_items) == len(benchmark_rows):
        partial_path.unlink(missing_ok=True)
    print(f"[{model_key}] saved {output_path}")

    del retriever
    gc.collect()
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run traffic RAG benchmark top-5 retrieval for baseline and hybrid indexes.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/benchmark/traffic_rag_gold_questions_v1/traffic_rag_gold_questions_v1.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark/traffic_rag_final_retrieval_answer_benchmark_v1/retrieval")
    parser.add_argument("--gazetteer-root", type=Path, default=ROOT / "ner_finetuning/data/preprocessed/expanded_gazetteer")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=["naive_bm25", "naive_dense", "no_embedding", "bge_m3", "minilm"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--semantic-entity-top-k", type=int, default=20)
    parser.add_argument("--semantic-entity-min-score", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    benchmark_rows = read_jsonl(args.benchmark)
    if args.limit is not None:
        benchmark_rows = benchmark_rows[: args.limit]

    print(f"benchmark rows: {len(benchmark_rows)}")
    print(f"models: {', '.join(args.models)}")

    outputs = []
    for model_key in args.models:
        outputs.append(
            run_model(
                model_key=model_key,
                config=MODEL_CONFIGS[model_key],
                benchmark_rows=benchmark_rows,
                benchmark_path=args.benchmark,
                output_dir=args.output_dir,
                gazetteer_root=args.gazetteer_root,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                semantic_entity_top_k=args.semantic_entity_top_k,
                semantic_entity_min_score=args.semantic_entity_min_score,
                resume=not args.no_resume,
                overwrite=args.overwrite,
            )
        )

    print("outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
