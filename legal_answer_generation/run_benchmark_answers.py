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

from legal_answer_generation.local_llm_answerer import (  # noqa: E402
    INSUFFICIENT_CONTEXT_ANSWER,
    PROMPT_VERSION,
    build_prompt,
    format_context,
    generate_answer,
    load_model,
    repair_mojibake,
    repair_mojibake_text,
)
from retrieval_pipelines.legal_linearrag_retriever.legal_linearrag_retriever import LinearRAGRetriever  # noqa: E402


DEFAULT_BENCHMARK_PATH = ROOT / "data/benchmark/traffic_rag_benchmark_v1/traffic_rag_benchmark_v1.jsonl"
DEFAULT_GAZETTEER_ROOT = ROOT / "ner_finetuning/data/preprocessed/expanded_gazetteer"


PIPELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "naive_bm25": {
        "display_name": "Naive BM25 RAG",
        "output_name": "traffic_rag_retrieval_naive_bm25_top5.json",
        "retrieval_file": ROOT / "data/benchmark/traffic_rag_retrieval_naive_bm25_top5.json",
        "index_dir": ROOT / "data/retrieval/index_bm25_graph",
        "weights": {"dense": 0.0, "bm25": 1.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "naive_dense": {
        "display_name": "Naive Dense RAG (BGE-M3)",
        "output_name": "traffic_rag_retrieval_naive_dense_top5.json",
        "retrieval_file": ROOT / "data/benchmark/traffic_rag_retrieval_naive_dense_top5.json",
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "weights": {"dense": 1.0, "bm25": 0.0, "graph": 0.0, "reference": 0.0},
        "use_reference_expansion": False,
    },
    "no_embedding": {
        "display_name": "No embedding",
        "output_name": "traffic_rag_retrieval_no_embedding_top5.json",
        "retrieval_file": ROOT / "data/benchmark/traffic_rag_retrieval_no_embedding_top5.json",
        "index_dir": ROOT / "data/retrieval/index_bm25_graph",
        "weights": {"dense": 0.0, "bm25": 0.25, "graph": 0.15, "reference": 0.60},
    },
    "bge_m3": {
        "display_name": "BGE-M3 hybrid",
        "output_name": "traffic_rag_retrieval_bge_m3_top5.json",
        "retrieval_file": ROOT / "data/benchmark/traffic_rag_retrieval_bge_m3_top5.json",
        "index_dir": ROOT / "data/retrieval/index_bge_m3_hybrid",
        "weights": {"dense": 0.25, "bm25": 0.25, "graph": 0.20, "reference": 0.30},
    },
    "minilm": {
        "display_name": "MiniLM hybrid",
        "output_name": "traffic_rag_retrieval_minilm_top5.json",
        "retrieval_file": ROOT / "data/benchmark/traffic_rag_retrieval_minilm_top5.json",
        "index_dir": ROOT / "data/retrieval/index_minilm_hybrid",
        "weights": {"dense": 0.20, "bm25": 0.30, "graph": 0.20, "reference": 0.30},
    },
}

GENERATION_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "llama_3_1_8b_instruct": {
        "display_name": "Llama-3.1-8B-Instruct",
        "model_name": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    },
    "qwen2_5_7b_instruct": {
        "display_name": "Qwen2.5-7B-Instruct",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen2_5_14b_instruct": {
        "display_name": "Qwen2.5-14B-Instruct",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
    },
}

MODEL_ALIASES = {
    # Legacy aliases from the previous command are mapped to Llama so old
    # benchmark commands keep running after Vistral became inaccessible.
    "Vistral-7B-Chat": "llama_3_1_8b_instruct",
    "vistral": "llama_3_1_8b_instruct",
    "vistral_7b_chat": "llama_3_1_8b_instruct",
    "Viet-Mistral/Vistral-7B-Chat": "llama_3_1_8b_instruct",
    "Llama-3.1-8B-Instruct": "llama_3_1_8b_instruct",
    "llama": "llama_3_1_8b_instruct",
    "llama_3_1_8b_instruct": "llama_3_1_8b_instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "llama_3_1_8b_instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama_3_1_8b_instruct",
    "Qwen2.5-7B-Instruct": "qwen2_5_7b_instruct",
    "qwen2.5-7b": "qwen2_5_7b_instruct",
    "qwen2_5_7b_instruct": "qwen2_5_7b_instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5_7b_instruct",
    "Qwen2.5-14B-Instruct": "qwen2_5_14b_instruct",
    "qwen2.5-14b": "qwen2_5_14b_instruct",
    "qwen2_5_14b_instruct": "qwen2_5_14b_instruct",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2_5_14b_instruct",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_items(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = read_json(path)
    rows = {}
    for row in data.get("items") or []:
        item_id = row.get("id")
        if item_id:
            rows[item_id] = row
    return rows


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
            item_id = row.get("id")
            if item_id:
                rows[item_id] = row
    return rows


def canonical_model_key(raw: str) -> str:
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]
    lowered = raw.lower()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]
    valid = ", ".join(sorted(GENERATION_MODEL_CONFIGS))
    raise ValueError(f"Unknown model alias: {raw}. Valid keys: {valid}")


def retrieval_output_path(config: dict[str, Any], args: argparse.Namespace) -> Path:
    existing_default = Path(config["retrieval_file"])
    if existing_default.exists() and not args.overwrite_retrieval:
        return existing_default
    if args.limit is not None:
        return args.output_dir / "_retrieval_cache" / config["output_name"]
    return args.retrieval_output_dir / config["output_name"]


def validate_retrieval_index(config: dict[str, Any]) -> dict[str, Any]:
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
            f"Pipeline requires dense retrieval but index is not dense-ready: {index_dir}. "
            f"index_summary skip_embeddings={summary.get('skip_embeddings')}, "
            f"embedding_model={summary.get('embedding_model')!r}"
        )
    return summary


def compact_retrieval_result(row: dict[str, Any], rank: int) -> dict[str, Any]:
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
        "question": repair_mojibake_text(item.get("question") or ""),
        "gold_doc_numbers": item.get("gold_doc_numbers") or [],
        "gold_citation_spans": repair_mojibake(item.get("gold_citation_spans") or []),
        "gold_reference_text": repair_mojibake_text(item.get("reference_text") or ""),
        "answer": repair_mojibake_text(item.get("answer") or ""),
    }


def generate_retrieval_file(
    pipeline_key: str,
    config: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")

    index_summary = validate_retrieval_index(config)
    rows = read_jsonl(benchmark_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    partial_path = output_path.with_suffix(".partial.jsonl")
    if args.overwrite_retrieval:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    completed = load_partial(partial_path)
    if output_path.exists() and not args.overwrite_retrieval and not completed:
        existing = read_json(output_path)
        if len(existing.get("items") or []) >= len(rows):
            return output_path

    print(f"[retrieve:{pipeline_key}] building missing retrieval file: {output_path}")
    print(f"[retrieve:{pipeline_key}] loading index: {config['index_dir']}")
    load_start = time.time()
    retriever = LinearRAGRetriever.from_index(config["index_dir"], args.gazetteer_root)
    load_seconds = round(time.time() - load_start, 3)
    print(f"[retrieve:{pipeline_key}] loaded in {load_seconds}s")

    for idx, item in enumerate(rows, start=1):
        item_id = item.get("id")
        if item_id in completed:
            print(f"[retrieve:{pipeline_key}] [{idx}/{len(rows)}] skip {item_id}")
            continue

        question = repair_mojibake_text(item.get("question") or "")
        record = benchmark_item_base(item)
        record["model_key"] = pipeline_key
        record["index_dir"] = str(config["index_dir"])
        record["weights"] = config["weights"]
        record["use_reference_expansion"] = bool(config.get("use_reference_expansion", True))
        record["reference_seed_weights"] = config.get("reference_seed_weights")
        record["top_k"] = args.top_k

        started = time.time()
        try:
            retrieved = retriever.retrieve(
                query=question,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                semantic_entity_top_k=args.semantic_entity_top_k,
                semantic_entity_min_score=args.semantic_entity_min_score,
                weights=config["weights"],
                use_reference_expansion=bool(config.get("use_reference_expansion", True)),
                reference_seed_weights=config.get("reference_seed_weights"),
            )
            record["activated_entities"] = retrieved.get("activated_entities") or []
            record["top_results"] = [
                compact_retrieval_result(result, rank)
                for rank, result in enumerate(retrieved.get("results") or [], start=1)
            ]
            record["debug"] = retrieved.get("debug") or {}
            record["error"] = None
        except Exception as exc:
            record["activated_entities"] = []
            record["top_results"] = []
            record["debug"] = {}
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}

        record["elapsed_seconds"] = round(time.time() - started, 3)
        completed[item_id] = record
        append_jsonl(partial_path, record)
        print(
            f"[retrieve:{pipeline_key}] [{idx}/{len(rows)}] "
            f"{item_id} top={len(record['top_results'])} err={record['error'] is not None}"
        )

    partial_rows = load_partial(partial_path)
    ordered = [partial_rows[item["id"]] for item in rows if item.get("id") in partial_rows]
    errors = [row for row in ordered if row.get("error")]
    output = {
        "metadata": {
            "model_key": pipeline_key,
            "index_dir": str(config["index_dir"]),
            "index_summary": {
                "passage_count": index_summary.get("passage_count"),
                "entity_count": index_summary.get("entity_count"),
                "embedding_model": index_summary.get("embedding_model"),
                "skip_embeddings": index_summary.get("skip_embeddings"),
            },
            "gazetteer_root": str(args.gazetteer_root),
            "benchmark": str(benchmark_path),
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "semantic_entity_top_k": args.semantic_entity_top_k,
            "semantic_entity_min_score": args.semantic_entity_min_score,
            "weights": config["weights"],
            "use_reference_expansion": bool(config.get("use_reference_expansion", True)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "load_seconds": load_seconds,
            "question_count": len(rows),
            "completed_count": len(ordered),
            "error_count": len(errors),
        },
        "items": ordered,
    }
    write_json(output_path, output)
    if len(ordered) == len(rows) and not errors:
        partial_path.unlink(missing_ok=True)

    del retriever
    gc.collect()
    print(f"[retrieve:{pipeline_key}] saved {output_path}")
    return output_path


def load_retrieval_file(
    pipeline_key: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = retrieval_output_path(config, args)
    if not path.exists():
        if args.no_auto_retrieve:
            raise FileNotFoundError(
                f"Missing retrieval file: {path}. "
                "Run retrieval first or omit --no-auto-retrieve."
            )
        path = generate_retrieval_file(
            pipeline_key=pipeline_key,
            config=config,
            output_path=path,
            args=args,
        )
    data = repair_mojibake(read_json(path))
    items = data.get("items") or []
    if args.limit is not None:
        items = items[: args.limit]

    non_empty_top_results = sum(1 for item in items if item.get("top_results"))
    positive_score_items = sum(
        1
        for item in items
        if any(float(result.get("score") or 0.0) > 0.0 for result in (item.get("top_results") or []))
    )
    if items and (non_empty_top_results == 0 or positive_score_items == 0):
        if args.no_auto_retrieve:
            raise RuntimeError(
                f"Retrieval file looks broken: {path} "
                f"(non_empty_top_results={non_empty_top_results}, positive_score_items={positive_score_items})."
            )
        print(
            f"[retrieve:{pipeline_key}] existing retrieval file looks broken; "
            f"regenerating {path}"
        )
        path.unlink(missing_ok=True)
        path.with_suffix(".partial.jsonl").unlink(missing_ok=True)
        path = generate_retrieval_file(
            pipeline_key=pipeline_key,
            config=config,
            output_path=path,
            args=args,
        )
        data = repair_mojibake(read_json(path))
        items = data.get("items") or []
        if args.limit is not None:
            items = items[: args.limit]

    metadata = data.get("metadata") or {}
    metadata["source_file"] = str(path)
    return {"metadata": metadata, "items": items}


def compact_context_results(results: list[dict[str, Any]], top_k: int, max_chars_per_passage: int) -> list[dict[str, Any]]:
    compact = []
    for raw in results[:top_k]:
        row = repair_mojibake(raw)
        text = (row.get("text") or "").strip()
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "..."
        compact.append({
            "rank": row.get("rank"),
            "passage_id": row.get("passage_id"),
            "score": row.get("score"),
            "score_components": row.get("score_components"),
            "document_number": row.get("document_number"),
            "document_id": row.get("document_id"),
            "package_id": row.get("package_id"),
            "path_text": row.get("path_text"),
            "text": text,
        })
    return compact


def output_path_for(output_dir: Path, pipeline_key: str, model_key: str) -> Path:
    return output_dir / f"traffic_rag_answers_{pipeline_key}__{model_key}.json"


def build_record_base(
    item: dict[str, Any],
    pipeline_key: str,
    model_key: str,
    model_name: str,
    top_k: int,
    max_chars_per_passage: int,
) -> tuple[dict[str, Any], str]:
    question = repair_mojibake_text(item.get("question") or "")
    top_results = item.get("top_results") or []
    context_used = format_context(
        top_results,
        max_passages=top_k,
        max_chars_per_passage=max_chars_per_passage,
    )
    context_passages = compact_context_results(
        top_results,
        top_k=top_k,
        max_chars_per_passage=max_chars_per_passage,
    )

    record = {
        "id": item.get("id"),
        "seq_id": item.get("seq_id"),
        "benchmark_group": item.get("benchmark_group"),
        "is_multi_reference": item.get("is_multi_reference"),
        "question": question,
        "gold_answer": repair_mojibake_text(item.get("answer") or ""),
        "gold_reference_text": repair_mojibake_text(item.get("gold_reference_text") or ""),
        "gold_doc_numbers": item.get("gold_doc_numbers") or [],
        "gold_citation_spans": repair_mojibake(item.get("gold_citation_spans") or []),
        "retrieval_pipeline": pipeline_key,
        "generation_model_key": model_key,
        "generation_model_name": model_name,
        "top_k": top_k,
        "context_passages": context_passages,
        "context_used": context_used,
    }
    return record, context_used


def write_final_output(
    output_path: Path,
    partial_path: Path,
    completed: dict[str, dict[str, Any]],
    retrieval_data: dict[str, Any],
    pipeline_key: str,
    model_key: str,
    model_config: dict[str, str],
    args: argparse.Namespace,
    started_at_utc: str,
) -> None:
    items = retrieval_data["items"]
    ordered = [completed[item["id"]] for item in items if item.get("id") in completed]
    errors = [row for row in ordered if row.get("error")]

    output = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at_utc,
            "retrieval_pipeline": pipeline_key,
            "retrieval_pipeline_display_name": PIPELINE_CONFIGS[pipeline_key]["display_name"],
            "retrieval_file": str((retrieval_data.get("metadata") or {}).get("source_file") or PIPELINE_CONFIGS[pipeline_key]["retrieval_file"]),
            "retrieval_metadata": retrieval_data.get("metadata") or {},
            "generation_model_key": model_key,
            "generation_model_display_name": model_config["display_name"],
            "generation_model_name": model_config["model_name"],
            "top_k": args.top_k,
            "max_chars_per_passage": args.max_chars_per_passage,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "load_4bit": args.load_4bit,
            "answer_mode": args.answer_mode,
            "prompt_version": PROMPT_VERSION,
            "question_count": len(items),
            "completed_count": len(ordered),
            "error_count": len(errors),
        },
        "items": ordered,
    }
    write_json(output_path, output)
    if len(ordered) == len(items) and not errors:
        partial_path.unlink(missing_ok=True)


def run_pair(
    tokenizer,
    model,
    retrieval_data: dict[str, Any],
    pipeline_key: str,
    model_key: str,
    model_config: dict[str, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_path = output_path_for(output_dir, pipeline_key, model_key)
    partial_path = output_path.with_suffix(".partial.jsonl")

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    completed = load_json_items(output_path)
    completed.update(load_partial(partial_path))

    items = retrieval_data["items"]
    if len(completed) >= len(items) and not args.overwrite:
        print(f"[skip] {pipeline_key} + {model_key}: existing complete output {output_path}")
        return output_path

    started_at_utc = datetime.now(timezone.utc).isoformat()
    for idx, item in enumerate(items, start=1):
        item_id = item.get("id")
        if item_id in completed:
            print(f"[{pipeline_key}/{model_key}] [{idx}/{len(items)}] skip {item_id}")
            continue

        started = time.time()
        record, context_used = build_record_base(
            item=item,
            pipeline_key=pipeline_key,
            model_key=model_key,
            model_name=model_config["model_name"],
            top_k=args.top_k,
            max_chars_per_passage=args.max_chars_per_passage,
        )
        record["answer_mode"] = args.answer_mode
        record["prompt_version"] = PROMPT_VERSION

        try:
            if not context_used.strip():
                answer = INSUFFICIENT_CONTEXT_ANSWER
            else:
                messages = build_prompt(record["question"], context_used, answer_mode=args.answer_mode)
                answer = generate_answer(
                    tokenizer=tokenizer,
                    model=model,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
            record["answer"] = answer
            record["error"] = None
        except Exception as exc:
            record["answer"] = ""
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}

        record["elapsed_seconds"] = round(time.time() - started, 3)
        completed[item_id] = record
        append_jsonl(partial_path, record)
        print(
            f"[{pipeline_key}/{model_key}] [{idx}/{len(items)}] "
            f"{item_id} err={record['error'] is not None} elapsed={record['elapsed_seconds']}s"
        )

    write_final_output(
        output_path=output_path,
        partial_path=partial_path,
        completed=completed,
        retrieval_data=retrieval_data,
        pipeline_key=pipeline_key,
        model_key=model_key,
        model_config=model_config,
        args=args,
        started_at_utc=started_at_utc,
    )
    print(f"[saved] {output_path}")
    return output_path


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Generate benchmark answers for baseline/hybrid retrieval pipelines and local Hugging Face LLMs."
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["naive_bm25", "naive_dense", "bge_m3"],
        choices=sorted(PIPELINE_CONFIGS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"],
        help="Model aliases or Hugging Face model ids for the supported default models.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark/traffic_rag_answer_generation_v1")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--retrieval-output-dir", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--gazetteer-root", type=Path, default=DEFAULT_GAZETTEER_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--semantic-entity-top-k", type=int, default=20)
    parser.add_argument("--semantic-entity-min-score", type=float, default=0.45)
    parser.add_argument("--max-chars-per-passage", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--answer-mode", default="extractive_multi_agent", choices=["direct", "extractive_multi_agent"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-retrieval", action="store_true")
    parser.add_argument("--no-auto-retrieve", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and prompt formatting without loading LLMs.")
    args = parser.parse_args()

    model_keys = [canonical_model_key(raw) for raw in args.models]
    retrieval_by_pipeline = {
        pipeline_key: load_retrieval_file(pipeline_key, PIPELINE_CONFIGS[pipeline_key], args)
        for pipeline_key in args.pipelines
    }

    print(f"pipelines: {', '.join(args.pipelines)}")
    print(f"models: {', '.join(model_keys)}")
    print(f"top_k: {args.top_k}")
    for pipeline_key, data in retrieval_by_pipeline.items():
        print(f"[input] {pipeline_key}: {len(data['items'])} questions")

    if args.dry_run:
        first_pipeline = args.pipelines[0]
        first_item = retrieval_by_pipeline[first_pipeline]["items"][0]
        _, context = build_record_base(
            item=first_item,
            pipeline_key=first_pipeline,
            model_key=model_keys[0],
            model_name=GENERATION_MODEL_CONFIGS[model_keys[0]]["model_name"],
            top_k=args.top_k,
            max_chars_per_passage=args.max_chars_per_passage,
        )
        messages = build_prompt(first_item.get("question") or "", context, answer_mode=args.answer_mode)
        print("[dry-run] first question:")
        print(repair_mojibake_text(first_item.get("question") or ""))
        print("[dry-run] answer_mode:", args.answer_mode)
        print("[dry-run] prompt_version:", PROMPT_VERSION)
        print("[dry-run] first prompt chars:", sum(len(m["content"]) for m in messages))
        print("[dry-run] no LLM loaded and no answer files written")
        return

    outputs = []
    for model_key in model_keys:
        model_config = GENERATION_MODEL_CONFIGS[model_key]
        print(f"[model] loading {model_config['display_name']} ({model_config['model_name']})")
        tokenizer, model = load_model(
            model_config["model_name"],
            load_4bit=args.load_4bit,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        try:
            for pipeline_key in args.pipelines:
                outputs.append(
                    run_pair(
                        tokenizer=tokenizer,
                        model=model,
                        retrieval_data=retrieval_by_pipeline[pipeline_key],
                        pipeline_key=pipeline_key,
                        model_key=model_key,
                        model_config=model_config,
                        output_dir=args.output_dir,
                        args=args,
                    )
                )
        finally:
            unload_model(model)
            del tokenizer
            gc.collect()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipelines": args.pipelines,
        "models": model_keys,
        "top_k": args.top_k,
        "outputs": [str(path) for path in outputs],
    }
    write_json(args.output_dir / "traffic_rag_answer_generation_manifest.json", manifest)
    print("outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
