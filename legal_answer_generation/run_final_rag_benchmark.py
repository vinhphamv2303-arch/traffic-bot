from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legal_answer_generation.run_benchmark_answers import canonical_model_key  # noqa: E402


DEFAULT_PIPELINES = ["naive_bm25", "naive_dense", "bge_m3"]
DEFAULT_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
]


def run_command(cmd: list[str], label: str) -> None:
    print(f"\n[{label}]", flush=True)
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_optional_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def add_optional_value(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run the final traffic RAG benchmark: 3 retrieval pipelines x 3 local LLMs, then evaluate retrieval and answers."
    )
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark/traffic_rag_benchmark_v1/traffic_rag_benchmark_v1.jsonl")
    parser.add_argument("--gazetteer-root", type=Path, default=ROOT / "ner_finetuning/data/preprocessed/expanded_gazetteer")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/benchmark/traffic_rag_final_three_pipeline_v1")
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-retrieval", action="store_true")
    parser.add_argument("--no-auto-retrieve", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    answer_dir = output_root / "answers"
    retrieval_dir = output_root / "retrieval"
    retrieval_eval_dir = output_root / "retrieval_eval"
    answer_eval_dir = output_root / "answer_eval"
    canonical_models = [canonical_model_key(model) for model in args.models]

    if not args.skip_generation:
        generation_cmd = [
            sys.executable,
            str(ROOT / "legal_answer_generation/run_benchmark_answers.py"),
            "--pipelines",
            *args.pipelines,
            "--models",
            *args.models,
            "--output-dir",
            str(answer_dir),
            "--retrieval-output-dir",
            str(retrieval_dir),
            "--benchmark",
            str(args.benchmark),
            "--gazetteer-root",
            str(args.gazetteer_root),
            "--top-k",
            str(args.top_k),
            "--candidate-k",
            str(args.candidate_k),
            "--semantic-entity-top-k",
            str(args.semantic_entity_top_k),
            "--semantic-entity-min-score",
            str(args.semantic_entity_min_score),
            "--max-chars-per-passage",
            str(args.max_chars_per_passage),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--answer-mode",
            args.answer_mode,
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
        ]
        add_optional_value(generation_cmd, "--limit", args.limit)
        add_optional_flag(generation_cmd, "--load-4bit", args.load_4bit)
        add_optional_flag(generation_cmd, "--overwrite", args.overwrite)
        add_optional_flag(generation_cmd, "--overwrite-retrieval", args.overwrite_retrieval)
        add_optional_flag(generation_cmd, "--no-auto-retrieve", args.no_auto_retrieve)
        add_optional_flag(generation_cmd, "--dry-run", args.dry_run)
        run_command(generation_cmd, "generate retrieval + answers")

    if args.dry_run or args.skip_eval:
        return

    retrieval_eval_cmd = [
        sys.executable,
        str(ROOT / "data/benchmark/traffic_rag_benchmark_v1/evaluate_retrieval_top5.py"),
        "--benchmark",
        str(args.benchmark),
        "--models",
        *args.pipelines,
        "--result-dir",
        str(retrieval_dir),
        "--output-dir",
        str(retrieval_eval_dir),
        "--ks",
        "1",
        "3",
        "5",
    ]
    run_command(retrieval_eval_cmd, "evaluate retrieval")

    answer_eval_cmd = [
        sys.executable,
        str(ROOT / "data/benchmark/traffic_rag_benchmark_v1/evaluate_answer_generation.py"),
        "--answer-dir",
        str(answer_dir),
        "--benchmark",
        str(args.benchmark),
        "--output-dir",
        str(answer_eval_dir),
        "--models",
        *canonical_models,
        "--pipelines",
        *args.pipelines,
        "--ks",
        "1",
        "3",
        "5",
    ]
    run_command(answer_eval_cmd, "evaluate answers")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipelines": args.pipelines,
        "models": canonical_models,
        "top_k": args.top_k,
        "answer_dir": str(answer_dir),
        "retrieval_dir": str(retrieval_dir),
        "retrieval_eval_dir": str(retrieval_eval_dir),
        "answer_eval_dir": str(answer_eval_dir),
        "retrieval_summary": str(retrieval_eval_dir / "traffic_rag_retrieval_eval_summary.csv"),
        "answer_summary": str(answer_eval_dir / "traffic_rag_answer_generation_eval_summary.csv"),
    }
    write_manifest(output_root / "final_benchmark_manifest.json", manifest)
    print("\n[done]")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
