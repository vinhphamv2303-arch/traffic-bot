from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from legal_answer_generation.local_llm_answerer import repair_mojibake  # noqa: E402


DEFAULT_MODELS = {
    "llama_3_1_8b_instruct",
    "qwen2_5_7b_instruct",
}

PIPELINE_DISPLAY_NAMES = {
    "naive_bm25": "Naive BM25 RAG",
    "naive_dense": "Naive Dense RAG (BGE-M3)",
    "no_embedding": "No embedding hybrid",
    "bge_m3": "BGE-M3 hybrid",
    "minilm": "MiniLM hybrid",
}

MODEL_DISPLAY_NAMES = {
    "llama_3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_5_14b_instruct": "Qwen2.5-14B-Instruct",
}

INSUFFICIENT_PATTERNS = [
    "khong tim thay can cu",
    "khong co can cu",
    "khong du can cu",
    "khong du ro",
    "khong co thong tin",
    "khong xac dinh",
]


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


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def norm(text: str) -> str:
    text = repair_mojibake(text or "")
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_text(haystack: str, needle: str) -> bool:
    hay = norm(haystack)
    ned = norm(needle)
    return bool(ned and ned in hay)


def combined_context_text(item: dict[str, Any], k: int = 5) -> str:
    chunks = []
    for passage in item.get("context_passages", [])[:k]:
        chunks.extend([
            passage.get("document_number") or "",
            passage.get("path_text") or "",
            passage.get("text") or "",
        ])
    return " ".join(chunks)


def extract_citation_requirements(span: str) -> list[tuple[str, str]]:
    n = norm(span)
    reqs: list[tuple[str, str]] = []
    for label, pattern in [
        ("diem", r"\bdiem\s+([a-z0-9]+)\b"),
        ("khoan", r"\bkhoan\s+([0-9]+[a-z]?)\b"),
        ("dieu", r"\bdieu\s+([0-9]+[a-z]?)\b"),
        ("muc", r"\bmuc\s+([0-9ivxlcdm]+)\b"),
        ("chuong", r"\bchuong\s+([0-9ivxlcdm]+)\b"),
        ("phu luc", r"\bphu luc\s+([0-9ivxlcdm]+)\b"),
    ]:
        for value in re.findall(pattern, n):
            reqs.append((label, value))
    return reqs


def citation_match(span: str, text: str) -> bool:
    span_n = norm(span)
    hay = norm(text)
    if span_n and span_n in hay:
        return True
    reqs = extract_citation_requirements(span)
    if not reqs:
        return bool(span_n and all(tok in hay for tok in span_n.split()))
    return all(f"{label} {value}" in hay for label, value in reqs)


def reciprocal_rank(ranks: list[int], k: int) -> float:
    ranks = [rank for rank in ranks if 1 <= rank <= k]
    return 0.0 if not ranks else 1.0 / min(ranks)


def ndcg_from_unique_doc_ranks(gold_docs: set[str], context_passages: list[dict[str, Any]], k: int) -> float:
    seen_docs: set[str] = set()
    rels = []
    for passage in context_passages[:k]:
        doc = passage.get("document_number")
        if doc in gold_docs and doc not in seen_docs:
            rels.append(1)
            seen_docs.add(doc)
        else:
            rels.append(0)
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))
    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def is_refusal(answer: str) -> bool:
    answer_n = norm(answer)
    return any(pattern in answer_n for pattern in INSUFFICIENT_PATTERNS)


def evaluate_item(item: dict[str, Any], benchmark_by_id: dict[str, dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    item = repair_mojibake(item)
    benchmark = repair_mojibake(benchmark_by_id.get(item.get("id"), {}))
    answer = item.get("answer") or ""
    gold_answer = item.get("gold_answer") or benchmark.get("answer") or ""
    answer_highlights = benchmark.get("answer_highlights") or []
    gold_docs = set(item.get("gold_doc_numbers") or benchmark.get("gold_doc_numbers") or [])
    gold_citations = item.get("gold_citation_spans") or benchmark.get("gold_citation_spans") or []
    context_passages = item.get("context_passages") or []

    doc_ranks = []
    for rank, passage in enumerate(context_passages, start=1):
        if passage.get("document_number") in gold_docs:
            doc_ranks.append(rank)

    context_text = combined_context_text(item, k=max(ks))
    answer_highlight_matches = [
        highlight for highlight in answer_highlights if contains_text(answer, highlight)
    ]
    context_answer_highlight_matches = [
        highlight for highlight in answer_highlights if contains_text(context_text, highlight)
    ]
    citation_answer_matches = [
        span for span in gold_citations if citation_match(span, answer)
    ]
    citation_context_matches = [
        span for span in gold_citations if citation_match(span, context_text)
    ]

    out = {
        "id": item.get("id"),
        "benchmark_group": item.get("benchmark_group") or benchmark.get("benchmark_group"),
        "is_multi_reference": item.get("is_multi_reference") if item.get("is_multi_reference") is not None else benchmark.get("is_multi_reference"),
        "question": item.get("question") or benchmark.get("question"),
        "retrieval_pipeline": item.get("retrieval_pipeline"),
        "generation_model_key": item.get("generation_model_key"),
        "gold_doc_numbers": sorted(gold_docs),
        "top_doc_numbers": [p.get("document_number") for p in context_passages[:max(ks)]],
        "answer": answer,
        "gold_answer": gold_answer,
        "answer_highlights": answer_highlights,
        "gold_citation_spans": gold_citations,
        "gold_answer_contained": contains_text(answer, gold_answer),
        "answer_contains_any_highlight": bool(answer_highlight_matches),
        "answer_highlight_recall": len(answer_highlight_matches) / len(answer_highlights) if answer_highlights else 0.0,
        "answer_contains_any_citation": bool(citation_answer_matches),
        "answer_citation_recall": len(citation_answer_matches) / len(gold_citations) if gold_citations else 0.0,
        "context_contains_any_highlight": bool(context_answer_highlight_matches),
        "context_highlight_recall": len(context_answer_highlight_matches) / len(answer_highlights) if answer_highlights else 0.0,
        "context_contains_any_citation": bool(citation_context_matches),
        "context_citation_recall": len(citation_context_matches) / len(gold_citations) if gold_citations else 0.0,
        "refusal": is_refusal(answer),
        "error": item.get("error"),
        "context_passage_count": len(context_passages),
    }

    for k in ks:
        top_k = context_passages[:k]
        retrieved_gold_docs = {p.get("document_number") for p in top_k if p.get("document_number") in gold_docs}
        out[f"doc_hit@{k}"] = bool(retrieved_gold_docs)
        out[f"doc_recall@{k}"] = len(retrieved_gold_docs) / len(gold_docs) if gold_docs else 0.0
        out[f"doc_mrr@{k}"] = reciprocal_rank(doc_ranks, k)
        out[f"doc_ndcg@{k}"] = ndcg_from_unique_doc_ranks(gold_docs, context_passages, k)
    return out


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows) if rows else 0.0


def aggregate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for key in [
        "gold_answer_contained",
        "answer_contains_any_highlight",
        "answer_highlight_recall",
        "answer_contains_any_citation",
        "answer_citation_recall",
        "context_contains_any_highlight",
        "context_highlight_recall",
        "context_contains_any_citation",
        "context_citation_recall",
        "refusal",
    ]:
        out[key] = round(mean(rows, key), 6)
    for k in ks:
        for key in [f"doc_hit@{k}", f"doc_recall@{k}", f"doc_mrr@{k}", f"doc_ndcg@{k}"]:
            out[key] = round(mean(rows, key), 6)
    out["error_count"] = sum(1 for row in rows if row.get("error"))
    out["empty_context_count"] = sum(1 for row in rows if not row.get("context_passage_count"))
    return out


def group_aggregates(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("benchmark_group") or "unknown"].append(row)
    return {group: aggregate(group_rows, ks) for group, group_rows in sorted(grouped.items())}


def read_answer_files(answer_dir: Path, model_keys: set[str]) -> list[tuple[Path, dict[str, Any]]]:
    files = []
    for path in sorted(answer_dir.glob("traffic_rag_answers_*.json")):
        if path.name.endswith(".partial.jsonl"):
            continue
        data = repair_mojibake(read_json(path))
        model_key = (data.get("metadata") or {}).get("generation_model_key")
        if model_key in model_keys:
            files.append((path, data))
    return files


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pipeline",
        "model",
        "count",
        "doc_hit@1",
        "doc_hit@5",
        "doc_recall@5",
        "doc_mrr@5",
        "doc_ndcg@5",
        "context_contains_any_highlight",
        "answer_contains_any_highlight",
        "answer_highlight_recall",
        "gold_answer_contained",
        "answer_contains_any_citation",
        "refusal",
        "error_count",
        "empty_context_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    lines = [
        "# Traffic RAG Answer Generation Evaluation",
        "",
        "This report evaluates retrieval context and generated answers from precomputed JSON files. The main answer metric follows a LinearRAG-style containment check: whether the gold answer/highlight appears in the generated answer after normalization.",
        "",
        "## Overall",
        "",
        "| Pipeline | Model | Doc Hit@5 | Doc Recall@5 | Context Highlight Hit | Answer Highlight Hit | Highlight Recall | Full Gold Containment | Citation In Answer | Refusal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['pipeline']} | {row['model']} "
            f"| {pct(row['doc_hit@5'])} "
            f"| {pct(row['doc_recall@5'])} "
            f"| {pct(row['context_contains_any_highlight'])} "
            f"| {pct(row['answer_contains_any_highlight'])} "
            f"| {pct(row['answer_highlight_recall'])} "
            f"| {pct(row['gold_answer_contained'])} "
            f"| {pct(row['answer_contains_any_citation'])} "
            f"| {pct(row['refusal'])} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `Context Highlight Hit` checks whether top-5 retrieved passages contain any gold answer highlight.",
        "- `Answer Highlight Hit` checks whether the generated answer contains any gold answer highlight.",
        "- `Full Gold Containment` is stricter: the full gold answer sentence must be contained in the model answer.",
        "- Low answer score with high context score usually means generation failed despite enough evidence.",
        "- Low context score means the retriever did not provide the answer evidence.",
        "",
        "## Miss Counts",
        "",
    ])
    for model_key, model_report in report["runs"].items():
        overall = model_report["overall"]
        lines.append(
            f"- {model_key}: answer_highlight_miss={overall['count'] - round(overall['answer_contains_any_highlight'] * overall['count'])}, "
            f"context_highlight_miss={overall['count'] - round(overall['context_contains_any_highlight'] * overall['count'])}, "
            f"errors={overall['error_count']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate answer generation outputs for traffic RAG benchmark.")
    parser.add_argument("--answer-dir", type=Path, default=ROOT / "data/benchmark/traffic_rag_answer_generation_v1")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark/traffic_rag_benchmark_v1/traffic_rag_benchmark_v1.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--models", nargs="+", default=sorted(DEFAULT_MODELS))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    benchmark_rows = repair_mojibake(read_jsonl(args.benchmark))
    benchmark_by_id = {row["id"]: row for row in benchmark_rows}
    answer_files = read_answer_files(args.answer_dir, set(args.models))

    report: dict[str, Any] = {
        "metadata": {
            "answer_dir": str(args.answer_dir),
            "benchmark": str(args.benchmark),
            "model_keys": args.models,
            "answer_files": [str(path) for path, _ in answer_files],
            "ks": args.ks,
        },
        "runs": {},
    }
    summary_rows = []

    for path, data in answer_files:
        metadata = data.get("metadata") or {}
        pipeline_key = metadata.get("retrieval_pipeline") or "unknown"
        model_key = metadata.get("generation_model_key") or "unknown"
        run_key = f"{pipeline_key}__{model_key}"
        rows = [
            evaluate_item(item, benchmark_by_id, args.ks)
            for item in data.get("items", [])
        ]
        overall = aggregate(rows, args.ks)
        report["runs"][run_key] = {
            "source_file": str(path),
            "metadata": metadata,
            "overall": overall,
            "by_group": group_aggregates(rows, args.ks),
            "per_question": rows,
        }
        summary_rows.append({
            "pipeline_key": pipeline_key,
            "model_key": model_key,
            "pipeline": PIPELINE_DISPLAY_NAMES.get(pipeline_key, pipeline_key),
            "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
            **overall,
        })

    summary_rows.sort(key=lambda row: (row["pipeline_key"], row["model_key"]))

    json_path = args.output_dir / "traffic_rag_answer_generation_eval.json"
    csv_path = args.output_dir / "traffic_rag_answer_generation_eval_summary.csv"
    md_path = args.output_dir / "traffic_rag_answer_generation_eval_report.md"
    write_json(json_path, report)
    write_summary_csv(csv_path, summary_rows)
    write_markdown(md_path, summary_rows, report)

    print("saved", json_path)
    print("saved", csv_path)
    print("saved", md_path)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
