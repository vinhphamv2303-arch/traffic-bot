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
DEFAULT_RESULT_FILES = {
    "naive_bm25": ROOT / "data/benchmark/traffic_rag_retrieval_naive_bm25_top5.json",
    "naive_dense": ROOT / "data/benchmark/traffic_rag_retrieval_naive_dense_top5.json",
    "no_embedding": ROOT / "data/benchmark/traffic_rag_retrieval_no_embedding_top5.json",
    "minilm_no_graph": ROOT / "data/benchmark/traffic_rag_retrieval_minilm_no_graph_top5.json",
    "minilm": ROOT / "data/benchmark/traffic_rag_retrieval_minilm_top5.json",
    "bge_m3_no_graph": ROOT / "data/benchmark/traffic_rag_retrieval_bge_m3_no_graph_top5.json",
    "bge_m3": ROOT / "data/benchmark/traffic_rag_retrieval_bge_m3_top5.json",
}

MODEL_DISPLAY_NAMES = {
    "naive_bm25": "Naive BM25 RAG",
    "naive_dense": "Naive Dense RAG (BGE-M3)",
    "no_embedding": "No embedding",
    "minilm_no_graph": "MiniLM no-graph",
    "minilm": "MiniLM hybrid",
    "bge_m3_no_graph": "BGE-M3 no-graph",
    "bge_m3": "BGE-M3 hybrid",
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


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def norm(text: str) -> str:
    text = strip_accents(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combined_text(result: dict[str, Any]) -> str:
    return " ".join([
        result.get("document_number") or "",
        result.get("path_text") or "",
        result.get("text") or "",
    ])


def extract_citation_requirements(span: str) -> list[tuple[str, str]]:
    n = norm(span)
    reqs: list[tuple[str, str]] = []
    for label, pattern in [
        ("diem", r"\bdiem\s+([a-z0-9]+)\b"),
        ("khoan", r"\bkhoan\s+([0-9]+[a-z]?)\b"),
        ("dieu", r"\bdieu\s+([0-9]+[a-z]?)\b"),
        ("muc", r"\bmuc\s+([0-9ivxlcdm]+)\b"),
        ("chuong", r"\bchuong\s+([0-9ivxlcdm]+)\b"),
        ("phu_luc", r"\bphu luc\s+([0-9ivxlcdm]+)\b"),
    ]:
        for value in re.findall(pattern, n):
            reqs.append((label, value))
    return reqs


def citation_match(span: str, result: dict[str, Any]) -> bool:
    hay = norm(combined_text(result))
    span_n = norm(span)
    if span_n and span_n in hay:
        return True
    reqs = extract_citation_requirements(span)
    if not reqs:
        return bool(span_n and all(tok in hay for tok in span_n.split()))
    for label, value in reqs:
        label_text = label.replace("_", " ")
        if f"{label_text} {value}" not in hay:
            return False
    return True


def text_match(needle: str, result: dict[str, Any]) -> bool:
    needle_n = norm(needle)
    if not needle_n:
        return False
    return needle_n in norm(combined_text(result))


def reciprocal_rank(ranks: list[int], k: int) -> float:
    ranks = [r for r in ranks if 1 <= r <= k]
    return 0.0 if not ranks else 1.0 / min(ranks)


def ndcg_from_unique_doc_ranks(gold_docs: set[str], results: list[dict[str, Any]], k: int) -> float:
    seen_docs: set[str] = set()
    rels = []
    for result in results[:k]:
        doc = result.get("document_number")
        if doc in gold_docs and doc not in seen_docs:
            rels.append(1)
            seen_docs.add(doc)
        else:
            rels.append(0)
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))
    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_item(item: dict[str, Any], benchmark_by_id: dict[str, dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    benchmark = benchmark_by_id.get(item.get("id"), {})
    top_results = item.get("top_results") or []
    gold_docs = set(item.get("gold_doc_numbers") or benchmark.get("gold_doc_numbers") or [])
    gold_spans = item.get("gold_citation_spans") or benchmark.get("gold_citation_spans") or []
    answer_highlights = benchmark.get("answer_highlights") or []

    doc_ranks = []
    for rank, result in enumerate(top_results, 1):
        if result.get("document_number") in gold_docs:
            doc_ranks.append(rank)

    citation_ranks_by_span: dict[str, int] = {}
    for span in gold_spans:
        for rank, result in enumerate(top_results, 1):
            if citation_match(span, result):
                citation_ranks_by_span[span] = rank
                break

    answer_ranks_by_highlight: dict[str, int] = {}
    for highlight in answer_highlights:
        for rank, result in enumerate(top_results, 1):
            if text_match(highlight, result):
                answer_ranks_by_highlight[highlight] = rank
                break

    out = {
        "id": item.get("id"),
        "benchmark_group": item.get("benchmark_group") or benchmark.get("benchmark_group"),
        "is_multi_reference": item.get("is_multi_reference") if item.get("is_multi_reference") is not None else benchmark.get("is_multi_reference"),
        "question": item.get("question") or benchmark.get("question"),
        "gold_doc_numbers": sorted(gold_docs),
        "gold_citation_spans": gold_spans,
        "answer_highlights": answer_highlights,
        "top_doc_numbers": [r.get("document_number") for r in top_results],
        "first_doc_hit_rank": min(doc_ranks) if doc_ranks else None,
        "first_citation_hit_rank": min(citation_ranks_by_span.values()) if citation_ranks_by_span else None,
        "first_answer_highlight_hit_rank": min(answer_ranks_by_highlight.values()) if answer_ranks_by_highlight else None,
        "matched_citation_spans": sorted(citation_ranks_by_span),
        "matched_answer_highlights": sorted(answer_ranks_by_highlight),
        "top_results_count": len(top_results),
        "error": item.get("error"),
    }

    for k in ks:
        top_k = top_results[:k]
        retrieved_gold_docs = {r.get("document_number") for r in top_k if r.get("document_number") in gold_docs}
        relevant_passage_count = sum(1 for r in top_k if r.get("document_number") in gold_docs)
        matched_spans = {span for span, rank in citation_ranks_by_span.items() if rank <= k}
        matched_highlights = {h for h, rank in answer_ranks_by_highlight.items() if rank <= k}
        out[f"doc_hit@{k}"] = bool(retrieved_gold_docs)
        out[f"doc_recall@{k}"] = len(retrieved_gold_docs) / len(gold_docs) if gold_docs else 0.0
        out[f"passage_doc_precision@{k}"] = relevant_passage_count / k if k else 0.0
        out[f"doc_mrr@{k}"] = reciprocal_rank(doc_ranks, k)
        out[f"doc_ndcg@{k}"] = ndcg_from_unique_doc_ranks(gold_docs, top_results, k)
        out[f"citation_hit@{k}"] = bool(matched_spans)
        out[f"citation_recall@{k}"] = len(matched_spans) / len(gold_spans) if gold_spans else 0.0
        out[f"answer_highlight_hit@{k}"] = bool(matched_highlights)
        out[f"answer_highlight_recall@{k}"] = len(matched_highlights) / len(answer_highlights) if answer_highlights else 0.0
    return out


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row.get(key) or 0.0) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for k in ks:
        for key in [
            f"doc_hit@{k}",
            f"doc_recall@{k}",
            f"passage_doc_precision@{k}",
            f"doc_mrr@{k}",
            f"doc_ndcg@{k}",
            f"citation_hit@{k}",
            f"citation_recall@{k}",
            f"answer_highlight_hit@{k}",
            f"answer_highlight_recall@{k}",
        ]:
            out[key] = round(mean(rows, key), 6)
    out["error_count"] = sum(1 for row in rows if row.get("error"))
    out["empty_top_results_count"] = sum(1 for row in rows if not row.get("top_results_count"))
    out["no_doc_hit@5"] = sum(1 for row in rows if not row.get("doc_hit@5"))
    out["no_citation_hit@5"] = sum(1 for row in rows if not row.get("citation_hit@5"))
    out["no_answer_highlight_hit@5"] = sum(1 for row in rows if not row.get("answer_highlight_hit@5"))
    return out


def group_aggregates(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("benchmark_group") or "unknown"].append(row)
    return {group: aggregate(group_rows, ks) for group, group_rows in sorted(grouped.items())}


def multi_reference_aggregates(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    grouped = {
        "single_reference": [row for row in rows if not row.get("is_multi_reference")],
        "multi_reference": [row for row in rows if row.get("is_multi_reference")],
    }
    return {group: aggregate(group_rows, ks) for group, group_rows in grouped.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_key",
        "count",
        "doc_hit@1", "doc_hit@3", "doc_hit@5",
        "doc_recall@1", "doc_recall@3", "doc_recall@5",
        "doc_mrr@5", "doc_ndcg@5",
        "citation_hit@1", "citation_hit@3", "citation_hit@5",
        "citation_recall@5",
        "answer_highlight_hit@1", "answer_highlight_hit@3", "answer_highlight_hit@5",
        "answer_highlight_recall@5",
        "error_count", "empty_top_results_count",
        "no_doc_hit@5", "no_citation_hit@5", "no_answer_highlight_hit@5",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def write_markdown_report(path: Path, report: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    clean_lines = []
    clean_lines.append("# Traffic RAG Retrieval Evaluation Top-5")
    clean_lines.append("")
    clean_lines.append("This report is generated from retrieval files in `data/benchmark`. The primary metrics are document-level metrics; citation and answer metrics are heuristic matches over `path_text + text`.")
    clean_lines.append("")
    clean_lines.append("## Overall")
    clean_lines.append("")
    clean_lines.append("| Model | Doc Hit@1 | Doc Hit@5 | Doc Recall@5 | MRR@5 | nDCG@5 | Citation Hit@5 | Answer Hit@5 | Errors | Empty Top5 | Doc Miss@5 |")
    clean_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        model_key = row["model_key"]
        clean_lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_key, model_key)} "
            f"| {pct(row['doc_hit@1'])} "
            f"| {pct(row['doc_hit@5'])} "
            f"| {pct(row['doc_recall@5'])} "
            f"| {row['doc_mrr@5']:.3f} "
            f"| {row['doc_ndcg@5']:.3f} "
            f"| {pct(row['citation_hit@5'])} "
            f"| {pct(row['answer_highlight_hit@5'])} "
            f"| {row['error_count']} "
            f"| {row['empty_top_results_count']} "
            f"| {row['no_doc_hit@5']} |"
        )
    clean_lines.append("")

    invalid_rows = [
        row for row in summary_rows
        if row.get("error_count") or row.get("empty_top_results_count")
    ]
    if invalid_rows:
        clean_lines.append("## Run Diagnostics")
        clean_lines.append("")
        clean_lines.append("Some runs produced errors or empty `top_results`. Their retrieval metrics should be treated as invalid until the corresponding JSON output is regenerated.")
        clean_lines.append("")
        for row in invalid_rows:
            model_key = row["model_key"]
            clean_lines.append(
                f"- {MODEL_DISPLAY_NAMES.get(model_key, model_key)}: "
                f"errors={row['error_count']}, empty_top_results={row['empty_top_results_count']}"
            )
        clean_lines.append("")

    comparisons = [
        ("minilm_no_graph", "minilm", "MiniLM"),
        ("bge_m3_no_graph", "bge_m3", "BGE-M3"),
    ]
    available = {row["model_key"]: row for row in summary_rows}
    if any(a in available and b in available for a, b, _ in comparisons):
        clean_lines.append("## Hybrid Gain")
        clean_lines.append("")
        clean_lines.append("| Family | Metric | No-graph | Hybrid | Gain |")
        clean_lines.append("|---|---|---:|---:|---:|")
        for dense_key, hybrid_key, label in comparisons:
            if dense_key not in available or hybrid_key not in available:
                continue
            dense = available[dense_key]
            hybrid = available[hybrid_key]
            for metric in ["doc_hit@1", "doc_hit@5", "doc_recall@5", "citation_hit@5", "answer_highlight_hit@5"]:
                gain = hybrid[metric] - dense[metric]
                clean_lines.append(f"| {label} | `{metric}` | {pct(dense[metric])} | {pct(hybrid[metric])} | {gain * 100:+.1f} pp |")
        clean_lines.append("")

    clean_lines.append("## Breakdown By Question Group")
    clean_lines.append("")
    clean_lines.append("| Model | Single Fact Doc Hit@5 | Comparison/Two Facts Doc Hit@5 | Multi-hop Doc Hit@5 | Multi-hop Doc Recall@5 |")
    clean_lines.append("|---|---:|---:|---:|---:|")
    for model_key in report["models"]:
        groups = report["models"][model_key]["by_group"]
        clean_lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_key, model_key)} "
            f"| {pct(groups['single_fact']['doc_hit@5'])} "
            f"| {pct(groups['comparison_or_two_facts']['doc_hit@5'])} "
            f"| {pct(groups['multi_hop_cross_doc']['doc_hit@5'])} "
            f"| {pct(groups['multi_hop_cross_doc']['doc_recall@5'])} |"
        )
    clean_lines.append("")

    clean_lines.append("## Doc Miss@5")
    clean_lines.append("")
    for model_key, model_report in report["models"].items():
        misses = [row for row in model_report["per_question"] if not row.get("doc_hit@5")]
        miss_ids = ", ".join(f"`{row['id']}`" for row in misses) if misses else "none"
        clean_lines.append(f"- {MODEL_DISPLAY_NAMES.get(model_key, model_key)}: {len(misses)} case: {miss_ids}")
    clean_lines.append("")

    clean_lines.append("## How To Read This")
    clean_lines.append("")
    clean_lines.append("- `no-graph` still uses dense embedding and reference expansion, but does not use entity graph score or graph-based reference seeds.")
    clean_lines.append("- If `no-graph` is lower than `hybrid`, the entity graph is adding useful signal beyond semantic embedding and reference expansion.")
    clean_lines.append("- If `no-graph` is close to or better than `hybrid` on a metric, inspect graph weights or group-specific reranking.")
    clean_lines.append("- `Doc Hit@5` measures whether the retriever puts a gold document into the context. `Answer Hit@5` is usually lower because a correct document can be split into many passages.")
    clean_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
    return

    lines = []
    lines.append("# Traffic RAG Retrieval Evaluation Top-5")
    lines.append("")
    lines.append("Báo cáo này được sinh tự động từ các file retrieval trong `data/benchmark`. Metric chính nên xem là document-level, vì citation/answer hiện vẫn là heuristic match trên `path_text + text`.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Doc Hit@1 | Doc Hit@5 | Doc Recall@5 | MRR@5 | nDCG@5 | Citation Hit@5 | Answer Hit@5 | Errors | Empty Top5 | Doc Miss@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        model_key = row["model_key"]
        lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_key, model_key)} "
            f"| {pct(row['doc_hit@1'])} "
            f"| {pct(row['doc_hit@5'])} "
            f"| {pct(row['doc_recall@5'])} "
            f"| {row['doc_mrr@5']:.3f} "
            f"| {row['doc_ndcg@5']:.3f} "
            f"| {pct(row['citation_hit@5'])} "
            f"| {pct(row['answer_highlight_hit@5'])} "
            f"| {row['error_count']} "
            f"| {row['empty_top_results_count']} "
            f"| {row['no_doc_hit@5']} |"
        )
    lines.append("")

    invalid_rows = [
        row for row in summary_rows
        if row.get("error_count") or row.get("empty_top_results_count")
    ]
    if invalid_rows:
        lines.append("## Run Diagnostics")
        lines.append("")
        lines.append("Some runs produced errors or empty `top_results`. Their retrieval metrics should be treated as invalid until the corresponding JSON output is regenerated.")
        lines.append("")
        for row in invalid_rows:
            model_key = row["model_key"]
            lines.append(
                f"- {MODEL_DISPLAY_NAMES.get(model_key, model_key)}: "
                f"errors={row['error_count']}, empty_top_results={row['empty_top_results_count']}"
            )
        lines.append("")

    comparisons = [
        ("minilm_no_graph", "minilm", "MiniLM"),
        ("bge_m3_no_graph", "bge_m3", "BGE-M3"),
    ]
    available = {row["model_key"]: row for row in summary_rows}
    if any(a in available and b in available for a, b, _ in comparisons):
        lines.append("## Hybrid Gain")
        lines.append("")
        lines.append("| Family | Metric | No-graph | Hybrid | Gain |")
        lines.append("|---|---|---:|---:|---:|")
        for dense_key, hybrid_key, label in comparisons:
            if dense_key not in available or hybrid_key not in available:
                continue
            dense = available[dense_key]
            hybrid = available[hybrid_key]
            for metric in ["doc_hit@1", "doc_hit@5", "doc_recall@5", "citation_hit@5", "answer_highlight_hit@5"]:
                gain = hybrid[metric] - dense[metric]
                lines.append(f"| {label} | `{metric}` | {pct(dense[metric])} | {pct(hybrid[metric])} | {gain * 100:+.1f} pp |")
        lines.append("")

    lines.append("## Breakdown Theo Nhóm Câu Hỏi")
    lines.append("")
    lines.append("| Model | Single Fact Doc Hit@5 | Comparison/Two Facts Doc Hit@5 | Multi-hop Doc Hit@5 | Multi-hop Doc Recall@5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for model_key in report["models"]:
        groups = report["models"][model_key]["by_group"]
        lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model_key, model_key)} "
            f"| {pct(groups['single_fact']['doc_hit@5'])} "
            f"| {pct(groups['comparison_or_two_facts']['doc_hit@5'])} "
            f"| {pct(groups['multi_hop_cross_doc']['doc_hit@5'])} "
            f"| {pct(groups['multi_hop_cross_doc']['doc_recall@5'])} |"
        )
    lines.append("")

    lines.append("## Doc Miss@5")
    lines.append("")
    for model_key, model_report in report["models"].items():
        misses = [row for row in model_report["per_question"] if not row.get("doc_hit@5")]
        miss_ids = ", ".join(f"`{row['id']}`" for row in misses) if misses else "none"
        lines.append(f"- {MODEL_DISPLAY_NAMES.get(model_key, model_key)}: {len(misses)} case: {miss_ids}")
    lines.append("")

    lines.append("## Cách Diễn Giải")
    lines.append("")
    lines.append("- `no-graph` vẫn dùng dense embedding và reference expansion, nhưng không dùng entity graph score hay graph seed cho reference.")
    lines.append("- Nếu `no-graph` thấp hơn `hybrid`, nghĩa là entity graph đang bổ sung tín hiệu hữu ích ngoài semantic embedding và reference.")
    lines.append("- Nếu `no-graph` gần bằng hoặc hơn `hybrid` ở một vài metric, cần xem lại weight của graph hoặc rerank theo từng nhóm câu hỏi.")
    lines.append("- `Doc Hit@5` dùng để đo retriever có đưa đúng văn bản vào context hay không. `Answer Hit@5` thấp hơn là bình thường vì một văn bản đúng có thể bị chia thành nhiều passage nhỏ.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate traffic RAG top-5 retrieval outputs.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/benchmark/traffic_rag_benchmark_v1/traffic_rag_benchmark_v1.jsonl",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive_bm25", "naive_dense", "bge_m3"],
        choices=sorted(DEFAULT_RESULT_FILES),
        help="Retrieval result keys to evaluate.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Optional directory containing traffic_rag_retrieval_*_top5.json files.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    benchmark_rows = read_jsonl(args.benchmark)
    benchmark_by_id = {row["id"]: row for row in benchmark_rows}

    report: dict[str, Any] = {
        "metadata": {
            "benchmark": str(args.benchmark),
            "question_count": len(benchmark_rows),
            "ks": args.ks,
            "models": args.models,
            "result_dir": str(args.result_dir) if args.result_dir else None,
            "notes": {
                "doc_metrics": "Compare retrieved document_number against gold_doc_numbers.",
                "citation_metrics": "Heuristic match of gold citation components in path_text + text.",
                "answer_highlight_metrics": "Heuristic normalized substring match of answer_highlights in path_text + text.",
            },
        },
        "models": {},
    }
    summary_rows = []

    for model_key in args.models:
        default_path = DEFAULT_RESULT_FILES[model_key]
        path = args.result_dir / default_path.name if args.result_dir else default_path
        if not path.exists():
            print(f"skip missing {model_key}: {path}")
            continue
        data = read_json(path)
        rows = [
            evaluate_item(item, benchmark_by_id, args.ks)
            for item in data.get("items", [])
        ]
        model_report = {
            "source_file": str(path),
            "metadata": data.get("metadata", {}),
            "overall": aggregate(rows, args.ks),
            "by_group": group_aggregates(rows, args.ks),
            "by_reference_type": multi_reference_aggregates(rows, args.ks),
            "per_question": rows,
        }
        report["models"][model_key] = model_report
        summary_row = {"model_key": model_key, **model_report["overall"]}
        summary_rows.append(summary_row)

    report_path = args.output_dir / "traffic_rag_retrieval_eval_top5.json"
    csv_path = args.output_dir / "traffic_rag_retrieval_eval_summary.csv"
    md_path = args.output_dir / "traffic_rag_retrieval_eval_report.md"
    write_json(report_path, report)
    write_csv(csv_path, summary_rows)
    write_markdown_report(md_path, report, summary_rows)

    print("saved", report_path)
    print("saved", csv_path)
    print("saved", md_path)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
