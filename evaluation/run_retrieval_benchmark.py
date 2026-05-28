from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark/traffic_rag_gold_questions_v1/traffic_rag_gold_questions_v1.jsonl")
    ap.add_argument("--retriever-dir", default="retrieval_pipelines_builder/legal_linearrag_retriever")
    ap.add_argument("--index-dir", default="data/retrieval/index_bm25_graph")
    ap.add_argument("--gazetteer-root", default="ner_finetuning/data/preprocessed/expanded_gazetteer")
    ap.add_argument("--output", default="data/benchmark/traffic_rag_final_retrieval_answer_benchmark_v1/retrieval/retrieval_results_v1.jsonl")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--dense-weight", default="0.0")
    ap.add_argument("--bm25-weight", default="0.25")
    ap.add_argument("--graph-weight", default="0.15")
    ap.add_argument("--reference-weight", default="0.60")
    args = ap.parse_args()

    retrieve_py = Path(args.retriever_dir) / "retrieve.py"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for item in read_jsonl(args.benchmark):
            q = item["question"]
            cmd = [
                sys.executable, str(retrieve_py),
                "--index-dir", args.index_dir,
                "--gazetteer-root", args.gazetteer_root,
                "--query", q,
                "--top-k", str(args.top_k),
                "--dense-weight", args.dense_weight,
                "--bm25-weight", args.bm25_weight,
                "--graph-weight", args.graph_weight,
                "--reference-weight", args.reference_weight,
            ]
            r = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rec = {
                "id": item["id"],
                "question": q,
                "gold_doc_numbers": item.get("gold_doc_numbers", []),
                "gold_reference_text": item.get("reference_text"),
                "returncode": r.returncode,
                "stderr": r.stderr,
            }
            try:
                rec["retrieval"] = json.loads(r.stdout)
            except Exception:
                rec["raw_stdout"] = r.stdout
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            print(f"[{total}] {item['id']} rc={r.returncode}: {q[:80]}")

    print("saved", out_path)

if __name__ == "__main__":
    main()
