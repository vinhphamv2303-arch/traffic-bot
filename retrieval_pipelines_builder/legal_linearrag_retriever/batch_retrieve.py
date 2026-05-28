import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_pipelines_builder.legal_linearrag_retriever.legal_linearrag_retriever import LinearRAGRetriever
from retrieval_pipelines_builder.legal_linearrag_retriever.legal_linearrag_retriever.utils import write_jsonl


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_queries(path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                qs.append(row)
            else:
                qs.append({"query": line})
    return qs

def main():
    ap = argparse.ArgumentParser(description="Batch LinearRAG-style retrieval.")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--queries", required=True, help="txt or jsonl. If txt, one query per line.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=300)
    ap.add_argument("--graph-only-penalty", type=float, default=0.65)
    args = ap.parse_args()

    retriever = LinearRAGRetriever.from_index(args.index_dir, args.gazetteer_root)
    rows = []
    for q in read_queries(args.queries):
        query = q.get("query") or q.get("question") or ""
        if not query:
            continue
        out = retriever.retrieve(
            query=query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            graph_only_penalty=args.graph_only_penalty,
        )
        rows.append({**q, "retrieval": out})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Saved: {args.output}")
    print(f"Queries: {len(rows)}")

if __name__ == "__main__":
    main()
