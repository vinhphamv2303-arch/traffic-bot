import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_pipelines.legal_linearrag_retriever.legal_linearrag_retriever import build_index


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build LinearRAG-style hybrid retrieval index.")
    ap.add_argument("--graph-root", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/retrieval/index_bge_m3_hybrid")
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--embedding-batch-size", type=int, default=64)
    ap.add_argument("--skip-embeddings", action="store_true", help="Build BM25 + graph index only.")
    args = ap.parse_args()

    summary = build_index(
        graph_root=args.graph_root,
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        skip_embeddings=args.skip_embeddings,
    )

    print("LinearRAG index build completed")
    print(f"Passages: {summary['passage_count']}")
    print(f"Entities: {summary['entity_count']}")
    print(f"Embedding model: {summary['embedding_model']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
