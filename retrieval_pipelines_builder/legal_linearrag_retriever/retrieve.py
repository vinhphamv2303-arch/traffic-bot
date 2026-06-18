import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_pipelines_builder.legal_linearrag_retriever.legal_linearrag_retriever import LinearRAGRetriever


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run LinearRAG-style hybrid retrieval.")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=300)
    ap.add_argument("--semantic-entity-top-k", type=int, default=20)
    ap.add_argument("--semantic-entity-min-score", type=float, default=0.60)
    ap.add_argument("--no-reference-expansion", action="store_true")
    ap.add_argument("--dense-weight", type=float, default=0.15)
    ap.add_argument("--bm25-weight", type=float, default=0.25)
    ap.add_argument("--graph-weight", type=float, default=0.15)
    ap.add_argument("--reference-weight", type=float, default=0.30)
    ap.add_argument("--reference-max-seed-passages", type=int, default=30)
    ap.add_argument("--graph-only-penalty", type=float, default=0.65)
    ap.add_argument("--use-query-gliner", action="store_true")
    ap.add_argument(
        "--query-gliner-model-dir",
        type=Path,
        default=ROOT / "ner_finetuning/data/models/gliner_traffic_ner/final_model",
    )
    ap.add_argument("--query-gliner-threshold", type=float, default=0.85)
    args = ap.parse_args()

    retriever = LinearRAGRetriever.from_index(args.index_dir, args.gazetteer_root)
    out = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        semantic_entity_top_k=args.semantic_entity_top_k,
        semantic_entity_min_score=args.semantic_entity_min_score,
        use_reference_expansion=not args.no_reference_expansion,
        reference_max_seed_passages=args.reference_max_seed_passages,
        weights={
            "dense": args.dense_weight,
            "bm25": args.bm25_weight,
            "graph": args.graph_weight,
            "reference": args.reference_weight,
        },
        graph_only_penalty=args.graph_only_penalty,
        use_query_gliner=args.use_query_gliner,
        query_gliner_model_dir=args.query_gliner_model_dir,
        query_gliner_threshold=args.query_gliner_threshold,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
