import argparse
from legal_xner_span_miner.scoring import score_candidates_embedding

def main():
    ap = argparse.ArgumentParser(description="Score candidate spans against seeds with embedding-based X-NER-style mining.")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/xner_mined_entities_v1")
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k-seeds-per-candidate", type=int, default=10)
    args = ap.parse_args()

    summary = score_candidates_embedding(
        seeds_path=args.seeds,
        candidates_path=args.candidates,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        min_score=args.min_score,
        top_k_seeds_per_candidate=args.top_k_seeds_per_candidate,
        device=args.device,
    )
    print("Candidates scored")
    print(summary)

if __name__ == "__main__":
    main()
