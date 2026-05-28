import argparse
import sys
from xner_span_mining_core import run_xner_mining


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run full X-NER-style span mining pipeline.")
    ap.add_argument("--sentences-root", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/xner_candidate_entities")
    ap.add_argument("--manual-seed-file", default=None)
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--max-ngram", type=int, default=14)
    ap.add_argument("--min-surface-count", type=int, default=1)
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-scoring", action="store_true", help="Run only seed + candidate generation; do not load embedding model.")
    ap.add_argument(
        "--quality-preset",
        choices=["balanced", "max"],
        default="max",
        help="max = recall/quality first: max_ngram=14, min_surface_count=1, min_score=0.35, batch_size as provided."
    )
    args = ap.parse_args()

    if args.quality_preset == "max":
        args.max_ngram = max(args.max_ngram, 14)
        args.min_surface_count = min(args.min_surface_count, 1)
        args.min_score = min(args.min_score, 0.35)

    summary = run_xner_mining(
        sentences_root=args.sentences_root,
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        manual_seed_file=args.manual_seed_file,
        max_sentences=args.max_sentences,
        max_ngram=args.max_ngram,
        min_surface_count=args.min_surface_count,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        min_score=args.min_score,
        device=args.device,
        skip_scoring=args.skip_scoring,
    )
    print("X-NER-style mining completed")
    print(summary)

if __name__ == "__main__":
    main()
