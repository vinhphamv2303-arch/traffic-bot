import argparse
from legal_xner_span_miner.candidates import collect_span_candidates

def main():
    ap = argparse.ArgumentParser(description="Generate candidate spans from sentences/path_text.")
    ap.add_argument("--sentences-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/xner_mined_entities_v1")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--max-ngram", type=int, default=10)
    ap.add_argument("--min-surface-count", type=int, default=2)
    ap.add_argument("--no-path-text", action="store_true")
    args = ap.parse_args()

    summary = collect_span_candidates(
        sentences_root=args.sentences_root,
        output_dir=args.output,
        max_sentences=args.max_sentences,
        max_ngram=args.max_ngram,
        include_path_text=not args.no_path_text,
        min_surface_count=args.min_surface_count,
    )
    print("Candidates generated")
    print(summary)

if __name__ == "__main__":
    main()
