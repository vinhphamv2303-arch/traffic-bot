import argparse
import sys
from legal_xner_span_miner.seeds import build_seeds


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build seed entities for X-NER-style mining.")
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/xner_mined_entities_v1")
    ap.add_argument("--manual-seed-file", default=None)
    ap.add_argument("--max-gazetteer-seeds-per-label", type=int, default=30)
    args = ap.parse_args()

    summary = build_seeds(
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        manual_seed_file=args.manual_seed_file,
        max_gazetteer_seeds_per_label=args.max_gazetteer_seeds_per_label,
    )
    print("Seeds built")
    print(summary)

if __name__ == "__main__":
    main()
