import argparse
from legal_entity_vocab.build_gazetteer import build_gazetteer

def main():
    ap = argparse.ArgumentParser(description="Build gazetteer from reviewed_surface_forms.csv")
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/gazetteers_v1")
    ap.add_argument("--include-conflicts", action="store_true", help="Keep accepted rows marked label_conflict=True")
    ap.add_argument("--min-count", type=int, default=8, help="Keep accepted surfaces with count >= this value; default 8 means count > 7")
    args = ap.parse_args()

    summary = build_gazetteer(args.reviewed, args.output, include_conflicts=args.include_conflicts, min_count=args.min_count)
    print("Gazetteer build completed")
    print(f"Accepted surfaces: {summary['accepted_surface_count']}")
    print(f"Canonical entities: {summary['canonical_entity_count']}")
    print(f"Skipped conflicts: {summary['skipped_conflict_count']}")
    print(f"Skipped low count: {summary['skipped_low_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
