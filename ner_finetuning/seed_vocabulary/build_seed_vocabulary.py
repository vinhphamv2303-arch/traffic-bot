import argparse
from seed_vocabulary_core.aggregate import aggregate_entity_vocab

def main():
    ap = argparse.ArgumentParser(description="Build reviewable seed vocabulary from bootstrap entity mentions.")
    ap.add_argument("--entity-mentions", required=True, help="Path to all_entity_mentions.jsonl")
    ap.add_argument("--output", "-o", default="./data/preprocessed/seed_vocabulary")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    summary = aggregate_entity_vocab(
        entity_mentions_path=args.entity_mentions,
        output_dir=args.output,
        max_examples=args.max_examples,
        min_count_for_summary=args.min_count,
    )
    print("Vocabulary aggregation completed")
    print(f"Total mentions: {summary['total_mentions']}")
    print(f"Surface forms: {summary['surface_form_count']}")
    print(f"Label conflicts: {summary['label_conflict_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
