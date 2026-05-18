import argparse
from legal_entity_vocab.prune import prune_gazetteer

def main():
    ap = argparse.ArgumentParser(description="Prune/downweight generic gazetteer aliases.")
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/gazetteers_v1_pruned")
    ap.add_argument("--keep-rejected", action="store_true", help="Keep rejected aliases in output with match_mode=reject")
    args = ap.parse_args()

    summary = prune_gazetteer(
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        drop_reject=not args.keep_rejected,
    )
    print("Gazetteer pruning completed")
    print(f"Input aliases: {summary['input_alias_count']}")
    print(f"Output aliases: {summary['output_alias_count']}")
    print(f"Kept: {summary['kept_alias_count']}")
    print(f"Downweighted: {summary['downweighted_alias_count']}")
    print(f"Rejected: {summary['rejected_alias_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
