import argparse
from legal_xner_span_miner.gazetteer_v2 import build_gazetteer_v2

def main():
    ap = argparse.ArgumentParser(description="Build gazetteer v2 from pruned gazetteer + reviewed mined candidates.")
    ap.add_argument("--base-gazetteer-root", required=True)
    ap.add_argument("--reviewed-mined-csv", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/gazetteers_v2")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument(
        "--accept-auto-candidates",
        action="store_true",
        help="Also accept rows still marked accept_candidate/auto_accept. Default requires human-reviewed accept/keep statuses.",
    )
    args = ap.parse_args()

    summary = build_gazetteer_v2(
        base_gazetteer_root=args.base_gazetteer_root,
        reviewed_mined_csv=args.reviewed_mined_csv,
        output_dir=args.output,
        min_score=args.min_score,
        accept_auto_candidates=args.accept_auto_candidates,
    )
    print("Gazetteer v2 built")
    print(summary)

if __name__ == "__main__":
    main()
