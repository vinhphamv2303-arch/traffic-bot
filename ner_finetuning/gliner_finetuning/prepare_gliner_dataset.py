import argparse
from gliner_finetuning_core.prepare import build_gliner_dataset

def main():
    ap = argparse.ArgumentParser(description="Build GLiNER training dataset from gazetteer pseudo-labels.")
    ap.add_argument("--entities-root", required=True, help="Root containing */sentence_entities.jsonl from gazetteer/entity extractor.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--negative-ratio", type=float, default=0.35)
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = build_gliner_dataset(
        entities_root=args.entities_root,
        output_dir=args.output,
        min_weight=args.min_weight,
        negative_ratio=args.negative_ratio,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    print("GLiNER dataset prepared")
    print(summary)

if __name__ == "__main__":
    main()
