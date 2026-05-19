\
import argparse
from legal_local_entity_extractor.prepare_data import build_local_ner_dataset

def main():
    ap = argparse.ArgumentParser(description="Build local NER training data from pruned gazetteer + sentences.")
    ap.add_argument("--sentences-root", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/local_ner_train_v1")
    ap.add_argument("--no-inherited", action="store_true")
    ap.add_argument("--negative-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = build_local_ner_dataset(
        sentences_root=args.sentences_root,
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        include_inherited=not args.no_inherited,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    print("Local NER dataset prepared")
    print(f"Rows: {summary['dataset_rows']} | train={summary['train_rows']} dev={summary['dev_rows']} test={summary['test_rows']}")
    print(f"Direct entities: {summary['direct_entity_count']}")
    print(f"Inherited entities all sentences: {summary['inherited_entity_count_all_sentences']}")
    print(f"By label: {summary['by_label']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
