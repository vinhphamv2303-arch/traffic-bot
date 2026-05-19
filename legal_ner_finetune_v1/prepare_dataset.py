import argparse
from legal_ner_finetune_v1.prepare import prepare_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entities-root', required=True)
    ap.add_argument('--output-dir', default='./data/preprocessed/ner_train_v1')
    ap.add_argument('--no-auto-clean', action='store_true')
    args = ap.parse_args()
    s = prepare_dataset(args.entities_root, args.output_dir, auto_clean=not args.no_auto_clean)
    print('Prepared dataset')
    print(f"Rows: {s['row_count']}")
    print(f"Rows with entities: {s['row_with_entity_count']}")
    print(f"Entities: {s['entity_count']}")
    print(f"Train file: {s['train_file']}")

if __name__ == '__main__':
    main()
