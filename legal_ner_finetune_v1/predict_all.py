import argparse
from pathlib import Path
from legal_ner_finetune_v1.predict import predict_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--sentences-root', required=True)
    ap.add_argument('--output-dir', default='./data/preprocessed/entities_model_v1')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--device', default=None)
    ap.add_argument('--min-confidence', type=float, default=0.0)
    args = ap.parse_args()
    s = predict_all(Path(args.model_dir), Path(args.sentences_root), Path(args.output_dir), batch_size=args.batch_size, max_length=args.max_length, device=args.device, min_confidence=args.min_confidence)
    print('Prediction completed')
    print(f"Sentences: {s['sentence_count']}")
    print(f"Sentences with entities: {s['sentence_with_entity_count']}")
    print(f"Entities: {s['entity_count']}")
    print(f"Output: {args.output_dir}")

if __name__ == '__main__':
    main()
