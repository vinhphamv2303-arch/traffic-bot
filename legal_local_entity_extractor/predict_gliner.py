\
import argparse
from legal_local_entity_extractor.predict_gliner import predict_all_gliner

def main():
    ap = argparse.ArgumentParser(description="Run local GLiNER extraction on all sentences.")
    ap.add_argument("--sentences-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/entities_gliner_local_v1")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    summary = predict_all_gliner(
        sentences_root=args.sentences_root,
        model_dir=args.model_dir,
        output_dir=args.output,
        threshold=args.threshold,
    )
    print("GLiNER prediction completed")
    print(f"Sentences: {summary['sentence_count']}")
    print(f"Sentences with entities: {summary['sentence_with_entity_count']}")
    print(f"Entities: {summary['entity_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
