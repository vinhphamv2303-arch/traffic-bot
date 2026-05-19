\
import argparse
from legal_local_entity_extractor.predict_gazetteer import predict_all_gazetteer

def main():
    ap = argparse.ArgumentParser(description="CPU deterministic gazetteer extractor baseline.")
    ap.add_argument("--sentences-root", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/entities_gazetteer_local_v1")
    args = ap.parse_args()

    summary = predict_all_gazetteer(args.sentences_root, args.gazetteer_root, args.output)
    print("Gazetteer local prediction completed")
    print(f"Sentences: {summary['sentence_count']}")
    print(f"Sentences with entities: {summary['sentence_with_entity_count']}")
    print(f"Direct entities: {summary['entity_count']}")
    print(f"Inherited entities: {summary['inherited_entity_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
