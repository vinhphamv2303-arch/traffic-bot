import argparse
from legal_entity_vocab.matcher import match_all_sentence_packages

def main():
    ap = argparse.ArgumentParser(description="Match pruned gazetteer over all sentence packages.")
    ap.add_argument("--sentences-root", "-i", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/entity_links_v1_pruned")
    args = ap.parse_args()

    summary = match_all_sentence_packages(args.sentences_root, args.gazetteer_root, args.output)
    print("Pruned gazetteer matching completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Sentences: {summary['sentence_count']}")
    print(f"Sentences with entities: {summary['sentence_with_entity_count']}")
    print(f"Entity links: {summary['entity_link_count']}")
    print(f"By mode: {summary['by_match_mode']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
