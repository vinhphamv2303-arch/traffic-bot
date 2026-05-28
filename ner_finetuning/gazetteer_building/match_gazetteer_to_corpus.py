import argparse
from gazetteer_building_core.matcher import match_all_sentence_packages

def main():
    ap = argparse.ArgumentParser(description="Match a gazetteer over all sentence packages.")
    ap.add_argument("--sentences-root", "-i", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/gazetteer_pseudo_labels")
    args = ap.parse_args()

    summary = match_all_sentence_packages(args.sentences_root, args.gazetteer_root, args.output)
    print("Gazetteer matching completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Sentences: {summary['sentence_count']}")
    print(f"Sentences with entities: {summary['sentence_with_entity_count']}")
    print(f"Entity links: {summary['entity_link_count']}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
