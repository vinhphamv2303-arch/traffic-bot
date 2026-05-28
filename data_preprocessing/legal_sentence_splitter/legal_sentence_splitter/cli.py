import argparse
from pathlib import Path
from .config import SentenceSplitterConfig
from .splitter import LegalSentenceSplitter

def main():
    ap = argparse.ArgumentParser(description="Split legal passages into sentences for NER/entity extraction.")
    ap.add_argument("--passages-root", "-i", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/sentences")
    ap.add_argument("--no-context-for-ner", action="store_true")
    args = ap.parse_args()

    config = SentenceSplitterConfig(
        passages_root=Path(args.passages_root),
        output_root=Path(args.output),
        include_context_for_ner=not args.no_context_for_ner,
    )
    summary = LegalSentenceSplitter(config).split_all()

    print("🎉 Sentence splitting completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Total sentences: {summary['total_sentences']}")
    print(f"Output: {config.output_root}")

if __name__ == "__main__":
    main()
