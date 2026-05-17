import argparse
import os
from pathlib import Path

from .config import LLMNERConfig
from .extractor import LegalLLMNER


def main():
    ap = argparse.ArgumentParser(description="LLM-based semantic NER for Vietnamese traffic law sentences.")
    ap.add_argument("--sentences-root", "-i", required=True, help="Path to data/preprocessed/sentences or one package folder")
    ap.add_argument("--output", "-o", default="./data/preprocessed/entities")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai_compatible", "mock"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint")
    ap.add_argument("--api-base", default="http://localhost:8000/v1", help="OpenAI-compatible API base")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    config = LLMNERConfig(
        sentences_root=Path(args.sentences_root),
        output_root=Path(args.output),
        provider=args.provider,
        model=args.model,
        endpoint=args.endpoint,
        api_base=args.api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        limit=args.limit,
        temperature=args.temperature,
        resume=not args.no_resume,
    )
    summary = LegalLLMNER(config).run_all()

    print("LLM NER completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Total sentences: {summary['total_sentences']}")
    print(f"Total entity mentions: {summary['total_entity_mentions']}")
    print(f"Provider/model: {summary['provider']} / {summary['model']}")
    print(f"Output: {config.output_root}")


if __name__ == "__main__":
    main()
