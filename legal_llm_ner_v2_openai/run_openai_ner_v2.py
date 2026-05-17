import argparse
from pathlib import Path

from legal_llm_ner_v2.config import LLMNERv2Config
from legal_llm_ner_v2.env import get_api_key, load_dotenv
from legal_llm_ner_v2.extractor import LLMNERv2


def main():
    ap = argparse.ArgumentParser(description="OpenAI/OpenRouter LLM NER v2 with 7 retrieval-oriented labels")
    ap.add_argument("--sentences-root", "-i", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/entities_llm_v2")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--gold-model", default="openai/gpt-4.1")
    ap.add_argument("--silver-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--quality-tier", choices=["gold", "silver", "mixed"], default="mixed")
    ap.add_argument("--gold-limit", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-base-seconds", type=float, default=2.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--all-sentences", action="store_true")
    ap.add_argument("--min-candidate-score", type=int, default=1)
    ap.add_argument("--no-json-schema", action="store_true")
    args = ap.parse_args()

    load_dotenv(args.env_file)
    cfg = LLMNERv2Config(
        sentences_root=Path(args.sentences_root),
        output_root=Path(args.output),
        api_key=get_api_key(args.api_key),
        api_base=args.api_base,
        gold_model=args.gold_model,
        silver_model=args.silver_model,
        quality_tier=args.quality_tier,
        gold_limit=args.gold_limit,
        batch_size=args.batch_size,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        resume=not args.no_resume,
        candidate_only=not args.all_sentences,
        min_candidate_score=args.min_candidate_score,
        use_json_schema=not args.no_json_schema,
    )
    summary = LLMNERv2(cfg).run_all()
    print("OpenAI/OpenRouter LLM NER v2 completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Input sentences: {summary['total_input_sentences']}")
    print(f"Selected sentences: {summary['total_selected_sentences']}")
    print(f"Annotated sentences: {summary['total_annotated_sentences']}")
    print(f"Entity mentions: {summary['total_entity_mentions']}")
    print(f"Output: {cfg.output_root}")


if __name__ == "__main__":
    main()
