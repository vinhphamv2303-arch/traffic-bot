from dataclasses import dataclass
from pathlib import Path

@dataclass
class LLMNERv2Config:
    sentences_root: Path
    output_root: Path = Path("./data/preprocessed/entities_llm_v2")
    api_key: str | None = None
    api_base: str = "https://openrouter.ai/api/v1"
    gold_model: str = "openai/gpt-4.1"
    silver_model: str = "openai/gpt-4.1-mini"
    quality_tier: str = "mixed"  # gold | silver | mixed
    gold_limit: int = 1000
    batch_size: int = 8
    limit: int | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 120
    max_retries: int = 5
    retry_base_seconds: float = 2.0
    candidate_only: bool = True
    min_candidate_score: int = 1
    resume: bool = True
    text_field: str = "text"
    context_field: str = "context_text"
    reject_unaligned: bool = True
    use_json_schema: bool = True
