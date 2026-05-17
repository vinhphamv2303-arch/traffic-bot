from dataclasses import dataclass
from pathlib import Path

@dataclass
class LLMNERConfig:
    sentences_root: Path
    output_root: Path = Path("./data/preprocessed/entities")

    provider: str = "ollama"  # ollama | openai_compatible | mock
    model: str = "qwen3:8b"
    endpoint: str = "http://localhost:11434"

    # For openai_compatible provider
    api_key: str | None = None
    api_base: str = "http://localhost:8000/v1"

    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 120

    batch_size: int = 8
    limit: int | None = None
    resume: bool = True

    input_field: str = "sentence_text_for_ner"
    fallback_input_field: str = "text"

    min_confidence: float = 0.0
    review_status: str = "silver"

    # Do not extract structural legal references as semantic entities.
    block_reference_like_entities: bool = True
