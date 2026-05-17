from dataclasses import dataclass
from pathlib import Path

@dataclass
class SentenceSplitterConfig:
    passages_root: Path
    output_root: Path = Path("./data/preprocessed/sentences")
    split_source_field: str = "content"
    keep_whole_unit_types: tuple[str, ...] = (
        "table_row",
        "form_field",
        "form_table",
        "appendix_table",
    )
    include_context_for_ner: bool = True
    min_sentence_chars: int = 2
