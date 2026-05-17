from dataclasses import dataclass
from pathlib import Path

@dataclass
class PassageBuilderConfig:
    parsed_root: Path
    output_root: Path = Path("./data/preprocessed/passages")
    effectivity_root: Path | None = None
    resolved_refs_root: Path | None = None
    include_container_passages: bool = True
    include_source_reference_text: bool = True
    max_ref_summary: int = 8
