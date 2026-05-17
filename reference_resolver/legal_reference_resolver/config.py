from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResolverConfig:
    parsed_root: Path
    output_root: Path = Path("./data/preprocessed/resolved_references")
    resolved_threshold: float = 0.90
    ambiguous_threshold: float = 0.65
    max_candidates: int = 8
