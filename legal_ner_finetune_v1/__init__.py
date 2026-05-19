from pathlib import Path

_inner_package = Path(__file__).resolve().parent / "legal_ner_finetune_v1"
if _inner_package.exists():
    __path__.append(str(_inner_package))

