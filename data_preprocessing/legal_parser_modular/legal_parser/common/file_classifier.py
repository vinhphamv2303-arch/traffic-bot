from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from .utils import strip_vietnamese_accents


ATTACHMENT_PREFIX_PATTERNS = [
    r"^phu\s*luc\b",
    r"^phụ\s*lục\b",
    r"^mau\b",
    r"^mẫu\b",
    r"^mau\s*so\b",
    r"^mẫu\s*số\b",
    r"^mau\s*dkx\b",
    r"^dkx\b",
    r"^qcvn\b",
    r"^quy\s*chuan\b",
    r"^quy\s*chuẩn\b",
]

MAIN_DOC_HINT_PATTERNS = [
    r"\bluat\b",
    r"\bluật\b",
    r"\bnghi\s*dinh\b",
    r"\bnghị\s*định\b",
    r"\bthong\s*tu\b",
    r"\bthông\s*tư\b",
    r"\bquyet\s*dinh\b",
    r"\bquyết\s*định\b",
    r"\bnghi\s*quyet\b",
    r"\bnghị\s*quyết\b",
    r"\bdoc\b",
]


def _normalized_candidates(path: Path) -> set[str]:
    name = path.stem.lower().replace("_", " ").replace("-", " ")
    name_ascii = strip_vietnamese_accents(name, keep_dd=False).lower()
    return {
        re.sub(r"\s+", " ", name).strip(),
        re.sub(r"\s+", " ", name_ascii).strip(),
    }


def is_probable_attachment(path: Path) -> bool:
    for candidate in _normalized_candidates(path):
        for pat in ATTACHMENT_PREFIX_PATTERNS:
            if re.search(pat, candidate, re.IGNORECASE):
                return True
    return False


def is_probable_main_document(path: Path) -> bool:
    if is_probable_attachment(path):
        return False
    return any(
        re.search(pat, candidate, re.IGNORECASE)
        for candidate in _normalized_candidates(path)
        for pat in MAIN_DOC_HINT_PATTERNS
    )


def select_main_documents(files: Iterable[Path]) -> List[Path]:
    """
    Select only main legal documents from a list of .docx files.

    Rule:
    - Exclude attachment-like files: Phu luc, Mau, DKX, QCVN...
    - Prefer files whose names contain Luat/ND/Thong tu/Quyet dinh/Nghi quyet/DOC.
    - If a folder has exactly one non-attachment .docx, keep it.
    """
    files = sorted([f for f in files if f.suffix.lower() == ".docx" and not f.name.startswith("~$")])
    non_attachments = [f for f in files if not is_probable_attachment(f)]
    hinted = [f for f in non_attachments if is_probable_main_document(f)]

    if hinted:
        return hinted
    if len(non_attachments) == 1:
        return non_attachments
    return []
