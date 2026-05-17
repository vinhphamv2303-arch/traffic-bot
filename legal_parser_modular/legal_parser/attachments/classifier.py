from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .common import collapse_ws, get_docx_texts, safe_dirname, strip_vietnamese_accents


@dataclass
class AttachmentKind:
    kind: str
    confidence: float
    reason: str


def classify_attachment(path: Union[str, Path]) -> AttachmentKind:
    path = Path(path)
    name = path.stem.lower().replace("_", " ").replace("-", " ")
    name_ascii = strip_vietnamese_accents(name, keep_dd=False).lower()
    head = "\n".join(get_docx_texts(path, limit=40)).lower()
    head_ascii = strip_vietnamese_accents(head, keep_dd=False).lower()

    combined = f"{name}\n{name_ascii}\n{head}\n{head_ascii}"

    qcvn_head = any(
        re.match(r"^\s*qcvn\b", line, re.I)
        or re.match(r"^\s*quy\s*chuan\s*ky\s*thuat\s*quoc\s*gia\b", strip_vietnamese_accents(line, keep_dd=False), re.I)
        for line in get_docx_texts(path, limit=12)
    )
    if re.search(r"\bqcvn\b", name_ascii, re.I) or qcvn_head:
        return AttachmentKind("qcvn", 0.95, "filename/content contains QCVN/quy chuẩn kỹ thuật quốc gia")

    # Direct form/mẫu file.
    if re.search(r"^(mau|mẫu)\b|^mau\s*so\b|^mẫu\s*số\b|^mau\s*dkx\b|^dkx\b", name_ascii, re.I):
        return AttachmentKind("form", 0.90, "filename starts with Mau/DKX")

    # Appendix that looks like a form.
    form_signals = 0
    for sig in ["họ và tên", "ho va ten", "ngày sinh", "so can cuoc", "số căn cước", "ký tên", "dong dau", "đóng dấu", "kính gửi", "noi nhan", "nơi nhận"]:
        if sig in combined:
            form_signals += 1

    if re.search(r"^(phu\s*luc|phụ\s*lục)\b", name_ascii, re.I):
        if form_signals >= 2:
            return AttachmentKind("appendix_form", 0.86, "appendix with form-like fields")
        return AttachmentKind("appendix_structured", 0.78, "filename starts with Phu luc")

    if form_signals >= 3:
        return AttachmentKind("form", 0.74, "content has form-like fields")

    return AttachmentKind("unknown_attachment", 0.35, "no strong rule matched")


def attachment_slug(path: Union[str, Path]) -> str:
    name = Path(path).stem
    name_ascii = safe_dirname(name)
    name_ascii = re.sub(r"[^a-zA-Z0-9]+", "_", name_ascii).strip("_").lower()
    if len(name_ascii) <= 96:
        return name_ascii or "attachment"
    digest = hashlib.md5(name_ascii.encode("utf-8")).hexdigest()[:8]
    return f"{name_ascii[:80].rstrip('_')}_{digest}"
