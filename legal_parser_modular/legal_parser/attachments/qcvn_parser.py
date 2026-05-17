from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .appendix_parser import AppendixParser
from .classifier import AttachmentKind, classify_attachment


class QCVNParser(AppendixParser):
    """
    MVP QCVN parser.

    QCVN parsing usually needs deeper technical table extraction.
    For now, inherit structured appendix parser and tag output as qcvn.
    Later this can specialize definitions, requirements, thresholds, and test methods.
    """

    parser_name = "qcvn_parser"

    def parse(
        self,
        *,
        docx_path: Union[str, Path],
        output_dir: Union[str, Path],
        package_id: str,
        document_id: Optional[str] = None,
        document_number: Optional[str] = None,
        kind: Optional[AttachmentKind] = None,
    ) -> Dict[str, Any]:
        kind = kind or classify_attachment(docx_path)
        result = super().parse(
            docx_path=docx_path,
            output_dir=output_dir,
            package_id=package_id,
            document_id=document_id,
            document_number=document_number,
            kind=kind,
        )
        return result
