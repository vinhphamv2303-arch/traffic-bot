from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .parser import LegalBodyParser
from ..common.doc_converter import convert_legacy_docs_under
from ..common.file_classifier import select_main_documents
from ..common.logging_utils import configure_logging, get_logger
from ..common.models import ParserConfig


def discover_docx_files(input_path: Union[str, Path], recursive: bool = False, include_attachments: bool = False) -> List[Path]:
    """
    Discover DOCX files.

    Default behavior:
    - If input is one .docx file: parse that file directly.
    - If input is a folder and recursive=False: select only main document(s) in that folder.
    - If input is a folder and recursive=True: treat each folder as a package and select only main document(s).
    - Attachments are skipped unless --include-attachments is passed.
    """
    p = Path(input_path)

    if p.is_file() and p.suffix.lower() == ".docx" and not p.name.startswith("~$"):
        return [p]
    if p.is_file() and p.suffix.lower() == ".doc" and p.with_suffix(".docx").exists():
        return [p.with_suffix(".docx")]

    pattern = "**/*.docx" if recursive else "*.docx"
    all_files = sorted([
        x for x in p.glob(pattern)
        if x.is_file() and x.suffix.lower() == ".docx" and not x.name.startswith("~$")
    ])

    if include_attachments:
        return all_files

    if not recursive:
        return select_main_documents(all_files)

    by_parent: Dict[Path, List[Path]] = {}
    for f in all_files:
        by_parent.setdefault(f.parent, []).append(f)

    selected: List[Path] = []
    for parent, files in sorted(by_parent.items(), key=lambda kv: str(kv[0])):
        selected.extend(select_main_documents(files))

    return sorted(selected)


def parse_path(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    recursive: bool = False,
    include_attachments: bool = False,
    *,
    convert_doc: bool = True,
    delete_converted_doc: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    log = get_logger(logger)
    if convert_doc:
        convert_legacy_docs_under(
            input_path,
            recursive=recursive,
            delete_source=delete_converted_doc,
            logger=log,
        )

    parser = LegalBodyParser(ParserConfig(output_base_dir=Path(output_dir)))
    files = discover_docx_files(input_path, recursive=recursive, include_attachments=include_attachments)

    if not files:
        log.warning("⚠️ Không tìm thấy văn bản chính .docx nào trong: %s", input_path)
        log.warning("   Nếu muốn parse cả phụ lục/mẫu, thêm flag: --include-attachments")
        return

    log.info("🔍 Bắt đầu xử lý %s văn bản chính .docx", len(files))
    if include_attachments:
        log.info("⚠️ Đang bật --include-attachments: sẽ parse cả phụ lục/mẫu bằng body parser.")
    log.info("-" * 60)

    success = 0
    errors: List[Tuple[str, str]] = []

    for file_path in files:
        try:
            log.info("⏳ Đang phân tích: %s", file_path)
            result = parser.parse_file(file_path)
            success += 1
            log.info(
                "✅ %s | units=%s | tables=%s | ref_mentions=%s | amendment_mentions=%s | out=%s",
                result.document_id,
                result.unit_count,
                result.table_count,
                result.ref_mention_count,
                result.amendment_mention_count,
                result.output_dir,
            )
        except Exception as e:
            errors.append((str(file_path), str(e)))
            log.error("❌ Lỗi: %s: %s", file_path, e)

    log.info("\n" + "=" * 60)
    log.info("🎉 Thành công: %s/%s", success, len(files))
    if errors:
        log.error("❌ Thất bại: %s", len(errors))
        for path, msg in errors:
            log.error("   - %s: %s", path, msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse only main Vietnamese legal DOCX bodies into tree + flat legal units.")
    ap.add_argument("--input", "-i", default="../data/doc_body", help="Input .docx file or folder")
    ap.add_argument("--output", "-o", default="../data/preprocessed/parsed", help="Output folder")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recursively scan package folders")
    ap.add_argument(
        "--include-attachments",
        action="store_true",
        help="Parse all .docx files, including Phu luc/Mau/QCVN attachments. Default is false.",
    )
    ap.add_argument("--keep-doc", action="store_true", help="Do not delete .doc files after successful conversion to .docx")
    ap.add_argument("--no-convert-doc", action="store_true", help="Do not convert legacy .doc files before parsing")
    ap.add_argument("--log-file", default="parse.log", help="Log filename written under the output folder")
    args = ap.parse_args()

    logger = configure_logging(args.output, log_filename=args.log_file)
    logger.info("🚀 Bắt đầu parse body | input=%s | output=%s", args.input, args.output)
    parse_path(
        args.input,
        args.output,
        recursive=args.recursive,
        include_attachments=args.include_attachments,
        convert_doc=not args.no_convert_doc,
        delete_converted_doc=not args.keep_doc,
        logger=logger,
    )


if __name__ == "__main__":
    main()
