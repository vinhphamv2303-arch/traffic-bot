from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .appendix_parser import AppendixParser
from .classifier import AttachmentKind, attachment_slug, classify_attachment
from .common import ensure_dir, read_jsonl, write_json, write_jsonl
from .form_parser import FormParser
from .qcvn_parser import QCVNParser
from ..common.doc_converter import convert_legacy_docs_under
from ..common.file_classifier import is_probable_attachment, is_probable_main_document
from ..common.logging_utils import configure_logging, get_logger


def discover_attachment_docx_files(
    input_path: Union[str, Path],
    *,
    recursive: bool = False,
    include_unknown: bool = False,
) -> List[Path]:
    """
    Discover attachment DOCX files.

    File input is always parsed directly. Folder input defaults to attachment-like
    filenames only; pass --include-unknown to include non-main DOCX files too.
    """
    p = Path(input_path)

    if p.suffix.lower() == ".doc" and p.with_suffix(".docx").exists():
        return [p.with_suffix(".docx")]
    if p.is_file() and p.suffix.lower() == ".docx" and not p.name.startswith("~$"):
        return [p]
    if not p.exists() or not p.is_dir():
        return []

    pattern = "**/*.docx" if recursive else "*.docx"
    files = sorted(
        x for x in p.glob(pattern)
        if x.is_file() and x.suffix.lower() == ".docx" and not x.name.startswith("~$")
    )

    if include_unknown:
        return [x for x in files if not is_probable_main_document(x)]
    return [x for x in files if is_probable_attachment(x)]


def parse_attachments(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    recursive: bool = False,
    include_unknown: bool = False,
    package_id: Optional[str] = None,
    document_id: Optional[str] = None,
    document_number: Optional[str] = None,
    convert_doc: bool = True,
    delete_converted_doc: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    log = get_logger(logger)
    source = Path(input_path)
    out_root = ensure_dir(output_dir)

    if convert_doc:
        convert_legacy_docs_under(
            source,
            recursive=recursive,
            delete_source=delete_converted_doc,
            logger=log,
        )

    files = discover_attachment_docx_files(
        source,
        recursive=recursive,
        include_unknown=include_unknown,
    )
    if not files:
        log.warning("Khong tim thay file dinh kem .docx trong: %s", source)
        log.warning("Neu folder co file dinh kem ten khong ro, thu them flag: --include-unknown")
        return

    inferred_package_id = package_id or (source.name if source.is_dir() else source.parent.name)
    log.info("Bat dau parse %s attachment(s) | package_id=%s", len(files), inferred_package_id)

    inventory: Dict[str, Any] = {
        "package_id": inferred_package_id,
        "document_id": document_id,
        "document_number": document_number,
        "source_path": str(source),
        "attachments": [],
    }
    parsed_dirs: List[Path] = []
    errors: List[Tuple[str, str]] = []

    for file_path in files:
        try:
            kind = classify_attachment(file_path)
            att_out = out_root / attachment_slug(file_path)
            parser = _parser_for_kind(kind)
            parsed = parser.parse(
                docx_path=file_path,
                output_dir=att_out,
                package_id=inferred_package_id,
                document_id=document_id,
                document_number=document_number,
                kind=kind,
            )
            parsed_dirs.append(att_out)
            inventory["attachments"].append({
                **parsed["attachment"],
                "parsed_dir": str(att_out.relative_to(out_root)),
                "unit_count": parsed.get("unit_count"),
                "table_count": parsed.get("table_count"),
                "table_row_count": parsed.get("table_row_count"),
                "form_field_count": parsed.get("form_field_count"),
                "ref_mention_count": parsed.get("ref_mention_count"),
            })
            log.info(
                "OK %s | kind=%s | units=%s | tables=%s | refs=%s",
                file_path.name,
                kind.kind,
                parsed.get("unit_count"),
                parsed.get("table_count"),
                parsed.get("ref_mention_count"),
            )
        except Exception as exc:
            errors.append((str(file_path), str(exc)))
            log.error("Loi attachment %s: %s", file_path, exc)

    counts = _aggregate_outputs(out_root, parsed_dirs)
    inventory["aggregate_counts"] = counts
    if errors:
        inventory["errors"] = [{"source_file": path, "message": msg} for path, msg in errors]
    write_json(out_root / "attachments_inventory.json", inventory)

    log.info("Hoan tat attachments | success=%s/%s | all_units=%s | all_tables=%s | all_refs=%s",
             len(parsed_dirs), len(files), counts["units"], counts["tables"], counts["ref_mentions"])


def _parser_for_kind(kind: AttachmentKind):
    if kind.kind == "qcvn":
        return QCVNParser()
    if kind.kind in {"form", "appendix_form"}:
        return FormParser()
    return AppendixParser()


def _aggregate_outputs(out_root: Path, parsed_dirs: List[Path]) -> Dict[str, int]:
    all_units: List[Dict[str, Any]] = []
    all_tables: List[Dict[str, Any]] = []
    all_refs: List[Dict[str, Any]] = []

    for att_dir in parsed_dirs:
        for fname, target in [
            ("units.jsonl", all_units),
            ("tables.jsonl", all_tables),
            ("ref_mentions.jsonl", all_refs),
        ]:
            p = att_dir / fname
            if p.exists():
                target.extend(list(read_jsonl(p)))

    write_jsonl(out_root / "all_units.jsonl", all_units)
    write_jsonl(out_root / "all_tables.jsonl", all_tables)
    write_jsonl(out_root / "all_ref_mentions.jsonl", all_refs)
    return {"units": len(all_units), "tables": len(all_tables), "ref_mentions": len(all_refs)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Vietnamese legal attachments DOCX into structured JSON/JSONL.")
    ap.add_argument("--input", "-i", required=True, help="Attachment .docx/.doc file or folder")
    ap.add_argument("--output", "-o", default="../data/preprocessed/parsed_attachments", help="Output folder")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recursively scan attachment folders")
    ap.add_argument("--include-unknown", action="store_true", help="Parse non-main DOCX files even if filename is not attachment-like")
    ap.add_argument("--package-id", help="Package/document folder id used in attachment metadata")
    ap.add_argument("--document-id", help="Main document id to attach metadata to")
    ap.add_argument("--document-number", help="Main document number to attach metadata to")
    ap.add_argument("--keep-doc", action="store_true", help="Do not delete .doc files after successful conversion to .docx")
    ap.add_argument("--no-convert-doc", action="store_true", help="Do not convert legacy .doc files before parsing")
    ap.add_argument("--log-file", default="parse_attachments.log", help="Log filename written under the output folder")
    args = ap.parse_args()

    logger = configure_logging(args.output, log_filename=args.log_file)
    logger.info("Bat dau parse attachments | input=%s | output=%s", args.input, args.output)
    parse_attachments(
        args.input,
        args.output,
        recursive=args.recursive,
        include_unknown=args.include_unknown,
        package_id=args.package_id,
        document_id=args.document_id,
        document_number=args.document_number,
        convert_doc=not args.no_convert_doc,
        delete_converted_doc=not args.keep_doc,
        logger=logger,
    )


if __name__ == "__main__":
    main()
