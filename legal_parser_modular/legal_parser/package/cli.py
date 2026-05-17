from __future__ import annotations

import argparse
from pathlib import Path

from ..common.logging_utils import configure_logging
from .parser import LegalPackageParser


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse legal dataset packages: main document + attachments.")
    ap.add_argument("--input", "-i", required=True, help="Dataset root or one package folder")
    ap.add_argument("--output", "-o", default="../data/preprocessed/parsed", help="Output parsed root")
    ap.add_argument("--single-package", action="store_true", help="Treat input as one package folder")
    ap.add_argument("--keep-doc", action="store_true", help="Do not delete .doc files after successful conversion to .docx")
    ap.add_argument("--no-convert-doc", action="store_true", help="Do not convert legacy .doc files before parsing")
    ap.add_argument("--log-file", default="parse.log", help="Log filename written under the output folder")
    args = ap.parse_args()

    logger = configure_logging(args.output, log_filename=args.log_file)
    logger.info("🚀 Bắt đầu parse package | input=%s | output=%s", args.input, args.output)

    parser = LegalPackageParser(
        args.output,
        logger=logger,
        convert_doc=not args.no_convert_doc,
        delete_converted_doc=not args.keep_doc,
    )
    input_path = Path(args.input)

    if args.single_package:
        results = [parser.parse_package(input_path)]
    else:
        # If input directly contains docx files, parse as one package; otherwise parse child folders.
        if any(input_path.glob("*.docx")):
            results = [parser.parse_package(input_path)]
        else:
            results = parser.parse_dataset(input_path)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 Parsed %s package(s)", len(results))
    for r in results:
        logger.info(
            "✅ %s | attachments=%s | all_units=%s | all_tables=%s | all_refs=%s | "
            "converted_doc=%s | failed_doc_conversion=%s | out=%s",
            r.package_id,
            r.attachment_count,
            r.all_unit_count,
            r.all_table_count,
            r.all_ref_mention_count,
            r.converted_doc_count,
            r.failed_doc_conversion_count,
            r.output_dir,
        )


if __name__ == "__main__":
    main()
