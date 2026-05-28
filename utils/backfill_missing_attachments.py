from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from docx import Document

from normalize_raw_dataset import (
    ROOT,
    attachment_heading_key,
    clean_filename,
    compact_text,
    docx_content_children,
    find_embedded_attachment_headings,
    is_attachment_name,
    save_docx_segment,
    unique_path,
)


DEFAULT_DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_LOG_FILE = ROOT / "data" / "raw" / "backfill_missing_attachments.log"
DEFAULT_SUMMARY = ROOT / "data" / "raw" / "backfill_missing_attachments_summary.json"


@dataclass
class AttachmentCandidate:
    title: str
    key: str
    source_main: str
    output_name: Optional[str] = None
    status: str = "pending"
    reason: str = ""


@dataclass
class PackageBackfillResult:
    package_id: str
    package_dir: str
    main_files: List[str] = field(default_factory=list)
    candidates: List[AttachmentCandidate] = field(default_factory=list)
    status: str = "ok"
    message: str = ""


def setup_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("backfill_missing_attachments")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_manifest_main_files(package_dir: Path) -> List[Path]:
    manifest = package_dir / "dataset_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    main_files: List[Path] = []
    for item in payload.get("files", []):
        if item.get("role") != "main":
            continue
        if item.get("suffix") != ".docx":
            continue
        path = package_dir / item.get("output_name", "")
        if path.exists() and path.is_file():
            main_files.append(path)
    return main_files


def select_main_docx_files(package_dir: Path) -> List[Path]:
    manifest_main = load_manifest_main_files(package_dir)
    if manifest_main:
        return manifest_main

    docx_files = sorted(
        path
        for path in package_dir.glob("*.docx")
        if path.is_file() and not path.name.startswith("~$") and not is_attachment_name(path.name)
    )
    if not docx_files:
        return []

    preferred = [
        path
        for path in docx_files
        if " doc" in compact_text(path.stem)
        or "thongtu" in compact_text(path.stem)
        or "nghidinh" in compact_text(path.stem)
        or "luat" in compact_text(path.stem)
        or "quyetdinh" in compact_text(path.stem)
        or "nghiquyet" in compact_text(path.stem)
    ]
    return preferred or docx_files


def existing_file_identities(package_dir: Path) -> Dict[str, str]:
    identities: Dict[str, str] = {}
    for path in package_dir.iterdir():
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in {".docx", ".doc", ".pdf", ".rtf", ".xls", ".xlsx"}:
            continue
        identities[compact_text(path.stem)] = path.name
    return identities


def existing_attachment_keys(package_dir: Path) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for path in package_dir.iterdir():
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in {".docx", ".doc", ".pdf", ".rtf", ".xls", ".xlsx"}:
            continue
        if not is_attachment_name(path.name):
            continue
        key = attachment_heading_key(path.stem)
        if key:
            keys.setdefault(key, path.name)
    return keys


def candidate_already_exists(
    title: str,
    key: str,
    identities: Dict[str, str],
    attachment_keys: Dict[str, str],
) -> Optional[str]:
    candidate = compact_text(Path(clean_filename(f"{title}.docx")).stem)
    if candidate in identities:
        return identities[candidate]

    for existing_identity, existing_name in identities.items():
        if len(candidate) >= 24 and (candidate in existing_identity or existing_identity in candidate):
            return existing_name

    # Appendix/QCVN headings are usually one document per key. Forms can repeat
    # across appendices, so they require a stronger title match above.
    if key.startswith("phu luc") or key.startswith("qcvn") or key.startswith("quy chuan"):
        if key in attachment_keys:
            return attachment_keys[key]
    return None


def process_package(
    package_dir: Path,
    *,
    dry_run: bool,
    min_main_chars: int,
    logger: logging.Logger,
) -> PackageBackfillResult:
    result = PackageBackfillResult(package_id=package_dir.name, package_dir=str(package_dir))
    main_files = select_main_docx_files(package_dir)
    result.main_files = [path.name for path in main_files]
    if not main_files:
        result.status = "skipped"
        result.message = "No main .docx file found."
        return result

    identities = existing_file_identities(package_dir)
    attachment_keys = existing_attachment_keys(package_dir)

    for main_path in main_files:
        try:
            document = Document(main_path)
        except Exception as exc:
            result.status = "warning"
            result.message = f"Cannot open {main_path.name}: {exc}"
            logger.warning("%s | cannot open %s: %s", package_dir.name, main_path.name, exc)
            continue

        blocks = docx_content_children(document)
        headings = find_embedded_attachment_headings(document, min_main_chars)
        if not headings:
            continue

        logger.info("%s | %s | embedded headings: %s", package_dir.name, main_path.name, len(headings))
        for index, heading in enumerate(headings):
            next_start = headings[index + 1].start if index + 1 < len(headings) else len(blocks)
            if next_start <= heading.start:
                continue
            existing_name = candidate_already_exists(heading.title, heading.key, identities, attachment_keys)
            candidate = AttachmentCandidate(
                title=heading.title,
                key=heading.key,
                source_main=main_path.name,
            )
            if existing_name:
                candidate.status = "skipped_existing"
                candidate.reason = existing_name
                result.candidates.append(candidate)
                continue

            output_name = clean_filename(f"{heading.title}.docx")
            output_path = unique_path(package_dir, output_name)
            candidate.output_name = output_path.name
            if dry_run:
                candidate.status = "would_add"
            else:
                save_docx_segment(blocks, output_path, heading.start, next_start, source_path=main_path)
                candidate.status = "added"
                identities[compact_text(output_path.stem)] = output_path.name
                if is_attachment_name(output_path.name):
                    attachment_keys.setdefault(attachment_heading_key(output_path.stem), output_path.name)
            result.candidates.append(candidate)

    if not result.candidates:
        result.status = "ok"
        result.message = "No missing embedded attachment found."
    return result


def write_summary(path: Path, results: Sequence[PackageBackfillResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_packages": len(results),
        "by_candidate_status": {},
        "by_package_status": {},
        "results": [asdict(result) for result in results],
    }
    for result in results:
        payload["by_package_status"][result.status] = payload["by_package_status"].get(result.status, 0) + 1
        for candidate in result.candidates:
            payload["by_candidate_status"][candidate.status] = (
                payload["by_candidate_status"].get(candidate.status, 0) + 1
            )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add missing appendix/form files from embedded sections in existing dataset main documents."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--min-main-chars-before-split", type=int, default=1200)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_file, args.verbose)
    dataset_root = args.dataset_root.resolve()

    packages = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if args.only:
        needles = [compact_text(value) for value in args.only if compact_text(value)]
        packages = [path for path in packages if any(needle in compact_text(path.name) for needle in needles)]

    logger.info("Dataset root: %s", dataset_root)
    logger.info("Packages to scan: %s", len(packages))
    if args.dry_run:
        logger.info("Dry-run mode: no attachment files will be written.")

    results: List[PackageBackfillResult] = []
    for index, package_dir in enumerate(packages, start=1):
        logger.info("[%s/%s] %s", index, len(packages), package_dir.name)
        results.append(
            process_package(
                package_dir,
                dry_run=args.dry_run,
                min_main_chars=args.min_main_chars_before_split,
                logger=logger,
            )
        )

    write_summary(args.summary, results)
    logger.info("Summary written to: %s", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
