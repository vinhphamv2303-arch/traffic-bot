from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .logging_utils import get_logger


WD_FORMAT_XML_DOCUMENT = 16


@dataclass
class DocConversionResult:
    source_path: Path
    target_path: Path
    status: str
    converter: Optional[str] = None
    message: str = ""
    deleted_source: bool = False


def find_legacy_doc_files(input_path: Union[str, Path], *, recursive: bool = False) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if _is_legacy_doc(path) else []
    if not path.exists():
        return []
    pattern = "**/*.doc" if recursive else "*.doc"
    return sorted(p for p in path.glob(pattern) if _is_legacy_doc(p))


def convert_legacy_doc_files(
    files: Iterable[Union[str, Path]],
    *,
    delete_source: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[DocConversionResult]:
    log = get_logger(logger)
    results: List[DocConversionResult] = []
    for file_path in files:
        result = convert_legacy_doc_file(file_path, delete_source=delete_source, logger=log)
        results.append(result)
    return results


def convert_legacy_docs_under(
    input_path: Union[str, Path],
    *,
    recursive: bool = False,
    delete_source: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[DocConversionResult]:
    files = find_legacy_doc_files(input_path, recursive=recursive)
    if not files:
        return []
    log = get_logger(logger)
    log.info("🧾 Tìm thấy %s file .doc cần chuyển đổi", len(files))
    return convert_legacy_doc_files(files, delete_source=delete_source, logger=log)


def convert_legacy_doc_file(
    source_path: Union[str, Path],
    *,
    delete_source: bool = True,
    logger: Optional[logging.Logger] = None,
) -> DocConversionResult:
    log = get_logger(logger)
    source = Path(source_path)
    target = source.with_suffix(".docx")

    if not _is_legacy_doc(source):
        return DocConversionResult(source, target, "skipped", message="Not a legacy .doc file.")
    if not source.exists():
        return DocConversionResult(source, target, "failed", message="Source file does not exist.")

    if _valid_docx(target):
        deleted = _delete_source_if_requested(source, target, delete_source, log)
        return DocConversionResult(
            source,
            target,
            "already_converted",
            message="Target .docx already exists.",
            deleted_source=deleted,
        )

    log.info("🔄 Chuyển .doc sang .docx: %s", source)
    errors: List[str] = []
    for converter in (_convert_with_word, _convert_with_libreoffice):
        converter_name = converter.__name__.replace("_convert_with_", "")
        try:
            converter(source, target)
            if _valid_docx(target):
                deleted = _delete_source_if_requested(source, target, delete_source, log)
                log.info("✅ Đã chuyển đổi: %s -> %s", source.name, target.name)
                return DocConversionResult(
                    source,
                    target,
                    "converted",
                    converter=converter_name,
                    deleted_source=deleted,
                )
            errors.append(f"{converter_name}: target .docx was not created or is empty")
        except Exception as exc:
            errors.append(f"{converter_name}: {exc}")
            log.debug("Converter failed for %s via %s", source, converter_name, exc_info=True)

    message = " | ".join(errors) if errors else "No converter was available."
    log.error("❌ Không chuyển được .doc: %s | %s", source, message)
    return DocConversionResult(source, target, "failed", message=message)


def _is_legacy_doc(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and path.suffix.lower() == ".doc" and not name.startswith("~$")


def _valid_docx(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _delete_source_if_requested(source: Path, target: Path, delete_source: bool, log: logging.Logger) -> bool:
    if not delete_source:
        return False
    if not _valid_docx(target):
        log.warning("⚠️ Không xoá .doc vì .docx chưa hợp lệ: %s", target)
        return False
    try:
        source.unlink()
        log.info("🗑️ Đã xoá file .doc sau khi có .docx: %s", source)
        return True
    except Exception as exc:
        log.warning("⚠️ Không xoá được .doc: %s | %s", source, exc)
        return False


def _convert_with_word(source: Path, target: Path) -> None:
    try:
        import win32com.client  # type: ignore
    except Exception as exc:  # pragma: no cover - platform/env dependent
        raise RuntimeError("pywin32/Word COM is not available") from exc

    word = None
    doc = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(
            str(source.resolve()),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
        )
        doc.SaveAs2(str(target.resolve()), FileFormat=WD_FORMAT_XML_DOCUMENT)
    finally:
        if doc is not None:
            doc.Close(False)
        if word is not None:
            word.Quit()


def _convert_with_libreoffice(source: Path, target: Path) -> None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("LibreOffice/soffice is not available")
    completed = subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(source.parent.resolve()),
            str(source.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(stderr or f"LibreOffice exited with code {completed.returncode}")
    if not target.exists():
        raise RuntimeError("LibreOffice reported success but target .docx is missing")
