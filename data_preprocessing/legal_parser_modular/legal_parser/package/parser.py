from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from ..body.parser import LegalBodyParser
    from ..common.models import ParserConfig
except Exception:  # pragma: no cover
    LegalBodyParser = None
    ParserConfig = None

from ..attachments.appendix_parser import AppendixParser
from ..attachments.classifier import AttachmentKind, attachment_slug, classify_attachment
from ..attachments.common import ensure_dir, read_jsonl, safe_dirname, slugify, write_json, write_jsonl
from ..attachments.form_parser import FormParser
from ..attachments.qcvn_parser import QCVNParser
from ..common.doc_converter import convert_legacy_docs_under, find_legacy_doc_files
from ..common.logging_utils import get_logger
from ..common.utils import strip_vietnamese_accents
from .attachment_linker import infer_appendix_form_links


@dataclass
class PackageParseResult:
    package_id: str
    output_dir: Path
    main_file: Optional[Path]
    attachment_count: int
    all_unit_count: int
    all_table_count: int
    all_ref_mention_count: int
    converted_doc_count: int = 0
    failed_doc_conversion_count: int = 0


ATTACHMENT_PREFIXES = ("phu luc", "phụ lục", "mau", "mẫu", "qcvn", "quy chuan", "quy chuẩn", "dkx")


class LegalPackageParser:
    """
    Parse one dataset folder as a legal package:

      package/
        main docx
        attachments docx...

    Output:
      parsed/<PACKAGE_ID>/
        package_inventory.json
        main/
        attachments/<attachment_slug>/
        all_units.jsonl
        all_tables.jsonl
        all_ref_mentions.jsonl
    """

    def __init__(
        self,
        output_base_dir: Union[str, Path],
        *,
        logger: Optional[logging.Logger] = None,
        convert_doc: bool = True,
        delete_converted_doc: bool = True,
    ):
        self.output_base_dir = Path(output_base_dir).resolve()
        self.logger = get_logger(logger)
        self.convert_doc = convert_doc
        self.delete_converted_doc = delete_converted_doc

    def parse_dataset(self, dataset_dir: Union[str, Path]) -> List[PackageParseResult]:
        dataset_dir = Path(dataset_dir).resolve()
        results = []
        for package_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            try:
                results.append(self.parse_package(package_dir))
            except Exception as e:
                self.logger.error("❌ Lỗi package %s: %s", package_dir, e)
        return results

    def parse_package(self, package_dir: Union[str, Path]) -> PackageParseResult:
        package_dir = Path(package_dir).resolve()
        package_id = package_dir.name
        out_dir = self.output_base_dir / package_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir = ensure_dir(out_dir)

        inventory: Dict[str, Any] = {
            "package_id": package_id,
            "source_dir": str(package_dir),
            "main_document": None,
            "attachments": [],
            "converted_files": [],
            "unsupported_files": [],
        }

        conversion_results = []
        if self.convert_doc:
            conversion_results = convert_legacy_docs_under(
                package_dir,
                recursive=False,
                delete_source=self.delete_converted_doc,
                logger=self.logger,
            )
            for item in conversion_results:
                inventory["converted_files"].append({
                    "source_file": str(item.source_path),
                    "target_file": str(item.target_path),
                    "status": item.status,
                    "converter": item.converter,
                    "message": item.message,
                    "deleted_source": item.deleted_source,
                })

        unsupported_doc_files = find_legacy_doc_files(package_dir, recursive=False)
        for unsupported in unsupported_doc_files:
            inventory["unsupported_files"].append({
                "source_file": str(unsupported),
                "reason": "Legacy .doc files are not parsed by python-docx and conversion did not produce a .docx.",
            })

        docx_files = sorted([p for p in package_dir.glob("*.docx") if not p.name.startswith("~$")])
        main_file, attachments = self._split_main_and_attachments(docx_files)

        self.logger.info(
            "📦 Package %s | main=%s | attachments=%s | unsupported_doc=%s",
            package_id,
            main_file.name if main_file else "null",
            len(attachments),
            len(unsupported_doc_files),
        )

        main_doc_id = None
        main_doc_number = None

        if main_file:
            main_info = self._parse_main(main_file, out_dir / "main")
            inventory["main_document"] = main_info
            main_doc_id = main_info.get("document_id")
            main_doc_number = main_info.get("document_number")

        for att in attachments:
            kind = classify_attachment(att)
            att_slug = attachment_slug(att)
            att_out = out_dir / "attachments" / att_slug
            parser = self._parser_for_kind(kind)
            parsed = parser.parse(
                docx_path=att,
                output_dir=att_out,
                package_id=package_id,
                document_id=main_doc_id,
                document_number=main_doc_number,
                kind=kind,
            )
            inventory["attachments"].append({
                **parsed["attachment"],
                "parsed_dir": str(att_out.relative_to(out_dir)),
                "unit_count": parsed.get("unit_count"),
                "table_count": parsed.get("table_count"),
                "table_row_count": parsed.get("table_row_count"),
                "form_field_count": parsed.get("form_field_count"),
            })

        inventory = infer_appendix_form_links(package_out_dir=out_dir, inventory=inventory, logger=self.logger)
        write_json(out_dir / "package_inventory.json", inventory)
        counts = self._aggregate_outputs(out_dir)

        return PackageParseResult(
            package_id=package_id,
            output_dir=out_dir,
            main_file=main_file,
            attachment_count=len(attachments),
            all_unit_count=counts["units"],
            all_table_count=counts["tables"],
            all_ref_mention_count=counts["ref_mentions"],
            converted_doc_count=sum(1 for item in conversion_results if item.status in {"converted", "already_converted"}),
            failed_doc_conversion_count=sum(1 for item in conversion_results if item.status == "failed"),
        )

    def _split_main_and_attachments(self, files: List[Path]) -> Tuple[Optional[Path], List[Path]]:
        if not files:
            return None, []

        attachments = []
        main_candidates = []
        for f in files:
            name = f.stem.lower().replace("_", " ").replace("-", " ")
            if name.startswith(ATTACHMENT_PREFIXES):
                attachments.append(f)
                continue
            # Do not classify a possible main document by content here. Legal
            # titles often contain "quy chuẩn", which would look like QCVN.
            main_candidates.append(f)

        # Prefer obvious legal main file names.
        if main_candidates:
            priority = []
            for f in main_candidates:
                n = self._normalized_main_candidate_name(f)
                score = 0
                for s in ["luat", "nghi dinh", "thong tu", "quyet dinh", "nghi quyet", "bo luat"]:
                    if s in n:
                        score += 5
                package_tokens = [t for t in re.split(r"[^a-z0-9]+", strip_vietnamese_accents(f.parent.name).lower()) if t]
                name_tokens = set(re.split(r"[^a-z0-9]+", n))
                if package_tokens and all(t in name_tokens for t in package_tokens if t.isdigit()):
                    score += 3
                if f.stat().st_size > 200_000:
                    score += 1
                priority.append((score, len(f.name), f))
            priority.sort(key=lambda x: (-x[0], x[1], str(x[2])))
            main = priority[0][2]
            rest_main = [x[2] for x in priority[1:]]
            # Do not silently discard extra non-attachment files; treat as attachments unknown.
            return main, sorted(attachments + rest_main)

        # If all files are attachments, no main.
        return None, sorted(attachments)

    @staticmethod
    def _normalized_main_candidate_name(path: Path) -> str:
        name = path.stem.lower()
        name = strip_vietnamese_accents(name, keep_dd=False).lower()
        name = re.sub(r"[_\-–—]+", " ", name)
        name = re.sub(r"[^a-z0-9]+", " ", name)
        return re.sub(r"\s+", " ", name).strip()

    def _parse_main(self, main_file: Path, out_dir: Path) -> Dict[str, Any]:
        if LegalBodyParser is None or ParserConfig is None:
            raise RuntimeError("LegalBodyParser/ParserConfig not importable. Keep package_parser inside your legal_parser package.")

        tmp_base = ensure_dir(out_dir / "_tmp")
        parser = LegalBodyParser(ParserConfig(output_base_dir=tmp_base))
        result = parser.parse_file(main_file)

        ensure_dir(out_dir)
        # Copy normalized names into main/.
        mapping = {
            result.tree_path: out_dir / "tree.json",
            result.units_path: out_dir / "units.jsonl",
            result.tables_path: out_dir / "tables.jsonl",
            result.ref_mentions_path: out_dir / "ref_mentions.jsonl",
            result.amendment_mentions_path: out_dir / "amendment_mentions.jsonl",
        }
        for src, dst in mapping.items():
            if src and Path(src).exists():
                shutil.copy2(src, dst)

        # Extract metadata from tree.
        metadata = {}
        tree_path = out_dir / "tree.json"
        if tree_path.exists():
            with open(tree_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
            metadata = tree.get("metadata", {})

        shutil.rmtree(tmp_base, ignore_errors=True)
        return {
            "document_id": metadata.get("document_id") or result.document_id,
            "document_number": metadata.get("document_number"),
            "document_type": metadata.get("document_type"),
            "document_title": metadata.get("document_title"),
            "source_file": str(main_file),
            "parsed_dir": "main",
            "unit_count": result.unit_count,
            "table_count": result.table_count,
            "ref_mention_count": result.ref_mention_count,
            "amendment_mention_count": result.amendment_mention_count,
        }

    def _parser_for_kind(self, kind: AttachmentKind):
        if kind.kind == "qcvn":
            return QCVNParser()
        if kind.kind in {"form", "appendix_form"}:
            return FormParser()
        return AppendixParser()

    def _aggregate_outputs(self, out_dir: Path) -> Dict[str, int]:
        all_units = []
        all_tables = []
        all_refs = []

        # Main.
        for rel in ["main/units.jsonl", "main/tables.jsonl", "main/ref_mentions.jsonl"]:
            p = out_dir / rel
            if not p.exists():
                continue
            if rel.endswith("units.jsonl"):
                for row in read_jsonl(p):
                    row.setdefault("package_id", out_dir.name)
                    row.setdefault("source_type", "main_document")
                    row.setdefault("attachment_id", None)
                    row.setdefault("attachment_type", None)
                    row.setdefault("unit_id", row.get("id"))
                    row.setdefault("unit_type", row.get("type"))
                    all_units.append(row)
            elif rel.endswith("tables.jsonl"):
                for row in read_jsonl(p):
                    row.setdefault("package_id", out_dir.name)
                    row.setdefault("source_type", "main_document")
                    all_tables.append(row)
            elif rel.endswith("ref_mentions.jsonl"):
                for row in read_jsonl(p):
                    row.setdefault("package_id", out_dir.name)
                    row.setdefault("source_type", "main_document")
                    all_refs.append(row)

        # Attachments.
        att_root = out_dir / "attachments"
        if att_root.exists():
            for att_dir in sorted([p for p in att_root.iterdir() if p.is_dir()]):
                for fname, target in [("units.jsonl", all_units), ("tables.jsonl", all_tables), ("ref_mentions.jsonl", all_refs)]:
                    p = att_dir / fname
                    if p.exists():
                        target.extend(list(read_jsonl(p)))

        write_jsonl(out_dir / "all_units.jsonl", all_units)
        write_jsonl(out_dir / "all_tables.jsonl", all_tables)
        write_jsonl(out_dir / "all_ref_mentions.jsonl", all_refs)

        return {"units": len(all_units), "tables": len(all_tables), "ref_mentions": len(all_refs)}
