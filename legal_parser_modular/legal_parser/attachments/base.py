from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import docx

from .classifier import AttachmentKind, attachment_slug, classify_attachment
from .common import (
    collapse_ws,
    ensure_dir,
    extract_attachment_header,
    get_docx_block_texts,
    extract_ref_mentions_light,
    get_docx_texts,
    get_mammoth_html_tables,
    make_text_for_embedding,
    normalize_html_table,
    read_jsonl,
    safe_dirname,
    slugify,
    write_json,
    write_jsonl,
)


class AttachmentParserBase:
    parser_name = "base"
    non_reference_unit_types = {
        "attachment_summary",
        "form_summary",
        "embedded_form_title",
    }

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
        raise NotImplementedError

    def build_metadata(
        self,
        *,
        docx_path: Union[str, Path],
        package_id: str,
        document_id: Optional[str],
        document_number: Optional[str],
        kind: Optional[AttachmentKind],
    ) -> Dict[str, Any]:
        docx_path = Path(docx_path)
        kind = kind or classify_attachment(docx_path)
        if kind.kind == "qcvn":
            texts = get_docx_block_texts(docx_path, limit=100)
        else:
            texts = get_docx_texts(docx_path, limit=80)
        header = extract_attachment_header(texts, docx_path)

        att_slug = attachment_slug(docx_path)
        attachment_id = f"{document_id or slugify(package_id)}.{slugify(att_slug)}"

        return {
            "package_id": package_id,
            "document_id": document_id,
            "document_number": document_number,
            "attachment_id": attachment_id,
            "attachment_slug": att_slug,
            "attachment_kind": kind.kind,
            "classifier_confidence": kind.confidence,
            "classifier_reason": kind.reason,
            "label": header.get("label"),
            "title": header.get("title"),
            "issued_with": header.get("issued_with"),
            "source_file": str(docx_path),
            "parser": self.parser_name,
        }

    def make_unit(
        self,
        *,
        metadata: Dict[str, Any],
        unit_type: str,
        local_id: str,
        path_parts: List[str],
        content: str,
        parent_id: Optional[str] = None,
        order: int = 0,
        structured_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        unit_id = f"{metadata['attachment_id']}.{local_id}"
        path_text = " > ".join([p for p in path_parts if p])
        ref_mentions = self.extract_ref_mentions_for_unit(metadata, unit_type, content)
        row = {
            "unit_id": unit_id,
            "id": unit_id,
            "package_id": metadata.get("package_id"),
            "document_id": metadata.get("document_id"),
            "document_number": metadata.get("document_number"),
            "source_type": "attachment",
            "attachment_id": metadata.get("attachment_id"),
            "attachment_type": metadata.get("attachment_kind"),
            "unit_type": unit_type,
            "type": unit_type,
            "parent_id": parent_id,
            "order": order,
            "path_text": path_text,
            "content": content,
            "structured_fields": structured_fields or {},
            "ref_mentions": ref_mentions,
            "text_for_embedding": make_text_for_embedding(
                title=metadata.get("title") or "",
                path_text=path_text,
                content=content,
            ),
            "source_file": metadata.get("source_file"),
        }
        return row

    def extract_ref_mentions_for_unit(self, metadata: Dict[str, Any], unit_type: str, content: str) -> List[Dict[str, Any]]:
        if unit_type in self.non_reference_unit_types:
            return []
        if self._is_attachment_self_label_text(metadata, content):
            return []
        return extract_ref_mentions_light(content)

    @staticmethod
    def _is_attachment_self_label_text(metadata: Dict[str, Any], content: str) -> bool:
        text = collapse_ws(content).strip(" .,:;")
        if not text:
            return False

        text_key = slugify(text)
        label = collapse_ws(metadata.get("label") or "").strip(" .,:;")
        title = collapse_ws(metadata.get("title") or "").strip(" .,:;")
        label_key = slugify(label) if label else ""
        title_key = slugify(title) if title else ""

        if label_key and text_key == label_key:
            return True
        if title_key and text_key == title_key:
            return True
        if label and title and text_key == slugify(f"{label}. {title}"):
            return True
        return False

    def extract_tables(self, metadata: Dict[str, Any], docx_path: Union[str, Path]) -> List[Dict[str, Any]]:
        tables = []
        html_tables = get_mammoth_html_tables(docx_path)
        for idx, html in enumerate(html_tables, start=1):
            rows = normalize_html_table(html)
            table_id = f"{metadata['attachment_id']}.table_{idx}"
            tables.append({
                "table_id": table_id,
                "package_id": metadata.get("package_id"),
                "document_id": metadata.get("document_id"),
                "attachment_id": metadata.get("attachment_id"),
                "attachment_type": metadata.get("attachment_kind"),
                "order": idx,
                "html": html,
                "normalized_rows": rows,
                "row_count": len(rows),
                "col_count": max([len(r) for r in rows], default=0),
                "source_file": metadata.get("source_file"),
            })
        return tables

    def write_outputs(
        self,
        *,
        output_dir: Union[str, Path],
        metadata: Dict[str, Any],
        units: List[Dict[str, Any]],
        tables: Optional[List[Dict[str, Any]]] = None,
        table_rows: Optional[List[Dict[str, Any]]] = None,
        form_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        out = ensure_dir(output_dir)
        tables = tables or []
        table_rows = table_rows or []
        form_fields = form_fields or []

        ref_mentions_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for unit in units:
            for ref in unit.get("ref_mentions", []) or []:
                source_unit_id = unit.get("unit_id") or unit.get("id")
                row = {
                    "source_unit_id": unit.get("unit_id") or unit.get("id"),
                    "package_id": unit.get("package_id"),
                    "document_id": unit.get("document_id"),
                    "attachment_id": unit.get("attachment_id"),
                    "source_path_text": unit.get("path_text"),
                    "source_text": unit.get("content"),
                    **ref,
                }
                key = self._ref_mention_key(row)
                if key not in ref_mentions_by_key:
                    row["source_unit_ids"] = [source_unit_id] if source_unit_id else []
                    row["source_path_texts"] = [unit.get("path_text")] if unit.get("path_text") else []
                    row["raw_values"] = [row.get("raw")] if row.get("raw") else []
                    row["occurrence_count"] = 1
                    ref_mentions_by_key[key] = row
                    continue

                existing = ref_mentions_by_key[key]
                if source_unit_id and source_unit_id not in existing["source_unit_ids"]:
                    existing["source_unit_ids"].append(source_unit_id)
                path_text = unit.get("path_text")
                if path_text and path_text not in existing["source_path_texts"]:
                    existing["source_path_texts"].append(path_text)
                raw = row.get("raw")
                if raw and raw not in existing["raw_values"]:
                    existing["raw_values"].append(raw)
                existing["occurrence_count"] += 1

        ref_mentions = list(ref_mentions_by_key.values())

        write_json(out / "attachment.json", metadata)
        write_jsonl(out / "units.jsonl", units)
        write_jsonl(out / "tables.jsonl", tables)
        write_jsonl(out / "table_rows.jsonl", table_rows)
        write_jsonl(out / "form_fields.jsonl", form_fields)
        write_jsonl(out / "ref_mentions.jsonl", ref_mentions)

        return {
            "attachment": metadata,
            "unit_count": len(units),
            "table_count": len(tables),
            "table_row_count": len(table_rows),
            "form_field_count": len(form_fields),
            "ref_mention_count": len(ref_mentions),
            "output_dir": str(out),
        }

    @staticmethod
    def _ref_mention_key(ref: Dict[str, Any]) -> Tuple[Any, ...]:
        label = AttachmentParserBase._canonical_ref_value(ref.get("label"))
        raw = AttachmentParserBase._canonical_ref_value(ref.get("raw"))
        return (
            ref.get("attachment_id"),
            ref.get("mention_type"),
            label or raw,
        )

    @staticmethod
    def _canonical_ref_value(value: Any) -> str:
        text = collapse_ws(str(value or "")).strip(".,;:)")
        return text.casefold()

    def table_rows_to_units(
        self,
        *,
        metadata: Dict[str, Any],
        tables: List[Dict[str, Any]],
        path_prefix: List[str],
        start_order: int = 100000,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        units = []
        table_rows = []
        order = start_order

        for table in tables:
            rows = table.get("normalized_rows") or []
            if not rows:
                continue

            # Very simple row-level representation.
            # Header is first non-empty row.
            header = None
            for r in rows:
                if any(c.strip() for c in r):
                    header = r
                    break
            for ridx, row in enumerate(rows, start=1):
                if not any(c.strip() for c in row):
                    continue
                content = " | ".join([c for c in row if c.strip()])
                if len(content) < 3:
                    continue

                local_id = f"{table['table_id'].split('.')[-1]}.row_{ridx}"
                fields = {"cells": row, "header": header, "table_id": table["table_id"], "row_index": ridx}

                unit = self.make_unit(
                    metadata=metadata,
                    unit_type="table_row",
                    local_id=local_id,
                    path_parts=path_prefix + [f"Bảng {table.get('order')}", f"Dòng {ridx}"],
                    content=content,
                    parent_id=table["table_id"],
                    order=order,
                    structured_fields=fields,
                )
                order += 1
                units.append(unit)
                table_rows.append({
                    "row_id": unit["unit_id"],
                    "table_id": table["table_id"],
                    "package_id": metadata.get("package_id"),
                    "document_id": metadata.get("document_id"),
                    "attachment_id": metadata.get("attachment_id"),
                    "row_index": ridx,
                    "cells": row,
                    "content": content,
                    "text_for_embedding": unit["text_for_embedding"],
                })

        return units, table_rows
