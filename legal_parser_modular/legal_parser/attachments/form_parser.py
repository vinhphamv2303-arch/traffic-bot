from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import docx

from .base import AttachmentParserBase
from .classifier import AttachmentKind, classify_attachment
from .common import (
    collapse_ws,
    field_label_from_line,
    is_probable_field_line,
    iter_docx_blocks,
    slugify,
)


class FormParser(AttachmentParserBase):
    parser_name = "form_parser"

    pat_form_no = re.compile(r"^(Mẫu|Mau)\s+(?:số\s+|so\s+)?(?P<no>[0-9A-Za-zĐđ_.-]+)\.?\s*(?P<title>.*)", re.I | re.U)
    pat_part = re.compile(r"^Phần\s+(?P<no>[IVXLCDM]+|\d+)\s*(?P<title>.*)", re.I | re.U)

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
        docx_path = Path(docx_path)
        kind = kind or classify_attachment(docx_path)
        metadata = self.build_metadata(
            docx_path=docx_path,
            package_id=package_id,
            document_id=document_id,
            document_number=document_number,
            kind=kind,
        )

        doc = docx.Document(str(docx_path))
        tables = self.extract_tables(metadata, docx_path)

        units: List[Dict[str, Any]] = []
        form_fields: List[Dict[str, Any]] = []
        order = 0
        current_section = metadata.get("title") or metadata.get("label") or "Biểu mẫu"
        prefix = [metadata.get("label") or "Biểu mẫu", metadata.get("title") or ""]

        units.append(self.make_unit(
            metadata=metadata,
            unit_type="form_summary",
            local_id="summary",
            path_parts=prefix,
            content=f"{metadata.get('label') or ''}. {metadata.get('title') or ''}".strip(),
            order=order,
        ))
        order += 1

        table_counter = 0
        for block in iter_docx_blocks(doc):
            if isinstance(block, docx.text.paragraph.Paragraph):
                text = collapse_ws(block.text)
                if not text:
                    continue
                if self._is_header_boilerplate(text):
                    continue

                m_form = self.pat_form_no.match(text)
                if m_form and re.search(r"\d", m_form.group("no") or ""):
                    current_section = collapse_ws(text)
                    units.append(self.make_unit(
                        metadata=metadata,
                        unit_type="embedded_form_title",
                        local_id=f"form_title_{order}",
                        path_parts=prefix + [current_section],
                        content=text,
                        order=order,
                    ))
                    order += 1
                    continue

                m_part = self.pat_part.match(text)
                if m_part:
                    current_section = collapse_ws(text)
                    units.append(self.make_unit(
                        metadata=metadata,
                        unit_type="form_section",
                        local_id=f"section_{slugify(m_part.group('no'))}_{order}",
                        path_parts=prefix + [current_section],
                        content=text,
                        order=order,
                    ))
                    order += 1
                    continue

                if is_probable_field_line(text):
                    order = self._append_field_unit(
                        metadata=metadata,
                        units=units,
                        form_fields=form_fields,
                        prefix=prefix,
                        current_section=current_section,
                        text=text,
                        order=order,
                    )
                else:
                    units.append(self.make_unit(
                        metadata=metadata,
                        unit_type="form_text",
                        local_id=f"text_{order}",
                        path_parts=prefix + [current_section, f"Đoạn {order}"],
                        content=text,
                        order=order,
                    ))
                    order += 1

            elif isinstance(block, docx.table.Table):
                table_counter += 1
                if table_counter <= len(tables):
                    table = tables[table_counter - 1]
                    preview_rows = table.get("normalized_rows") or []
                    preview = "\n".join([" | ".join([c for c in r if c.strip()]) for r in preview_rows[:8]])
                    table_unit = self.make_unit(
                        metadata=metadata,
                        unit_type="form_table",
                        local_id=f"table_{table_counter}",
                        path_parts=prefix + [current_section, f"Bảng {table_counter}"],
                        content=preview,
                        order=order,
                        structured_fields={"table_id": table.get("table_id")},
                    )
                    units.append(table_unit)
                    order += 1
                    order = self._append_table_field_units(
                        metadata=metadata,
                        units=units,
                        form_fields=form_fields,
                        prefix=prefix,
                        current_section=current_section,
                        table=table,
                        table_order=table_counter,
                        parent_id=table_unit["unit_id"],
                        order=order,
                    )

        while table_counter < len(tables):
            table_counter += 1
            table = tables[table_counter - 1]
            preview_rows = table.get("normalized_rows") or []
            preview = "\n".join([" | ".join([c for c in r if c.strip()]) for r in preview_rows[:8]])
            table_unit = self.make_unit(
                metadata=metadata,
                unit_type="form_table",
                local_id=f"table_{table_counter}",
                path_parts=prefix + [current_section, f"Bảng {table_counter}"],
                content=preview,
                order=order,
                structured_fields={"table_id": table.get("table_id"), "generated_placeholder": True},
            )
            units.append(table_unit)
            order += 1
            order = self._append_table_field_units(
                metadata=metadata,
                units=units,
                form_fields=form_fields,
                prefix=prefix,
                current_section=current_section,
                table=table,
                table_order=table_counter,
                parent_id=table_unit["unit_id"],
                order=order,
            )

        row_units, table_rows = self.table_rows_to_units(
            metadata=metadata,
            tables=tables,
            path_prefix=prefix,
            start_order=order,
        )
        units.extend(row_units)

        return self.write_outputs(
            output_dir=output_dir,
            metadata=metadata,
            units=units,
            tables=tables,
            table_rows=table_rows,
            form_fields=form_fields,
        )

    @staticmethod
    def _is_header_boilerplate(text: str) -> bool:
        low = text.lower()
        return any(x in low for x in ["ban hành kèm theo", "của bộ trưởng", "________________"])

    def _append_field_unit(
        self,
        *,
        metadata: Dict[str, Any],
        units: List[Dict[str, Any]],
        form_fields: List[Dict[str, Any]],
        prefix: List[str],
        current_section: str,
        text: str,
        order: int,
        parent_id: Optional[str] = None,
        source_location: Optional[Dict[str, Any]] = None,
    ) -> int:
        label = field_label_from_line(text) or text
        if self._is_field_label_noise(label):
            return order
        fields = {"label": label, "section": current_section}
        if source_location:
            fields.update(source_location)
        field_id = f"field_{slugify(label)}_{order}"
        unit = self.make_unit(
            metadata=metadata,
            unit_type="form_field",
            local_id=field_id,
            path_parts=prefix + [current_section, label],
            content=text,
            parent_id=parent_id,
            order=order,
            structured_fields=fields,
        )
        units.append(unit)
        row = {
            "field_id": unit["unit_id"],
            "package_id": metadata.get("package_id"),
            "document_id": metadata.get("document_id"),
            "attachment_id": metadata.get("attachment_id"),
            "label": label,
            "section": current_section,
            "raw_text": text,
            "source_file": metadata.get("source_file"),
        }
        if source_location:
            row.update(source_location)
        form_fields.append(row)
        return order + 1

    def _append_table_field_units(
        self,
        *,
        metadata: Dict[str, Any],
        units: List[Dict[str, Any]],
        form_fields: List[Dict[str, Any]],
        prefix: List[str],
        current_section: str,
        table: Dict[str, Any],
        table_order: int,
        parent_id: str,
        order: int,
    ) -> int:
        for row_idx, row in enumerate(table.get("normalized_rows") or [], start=1):
            for col_idx, cell in enumerate(row, start=1):
                text = collapse_ws(cell)
                if not text or not is_probable_field_line(text):
                    continue
                order = self._append_field_unit(
                    metadata=metadata,
                    units=units,
                    form_fields=form_fields,
                    prefix=prefix + [f"Bảng {table_order}"],
                    current_section=current_section,
                    text=text,
                    parent_id=parent_id,
                    order=order,
                    source_location={
                        "table_id": table.get("table_id"),
                        "table_order": table_order,
                        "row_index": row_idx,
                        "col_index": col_idx,
                    },
                )
        return order

    @staticmethod
    def _is_field_label_noise(label: str) -> bool:
        low = collapse_ws(label).lower()
        low_ascii = re.sub(r"\s+", " ", low)
        if not low or re.fullmatch(r"[.\s…;:_-]+", low):
            return True
        return any(x in low_ascii for x in ["cộng hòa xã hội", "độc lập", "ban hành kèm theo"])
