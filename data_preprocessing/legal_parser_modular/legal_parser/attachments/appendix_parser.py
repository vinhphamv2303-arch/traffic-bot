from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import docx

from .base import AttachmentParserBase
from .classifier import AttachmentKind, classify_attachment
from .common import collapse_ws, iter_docx_blocks, normalize_qcvn_code, slugify, strip_vietnamese_accents


class AppendixParser(AttachmentParserBase):
    parser_name = "appendix_parser"

    # Structured appendix headings.
    pat_alpha = re.compile(r"^([A-ZĐ])\.\s+(.+)")
    pat_roman = re.compile(r"^([IVXLCDM]+)\.\s+(.+)", re.I)
    pat_decimal = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)")
    pat_point = re.compile(r"^([a-zđ])\)\s*(.+)", re.I)
    pat_bullet = re.compile(r"^[-–•]\s+(.+)")

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
        order = 0
        stack: Dict[int, Dict[str, Any]] = {}
        path_stack: Dict[int, str] = {}
        prefix = [metadata.get("label") or "Phụ lục", metadata.get("title") or ""]
        in_qcvn_toc = False

        # Add top summary unit.
        units.append(self.make_unit(
            metadata=metadata,
            unit_type="attachment_summary",
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

                if kind.kind == "qcvn":
                    if self._is_qcvn_toc_heading(text):
                        in_qcvn_toc = True
                        continue
                    if in_qcvn_toc:
                        if self._is_qcvn_toc_end(text, metadata):
                            in_qcvn_toc = False
                        else:
                            continue

                # Skip boilerplate header already represented in metadata.
                if self._is_header_boilerplate(text, metadata):
                    continue

                level, no, title, unit_type = self._classify_line(text)
                if level is None:
                    # Continuation text: attach as raw paragraph under current deepest path.
                    parent = self._deepest_parent(stack)
                    local_id = f"p_{order}"
                    path_parts = prefix + self._current_path_parts(path_stack) + [f"Đoạn {order}"]
                    units.append(self.make_unit(
                        metadata=metadata,
                        unit_type="appendix_paragraph",
                        local_id=local_id,
                        path_parts=path_parts,
                        content=text,
                        parent_id=parent.get("unit_id") if parent else None,
                        order=order,
                    ))
                    order += 1
                    continue

                # Reset deeper levels.
                for k in list(stack):
                    if k >= level:
                        stack.pop(k, None)
                        path_stack.pop(k, None)

                parent = self._parent_for_level(stack, level)
                slug = slugify(no)
                local_id = f"{unit_type}_{slug}"
                if any(u["unit_id"].endswith("." + local_id) for u in units):
                    local_id = f"{local_id}_{order}"

                path_label = self._label_for(unit_type, no, title)
                path_parts = prefix + self._path_parts_before(path_stack, level) + [path_label]
                unit = self.make_unit(
                    metadata=metadata,
                    unit_type=unit_type,
                    local_id=local_id,
                    path_parts=path_parts,
                    content=title,
                    parent_id=parent.get("unit_id") if parent else None,
                    order=order,
                    structured_fields={"no": no},
                )
                units.append(unit)
                stack[level] = unit
                path_stack[level] = path_label
                order += 1

            elif isinstance(block, docx.table.Table):
                table_counter += 1
                # Table content already extracted globally; create a table placeholder unit
                if table_counter <= len(tables):
                    table = tables[table_counter - 1]
                    parent = self._deepest_parent(stack)
                    path_parts = prefix + self._current_path_parts(path_stack) + [f"Bảng {table_counter}"]
                    preview_rows = table.get("normalized_rows") or []
                    preview = "\n".join([" | ".join([c for c in r if c.strip()]) for r in preview_rows[:6]])
                    units.append(self.make_unit(
                        metadata=metadata,
                        unit_type="appendix_table",
                        local_id=f"table_{table_counter}",
                        path_parts=path_parts,
                        content=preview,
                        parent_id=parent.get("unit_id") if parent else None,
                        order=order,
                        structured_fields={"table_id": table.get("table_id")},
                    ))
                    order += 1

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
            form_fields=[],
        )

    def _is_header_boilerplate(self, text: str, metadata: Dict[str, Any]) -> bool:
        low = text.lower()
        if (
            metadata.get("attachment_kind") == "qcvn"
            and normalize_qcvn_code(text) == metadata.get("label")
            and self._is_qcvn_code_only(text)
        ):
            return True
        if metadata.get("label") and low.startswith(metadata["label"].lower()):
            if metadata.get("attachment_kind") == "qcvn":
                return self._is_qcvn_code_only(text)
            return True
        title = (metadata.get("title") or "").lower()
        if title and (low == title or (len(low) >= 8 and low in title)):
            return True
        if re.match(r"^ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", low, re.U):
            return True
        if low.startswith("của "):
            return True
        if "cộng hòa xã hội" in low or "độc lập" in low:
            return True
        if metadata.get("attachment_kind") == "qcvn":
            return "___" in low
        return any(s in low for s in ["ban hành kèm theo", "kèm theo", "của bộ trưởng", "___"])

    def _classify_line(self, text: str):
        m = self.pat_alpha.match(text)
        if m and not self._looks_like_roman_heading(m.group(1)):
            return 1, m.group(1), collapse_ws(m.group(2)), "appendix_section_alpha"

        m = self.pat_roman.match(text)
        if m and len(m.group(1)) <= 8 and self._looks_like_roman_heading(m.group(1)):
            return 2, m.group(1).upper(), collapse_ws(m.group(2)), "appendix_section_roman"

        m = self.pat_decimal.match(text)
        if m:
            no = m.group(1)
            depth = no.count(".")
            return 3 + depth, no, collapse_ws(m.group(2)), "appendix_item_decimal"

        m = self.pat_point.match(text)
        if m:
            return 8, m.group(1), collapse_ws(m.group(2)), "appendix_point"

        m = self.pat_bullet.match(text)
        if m:
            return 9, "-", collapse_ws(m.group(1)), "appendix_bullet"

        return None, None, None, None

    @staticmethod
    def _looks_like_roman_heading(value: str) -> bool:
        token = (value or "").upper()
        if not re.fullmatch(r"[IVXLCDM]+", token):
            return False
        # Single C/D/L/M headings in appendices are usually alphabetic sections.
        return len(token) > 1 or token in {"I", "V", "X"}

    @staticmethod
    def _is_qcvn_code_only(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*QCVN\s*\d+[A-Z]?\s*:\s*\d{4}\s*/\s*[A-ZĐ]+\s*",
            text or "",
            re.I | re.U,
        ))

    @staticmethod
    def _is_qcvn_toc_heading(text: str) -> bool:
        ascii_text = strip_vietnamese_accents(text, keep_dd=False).lower()
        return bool(re.fullmatch(r"muc\s+luc", ascii_text.strip()))

    @staticmethod
    def _is_qcvn_toc_end(text: str, metadata: Dict[str, Any]) -> bool:
        ascii_text = strip_vietnamese_accents(text, keep_dd=False).lower()
        if "quy chuan ky thuat quoc gia" in ascii_text:
            return True
        if normalize_qcvn_code(text) == metadata.get("label"):
            return True
        if ascii_text == "loi noi dau":
            return True
        return False

    @staticmethod
    def _label_for(unit_type: str, no: str, title: str) -> str:
        if unit_type == "appendix_section_alpha":
            return f"{no}. {title}"
        if unit_type == "appendix_section_roman":
            return f"{no}. {title}"
        if unit_type == "appendix_item_decimal":
            return f"{no}. {title}"
        if unit_type == "appendix_point":
            return f"{no}) {title}"
        if unit_type == "appendix_bullet":
            return f"- {title}"
        return title

    @staticmethod
    def _parent_for_level(stack: Dict[int, Dict[str, Any]], level: int) -> Optional[Dict[str, Any]]:
        lower = [k for k in stack if k < level]
        if not lower:
            return None
        return stack[max(lower)]

    @staticmethod
    def _deepest_parent(stack: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not stack:
            return None
        return stack[max(stack)]

    @staticmethod
    def _path_parts_before(path_stack: Dict[int, str], level: int) -> List[str]:
        return [path_stack[k] for k in sorted(path_stack) if k < level]

    @staticmethod
    def _current_path_parts(path_stack: Dict[int, str]) -> List[str]:
        return [path_stack[k] for k in sorted(path_stack)]
