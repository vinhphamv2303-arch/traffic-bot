from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def clean_xml_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text[:32767]


def cell_xml(row_idx: int, col_idx: int, value: Any, style: int = 0) -> str:
    ref = f"{col_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style else ""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(clean_xml_text(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def sheet_xml(
    rows: list[list[Any]],
    widths: dict[int, float] | None = None,
    freeze_top_row: bool = True,
    wrap_cols: set[int] | None = None,
) -> str:
    widths = widths or {}
    wrap_cols = wrap_cols or set()
    max_col = max((len(r) for r in rows), default=1)
    max_row = max(len(rows), 1)

    cols = []
    for idx in range(1, max_col + 1):
        width = widths.get(idx)
        if width:
            cols.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')
    cols_xml = f"<cols>{''.join(cols)}</cols>" if cols else ""

    views_xml = ""
    if freeze_top_row:
        views_xml = (
            "<sheetViews><sheetView workbookViewId=\"0\">"
            "<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/>"
            "<selection pane=\"bottomLeft\"/>"
            "</sheetView></sheetViews>"
        )

    row_parts = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            if r_idx == 1:
                style = 1
            elif c_idx in wrap_cols:
                style = 2
            else:
                style = 0
            cells.append(cell_xml(r_idx, c_idx, value, style=style))
        row_parts.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    auto_filter = f'<autoFilter ref="A1:{col_name(max_col)}{max_row}"/>' if rows else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{views_xml}{cols_xml}<sheetData>{''.join(row_parts)}</sheetData>{auto_filter}"
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(
            f'<sheet name="{escape(name)}" sheetId="{idx}" '
            f'r:id="rId{idx}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(sheets)}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{sheets}"
        "</Types>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<fonts count=\"2\">"
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        "</fonts>"
        "<fills count=\"2\">"
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"3\">"
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
        "</cellXfs>"
        "</styleSheet>"
    )


def format_entities(entities: list[dict[str, Any]]) -> str:
    lines = []
    for idx, ent in enumerate(entities, start=1):
        lines.append(
            f"{idx}. [{ent.get('label')}] {ent.get('text')} "
            f"({ent.get('start')}-{ent.get('end')})"
        )
    return "\n".join(lines)


def build_rows(data: list[dict[str, Any]]) -> dict[str, list[list[Any]]]:
    case_rows = [[
        "case_id",
        "text",
        "entity_count",
        "labels",
        "entities",
        "review_status",
        "review_note",
    ]]
    entity_rows = [[
        "case_id",
        "entity_id",
        "label",
        "text",
        "start",
        "end",
        "keep",
        "corrected_label",
        "corrected_text",
        "note",
        "sentence_text",
    ]]

    label_counter = Counter()
    for case_idx, item in enumerate(data, start=1):
        case_id = f"case_{case_idx:03d}"
        text = item.get("text") or ""
        entities = item.get("entities") or []
        label_counter.update(ent.get("label") for ent in entities)
        labels = ", ".join(f"{label}:{count}" for label, count in sorted(Counter(ent.get("label") for ent in entities).items()))
        case_rows.append([
            case_id,
            text,
            len(entities),
            labels,
            format_entities(entities),
            "",
            "",
        ])
        for ent_idx, ent in enumerate(entities, start=1):
            entity_rows.append([
                case_id,
                f"{case_id}_ent_{ent_idx:02d}",
                ent.get("label"),
                ent.get("text"),
                ent.get("start"),
                ent.get("end"),
                "",
                "",
                "",
                "",
                text,
            ])

    summary_rows = [
        ["metric", "value"],
        ["case_count", len(data)],
        ["entity_count", sum(len(item.get("entities") or []) for item in data)],
    ]
    for label, count in sorted(label_counter.items()):
        summary_rows.append([f"label_{label}", count])

    return {
        "cases": case_rows,
        "entities": entity_rows,
        "summary": summary_rows,
    }


def write_xlsx(output: str | Path, sheets: dict[str, list[list[Any]]]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheet_names)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheet_names)))
        zf.writestr("xl/styles.xml", styles_xml())

        for idx, name in enumerate(sheet_names, start=1):
            if name == "cases":
                widths = {1: 12, 2: 80, 3: 14, 4: 32, 5: 70, 6: 18, 7: 35}
                wrap_cols = {2, 5, 7}
            elif name == "entities":
                widths = {1: 12, 2: 18, 3: 34, 4: 45, 5: 10, 6: 10, 7: 10, 8: 24, 9: 35, 10: 35, 11: 80}
                wrap_cols = {4, 9, 10, 11}
            else:
                widths = {1: 38, 2: 16}
                wrap_cols = set()
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(sheets[name], widths=widths, wrap_cols=wrap_cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NER benchmark JSON to a simple review XLSX file.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = read_json(args.input)
    if not isinstance(data, list):
        raise TypeError("Input benchmark must be a JSON list.")

    sheets = build_rows(data)
    write_xlsx(args.output, sheets)
    print(f"saved: {args.output}")
    print(f"cases: {len(data)}")
    print(f"entities: {len(sheets['entities']) - 1}")


if __name__ == "__main__":
    main()
