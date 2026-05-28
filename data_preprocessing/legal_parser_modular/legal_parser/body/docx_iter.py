from __future__ import annotations
from typing import Any, Iterator, Union
import docx

def iter_blocks_recursive(doc: docx.Document) -> Iterator[Union[docx.text.paragraph.Paragraph, docx.table.Table]]:
    def walk(element: Any):
        for child in element:
            if child.tag.endswith("}p"):
                yield docx.text.paragraph.Paragraph(child, doc)
            elif child.tag.endswith("}tbl"):
                yield docx.table.Table(child, doc)
            else:
                yield from walk(child)
    yield from walk(doc.element.body)
