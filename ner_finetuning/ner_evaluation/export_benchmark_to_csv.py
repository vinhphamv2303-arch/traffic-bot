from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_entities(entities: list[dict[str, Any]]) -> str:
    lines = []
    for idx, ent in enumerate(entities, start=1):
        lines.append(
            f"{idx}. [{ent.get('label')}] {ent.get('text')} "
            f"({ent.get('start')}-{ent.get('end')})"
        )
    return "\n".join(lines)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig giup Excel tren Windows doc tieng Viet dung hon.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows = []
    entity_rows = []

    for case_idx, item in enumerate(data, start=1):
        case_id = f"case_{case_idx:03d}"
        text = item.get("text") or ""
        entities = item.get("entities") or []
        label_counts = Counter(ent.get("label") for ent in entities)

        case_rows.append({
            "case_id": case_id,
            "text": text,
            "entity_count": len(entities),
            "labels": ", ".join(f"{label}:{count}" for label, count in sorted(label_counts.items())),
            "entities_text": format_entities(entities),
            "entities_json": json.dumps(entities, ensure_ascii=False),
            "review_status": "",
            "review_note": "",
        })

        for ent_idx, ent in enumerate(entities, start=1):
            entity_rows.append({
                "case_id": case_id,
                "entity_id": f"{case_id}_ent_{ent_idx:02d}",
                "label": ent.get("label"),
                "text": ent.get("text"),
                "start": ent.get("start"),
                "end": ent.get("end"),
                "keep": "",
                "corrected_label": "",
                "corrected_text": "",
                "note": "",
                "sentence_text": text,
            })

    return case_rows, entity_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NER benchmark JSON to review-friendly CSV files.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--cases-output", required=True)
    ap.add_argument("--entities-output", required=True)
    args = ap.parse_args()

    data = read_json(args.input)
    if not isinstance(data, list):
        raise TypeError("Input benchmark must be a JSON list.")

    case_rows, entity_rows = build_rows(data)

    write_csv(args.cases_output, case_rows, [
        "case_id",
        "text",
        "entity_count",
        "labels",
        "entities_text",
        "entities_json",
        "review_status",
        "review_note",
    ])
    write_csv(args.entities_output, entity_rows, [
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
    ])

    print(f"cases csv: {args.cases_output}")
    print(f"entities csv: {args.entities_output}")
    print(f"cases: {len(case_rows)}")
    print(f"entities: {len(entity_rows)}")


if __name__ == "__main__":
    main()
