\
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .common import LABELS, ensure_dir, read_jsonl, write_json, write_jsonl
from .gliner_data import row_to_gliner


def convert_dataset(input_jsonl: str | Path, output_jsonl: str | Path) -> Dict[str, Any]:
    rows = []
    skipped = 0
    for r in read_jsonl(input_jsonl):
        gr = row_to_gliner(r)
        if gr is None:
            skipped += 1
            continue
        rows.append(gr)
    write_jsonl(output_jsonl, rows)
    return {"input": str(input_jsonl), "output": str(output_jsonl), "rows": len(rows), "skipped": skipped}


def train_gliner_model(
    train_file: str | Path,
    dev_file: str | Path,
    output_dir: str | Path,
    model_name: str = "urchade/gliner_medium-v2.1",
    num_steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 5e-6,
):
    """
    Thin wrapper. GLiNER APIs have changed across versions, so this function provides
    the expected flow and fails with actionable error if installed API differs.
    """
    output_dir = ensure_dir(output_dir)

    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("Install GLiNER first: pip install gliner") from e

    # Convert to GLiNER jsonl format.
    train_gliner = output_dir / "train_gliner.jsonl"
    dev_gliner = output_dir / "dev_gliner.jsonl"
    train_meta = convert_dataset(train_file, train_gliner)
    dev_meta = convert_dataset(dev_file, dev_gliner)

    # Most GLiNER releases support finetuning through model.train_model or Trainer utilities,
    # but APIs vary. We provide a robust fallback: save converted data + config.
    config = {
        "model_name": model_name,
        "labels": LABELS,
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "train_gliner": str(train_gliner),
        "dev_gliner": str(dev_gliner),
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "note": "If your installed gliner version exposes a training CLI, use train_gliner.jsonl/dev_gliner.jsonl. This module also supports zero-shot/fine-tuned inference through predict_gliner.py.",
        "train_meta": train_meta,
        "dev_meta": dev_meta,
    }
    write_json(output_dir / "gliner_training_config.json", config)

    model = GLiNER.from_pretrained(model_name)

    if hasattr(model, "train_model"):
        model.train_model(
            str(train_gliner),
            str(output_dir),
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        model.save_pretrained(str(output_dir / "final_model"))
        config["final_model"] = str(output_dir / "final_model")
        write_json(output_dir / "gliner_training_config.json", config)
        return config

    # API-compatible fallback: save pretrained model for zero-shot/local inference.
    model.save_pretrained(str(output_dir / "base_model_copy"))
    config["warning"] = "Installed GLiNER package does not expose train_model. Converted dataset was created; use your GLiNER version's training CLI/API with train_gliner.jsonl."
    write_json(output_dir / "gliner_training_config.json", config)
    return config
