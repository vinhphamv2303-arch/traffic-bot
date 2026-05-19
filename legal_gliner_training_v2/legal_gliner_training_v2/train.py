\
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .common import LABELS, ensure_dir, read_json, write_json


def train_gliner(
    train_file: str,
    dev_file: str | None,
    output_dir: str,
    base_model: str = "urchade/gliner_medium-v2.1",
    steps: int = 3000,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    learning_rate: float = 5e-6,
    others_learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    logging_steps: int = 50,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    bf16: bool = True,
    use_minimal_data: bool = False,
) -> Dict[str, Any]:
    """
    Fine-tune GLiNER using its current public train_model API.

    This follows the official repository pattern:
    GLiNER.from_pretrained(...).train_model(train_dataset=[...], eval_dataset=[...], ...)
    The repo README states GLiNER is fine-tunable and optimized for CPU/consumer hardware,
    while train.py in the current repo uses model.train_model.
    """
    ensure_dir(output_dir)

    try:
        import torch
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("Install dependencies first: pip install gliner torch transformers accelerate") from e

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")

    train_dataset = read_json(train_file)
    eval_dataset = read_json(dev_file) if dev_file else None

    print(f"[gliner:train] base_model={base_model}")
    print(f"[gliner:train] train_samples={len(train_dataset)} eval_samples={len(eval_dataset) if eval_dataset else 0}")
    print(f"[gliner:train] device={device} steps={steps} batch={train_batch_size}")

    model = GLiNER.from_pretrained(base_model)
    model = model.to(device)

    # GLiNER train_model signature differs slightly by version.
    # We try the current official signature first, then a smaller fallback.
    kwargs = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        max_steps=steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        others_lr=others_learning_rate,
        weight_decay=weight_decay,
        others_weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type="linear",
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        negatives=1.5,
        masking="global",
        loss_reduction="sum",
        bf16=bf16,
    )

    try:
        model.train_model(**kwargs)
    except TypeError as e:
        print("[gliner:train] train_model signature mismatch, retrying fallback args.")
        fallback = dict(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            max_steps=steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
        )
        model.train_model(**fallback)

    final_dir = Path(output_dir) / "final_model"
    model.save_pretrained(str(final_dir))

    metadata = {
        "base_model": base_model,
        "output_dir": output_dir,
        "final_model": str(final_dir),
        "train_file": train_file,
        "dev_file": dev_file,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "steps": steps,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "others_learning_rate": others_learning_rate,
        "device": device,
        "bf16": bf16,
        "labels": LABELS,
    }
    write_json(Path(output_dir) / "training_metadata.json", metadata)
    return metadata


def main():
    ap = argparse.ArgumentParser(description="Fine-tune GLiNER for Vietnamese traffic-law NER.")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default="urchade/gliner_medium-v2.1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--others-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    meta = train_gliner(
        train_file=args.train_file,
        dev_file=args.dev_file,
        output_dir=args.output_dir,
        base_model=args.base_model,
        steps=args.steps,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        others_learning_rate=args.others_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        device=args.device,
        bf16=not args.no_bf16,
    )
    print("GLiNER fine-tuning completed")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
