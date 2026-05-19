from __future__ import annotations
import inspect
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
from .config import TrainConfig
from .dataset import TokenDataset, encode_rows
from .io_utils import balance_negatives, clean_row, ensure_dir, label_summary, read_jsonl, split_train_dev, write_json, write_jsonl
from .metrics import make_compute_metrics
from .schema import LABELS, bio_labels

def make_training_args(config):
    kwargs = dict(
        output_dir=str(config.output_dir), learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit, save_strategy="epoch", report_to="none",
        seed=config.seed, fp16=config.fp16, bf16=config.bf16,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch" if config.eval_ratio > 0 else "no"
    else:
        kwargs["evaluation_strategy"] = "epoch" if config.eval_ratio > 0 else "no"
    return TrainingArguments(**kwargs)

def load_rows(path, auto_clean):
    rows = []
    for r in read_jsonl(path):
        if r.get("text"): rows.append(clean_row(r, auto_clean=auto_clean))
    return rows

def train(config: TrainConfig):
    out_dir = ensure_dir(config.output_dir)
    rows = load_rows(config.train_file, auto_clean=config.auto_clean)
    rows = balance_negatives(rows, config.negative_ratio, config.seed)
    labels = bio_labels(LABELS)
    label2id = {lab:i for i,lab in enumerate(labels)}
    id2label = {i:lab for lab,i in label2id.items()}
    train_rows, dev_rows = split_train_dev(rows, config.eval_ratio, config.seed)
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if not tok.is_fast: raise RuntimeError("Fast tokenizer required for offset_mapping. Use XLM-RoBERTa or another fast-tokenizer model.")
    tr_enc, tr_labels = encode_rows(train_rows, tok, label2id, config.max_length, config.text_field)
    train_ds = TokenDataset(tr_enc, tr_labels, train_rows)
    dev_ds = None
    if dev_rows:
        dev_enc, dev_labels = encode_rows(dev_rows, tok, label2id, config.max_length, config.text_field)
        dev_ds = TokenDataset(dev_enc, dev_labels, dev_rows)
    model = AutoModelForTokenClassification.from_pretrained(config.model_name_or_path, num_labels=len(labels), id2label=id2label, label2id=label2id)
    args = make_training_args(config)
    collator = DataCollatorForTokenClassification(tokenizer=tok)
    trainer_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator, compute_metrics=make_compute_metrics(id2label) if dev_ds else None)
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters: trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in sig.parameters: trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir)); tok.save_pretrained(str(final_dir))
    meta = {"model_name_or_path": config.model_name_or_path, "train_file": str(config.train_file), "output_dir": str(config.output_dir), "final_model_dir": str(final_dir), "labels": labels, "label2id": label2id, "id2label": id2label, "row_count": len(rows), "train_count": len(train_rows), "dev_count": len(dev_rows), "entity_count": sum(len(r.get("entities") or []) for r in rows), "by_label": label_summary(rows), "max_length": config.max_length, "auto_clean": config.auto_clean, "negative_ratio": config.negative_ratio}
    write_json(out_dir / "training_metadata.json", meta)
    write_jsonl(out_dir / "train_rows.jsonl", train_rows); write_jsonl(out_dir / "dev_rows.jsonl", dev_rows)
    return meta
