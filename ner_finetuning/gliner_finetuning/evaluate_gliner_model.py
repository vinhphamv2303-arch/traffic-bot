
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gliner_finetuning_core.common import LABELS, read_json, write_json


def span_key_from_gold(item: Dict[str, Any]) -> set[tuple[int, int, str]]:
    """
    Gold format:
    {
      "tokenized_text": [...],
      "ner": [[start_token, end_token_inclusive, label], ...]
    }
    """
    out = set()
    for s, e, lab in item.get("ner") or []:
        out.add((int(s), int(e), str(lab)))
    return out


def char_offsets_to_token_span(text: str, tokenized_text: list[str], start: int, end: int) -> tuple[int, int] | None:
    """
    Reconstruct token offsets from tokenized_text by searching sequentially in text.
    GLiNER prediction returns char offsets; gold is token span.
    """
    offsets = []
    pos = 0
    for tok in tokenized_text:
        idx = text.find(tok, pos)
        if idx < 0:
            # fallback: approximate by whitespace tokenization
            return None
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)

    idxs = []
    for i, (s, e) in enumerate(offsets):
        if e <= start or s >= end:
            continue
        idxs.append(i)
    if not idxs:
        return None
    return idxs[0], idxs[-1]


def micro_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_gliner(
    model_dir: str,
    test_file: str,
    output_file: str,
    threshold: float = 0.35,
    device: str = "cuda",
    batch_size: int = 16,
) -> Dict[str, Any]:
    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("Install GLiNER first: pip install gliner") from e

    data = read_json(test_file)
    model = GLiNER.from_pretrained(model_dir)
    model = model.to(device)

    total_gold = 0
    total_pred = 0
    total_tp = 0

    by_label = {lab: Counter({"tp": 0, "fp": 0, "fn": 0}) for lab in LABELS}
    examples = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        texts = [" ".join(item["tokenized_text"]) for item in batch]

        try:
            pred_batch = model.batch_predict_entities(texts, LABELS, threshold=threshold)
        except Exception:
            pred_batch = [model.predict_entities(t, LABELS, threshold=threshold) for t in texts]

        for item, text, preds in zip(batch, texts, pred_batch):
            gold = span_key_from_gold(item)
            pred = set()

            for p in preds:
                span = char_offsets_to_token_span(
                    text,
                    item["tokenized_text"],
                    int(p.get("start", 0)),
                    int(p.get("end", 0)),
                )
                if span is None:
                    continue
                lab = p.get("label")
                if lab not in LABELS:
                    continue
                pred.add((span[0], span[1], lab))

            tp_set = gold & pred
            fp_set = pred - gold
            fn_set = gold - pred

            total_gold += len(gold)
            total_pred += len(pred)
            total_tp += len(tp_set)

            for _, _, lab in tp_set:
                by_label[lab]["tp"] += 1
            for _, _, lab in fp_set:
                by_label[lab]["fp"] += 1
            for _, _, lab in fn_set:
                by_label[lab]["fn"] += 1

            if len(examples) < 50 and (fp_set or fn_set):
                examples.append({
                    "text": text,
                    "gold": sorted(list(gold)),
                    "pred": sorted(list(pred)),
                    "false_positive": sorted(list(fp_set)),
                    "false_negative": sorted(list(fn_set)),
                })

    total_fp = total_pred - total_tp
    total_fn = total_gold - total_tp

    by_label_metrics = {}
    for lab, c in by_label.items():
        by_label_metrics[lab] = {
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            **micro_prf(c["tp"], c["fp"], c["fn"]),
        }

    result = {
        "model_dir": model_dir,
        "test_file": test_file,
        "threshold": threshold,
        "device": device,
        "samples": len(data),
        "gold_entities": total_gold,
        "pred_entities": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro": micro_prf(total_tp, total_fp, total_fn),
        "by_label": by_label_metrics,
        "error_examples": examples,
        "note": "This evaluates against silver labels generated from gazetteer_v2, not manually annotated gold labels.",
    }

    write_json(output_file, result)
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained GLiNER on silver-label test set.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    result = evaluate_gliner(
        model_dir=args.model_dir,
        test_file=args.test_file,
        output_file=args.output,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["micro"], ensure_ascii=False, indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
