
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, List


LABELS = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[“”\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_offsets_from_joined_tokens(tokens: List[str]) -> tuple[str, list[tuple[int, int]]]:
    text = " ".join(tokens)
    offsets = []
    pos = 0
    for tok in tokens:
        start = pos
        end = start + len(tok)
        offsets.append((start, end))
        pos = end + 1
    return text, offsets


def char_to_token_span(start: int, end: int, offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if e <= start or s >= end:
            continue
        idxs.append(i)
    if not idxs:
        return None
    return idxs[0], idxs[-1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def load_gazetteer(gazetteer_root: str | Path) -> list[dict[str, str]]:
    root = Path(gazetteer_root)
    terms: list[dict[str, str]] = []

    aliases = first_existing(root, ["aliases.jsonl", "entity_aliases.jsonl"])
    csv_terms = first_existing(root, ["gazetteer_terms.csv", "terms.csv"])
    canon = first_existing(root, ["canonical_entities.jsonl", "entities.jsonl"])

    if aliases:
        for r in load_jsonl(aliases):
            surface = r.get("surface") or r.get("alias") or r.get("text") or r.get("term")
            label = r.get("label") or r.get("entity_type") or r.get("type")
            canonical = r.get("canonical") or r.get("canonical_surface") or r.get("name") or surface
            entity_id = r.get("entity_id") or r.get("id") or f"{label}:{canonical}"
            if surface and label in LABELS:
                terms.append({"surface": str(surface), "label": str(label), "canonical": str(canonical), "entity_id": str(entity_id)})

    if not terms and csv_terms:
        with open(csv_terms, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                surface = r.get("surface") or r.get("alias") or r.get("text") or r.get("term")
                label = r.get("label") or r.get("entity_type") or r.get("type")
                canonical = r.get("canonical") or r.get("canonical_surface") or r.get("name") or surface
                entity_id = r.get("entity_id") or r.get("id") or f"{label}:{canonical}"
                if surface and label in LABELS:
                    terms.append({"surface": str(surface), "label": str(label), "canonical": str(canonical), "entity_id": str(entity_id)})

    if not terms and canon:
        for r in load_jsonl(canon):
            surface = r.get("surface") or r.get("canonical") or r.get("text") or r.get("name")
            label = r.get("label") or r.get("entity_type") or r.get("type")
            canonical = r.get("canonical") or r.get("name") or surface
            entity_id = r.get("entity_id") or r.get("id") or f"{label}:{canonical}"
            for s in [surface] + list(r.get("aliases") or []):
                if s and label in LABELS:
                    terms.append({"surface": str(s), "label": str(label), "canonical": str(canonical), "entity_id": str(entity_id)})

    if not terms:
        raise FileNotFoundError(f"Cannot load gazetteer terms from {root}. Expected aliases.jsonl or gazetteer_terms.csv.")

    seen = set()
    clean = []
    for t in terms:
        surface = re.sub(r"\s+", " ", t["surface"].strip())
        label = t["label"]
        key = (surface.lower(), label)
        if not surface or len(surface) < 2 or key in seen:
            continue
        seen.add(key)
        t["surface"] = surface
        t["norm_surface"] = normalize_for_match(surface)
        if t["norm_surface"]:
            clean.append(t)

    clean.sort(key=lambda x: (-len(x["norm_surface"].split()), -len(x["norm_surface"]), x["label"], x["surface"]))
    return clean


def build_regex_terms(terms: list[dict[str, str]]) -> list[dict[str, Any]]:
    compiled = []
    for t in terms:
        parts = [re.escape(p) for p in t["norm_surface"].split()]
        if not parts:
            continue
        pat = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
        except re.error:
            continue
        compiled.append({**t, "regex": rx})
    return compiled


def map_norm_to_original(text: str) -> tuple[str, list[int]]:
    out, mp = [], []
    last_space = False
    for i, ch in enumerate(text):
        c = ch.lower()
        if c in "“”\"'":
            c = " "
        if c.isspace():
            c = " "
        if c == " ":
            if last_space:
                continue
            last_space = True
            out.append(c)
            mp.append(i)
        else:
            last_space = False
            out.append(c)
            mp.append(i)
    start = 0
    while start < len(out) and out[start] == " ":
        start += 1
    end = len(out)
    while end > start and out[end - 1] == " ":
        end -= 1
    return "".join(out[start:end]), mp[start:end]


def match_gazetteer(text: str, token_offsets: list[tuple[int, int]], compiled_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    norm_text, norm_to_orig = map_norm_to_original(text)
    candidates = []

    for t in compiled_terms:
        for m in t["regex"].finditer(norm_text):
            ns, ne = m.start(), m.end()
            if ns >= len(norm_to_orig) or ne - 1 >= len(norm_to_orig):
                continue
            os = norm_to_orig[ns]
            oe = norm_to_orig[ne - 1] + 1
            tok_span = char_to_token_span(os, oe, token_offsets)
            if tok_span is None:
                continue
            candidates.append({
                "start_char": os,
                "end_char": oe,
                "start_token": tok_span[0],
                "end_token": tok_span[1],
                "text": text[os:oe],
                "label": t["label"],
                "score": 0.85,
                "source": "gazetteer",
                "canonical": t["canonical"],
                "surface": t["surface"],
            })

    candidates.sort(key=lambda x: (-(x["end_char"] - x["start_char"]), x["start_char"], x["label"]))
    occupied = [False] * (len(text) + 1)
    selected = []
    for c in candidates:
        if any(occupied[c["start_char"]:c["end_char"]]):
            continue
        for i in range(c["start_char"], c["end_char"]):
            occupied[i] = True
        selected.append(c)
    selected.sort(key=lambda x: (x["start_token"], x["end_token"], x["label"]))
    return selected


def char_offsets_to_token_span(text: str, tokenized_text: list[str], start: int, end: int) -> tuple[int, int] | None:
    offsets = []
    pos = 0
    for tok in tokenized_text:
        idx = text.find(tok, pos)
        if idx < 0:
            return None
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return char_to_token_span(start, end, offsets)


def predict_gliner_batch(model, texts: list[str], labels: list[str], threshold: float):
    try:
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    except Exception:
        return [model.predict_entities(t, labels, threshold=threshold) for t in texts]


def gliner_preds_to_spans(text: str, tokens: list[str], preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in preds:
        label = p.get("label")
        if label not in LABELS:
            continue
        try:
            start, end = int(p.get("start", 0)), int(p.get("end", 0))
        except Exception:
            continue
        span = char_offsets_to_token_span(text, tokens, start, end)
        if span is None:
            continue
        out.append({
            "start_char": start,
            "end_char": end,
            "start_token": span[0],
            "end_token": span[1],
            "text": text[start:end],
            "label": label,
            "score": float(p.get("score", 0.0)),
            "source": "gliner",
            "canonical": text[start:end],
            "surface": text[start:end],
        })
    return out


def overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["start_token"] <= b["end_token"] and b["start_token"] <= a["end_token"]


def span_len(x: dict[str, Any]) -> int:
    return int(x["end_token"]) - int(x["start_token"]) + 1


def merge_hybrid(
    gazetteer_preds: list[dict[str, Any]],
    gliner_preds: list[dict[str, Any]],
    mode: str = "union",
    gliner_score_bonus: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Modes:
    - union: keep both sources, resolve overlap by priority.
    - gliner_priority: if overlap, prefer GLiNER unless gazetteer span is much longer exact known phrase.
    - agreement_boost: keep union, but if same/overlap same label appears in both, mark source=hybrid and boost.
    """
    candidates = []

    # Mark agreement.
    for g in gazetteer_preds:
        g2 = dict(g)
        for p in gliner_preds:
            if g["label"] == p["label"] and overlap(g, p):
                g2["source"] = "hybrid_agree"
                g2["score"] = max(float(g2.get("score", 0.85)), float(p.get("score", 0.0))) + 0.15
                break
        candidates.append(g2)

    for p in gliner_preds:
        p2 = dict(p)
        p2["score"] = float(p2.get("score", 0.0)) + gliner_score_bonus
        for g in gazetteer_preds:
            if g["label"] == p["label"] and overlap(g, p):
                p2["source"] = "hybrid_agree"
                p2["score"] = max(float(p2.get("score", 0.0)), float(g.get("score", 0.85))) + 0.10
                break
        candidates.append(p2)

    def priority(c):
        src = c.get("source")
        score = float(c.get("score", 0.0))
        length = span_len(c)
        # Agreement is best. Then mode-specific priority.
        agree_bonus = 2.0 if src == "hybrid_agree" else 0.0
        if mode == "gliner_priority":
            source_bonus = 1.0 if src in {"gliner", "hybrid_agree"} else 0.4
        else:
            source_bonus = 0.8 if src in {"gazetteer", "hybrid_agree"} else 0.6
        # Moderate spans favored slightly; very long spans can dominate by length otherwise.
        moderate_bonus = 0.3 if 1 <= length <= 10 else 0.0
        return agree_bonus + source_bonus + score + moderate_bonus + min(length, 12) * 0.01

    candidates.sort(key=lambda c: (-priority(c), c["start_token"], -span_len(c), c["label"]))

    selected = []
    for c in candidates:
        has_overlap = False
        for s in selected:
            if overlap(c, s):
                has_overlap = True
                break
        if not has_overlap:
            selected.append(c)

    selected.sort(key=lambda x: (x["start_token"], x["end_token"], x["label"]))
    return selected


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def score_exact(pred_set: set[tuple[int, int, str]], gold_set: set[tuple[int, int, str]]):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def relaxed_match(pred: tuple[int, int, str], gold: tuple[int, int, str]) -> bool:
    ps, pe, pl = pred
    gs, ge, gl = gold
    return pl == gl and ps <= ge and gs <= pe


def score_relaxed(preds: list[tuple[int, int, str]], golds: list[tuple[int, int, str]]):
    matched_gold = set()
    tp = 0
    for p in preds:
        best, best_overlap = None, -1
        for gi, g in enumerate(golds):
            if gi in matched_gold or not relaxed_match(p, g):
                continue
            ov = min(p[1], g[1]) - max(p[0], g[0]) + 1
            if ov > best_overlap:
                best_overlap = ov
                best = gi
        if best is not None:
            matched_gold.add(best)
            tp += 1
    fp = len(preds) - tp
    fn = len(golds) - tp
    return tp, fp, fn


def update_label_counters(by_label, preds, golds):
    for lab in LABELS:
        p_lab = [p for p in preds if p[2] == lab]
        g_lab = [g for g in golds if g[2] == lab]
        etp, efp, efn = score_exact(set(p_lab), set(g_lab))
        by_label["exact"][lab].update({"tp": etp, "fp": efp, "fn": efn})
        rtp, rfp, rfn = score_relaxed(p_lab, g_lab)
        by_label["relaxed"][lab].update({"tp": rtp, "fp": rfp, "fn": rfn})


def metrics_from_counter(c: Counter):
    return {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], **prf(c["tp"], c["fp"], c["fn"])}


def evaluate_hybrid(
    benchmark_file: str | Path,
    gazetteer_root: str | Path,
    model_dir: str,
    output_file: str | Path,
    threshold: float = 0.70,
    device: str = "cuda",
    batch_size: int = 32,
    merge_mode: str = "union",
    max_examples: int = 80,
):
    from gliner import GLiNER

    data = read_json(benchmark_file)
    terms = load_gazetteer(gazetteer_root)
    compiled_terms = build_regex_terms(terms)

    model = GLiNER.from_pretrained(model_dir)
    model = model.to(device)

    totals = {
        "gazetteer": {"exact": Counter(), "relaxed": Counter()},
        "gliner": {"exact": Counter(), "relaxed": Counter()},
        "hybrid": {"exact": Counter(), "relaxed": Counter()},
    }
    by_label = {
        sys: {"exact": {lab: Counter() for lab in LABELS}, "relaxed": {lab: Counter() for lab in LABELS}}
        for sys in totals
    }

    error_examples = []

    for start_i in range(0, len(data), batch_size):
        batch = data[start_i:start_i + batch_size]
        texts = [" ".join(item["tokenized_text"]) for item in batch]
        gliner_batch = predict_gliner_batch(model, texts, LABELS, threshold=threshold)

        for j, (item, text, gliner_raw) in enumerate(zip(batch, texts, gliner_batch)):
            idx = start_i + j
            tokens = item["tokenized_text"]
            rebuilt_text, offsets = token_offsets_from_joined_tokens(tokens)
            assert rebuilt_text == text

            golds = [(int(s), int(e), str(lab)) for s, e, lab in (item.get("ner") or [])]

            gaz_raw = match_gazetteer(text, offsets, compiled_terms)
            gli_raw = gliner_preds_to_spans(text, tokens, gliner_raw)
            hyb_raw = merge_hybrid(gaz_raw, gli_raw, mode=merge_mode)

            systems = {
                "gazetteer": gaz_raw,
                "gliner": gli_raw,
                "hybrid": hyb_raw,
            }

            for sys_name, raw_preds in systems.items():
                preds = [(p["start_token"], p["end_token"], p["label"]) for p in raw_preds]

                tp, fp, fn = score_exact(set(preds), set(golds))
                totals[sys_name]["exact"].update({"tp": tp, "fp": fp, "fn": fn})

                rtp, rfp, rfn = score_relaxed(preds, golds)
                totals[sys_name]["relaxed"].update({"tp": rtp, "fp": rfp, "fn": rfn})

                update_label_counters(by_label[sys_name], preds, golds)

            hybrid_preds = [(p["start_token"], p["end_token"], p["label"]) for p in hyb_raw]
            htp, hfp, hfn = score_exact(set(hybrid_preds), set(golds))
            if len(error_examples) < max_examples and (hfp or hfn):
                error_examples.append({
                    "index": idx,
                    "text": text,
                    "gold": sorted(golds),
                    "gazetteer_pred": sorted([(p["start_token"], p["end_token"], p["label"]) for p in gaz_raw]),
                    "gliner_pred": sorted([(p["start_token"], p["end_token"], p["label"]) for p in gli_raw]),
                    "hybrid_pred": sorted(hybrid_preds),
                    "hybrid_pred_text": [
                        {
                            "span": [p["start_token"], p["end_token"], p["label"]],
                            "text": p["text"],
                            "source": p["source"],
                            "score": p.get("score"),
                        }
                        for p in hyb_raw
                    ],
                    "false_positive_exact": sorted(list(set(hybrid_preds) - set(golds))),
                    "false_negative_exact": sorted(list(set(golds) - set(hybrid_preds))),
                })

    result = {
        "benchmark_file": str(benchmark_file),
        "gazetteer_root": str(gazetteer_root),
        "model_dir": str(model_dir),
        "sample_count": len(data),
        "gazetteer_term_count": len(terms),
        "threshold": threshold,
        "device": device,
        "batch_size": batch_size,
        "merge_mode": merge_mode,
        "systems": {},
        "error_examples": error_examples,
        "note": "Exact requires identical token span+label. Relaxed requires same label and token-span overlap. Hybrid merges gazetteer and GLiNER with overlap resolution.",
    }

    for sys_name in ["gazetteer", "gliner", "hybrid"]:
        result["systems"][sys_name] = {
            "exact": metrics_from_counter(totals[sys_name]["exact"]),
            "relaxed_overlap": metrics_from_counter(totals[sys_name]["relaxed"]),
            "by_label_exact": {
                lab: metrics_from_counter(c) for lab, c in by_label[sys_name]["exact"].items()
            },
            "by_label_relaxed": {
                lab: metrics_from_counter(c) for lab, c in by_label[sys_name]["relaxed"].items()
            },
        }

    write_json(output_file, result)
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate Gazetteer, GLiNER, and Hybrid NER on GLiNER-format JSON.")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--merge-mode", default="union", choices=["union", "gliner_priority", "agreement_boost"])
    ap.add_argument("--max-examples", type=int, default=80)
    args = ap.parse_args()

    result = evaluate_hybrid(
        benchmark_file=args.benchmark,
        gazetteer_root=args.gazetteer_root,
        model_dir=args.model_dir,
        output_file=args.output,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
        merge_mode=args.merge_mode,
        max_examples=args.max_examples,
    )

    for name in ["gazetteer", "gliner", "hybrid"]:
        print(f"\n{name.upper()}")
        print("Exact:", json.dumps(result["systems"][name]["exact"], ensure_ascii=False, indent=2))
        print("Relaxed:", json.dumps(result["systems"][name]["relaxed_overlap"], ensure_ascii=False, indent=2))
    print("\nsaved:", args.output)


if __name__ == "__main__":
    main()
