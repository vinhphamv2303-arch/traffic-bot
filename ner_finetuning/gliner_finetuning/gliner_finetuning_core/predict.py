
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

from .common import LABELS, ensure_dir, iter_sentence_entity_files, read_jsonl, stable_id, write_json, write_jsonl


LABELS_FOR_GLINER = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]


def predict_all(
    sentences_or_entities_root: str | Path,
    model_dir: str,
    output_dir: str | Path,
    threshold: float = 0.35,
    device: str = "cpu",
    batch_size: int = 16,
    max_words_per_chunk: int = 220,
    chunk_overlap_words: int = 40,
    max_chars_per_chunk: int = 0,
) -> Dict[str, Any]:
    """
    Run GLiNER on sentence rows with entity annotations.

    It only uses the text fields, so the input root can contain
    */sentences_with_entities.jsonl. Legacy names such as
    */sentence_entities.jsonl and */sentences_with_entity_links.jsonl
    are still supported by iter_sentence_entity_files().
    """
    try:
        from gliner import GLiNER
    except Exception as e:
        raise RuntimeError("Install GLiNER first: pip install gliner") from e

    out_root = ensure_dir(output_dir)
    model = GLiNER.from_pretrained(model_dir)
    model = model.to(device)
    words_splitter = getattr(getattr(model, "data_processor", None), "words_splitter", None)

    all_mentions = []
    summary = {
        "sentence_count": 0,
        "sentence_with_entity_count": 0,
        "entity_count": 0,
        "threshold": threshold,
        "model_dir": model_dir,
        "device": device,
        "max_words_per_chunk": max_words_per_chunk,
        "chunk_overlap_words": chunk_overlap_words,
        "max_chars_per_chunk": max_chars_per_chunk,
        "token_splitter": type(words_splitter).__name__ if words_splitter is not None else "fallback_regex",
        "chunked_sentence_count": 0,
        "prediction_chunk_count": 0,
        "packages": {},
    }

    for f in iter_sentence_entity_files(sentences_or_entities_root):
        pkg = f.parent.name
        pkg_out = ensure_dir(out_root / pkg)
        rows_out = []
        mentions = []

        batch_rows = []
        def flush():
            nonlocal batch_rows, rows_out, mentions
            if not batch_rows:
                return

            chunk_texts = []
            chunk_refs = []
            row_preds = [[] for _ in batch_rows]
            for row_idx, r in enumerate(batch_rows):
                chunks = chunk_text(
                    r.get("text") or "",
                    max_words_per_chunk=max_words_per_chunk,
                    overlap_words=chunk_overlap_words,
                    max_chars_per_chunk=max_chars_per_chunk,
                    token_splitter=words_splitter,
                )
                if len(chunks) > 1:
                    summary["chunked_sentence_count"] += 1
                summary["prediction_chunk_count"] += len(chunks)
                for chunk in chunks:
                    chunk_texts.append(chunk["text"])
                    chunk_refs.append((row_idx, chunk))

            # Some GLiNER versions support batch_predict_entities; fallback to loop.
            try:
                preds_batch = model.batch_predict_entities(chunk_texts, LABELS_FOR_GLINER, threshold=threshold)
            except Exception:
                preds_batch = [model.predict_entities(t, LABELS_FOR_GLINER, threshold=threshold) for t in chunk_texts]

            for (row_idx, chunk), preds in zip(chunk_refs, preds_batch):
                source_text = batch_rows[row_idx].get("text") or ""
                for p in preds:
                    normalized = normalize_prediction(p, source_text, chunk)
                    if normalized:
                        row_preds[row_idx].append(normalized)

            for r, preds in zip(batch_rows, row_preds):
                preds = dedupe_predictions(preds)
                ents = []
                for p in preds:
                    e = {
                        "text": p.get("text"),
                        "label": p.get("label"),
                        "start": p.get("start"),
                        "end": p.get("end"),
                        "confidence": float(p.get("score", 0.0)),
                        "scope": "direct",
                        "source": "gliner_v2",
                        "graph_weight": float(p.get("score", 0.0)),
                    }
                    ents.append(e)
                    mentions.append({
                        "mention_id": stable_id(r.get("sentence_id") or "", e["label"] or "", e["text"] or "", str(e["start"]), prefix="ment"),
                        "sentence_id": r.get("sentence_id"),
                        "passage_id": r.get("passage_id"),
                        "source_unit_id": r.get("source_unit_id"),
                        "package_id": r.get("package_id"),
                        "document_id": r.get("document_id"),
                        "document_number": r.get("document_number"),
                        "document_title": r.get("document_title"),
                        "path_text": r.get("path_text"),
                        **e,
                    })
                rows_out.append({**r, "entities": ents, "entity_count": len(ents)})
            batch_rows = []

        for r in read_jsonl(f):
            batch_rows.append(r)
            if len(batch_rows) >= batch_size:
                flush()
        flush()

        write_jsonl(pkg_out / "sentences_with_entities.jsonl", rows_out)
        write_jsonl(pkg_out / "entity_mentions.jsonl", mentions)

        pkg_summary = {
            "sentence_count": len(rows_out),
            "sentence_with_entity_count": sum(1 for r in rows_out if r.get("entity_count", 0) > 0),
            "entity_count": len(mentions),
        }
        write_json(pkg_out / "entity_summary.json", pkg_summary)

        summary["packages"][pkg] = pkg_summary
        summary["sentence_count"] += pkg_summary["sentence_count"]
        summary["sentence_with_entity_count"] += pkg_summary["sentence_with_entity_count"]
        summary["entity_count"] += pkg_summary["entity_count"]
        all_mentions.extend(mentions)

    write_jsonl(out_root / "all_entity_mentions.jsonl", all_mentions)
    write_json(out_root / "entity_summary.json", summary)
    return summary


def chunk_text(
    text: str,
    max_words_per_chunk: int = 220,
    overlap_words: int = 40,
    max_chars_per_chunk: int = 0,
    token_splitter: Any | None = None,
) -> List[Dict[str, Any]]:
    if not text:
        return [{"text": "", "offset": 0, "chunk_index": 0, "word_start": 0, "word_end": 0}]
    if max_words_per_chunk <= 0:
        chunks = [{"text": text, "offset": 0, "chunk_index": 0, "word_start": 0, "word_end": None}]
        return split_oversized_chunks_by_chars(chunks, max_chars_per_chunk)

    spans = get_token_spans(text, token_splitter=token_splitter)
    if len(spans) <= max_words_per_chunk and len(text) <= max_chars_per_chunk:
        return [{"text": text, "offset": 0, "chunk_index": 0, "word_start": 0, "word_end": len(spans)}]

    overlap_words = max(0, min(overlap_words, max_words_per_chunk - 1))
    step = max(1, max_words_per_chunk - overlap_words)
    chunks = []
    start_word = 0
    chunk_index = 0
    while start_word < len(spans):
        end_word = min(len(spans), start_word + max_words_per_chunk)
        char_start = spans[start_word][0]
        char_end = spans[end_word - 1][1]
        chunks.append({
            "text": text[char_start:char_end],
            "offset": char_start,
            "chunk_index": chunk_index,
            "word_start": start_word,
            "word_end": end_word,
        })
        if end_word >= len(spans):
            break
        start_word += step
        chunk_index += 1
    return split_oversized_chunks_by_chars(chunks, max_chars_per_chunk)


def get_token_spans(text: str, token_splitter: Any | None = None) -> List[tuple[int, int]]:
    if token_splitter is not None:
        try:
            spans = []
            for item in token_splitter(text):
                if len(item) < 3:
                    continue
                start = int(item[1])
                end = int(item[2])
                if end > start:
                    spans.append((start, end))
            if spans:
                return spans
        except Exception:
            pass

    # Match GLiNER's WhitespaceTokenSplitter fallback: words plus standalone symbols.
    return [(m.start(), m.end()) for m in re.finditer(r"\w+(?:[-_]\w+)*|\S", text)]


def split_oversized_chunks_by_chars(chunks: List[Dict[str, Any]], max_chars_per_chunk: int) -> List[Dict[str, Any]]:
    if max_chars_per_chunk <= 0:
        return chunks

    out = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        offset = int(chunk.get("offset") or 0)
        if len(text) <= max_chars_per_chunk:
            out.append(chunk)
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars_per_chunk)
            if end < len(text):
                split_at = _best_char_split(text, start, end)
                if split_at > start:
                    end = split_at
            piece = text[start:end].strip()
            leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
            if piece:
                out.append({
                    **chunk,
                    "text": piece,
                    "offset": offset + start + leading_trim,
                    "char_chunked": True,
                })
            if end <= start:
                break
            start = end

    for idx, chunk in enumerate(out):
        chunk["chunk_index"] = idx
    return out


def _best_char_split(text: str, start: int, end: int) -> int:
    window_start = max(start + int((end - start) * 0.6), start)
    segment = text[window_start:end]
    for pattern in [r"[.;:]\s+", r"[,]\s+", r"\s+"]:
        matches = list(re.finditer(pattern, segment))
        if matches:
            return window_start + matches[-1].end()
    return end


def normalize_prediction(pred: Dict[str, Any], source_text: str, chunk: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        start = int(pred.get("start"))
        end = int(pred.get("end"))
    except Exception:
        return None
    if start < 0 or end <= start or end > len(chunk.get("text") or ""):
        return None

    source_start = start + int(chunk.get("offset") or 0)
    source_end = end + int(chunk.get("offset") or 0)
    if source_start < 0 or source_end <= source_start or source_end > len(source_text or ""):
        return None

    score = pred.get("score", pred.get("confidence", 0.0))
    return {
        "text": source_text[source_start:source_end],
        "label": pred.get("label"),
        "start": source_start,
        "end": source_end,
        "score": float(score or 0.0),
        "chunk_index": chunk.get("chunk_index"),
    }


def dedupe_predictions(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for p in preds:
        key = (p.get("label"), p.get("start"), p.get("end"), p.get("text"))
        if key not in best or float(p.get("score", 0.0)) > float(best[key].get("score", 0.0)):
            best[key] = p
    out = list(best.values())
    out.sort(key=lambda x: (int(x.get("start") or 0), -(int(x.get("end") or 0) - int(x.get("start") or 0))))
    return out


def main():
    ap = argparse.ArgumentParser(description="Run trained GLiNER over all sentence files.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-words-per-chunk", type=int, default=220)
    ap.add_argument("--chunk-overlap-words", type=int, default=40)
    ap.add_argument("--max-chars-per-chunk", type=int, default=0)
    args = ap.parse_args()

    summary = predict_all(
        sentences_or_entities_root=args.input_root,
        model_dir=args.model_dir,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
        max_words_per_chunk=args.max_words_per_chunk,
        chunk_overlap_words=args.chunk_overlap_words,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )
    print("GLiNER prediction completed")
    print(summary)


if __name__ == "__main__":
    main()
