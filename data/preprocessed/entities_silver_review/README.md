# Legal NER Silver Review Dataset

This folder is generated from LLM NER probe outputs.

Files:

- `sentence_entities_review.jsonl`: merged sentence-level silver annotations, including rows with overlap conflicts for manual review.
- `entity_mentions_review.jsonl`: one row per entity mention.
- `entities_review.csv`: Excel-friendly UTF-8-BOM review sheet. Fill `review_decision`, `corrected_label`, `corrected_text`, and `notes`.
- `sentence_entities_trainable_flat.jsonl`: unreviewed flat NER candidate data with overlap-conflict sentences removed.
- `entity_mentions_trainable_flat.jsonl`: mention rows matching the flat candidate data.
- `label_summary.csv`: label coverage summary.
- `dataset_summary.json`: counts, source files, and rejected rows.

Recommended review values:

- `accept`: keep the span and label.
- `reject`: remove the entity.
- `fix`: use `corrected_label` and/or `corrected_text`.

Important: all annotations are silver/unreviewed. Use them for fine-tuning only after review, or keep the model output provenance as weak supervision.
