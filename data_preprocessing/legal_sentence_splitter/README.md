# Legal Sentence Splitter

Tách `passages.jsonl` thành `sentences.jsonl` để chạy NER/entity extraction.

## Chạy

```bash
python split_sentences.py \
  -i "../data/preprocessed/passages" \
  -o "../data/preprocessed/sentences"
```

Một package:

```bash
python split_sentences.py \
  -i "../data/preprocessed/passages/12_2025_TTBCA" \
  -o "../data/preprocessed/sentences"
```

Không thêm context vào câu NER:

```bash
python split_sentences.py \
  -i "../data/preprocessed/passages" \
  -o "../data/preprocessed/sentences" \
  --no-context-for-ner
```

## Output

```text
data/preprocessed/sentences/
  all_sentences.jsonl
  sentence_summary.json

data/preprocessed/sentences/<PACKAGE_ID>/
  sentences.jsonl
  sentence_summary.json
```

## Ghi chú

- `table_row`, `form_field`, `form_table`, `appendix_table` được giữ nguyên một sentence.
- Splitter tránh cắt sai:
  - `5.000.000 đồng`
  - `TT-BCA`, `NĐ-CP`
  - `QCVN`
  - số thứ tự `1.1.`, `2.2.3.`
