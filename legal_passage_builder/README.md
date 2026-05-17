# Legal Passage Builder

Tạo `passages.jsonl` theo tinh thần LinearRAG từ:

```text
parsed/<PACKAGE_ID>/all_units.jsonl
effectivity/effectivity_index.csv
effectivity/effectivity_unit_overrides.csv
resolved_references/<PACKAGE_ID>/resolved_references.jsonl
```

## Chạy

```bash
python build_passages.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/passages" \
  --effectivity-root "../data/preprocessed/effectivity" \
  --resolved-refs-root "../data/preprocessed/resolved_references"
```

Một package:

```bash
python build_passages.py \
  -i "../data/preprocessed/parsed/12_2025_TTBCA" \
  -o "../data/preprocessed/passages" \
  --effectivity-root "../data/preprocessed/effectivity" \
  --resolved-refs-root "../data/preprocessed/resolved_references"
```

Chỉ tạo atomic passages:

```bash
python build_passages.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/passages_atomic" \
  --effectivity-root "../data/preprocessed/effectivity" \
  --resolved-refs-root "../data/preprocessed/resolved_references" \
  --no-container-passages
```

## Output

```text
passages/<PACKAGE_ID>/passages.jsonl
passages/<PACKAGE_ID>/passage_summary.json
passages/all_passages.jsonl
passages/passage_summary.json
```

## Passage schema chính

- `passage_id`
- `source_unit_id`
- `passage_kind`: `atomic` hoặc `container`
- `path_text`
- `content`
- `passage_text`: text để embedding
- `effective_from`, `ceased_from`
- `outgoing_refs`, `incoming_refs`
- `reference_expansion_policies`

Tham chiếu dài tới Phụ lục/QCVN không được inline vào passage. Nó được đánh dấu:

```json
"reference_expansion_policies": ["search_within_target"]
```
