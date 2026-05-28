# Seed Vocabulary

Module này gom các entity mention bootstrap thành vocabulary seed theo surface form.

Entrypoint:

```bash
python seed_vocabulary/build_seed_vocabulary.py \
  --entity-mentions data/preprocessed/bootstrap_llm_entities/all_entity_mentions.jsonl \
  --output data/preprocessed/seed_vocabulary
```

Output chính là `data/preprocessed/seed_vocabulary/reviewed_surface_forms.csv`. Xem README tổng ở `../README.md` để biết toàn bộ pipeline và thống kê dữ liệu.
