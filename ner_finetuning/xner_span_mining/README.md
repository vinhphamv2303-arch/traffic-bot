# X-NER Span Mining

Module này mở rộng vocabulary bằng X-NER-style span mining. Seed ban đầu đến từ `data/preprocessed/pruned_seed_gazetteer`; candidate sau mining cần được review trước khi đưa vào gazetteer chính.

Entrypoints:

```bash
python xner_span_mining/mine_entity_spans.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/pruned_seed_gazetteer \
  --output data/preprocessed/xner_candidate_entities \
  --quality-preset max \
  --embedding-model BAAI/bge-m3 \
  --batch-size 256 \
  --device cuda

python xner_span_mining/build_expanded_gazetteer.py \
  --base-gazetteer-root data/preprocessed/pruned_seed_gazetteer \
  --reviewed-mined-csv data/preprocessed/xner_candidate_entities/reviewed_candidates.csv \
  --output data/preprocessed/expanded_gazetteer
```

Nếu chỉ smoke test trên CPU, dùng `--skip-scoring` và `--max-sentences` để không load embedding model lớn. Xem README tổng ở `../README.md` để biết toàn bộ pipeline.
