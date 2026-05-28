# Gazetteer Pruning

Module này lọc seed gazetteer để giảm false positive: loại alias quá chung, alias nhiều nghĩa và downweight các surface form có rủi ro.

Entrypoint chính:

```bash
python gazetteer_pruning/prune_gazetteer.py \
  --gazetteer-root data/preprocessed/seed_gazetteer \
  --output data/preprocessed/pruned_seed_gazetteer
```

Có thể match pruned gazetteer để audit bằng:

```bash
python gazetteer_pruning/match_pruned_gazetteer_to_corpus.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/pruned_seed_gazetteer \
  --output data/archive/pruned_gazetteer_sentence_matches
```

Xem README tổng ở `../README.md` để biết toàn bộ pipeline.
