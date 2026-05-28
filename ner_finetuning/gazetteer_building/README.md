# Gazetteer Building

Module này tạo gazetteer từ reviewed seed vocabulary và match gazetteer lên sentence corpus.

Entrypoints:

```bash
python gazetteer_building/build_seed_gazetteer.py \
  --reviewed data/preprocessed/seed_vocabulary/reviewed_surface_forms.csv \
  --output data/preprocessed/seed_gazetteer \
  --min-count 8

python gazetteer_building/match_gazetteer_to_corpus.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/expanded_gazetteer \
  --output data/preprocessed/gazetteer_pseudo_labels
```

Script `match_gazetteer_to_corpus.py` là matcher tổng quát, có thể dùng với seed gazetteer, pruned gazetteer hoặc expanded gazetteer. Xem README tổng ở `../README.md` để biết toàn bộ pipeline.
