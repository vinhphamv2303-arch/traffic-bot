# NER Evaluation

Module này đánh giá kết quả NER khi kết hợp GLiNER và gazetteer.

Entrypoint:

```bash
python ner_evaluation/evaluate_hybrid_ner.py \
  --benchmark data/preprocessed/gliner_training_data/test.json \
  --gazetteer-root data/preprocessed/expanded_gazetteer \
  --model-dir data/models/gliner_traffic_ner/final_model \
  --output data/models/gliner_traffic_ner/eval_hybrid_on_silver_test.json \
  --threshold 0.70 \
  --device cuda \
  --batch-size 32 \
  --merge-mode union
```

Benchmark gold nhỏ nằm ở `data/benchmark/ner_gold_benchmark/ner_benchmark_gemini_clean.json`. Xem README tổng ở `../README.md` để biết toàn bộ pipeline.
