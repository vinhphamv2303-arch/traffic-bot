# eval_hybrid_ner_v2

Đánh giá 3 cấu hình NER trên cùng một benchmark GLiNER-format:

1. `gazetteer`: match bằng `gazetteers_v2`
2. `gliner`: predict bằng model GLiNER fine-tuned
3. `hybrid`: merge gazetteer + GLiNER

## Test trên silver test.json

```bash
python eval_hybrid_ner_v2.py \
  --benchmark "../data/preprocessed/gliner_train_v2/test.json" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --model-dir "../data/models/legal_gliner_v2/final_model" \
  --output "../data/models/legal_gliner_v2/eval_hybrid_v2_on_silver_test.json" \
  --threshold 0.70 \
  --device cuda \
  --batch-size 32 \
  --merge-mode union
```

## Test trên Gemini reviewed non-nested

```bash
python eval_hybrid_ner_v2.py \
  --benchmark "../data/benchmarks/ner_benchmark_gemini_reviewed_non_nested_gliner.json" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --model-dir "../data/models/legal_gliner_v2/final_model" \
  --output "../data/models/legal_gliner_v2/eval_hybrid_v2_on_gemini_reviewed.json" \
  --threshold 0.70 \
  --device cuda \
  --batch-size 32
```

## Merge modes

- `union`: giữ cả 2 nguồn, resolve overlap bằng priority.
- `gliner_priority`: ưu tiên GLiNER khi overlap.
- `agreement_boost`: vẫn union, nhưng entity được cả hai nguồn bắt được tăng score.

Trong thực tế nên dùng `union` hoặc `agreement_boost`.
