# Legal Local Entity Extractor

Module nối tiếp từ `legal vocab step 4 prune`.

Mục tiêu:
- Không dùng LLM khi inference.
- Dùng `gazetteers_v1_pruned` để tạo training data local.
- Có baseline extractor deterministic chạy CPU.
- Có pipeline train/predict GLiNER local.

## Input

```text
data/preprocessed/sentences/
data/preprocessed/gazetteers_v1_pruned/
```

## Step 5: tạo training data local từ gazetteer đã prune

```powershell
python prepare_local_ner_data.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/local_ner_train_v1" `
  --negative-ratio 0.5
```

Output:

```text
data/preprocessed/local_ner_train_v1/
  train.jsonl
  dev.jsonl
  test.jsonl
  all_trainable.jsonl
  train_gliner_raw.jsonl
  dev_gliner_raw.jsonl
  test_gliner_raw.jsonl
  dataset_summary.json
```

`entities` = direct spans trong sentence text.

`inherited_entities` = entities lấy từ `path_text`, dùng cho graph/context, không dùng trực tiếp train BIO/GLiNER vì không nằm trong text.

## CPU baseline extractor

Dùng gazetteer exact để extract local, chạy được CPU rất nhanh:

```powershell
python predict_gazetteer_local.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/entities_gazetteer_local_v1"
```

Output:

```text
data/preprocessed/entities_gazetteer_local_v1/
  all_entity_mentions.jsonl
  entity_summary.json
  <PACKAGE_ID>/
    sentence_entities.jsonl
    entity_mentions.jsonl
```

## GLiNER fine-tuning

Cài đặt:

```bash
pip install gliner torch transformers accelerate
```

Chuẩn bị/fine-tune:

```powershell
python train_gliner.py `
  --train-file "../data/preprocessed/local_ner_train_v1/train.jsonl" `
  --dev-file "../data/preprocessed/local_ner_train_v1/dev.jsonl" `
  --output-dir "../data/models/legal_gliner_v1" `
  --model "urchade/gliner_medium-v2.1" `
  --steps 1000 `
  --batch-size 8 `
  --lr 5e-6
```

Lưu ý: GLiNER API thay đổi theo version. Script này sẽ:
1. convert data sang GLiNER format
2. nếu package có `train_model`, nó sẽ train
3. nếu không, nó vẫn xuất `train_gliner.jsonl`, `dev_gliner.jsonl`, `gliner_training_config.json` để bạn dùng training CLI/API của version GLiNER đang cài.

## GLiNER local inference CPU/GPU

```powershell
python predict_gliner.py `
  --sentences-root "../data/preprocessed/sentences" `
  --model-dir "../data/models/legal_gliner_v1/final_model" `
  --output "../data/preprocessed/entities_gliner_local_v1" `
  --threshold 0.35
```

Nếu chưa fine-tune được, có thể dùng model base zero-shot:

```powershell
python predict_gliner.py `
  --sentences-root "../data/preprocessed/sentences" `
  --model-dir "urchade/gliner_medium-v2.1" `
  --output "../data/preprocessed/entities_gliner_zeroshot_v1" `
  --threshold 0.35
```

## Vì sao đây là bước nối tiếp step 4 prune?

```text
gazetteers_v1_pruned
  ↓
prepare_local_ner_data
  ↓
local CPU gazetteer extractor baseline
  ↓
GLiNER train/predict local
```

Không dùng LLM runtime.

## Bước tiếp theo sau module này

Sau khi có `entities_gazetteer_local_v1` hoặc `entities_gliner_local_v1`, ta sẽ làm:

```text
export_entity_links_from_mentions.py
→ rebuild graph
→ retrieval test
```

Nếu GLiNER chưa tốt, ta bổ sung Step 5b:

```text
X-NER-style seed miner
→ mined_candidates.csv
→ review
→ gazetteers_v2
→ train lại local model
```
