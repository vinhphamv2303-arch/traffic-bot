# legal_gliner_training_v2

Pipeline train GLiNER cho NER pháp luật giao thông.

## Vì sao GLiNER?

GLiNER là mô hình NER generalist, cho phép extract entity type tùy biến, fine-tunable, và inference được trên CPU/consumer hardware. Repo chính thức dùng `GLiNER.from_pretrained(...).predict_entities(...)` cho inference và `model.train_model(...)` trong training script.

## Input

Dùng output từ bước match gazetteer v2:

```text
data/preprocessed/entities_gazetteer_v2/
  all_entity_mentions.jsonl
  <package>/
    sentence_entities.jsonl
```

Lưu ý:
- GLiNER train từ direct entity nằm trong `text`.
- `inherited_entities` từ `path_text` không dùng làm span train vì không nằm trong câu.

## 1. Prepare GLiNER dataset

```bash
python prepare_gliner_data.py \
  --entities-root "../data/preprocessed/entities_gazetteer_v2" \
  --output "../data/preprocessed/gliner_train_v2" \
  --negative-ratio 0.35
```

Output:

```text
data/preprocessed/gliner_train_v2/
  train.json
  dev.json
  test.json
  train_gliner_min.json
  dev_gliner_min.json
  test_gliner_min.json
  dataset_summary.json
```

Format GLiNER:

```json
{
  "tokenized_text": ["không", "đội", "mũ", "bảo", "hiểm"],
  "ner": [[0, 4, "BEHAVIOR"]]
}
```

## 2. Train trên GPU

Cài:

```bash
pip install gliner torch transformers accelerate
```

Với RTX 5090, cần PyTorch build hỗ trợ sm_120/cu128 nếu build hiện tại báo không compatible.

Train recommended:

```bash
python train_gliner_v2.py \
  --train-file "../data/preprocessed/gliner_train_v2/train.json" \
  --dev-file "../data/preprocessed/gliner_train_v2/dev.json" \
  --output-dir "../data/models/legal_gliner_v2" \
  --base-model "urchade/gliner_medium-v2.1" \
  --steps 3000 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 5e-6 \
  --others-lr 1e-5 \
  --device cuda
```

Nếu VRAM dư:

```bash
--batch-size 16 --eval-batch-size 16
```

Nếu OOM:

```bash
--batch-size 4 --eval-batch-size 4
```

Model cuối:

```text
data/models/legal_gliner_v2/final_model
```

## 3. Predict local / CPU

```bash
python predict_gliner_v2.py \
  --input-root "../data/preprocessed/entities_gazetteer_v2" \
  --model-dir "../data/models/legal_gliner_v2/final_model" \
  --output "../data/preprocessed/entities_gliner_v2" \
  --threshold 0.35 \
  --device cpu \
  --batch-size 8
```

Nếu dùng GPU để predict nhanh:

```bash
--device cuda --batch-size 32
```

## 4. Sau đó

Dùng:

```text
data/preprocessed/entities_gliner_v2/all_entity_mentions.jsonl
```

để build `entity_links_gliner_v2` hoặc graph v2.

## Gợi ý báo cáo

Mô tả dữ liệu train như sau:

```text
Tập huấn luyện NER được tạo theo hướng weak supervision: từ tập entity vocabulary mở rộng bằng X-NER-inspired mining, hệ thống match lại toàn bộ corpus để sinh các span nhãn bạc. Các span direct trong câu được chuyển sang định dạng GLiNER gồm tokenized_text và danh sách span [start_token, end_token, label]. Mô hình GLiNER được fine-tune trên tập nhãn bạc này để tạo extractor local, không phụ thuộc LLM ở giai đoạn inference.
```
