# Legal NER Fine-tune v1

Fine-tune NER với schema 7 nhãn từ output `entities_llm_v2`.

## Model khuyến nghị

- GPU tốt: `FacebookAI/xlm-roberta-large`
- CPU/GPU yếu: `FacebookAI/xlm-roberta-base`

## Cài đặt

```bash
pip install -r requirements.txt
```

## 1. Prepare dataset

```powershell
python prepare_dataset.py `
  --entities-root "../data/preprocessed/entities_llm_v2" `
  --output-dir "../data/preprocessed/ner_train_v1"
```

Output:

```text
data/preprocessed/ner_train_v1/
  sentence_entities_trainable_flat.jsonl
  dataset_summary.json
  label_summary.csv
```

## 2. Train bằng xlm-roberta-large

GPU 16GB+ nên dùng batch nhỏ + gradient accumulation:

```powershell
python train_ner.py `
  --train-file "../data/preprocessed/ner_train_v1/sentence_entities_trainable_flat.jsonl" `
  --output-dir "../data/models/legal_ner_xlmr_large_v1" `
  --model "FacebookAI/xlm-roberta-large" `
  --epochs 5 `
  --batch-size 2 `
  --grad-accum 4 `
  --max-length 256 `
  --eval-ratio 0.1 `
  --fp16
```

Nếu GPU hỗ trợ BF16:

```powershell
python train_ner.py `
  --train-file "../data/preprocessed/ner_train_v1/sentence_entities_trainable_flat.jsonl" `
  --output-dir "../data/models/legal_ner_xlmr_large_v1" `
  --model "FacebookAI/xlm-roberta-large" `
  --epochs 5 `
  --batch-size 2 `
  --grad-accum 4 `
  --max-length 256 `
  --eval-ratio 0.1 `
  --bf16
```

## 3. CPU / nhẹ hơn

```powershell
python train_ner.py `
  --train-file "../data/preprocessed/ner_train_v1/sentence_entities_trainable_flat.jsonl" `
  --output-dir "../data/models/legal_ner_xlmr_base_v1" `
  --model "FacebookAI/xlm-roberta-base" `
  --epochs 5 `
  --batch-size 4 `
  --max-length 192 `
  --eval-ratio 0.1
```

## 4. Predict toàn bộ sentences

```powershell
python predict_all.py `
  --model-dir "../data/models/legal_ner_xlmr_large_v1/final_model" `
  --sentences-root "../data/preprocessed/sentences" `
  --output-dir "../data/preprocessed/entities_model_v1" `
  --batch-size 16 `
  --max-length 256
```

CPU predict:

```powershell
python predict_all.py `
  --model-dir "../data/models/legal_ner_xlmr_base_v1/final_model" `
  --sentences-root "../data/preprocessed/sentences" `
  --output-dir "../data/preprocessed/entities_model_v1" `
  --batch-size 4 `
  --max-length 192 `
  --device cpu
```
