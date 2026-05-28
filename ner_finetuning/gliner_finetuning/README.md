# GLiNER Fine-tuning

Module này chuyển pseudo-label từ gazetteer thành train/dev/test, fine-tune GLiNER, predict toàn corpus và evaluate.

Entrypoints:

```bash
python gliner_finetuning/prepare_gliner_dataset.py \
  --entities-root data/preprocessed/gazetteer_pseudo_labels \
  --output data/preprocessed/gliner_training_data \
  --negative-ratio 0.35

python gliner_finetuning/train_gliner_model.py \
  --train-file data/preprocessed/gliner_training_data/train.json \
  --dev-file data/preprocessed/gliner_training_data/dev.json \
  --output-dir data/models/gliner_traffic_ner \
  --base-model urchade/gliner_medium-v2.1 \
  --steps 3000 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --device cuda

python gliner_finetuning/predict_entities.py \
  --input-root data/preprocessed/gazetteer_pseudo_labels \
  --model-dir data/models/gliner_traffic_ner/final_model \
  --output data/preprocessed/gliner_predictions_th070 \
  --threshold 0.70 \
  --device cuda
```

Xem README tổng ở `../README.md` để biết thống kê dữ liệu và cách đánh giá.
