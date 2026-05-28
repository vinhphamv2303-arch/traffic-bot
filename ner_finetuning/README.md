# Legal NER Fine-tuning

Thư mục này là một project riêng cho pipeline fine-tune NER pháp lý. Pipeline hiện tại không dùng LLM làm bộ trích xuất cuối cùng. LLM chỉ được dùng ở bước bootstrap ban đầu để tạo seed entity dạng silver, sau đó hệ thống mở rộng vocabulary bằng X-NER-style span mining, tạo pseudo-label bằng gazetteer và fine-tune GLiNER để chạy local.

## Mục tiêu

Input chính của project là sentence corpus đã được tách từ các văn bản pháp luật:

```text
../data/preprocessed/sentences/
```

Output chính là model GLiNER, gazetteer mở rộng và kết quả dự đoán entity trên toàn corpus:

```text
data/models/gliner_traffic_ner/final_model/
data/preprocessed/expanded_gazetteer/
data/preprocessed/gliner_predictions_th070/
```

## Cấu trúc thư mục

```text
ner_finetuning/
  llm_bootstrap_openai/      # chạy LLM để tạo seed entity ban đầu
  seed_vocabulary/           # gom entity mention bootstrap thành vocabulary seed
  gazetteer_building/        # tạo và match gazetteer lên corpus
  gazetteer_pruning/         # lọc alias quá chung, nhiều nghĩa hoặc dễ gây nhiễu
  xner_span_mining/          # mở rộng candidate bằng X-NER-style span mining
  gliner_finetuning/         # tạo train/dev/test, train, predict, evaluate GLiNER
  ner_evaluation/            # đánh giá kết hợp GLiNER + gazetteer
  data/
    preprocessed/
    models/
    benchmark/
    review/
    archive/
```

Tên thư mục và script được đặt theo nhiệm vụ thay vì theo số bước cũ. Các tên cũ như `entity_vocab_v1`, `gazetteers_v2`, `gliner_train_v2` hoặc `legal_gliner_v2` chỉ nên xem là tên lịch sử, không dùng cho pipeline hiện tại.

## Luồng xử lý

```text
llm_bootstrap_openai
  -> bootstrap_llm_entities
  -> seed_vocabulary
  -> seed_gazetteer
  -> pruned_seed_gazetteer
  -> xner_candidate_entities
  -> expanded_gazetteer
  -> gazetteer_pseudo_labels
  -> gliner_training_data
  -> gliner_traffic_ner
  -> gliner_predictions_th070
```

## Nguồn seed ban đầu

Seed ban đầu được tạo bởi module:

```text
llm_bootstrap_openai/run_llm_bootstrap.py
```

Artifact seed hiện có nằm ở:

```text
data/preprocessed/bootstrap_llm_entities/all_entity_mentions.jsonl
```

File này đến từ một lần chạy LLM-assisted NER trên một phần sentence corpus. Nó chỉ dùng để khởi tạo vocabulary, không phải phương pháp inference cuối cùng. Lần bootstrap hiện có có thống kê:

| Chỉ số | Giá trị |
| --- | ---: |
| Gói văn bản đã quét | 12 |
| Câu đầu vào đã xem | 6,482 |
| Câu đã annotate | 4,662 |
| Câu có entity | 3,729 |
| Entity mention | 8,095 |
| Label | ACTOR, BEHAVIOR, CONDITION, DOCUMENT, INFRASTRUCTURE, VEHICLE, VEHICLE_CONDITION_OR_EQUIPMENT |

## Thống kê artifact hiện tại

| Bước | Thư mục output | Thống kê chính |
| --- | --- | --- |
| Bootstrap LLM seed | `data/preprocessed/bootstrap_llm_entities` | 8,095 mentions, 3,729 câu có entity |
| Gom vocabulary seed | `data/preprocessed/seed_vocabulary` | 3,404 surface forms, 622 accept, 2,782 review, 221 conflict |
| Tạo seed gazetteer | `data/preprocessed/seed_gazetteer` | 119 accepted aliases, 119 canonical entities |
| Prune gazetteer | `data/preprocessed/pruned_seed_gazetteer` | 116 aliases output, 101 kept, 15 downweighted, 3 rejected |
| Candidate expansion | `data/preprocessed/xner_candidate_entities` | 1,770 clean reviewed candidates sau khi lọc trùng và conflict |
| Expanded gazetteer | `data/preprocessed/expanded_gazetteer` | 116 base aliases + 1,770 candidates, output 1,830 aliases |
| Pseudo-label corpus | `data/preprocessed/gazetteer_pseudo_labels` | 66,308 câu, 53,560 câu có entity, 84,784 direct entities |
| GLiNER train data | `data/preprocessed/gliner_training_data` | 40,981 rows: 33,746 train, 4,638 dev, 2,597 test |
| GLiNER model | `data/models/gliner_traffic_ner` | base `urchade/gliner_medium-v2.1`, 3,000 steps, batch 16 |
| GLiNER prediction | `data/preprocessed/gliner_predictions_th070` | threshold 0.70, 66,308 câu, 41,606 câu có entity, 80,510 entities |

Phân bố entity trong train data:

| Label | Số entity |
| --- | ---: |
| ACTOR | 13,430 |
| BEHAVIOR | 5,907 |
| CONDITION | 5,419 |
| DOCUMENT | 10,156 |
| INFRASTRUCTURE | 10,854 |
| VEHICLE | 7,830 |
| VEHICLE_CONDITION_OR_EQUIPMENT | 11,575 |

Kết quả đánh giá hiện có:

| Tập đánh giá | Precision | Recall | F1 | Ghi chú |
| --- | ---: | ---: | ---: | --- |
| Silver test | 0.619 | 0.604 | 0.612 | test split từ pseudo-label |
| Gemini reviewed benchmark | 0.585 | 0.351 | 0.439 | 72 case gold đã làm sạch |

## Chạy lại pipeline

Các lệnh dưới đây giả định đang đứng tại thư mục `ner_finetuning`. Ví dụ dùng cú pháp Linux/macOS. Với PowerShell có thể đổi `\` thành backtick hoặc viết thành một dòng.

### 1. Tạo bootstrap seed bằng LLM

Module này dùng OpenRouter/OpenAI-compatible API. API key được đọc từ `.env` ở thư mục hiện tại hoặc repo root. Các biến được hỗ trợ:

```text
OPEN_ROUTER_API=...
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```

Smoke test:

```bash
python llm_bootstrap_openai/run_llm_bootstrap.py \
  --sentences-root ../data/preprocessed/sentences/12_2025_TTBCA \
  --output data/preprocessed/bootstrap_llm_entities_smoke \
  --quality-tier silver \
  --silver-model openai/gpt-4.1-mini \
  --limit 2 \
  --batch-size 1 \
  --all-sentences \
  --no-resume
```

Chạy tạo seed:

```bash
python llm_bootstrap_openai/run_llm_bootstrap.py \
  --sentences-root ../data/preprocessed/sentences \
  --output data/preprocessed/bootstrap_llm_entities \
  --quality-tier silver \
  --silver-model openai/gpt-4.1-mini \
  --batch-size 8
```

Nếu model hoặc router không hỗ trợ strict JSON schema, thêm `--no-json-schema`.

Output:

```text
data/preprocessed/bootstrap_llm_entities/
  all_entity_mentions.jsonl
  entity_summary.json
  <PACKAGE_ID>/sentence_entities.jsonl
  <PACKAGE_ID>/entity_mentions.jsonl
```

### 2. Gom seed vocabulary

```bash
python seed_vocabulary/build_seed_vocabulary.py \
  --entity-mentions data/preprocessed/bootstrap_llm_entities/all_entity_mentions.jsonl \
  --output data/preprocessed/seed_vocabulary
```

Output quan trọng:

```text
data/preprocessed/seed_vocabulary/reviewed_surface_forms.csv
data/preprocessed/seed_vocabulary/vocab_summary.json
```

Sau bước này cần review `reviewed_surface_forms.csv`. Rule đang dùng cho seed chất lượng cao là `status = accept`, `conflict = false` và `count > 7`.

### 3. Tạo seed gazetteer

```bash
python gazetteer_building/build_seed_gazetteer.py \
  --reviewed data/preprocessed/seed_vocabulary/reviewed_surface_forms.csv \
  --output data/preprocessed/seed_gazetteer \
  --min-count 8
```

### 4. Prune gazetteer

```bash
python gazetteer_pruning/prune_gazetteer.py \
  --gazetteer-root data/preprocessed/seed_gazetteer \
  --output data/preprocessed/pruned_seed_gazetteer
```

Bước này loại các alias quá chung, alias có nguy cơ match sai và downweight các surface form nhiều nghĩa.

### 5. Mở rộng candidate bằng X-NER-style mining

```bash
python xner_span_mining/mine_entity_spans.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/pruned_seed_gazetteer \
  --output data/preprocessed/xner_candidate_entities \
  --quality-preset max \
  --embedding-model BAAI/bge-m3 \
  --batch-size 256 \
  --device cuda
```

Nếu chỉ kiểm tra logic trên máy không có GPU, có thể thêm `--skip-scoring` hoặc giảm `--max-sentences`.

Output cần review:

```text
data/preprocessed/xner_candidate_entities/reviewed_candidates.csv
data/preprocessed/xner_candidate_entities/candidate_review_summary.json
```

### 6. Build expanded gazetteer

```bash
python xner_span_mining/build_expanded_gazetteer.py \
  --base-gazetteer-root data/preprocessed/pruned_seed_gazetteer \
  --reviewed-mined-csv data/preprocessed/xner_candidate_entities/reviewed_candidates.csv \
  --output data/preprocessed/expanded_gazetteer
```

Output này là gazetteer chính cho các bước sau.

### 7. Match expanded gazetteer lên corpus

```bash
python gazetteer_building/match_gazetteer_to_corpus.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/expanded_gazetteer \
  --output data/preprocessed/gazetteer_pseudo_labels
```

Output là các file `sentence_entities.jsonl` theo từng gói văn bản, dùng làm pseudo-label cho GLiNER.

### 8. Tạo train/dev/test cho GLiNER

```bash
python gliner_finetuning/prepare_gliner_dataset.py \
  --entities-root data/preprocessed/gazetteer_pseudo_labels \
  --output data/preprocessed/gliner_training_data \
  --negative-ratio 0.35
```

Output:

```text
data/preprocessed/gliner_training_data/train.json
data/preprocessed/gliner_training_data/dev.json
data/preprocessed/gliner_training_data/test.json
data/preprocessed/gliner_training_data/dataset_summary.json
```

### 9. Fine-tune GLiNER

```bash
python gliner_finetuning/train_gliner_model.py \
  --train-file data/preprocessed/gliner_training_data/train.json \
  --dev-file data/preprocessed/gliner_training_data/dev.json \
  --output-dir data/models/gliner_traffic_ner \
  --base-model urchade/gliner_medium-v2.1 \
  --steps 3000 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --lr 5e-6 \
  --others-lr 1e-5 \
  --device cuda
```

Mặc định script dùng bf16 nếu chạy CUDA. Nếu GPU không hỗ trợ bf16, thêm `--no-bf16`.

### 10. Predict toàn corpus

```bash
python gliner_finetuning/predict_entities.py \
  --input-root data/preprocessed/gazetteer_pseudo_labels \
  --model-dir data/models/gliner_traffic_ner/final_model \
  --output data/preprocessed/gliner_predictions_th070 \
  --threshold 0.70 \
  --device cuda \
  --batch-size 32
```

### 11. Đánh giá model

Đánh giá trên silver test:

```bash
python gliner_finetuning/evaluate_gliner_model.py \
  --model-dir data/models/gliner_traffic_ner/final_model \
  --test-file data/preprocessed/gliner_training_data/test.json \
  --output data/models/gliner_traffic_ner/eval_test_silver_th_0.70.json \
  --threshold 0.70 \
  --device cuda \
  --batch-size 32
```

Đánh giá hybrid GLiNER + gazetteer:

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

Benchmark gold đã làm sạch nằm ở:

```text
data/benchmark/ner_gold_benchmark/ner_benchmark_gemini_clean.json
```

## Ý nghĩa các thư mục data

| Thư mục | Ý nghĩa |
| --- | --- |
| `data/preprocessed/bootstrap_llm_entities` | seed silver từ lần chạy LLM ban đầu |
| `data/preprocessed/seed_vocabulary` | vocabulary gom theo surface form để review |
| `data/preprocessed/seed_gazetteer` | gazetteer seed từ surface form đã accept |
| `data/preprocessed/pruned_seed_gazetteer` | gazetteer seed sau khi prune alias nguy hiểm |
| `data/preprocessed/xner_candidate_entities` | candidate mở rộng cần review |
| `data/preprocessed/expanded_gazetteer` | gazetteer sau khi hợp nhất seed và candidate đã review |
| `data/preprocessed/gazetteer_pseudo_labels` | pseudo-label trên corpus để train GLiNER |
| `data/preprocessed/gliner_training_data` | train/dev/test cho GLiNER |
| `data/preprocessed/gliner_predictions_th070` | dự đoán GLiNER trên corpus ở threshold 0.70 |
| `data/preprocessed/hybrid_entity_links` | entity links sau khi hợp nhất gazetteer và GLiNER |
| `data/models/gliner_traffic_ner` | checkpoint, final model và evaluation |
| `data/benchmark/ner_gold_benchmark` | benchmark gold nhỏ để kiểm tra chất lượng |
| `data/archive` | artifact cũ để đối chiếu, không phải output chính |

## Quy ước đặt tên

- Tên script là động từ + đối tượng, ví dụ `build_seed_vocabulary.py`, `mine_entity_spans.py`, `train_gliner_model.py`.
- Tên output nói rõ vai trò dữ liệu, ví dụ `expanded_gazetteer`, `gazetteer_pseudo_labels`, `gliner_training_data`.
- Không dùng hậu tố đánh số kiểu `v1`, `v2`, `step1`, `steps2_3` cho pipeline chính. Nếu cần versioning thì ghi trong metadata hoặc README.
- Các output thử nghiệm, đối chiếu hoặc không còn là pipeline chính đặt trong `data/archive`.
