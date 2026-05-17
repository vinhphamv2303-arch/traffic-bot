# Legal LLM NER

Module dùng LLM local/API để trích xuất **semantic entities** từ `sentences.jsonl`.

Quan trọng: module này **không trích xuất** các tham chiếu cấu trúc như Điều/khoản/điểm/Phụ lục/Mẫu số/số hiệu văn bản. Các thứ đó thuộc `reference_resolver`.

## Entity labels

- `VIOLATION_OR_BEHAVIOR`
- `VEHICLE_TYPE`
- `VEHICLE_IDENTIFIER`
- `REGULATED_SUBJECT`
- `DOCUMENT_OR_PERMIT`
- `LICENSE_CLASS`
- `SANCTION`
- `FINE_AMOUNT`
- `FEE_OR_PAYMENT`
- `AUTHORITY`
- `FACILITY_OR_INFRASTRUCTURE`
- `EQUIPMENT_OR_SYSTEM`
- `PLAN_OR_PROJECT`
- `LOCATION_OR_ROAD_CONTEXT`
- `TRAFFIC_SIGNAL_OR_SIGN`
- `TECHNICAL_REQUIREMENT`
- `PROCEDURE`
- `CONDITION`
- `CONSEQUENCE_OR_HARM`
- `TIME_OR_DURATION`

Legacy aliases:

- `TRAFFIC_BEHAVIOR` -> `VIOLATION_OR_BEHAVIOR`
- `ROAD_USER` -> `REGULATED_SUBJECT`

## Chạy với Ollama

Cài model:

```bash
ollama pull qwen3:8b
```

Chạy NER:

```bash
python run_llm_ner.py \
  -i "../data/preprocessed/sentences" \
  -o "../data/preprocessed/entities" \
  --provider ollama \
  --model qwen3:8b \
  --batch-size 8
```

Nếu Ollama chạy qua HTTP proxy/RunPod, truyền thêm endpoint:

```bash
python run_llm_ner.py \
  -i "../data/preprocessed/sentences" \
  -o "../data/preprocessed/entities" \
  --provider ollama \
  --model qwen3:8b \
  --endpoint "https://your-ollama-proxy.example.com" \
  --batch-size 8
```

Test nhanh 100 câu:

```bash
python run_llm_ner.py \
  -i "../data/preprocessed/sentences" \
  -o "../data/preprocessed/entities_test" \
  --provider ollama \
  --model qwen3:8b \
  --batch-size 4 \
  --limit 100
```

`--limit` giới hạn theo tổng số câu khi input là thư mục root chứa nhiều package.

## Chạy với OpenAI-compatible local server

Ví dụ vLLM/LM Studio/llama.cpp server:

```bash
python run_llm_ner.py \
  -i "../data/preprocessed/sentences" \
  -o "../data/preprocessed/entities" \
  --provider openai_compatible \
  --api-base "http://localhost:8000/v1" \
  --model "Qwen/Qwen3-8B"
```

## Dry-run/mock

```bash
python run_llm_ner.py \
  -i "../data/preprocessed/sentences" \
  -o "../data/preprocessed/entities_mock" \
  --provider mock \
  --limit 20
```

## Output

```text
data/preprocessed/entities/
  all_entity_mentions.jsonl
  entity_summary.json

data/preprocessed/entities/<PACKAGE_ID>/
  sentence_entities.jsonl
  entity_mentions.jsonl
  annotation_silver.jsonl
  entity_summary.json
```

### `sentence_entities.jsonl`

Mỗi dòng là một câu + danh sách entity:

```json
{
  "sentence_id": "...",
  "text": "...",
  "entities": [
    {
      "entity_id": "ent_...",
      "text": "xe mô tô",
      "label": "VEHICLE_TYPE",
      "start": 10,
      "end": 16,
      "confidence": 0.88,
      "alignment_status": "aligned"
    }
  ],
  "review_status": "silver"
}
```

### `entity_mentions.jsonl`

Một dòng cho mỗi entity mention, tiện để build graph/index.

### `annotation_silver.jsonl`

Dữ liệu silver để review và chuyển sang BIO/span dataset fine-tune sau này.

## Build review/fine-tune dataset

Gop cac output probe thanh mot bo silver can review:

```bash
python build_review_dataset.py \
  -i "../data/preprocessed/entities_probe_2000" "../data/preprocessed/entities_large_probe" \
     "../data/preprocessed/entities_diverse_probe" "../data/preprocessed/entities_ollama_probe" \
  -o "../data/preprocessed/entities_silver_review" \
  --license-class-supplement-limit 12 \
  --regex-supplement-sentence-root "../data/preprocessed/sentences" \
  --regex-supplement-limit-per-label 12
```

`--license-class-supplement-limit` chi them mot so nho span `LICENSE_CLASS` bang regex va gan
`quality_flags=regex_supplement`, nen can review truoc khi dung de fine-tune.

`--regex-supplement-limit-per-label` bo sung mot so nho span cho cac nhan thua du lieu nhu
`FINE_AMOUNT`, `TIME_OR_DURATION`, `LOCATION_OR_ROAD_CONTEXT`, `TRAFFIC_SIGNAL_OR_SIGN`,
`CONSEQUENCE_OR_HARM`, va `PLAN_OR_PROJECT`. Cac span nay cung duoc gan
`quality_flags=regex_supplement`.
