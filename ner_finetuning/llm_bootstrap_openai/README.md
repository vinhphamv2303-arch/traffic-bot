# LLM Bootstrap OpenAI

Module này dùng OpenRouter/OpenAI-compatible API để tạo seed entity ban đầu cho pipeline NER. Đây là bước bootstrap dữ liệu silver, không phải mô hình inference cuối cùng.

Các nhãn được dùng:

```text
BEHAVIOR
VEHICLE
ACTOR
INFRASTRUCTURE
DOCUMENT
VEHICLE_CONDITION_OR_EQUIPMENT
CONDITION
```

## API key

Module tự tìm `.env` ở thư mục hiện tại hoặc các thư mục cha, nên có thể chạy từ repo root hoặc từ `ner_finetuning`. Các biến được hỗ trợ:

```text
OPEN_ROUTER_API=...
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```

Mặc định CLI dùng OpenRouter:

```text
https://openrouter.ai/api/v1
```

## Test nhanh

Từ thư mục `ner_finetuning`:

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

Nếu model/router không hỗ trợ JSON schema strict, thêm:

```bash
--no-json-schema
```

## Chạy tạo seed

```bash
python llm_bootstrap_openai/run_llm_bootstrap.py \
  --sentences-root ../data/preprocessed/sentences \
  --output data/preprocessed/bootstrap_llm_entities \
  --quality-tier silver \
  --silver-model openai/gpt-4.1-mini \
  --batch-size 8
```

`--limit` là giới hạn tổng số câu của cả run, không phải giới hạn mỗi package.

## Output

```text
data/preprocessed/bootstrap_llm_entities/
  all_entity_mentions.jsonl
  entity_summary.json

data/preprocessed/bootstrap_llm_entities/<PACKAGE_ID>/
  selected_candidates.jsonl
  sentence_entities.jsonl
  entity_mentions.jsonl
  entity_summary.json
```

File `all_entity_mentions.jsonl` là đầu vào cho bước `seed_vocabulary/build_seed_vocabulary.py`.
