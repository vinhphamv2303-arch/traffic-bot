# Legal LLM NER v2 - OpenRouter/OpenAI Compatible

Module trich xuat semantic entities voi 7 nhan retrieval-oriented:

```text
BEHAVIOR
VEHICLE
ACTOR
INFRASTRUCTURE
DOCUMENT
VEHICLE_CONDITION_OR_EQUIPMENT
CONDITION
```

LLM chi nhan `id + text + context` va chi tra `text + label`. Code tu validate, align offset, reject reference-like/generic spans, roi gan metadata tu `sentences.jsonl`.

## API key

Module tu load `.env` o repo root. Cac ten bien duoc ho tro:

```text
OPEN_ROUTER_API=...
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```

Mac dinh CLI dung OpenRouter:

```text
https://openrouter.ai/api/v1
```

## Test nhanh

Tu repo root:

```powershell
python legal_llm_ner_v2_openai/run_openai_ner_v2.py `
  -i "data/preprocessed/sentences/12_2025_TTBCA" `
  -o "data/preprocessed/entities_llm_v2_smoke" `
  --quality-tier silver `
  --silver-model "openai/gpt-4.1-mini" `
  --limit 2 `
  --batch-size 1 `
  --all-sentences `
  --no-resume
```

Neu model/router khong ho tro JSON schema strict, them:

```powershell
--no-json-schema
```

## Chay toan corpus

```powershell
python legal_llm_ner_v2_openai/run_openai_ner_v2.py `
  -i "data/preprocessed/sentences" `
  -o "data/preprocessed/entities_llm_v2" `
  --quality-tier silver `
  --silver-model "openai/gpt-4.1-mini" `
  --batch-size 8
```

`--limit` la gioi han tong so cau cua ca run, khong phai moi package.

## Output

```text
data/preprocessed/entities_llm_v2/
  all_entity_mentions.jsonl
  entity_summary.json

data/preprocessed/entities_llm_v2/<PACKAGE_ID>/
  selected_candidates.jsonl
  sentence_entities.jsonl
  entity_mentions.jsonl
  entity_summary.json
```
