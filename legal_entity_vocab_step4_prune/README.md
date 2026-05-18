# legal_entity_vocab — Step 4: prune gazetteer

Bước này xử lý các hub/generic surface như:

```text
sản xuất
vận chuyển
tài liệu
giấy phép
đường bộ
động cơ
nhãn hiệu
bánh xe
cán bộ
học viên
```

Không nhất thiết xoá hết. Module gán:

```text
match_mode = keep | downweight | reject
graph_weight = 1.0 | 0.5 | 0.25 | 0.0
```

## 1. Prune gazetteer

```powershell
python prune_gazetteer.py `
  --gazetteer-root "../data/preprocessed/gazetteers_v1" `
  --output "../data/preprocessed/gazetteers_v1_pruned"
```

If `match_blocklist.txt` exists in the input gazetteer directory, its terms are
rejected during pruning. This keeps manual blocklist decisions from Step 2/3.

Output:

```text
data/preprocessed/gazetteers_v1_pruned/
  aliases.jsonl
  canonical_entities.jsonl
  generic_hubs.jsonl
  rejected_aliases.jsonl
  gazetteer_terms_pruned.csv
  gazetteer_summary.json
```

## 2. Match lại toàn bộ sentences

```powershell
python gazetteer_matcher_pruned.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/entity_links_v1_pruned"
```

Output links có thêm:

```json
{
  "match_mode": "downweight",
  "graph_weight": 0.25,
  "prune_reason": "manual_downweight_generic_hub"
}
```

## Gợi ý dùng trong graph

- `keep`: dùng bình thường.
- `downweight`: vẫn tạo edge nhưng weight thấp, ví dụ `edge_weight *= graph_weight`.
- `reject`: mặc định không match.
