# Legal X-NER Span Miner

Bước nối tiếp **legal vocab step 4 prune**.

Mục tiêu:
- Chưa train NER ngay.
- Mở rộng entity vocabulary trước bằng X-NER-style span mining.
- Không dùng LLM.
- Dùng GPU với embedding model để score candidates.
- Output là `mined_candidates.csv` để review.
- Sau review mới build `gazetteers_v2`, rồi mới tạo train data và train local extractor.

## Pipeline đúng

```text
gazetteers_v1_pruned
  ↓
build seeds
  ↓
generate span candidates from sentences + path_text
  ↓
score candidates against seeds
  ↓
review mined_candidates.csv
  ↓
build gazetteers_v2
  ↓
match corpus again
  ↓
prepare local NER train v2
  ↓
train local extractor
```

## Cài đặt

```bash
pip install sentence-transformers numpy torch
```

Nếu dùng `BAAI/bge-m3`, cần tải model khá lớn. Có GPU thì nên dùng.

## 1. Chạy full mining

```powershell
python run_xner_mining.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/xner_mined_entities_v1" `
  --embedding-model "BAAI/bge-m3" `
  --batch-size 64 `
  --min-surface-count 2 `
  --min-score 0.45
```

Test nhanh 5000 sentences:

```powershell
python run_xner_mining.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/xner_mined_entities_v1_test" `
  --embedding-model "BAAI/bge-m3" `
  --max-sentences 5000 `
  --batch-size 64
```

Nếu chỉ muốn kiểm tra local mà không tải embedding model / không dùng GPU:

```powershell
python run_xner_mining.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/xner_mined_entities_v1_test" `
  --max-sentences 5000 `
  --skip-scoring
```

Khi chạy, chương trình sẽ log từng bước bằng prefix `[xner]`, `[xner:candidates]`, `[xner:scoring]`.

## 2. Output

```text
data/preprocessed/xner_mined_entities_v1/
  seeds.jsonl
  seed_summary.json
  span_candidates.jsonl
  candidate_summary.json
  mined_candidates.jsonl
  mined_candidates.csv
  mining_summary.json
```

Bạn review file:

```text
mined_candidates.csv
```

Các cột quan trọng:

```text
surface
label
canonical
count
score
status
best_seed
example_text
example_path
```

Giá trị status:
- `accept_candidate`: điểm cao, count đủ, có thể giữ nếu nhìn ổn
- `review`: cần xem

Khi review:
- đổi status thành `accept` nếu muốn giữ
- đổi status thành `reject` nếu bỏ
- sửa `canonical` nếu muốn gom alias

## 3. Build gazetteer v2 sau review

Sau khi review `mined_candidates.csv`, chạy:

```powershell
python build_gazetteer_v2.py `
  --base-gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --reviewed-mined-csv "../data/preprocessed/xner_mined_entities_v1/mined_candidates.csv" `
  --output "../data/preprocessed/gazetteers_v2"
```

Mặc định bước build chỉ nhận các dòng đã review với `status` là `accept`, `accepted`, hoặc `keep`.
Các dòng còn giữ `accept_candidate`/`auto_accept` sẽ không được đưa vào gazetteer v2 trừ khi chạy thêm:

```powershell
--accept-auto-candidates
```

Output:

```text
data/preprocessed/gazetteers_v2/
  aliases.jsonl
  canonical_entities.jsonl
  gazetteer_terms.csv
  gazetteer_summary.json
```

## 4. Sau đó làm gì?

Dùng lại module `legal_entity_vocab_steps2_3` hoặc matcher hiện có để match lại corpus với `gazetteers_v2`:

```text
gazetteers_v2
  ↓
entity_links_v2
  ↓
legal_graph_v2
  ↓
local_ner_train_v2
  ↓
train model
```

## Lưu ý quan trọng

Module này là X-NER-style adapted, không phải full X-NER original:
- Có seed theo nhãn.
- Có candidate span mining từ raw corpus.
- Có scoring bằng semantic similarity với seed/context.
- Không tạo BIO labels tự động ngay.
- Không train model ngay.

Lý do: cần kiểm soát chất lượng vocabulary trước, tránh train model từ pseudo-label nhiễu.


## Max-quality preset

Bản này đã chỉnh default theo hướng ưu tiên recall/chất lượng:

```text
device: cuda
embedding_model: BAAI/bge-m3
batch_size: 256
max_ngram: 14
min_surface_count: 1
min_score: 0.35
top_k_seeds_per_candidate: 10
```

Chạy full corpus max-quality:

```powershell
python run_xner_mining.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/xner_mined_entities_v1" `
  --quality-preset max `
  --device cuda `
  --embedding-model "BAAI/bge-m3" `
  --batch-size 256
```

Nếu GPU còn dư VRAM, tăng:

```powershell
--batch-size 512
```

Nếu bị CUDA out of memory, giảm về:

```powershell
--batch-size 128
```
