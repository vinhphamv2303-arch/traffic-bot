# Legal LinearRAG Retriever Baseline

Retriever lai cho traffic-bot, dùng graph baseline và gazetteer đã prune:

```text
data/preprocessed/legal_graph_v1/
data/preprocessed/gazetteers_v1_pruned/
```

Luồng chính:

```text
query
  -> local entity activation
     - exact gazetteer match, hỗ trợ có dấu và không dấu
     - optional semantic query -> entity similarity nếu có embeddings
  -> global passage aggregation
     - BM25 passage retrieval
     - optional dense passage retrieval
     - entity -> passage graph propagation
     - reference expansion nhẹ qua PASSAGE_REFERS_TO_PASSAGE
  -> hybrid final score
```

## Install

BM25 + graph chỉ cần stdlib và `numpy`.

```bash
pip install numpy
```

Nếu build dense embeddings:

```bash
pip install sentence-transformers
```

Model mặc định là `BAAI/bge-m3`. Nếu máy yếu, dùng:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Build Index

Nhanh, không dùng dense embeddings:

```powershell
python build_index.py `
  --graph-root "../data/preprocessed/legal_graph_v1" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/linearrag_index_v1_bm25_graph" `
  --skip-embeddings
```

Đầy đủ BM25 + graph + dense:

```powershell
python build_index.py `
  --graph-root "../data/preprocessed/legal_graph_v1" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/linearrag_index_v1" `
  --embedding-model "BAAI/bge-m3" `
  --embedding-batch-size 32
```

## Retrieve

```powershell
python retrieve.py `
  --index-dir "../data/preprocessed/linearrag_index_v1_bm25_graph" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --query "không có giấy phép lái xe bị xử phạt thế nào" `
  --top-k 10
```

Query không dấu cũng match gazetteer:

```powershell
python retrieve.py `
  --index-dir "../data/preprocessed/linearrag_index_v1_bm25_graph" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --query "khong co giay phep lai xe bi xu phat the nao" `
  --top-k 10
```

## Batch Retrieve

```powershell
python batch_retrieve.py `
  --index-dir "../data/preprocessed/linearrag_index_v1_bm25_graph" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --queries "../data/eval/queries.txt" `
  --output "../data/preprocessed/retrieval_runs/linearrag_v1_results.jsonl" `
  --top-k 10
```

`queries` có thể là `.txt` mỗi dòng một query hoặc `.jsonl` có field `query`/`question`.

## Scoring

Default:

```text
dense:     0.35
BM25:      0.25
graph:     0.35
reference: 0.05
```

Khi một candidate chỉ đến từ graph mà không có BM25/dense support, graph score bị nhân `graph_only_penalty`, mặc định `0.65`. Cơ chế này giảm trường hợp entity rộng như `xe mô tô` đẩy passage không khớp hành vi lên quá cao.

Chỉnh weight:

```powershell
python retrieve.py `
  --index-dir "../data/preprocessed/linearrag_index_v1_bm25_graph" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --query "..." `
  --dense-weight 0.30 `
  --bm25-weight 0.30 `
  --graph-weight 0.35 `
  --reference-weight 0.05 `
  --graph-only-penalty 0.65
```

## Output

Mỗi result có dạng:

```json
{
  "passage_id": "...",
  "score": 0.92,
  "score_components": {
    "dense": 0.8,
    "bm25": 0.6,
    "graph": 1.0,
    "reference": 0.0
  },
  "document_number": "...",
  "path_text": "...",
  "text": "...",
  "entities": []
}
```

Debug query gồm:

```json
{
  "activated_entities": [],
  "entity_evidence": [],
  "debug": {
    "dense_candidates": 0,
    "bm25_candidates": 300,
    "graph_candidates": 120,
    "reference_candidates": 50,
    "final_candidates": 520
  }
}
```

## Notes

- `PASSAGE_REFERS_TO_DOCUMENT` không được đưa vào `passage_neighbors`, vì retrieval hiện trả về passage, không trả về document node.
- Nếu `activated_entities` rỗng với query đáng lẽ có anchor, cần bổ sung alias/gazetteer.
- Nếu entity đúng nhưng graph candidate sai, cần kiểm tra entity links hoặc giảm `graph_weight`.
- Nếu BM25 đúng nhưng graph làm nhiễu, tăng `bm25_weight` hoặc giảm `graph_weight`/`graph_only_penalty`.
