# graph_hybrid_to_linearrag_compat_v2

Bản v2 sửa lỗi quan trọng: reference resolver thường dùng unit id không có `.passage`, trong khi retriever/index dùng passage id có `.passage`.

Ví dụ:

```text
source_unit_id: 158_2024_nd_cp.dieu_25.khoan_2.diem_b
passage_id:    158_2024_nd_cp.dieu_25.khoan_2.diem_b.passage
```

Nếu không remap, retrieval sẽ có `reference_candidates = 0`.

## 1. Convert lại graph

```bash
python convert_graph_to_linearrag_compat_v2.py \
  --graph-root "../data/graphs/legal_graph_hybrid_v2" \
  --output "../data/preprocessed/legal_graph_hybrid_v2_linearrag_compat_v2"
```

## 2. Build lại index

```bash
cd ../legal_linearrag_retriever

python build_index.py \
  --graph-root "../data/preprocessed/legal_graph_hybrid_v2_linearrag_compat_v2" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --output "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph_v2" \
  --skip-embeddings
```

## 3. Test lại query

```bash
python retrieve.py \
  --index-dir "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph_v2" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --query "Thời gian lái xe liên tục tối đa của người lái xe là bao nhiêu?" \
  --top-k 10
```

Kỳ vọng debug không còn:

```text
reference_candidates: 0
```

Nếu vẫn top 1 là passage dẫn chiếu từ Nghị định 158, thì vẫn chấp nhận được ở tầng retrieve nếu top-k có cả `Khoản 1 Điều 64`. Nhưng để QA trả lời đúng, reranker/answerer phải ưu tiên passage có giá trị trả lời trực tiếp, tức passage chứa “không quá 04 giờ”.
