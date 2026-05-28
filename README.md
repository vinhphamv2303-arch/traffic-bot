# Entities extraction

```bash
cd ner_finetuning
```

```bash
python gazetteer_building/match_gazetteer_to_corpus.py \
  --sentences-root ../data/preprocessed/sentences \
  --gazetteer-root data/preprocessed/expanded_gazetteer \
  --output ../data/entities/gazetteer_matches
```






[//]: # (# Retrieve Hybrid V2)

[//]: # ()
[//]: # (README này chỉ hướng dẫn chạy retriever trên index đã build sẵn.)

[//]: # ()
[//]: # (## 1. Chạy nhanh trên CPU)

[//]: # ()
[//]: # (Dùng bản BM25 + graph + reference, không cần dense embedding khi retrieve.)

[//]: # ()
[//]: # (```powershell)

[//]: # (conda activate kltn)

[//]: # ()
[//]: # (python .\legal_linearrag_retriever\retrieve.py `)

[//]: # (  --index-dir .\data\preprocessed\linearrag_index_hybrid_v2_bm25_graph_v2 `)

[//]: # (  --gazetteer-root .\data\preprocessed\gazetteers_v2 `)

[//]: # (  --query "Thời gian lái xe liên tục tối đa của người lái xe là bao nhiêu?" `)

[//]: # (  --top-k 20 `)

[//]: # (  --dense-weight 0.0 `)

[//]: # (  --bm25-weight 0.25 `)

[//]: # (  --graph-weight 0.15 `)

[//]: # (  --reference-weight 0.60)

[//]: # (```)

[//]: # ()
[//]: # (Nên dùng cách này khi chạy local không có GPU.)

[//]: # ()
[//]: # (## 2. Chạy MiniLM trên CPU)

[//]: # ()
[//]: # (Dùng bản dense nhẹ đã build với `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.)

[//]: # ()
[//]: # (```powershell)

[//]: # (conda activate kltn)

[//]: # ()
[//]: # (python .\legal_linearrag_retriever\retrieve.py `)

[//]: # (  --index-dir .\data\preprocessed\linearrag_index_hybrid_v2_minilm_cpu `)

[//]: # (  --gazetteer-root .\data\preprocessed\gazetteers_v2 `)

[//]: # (  --query "Thời gian lái xe liên tục tối đa của người lái xe là bao nhiêu?" `)

[//]: # (  --top-k 20 `)

[//]: # (  --dense-weight 0.20 `)

[//]: # (  --bm25-weight 0.30 `)

[//]: # (  --graph-weight 0.20 `)

[//]: # (  --reference-weight 0.30)

[//]: # (```)

[//]: # ()
[//]: # (Đây là lựa chọn cân bằng nếu muốn có dense retrieval nhưng vẫn chạy được trên CPU.)

[//]: # ()
[//]: # (## 3. Chạy full dense hybrid)

[//]: # ()
[//]: # (Dùng bản full dense đã build với `BAAI/bge-m3`.)

[//]: # ()
[//]: # (```powershell)

[//]: # (conda activate kltn)

[//]: # ()
[//]: # (python .\legal_linearrag_retriever\retrieve.py `)

[//]: # (  --index-dir .\data\preprocessed\linearrag_index_hybrid_v2_full_dense `)

[//]: # (  --gazetteer-root .\data\preprocessed\gazetteers_v2 `)

[//]: # (  --query "Thời gian lái xe liên tục tối đa của người lái xe là bao nhiêu?" `)

[//]: # (  --top-k 20 `)

[//]: # (  --dense-weight 0.25 `)

[//]: # (  --bm25-weight 0.25 `)

[//]: # (  --graph-weight 0.20 `)

[//]: # (  --reference-weight 0.30)

[//]: # (```)

[//]: # ()
[//]: # (Cách này cần cài `sentence-transformers` trong môi trường chạy:)

[//]: # ()
[//]: # (```powershell)

[//]: # (python -m pip install -r .\legal_linearrag_retriever\requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (Nếu chạy CPU, lần đầu load và encode query bằng `BAAI/bge-m3` có thể chậm. Nên dùng GPU/cloud cho bản full dense.)

[//]: # ()
[//]: # (## Output)

[//]: # ()
[//]: # (Kết quả trả về là JSON. Các passage liên quan nằm trong trường `results`.)

[//]: # ()
[//]: # (Các trường quan trọng trong mỗi result:)

[//]: # ()
[//]: # (- `passage_id`: ID của passage.)

[//]: # (- `score`: điểm tổng hợp.)

[//]: # (- `score_components`: điểm thành phần `dense`, `bm25`, `graph`, `reference`.)

[//]: # (- `document_number`: số hiệu văn bản.)

[//]: # (- `path_text`: đường dẫn pháp lý trong văn bản.)

[//]: # (- `text`: nội dung passage.)
