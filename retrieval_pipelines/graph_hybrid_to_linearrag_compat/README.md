# graph_hybrid_to_linearrag_compat

Converter để dùng `legal_linearrag_retriever` cũ với graph hybrid mới.

Retriever cũ kỳ vọng graph gồm:

```text
passage_nodes.jsonl
entity_nodes.jsonl
mention_edges.jsonl
reference_edges.jsonl
```

Trong khi graph hybrid mới đang có:

```text
nodes.jsonl
edges.jsonl
```

Script này chuyển format mới sang format cũ.

## 1. Convert graph

```bash
python convert_graph_to_linearrag_compat.py \
  --graph-root "../data/graphs/legal_graph_hybrid_v2" \
  --output "../data/preprocessed/legal_graph_hybrid_v2_linearrag_compat"
```

## 2. Build LinearRAG index bằng retriever cũ

Không dùng dense, chạy nhanh:

```bash
cd ../legal_linearrag_retriever

python build_index.py \
  --graph-root "../data/preprocessed/legal_graph_hybrid_v2_linearrag_compat" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --output "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph" \
  --skip-embeddings
```

Có dense BGE-M3, cần GPU/CPU mạnh hơn:

```bash
python build_index.py \
  --graph-root "../data/preprocessed/legal_graph_hybrid_v2_linearrag_compat" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --output "../data/preprocessed/linearrag_index_hybrid_v2" \
  --embedding-model "BAAI/bge-m3" \
  --embedding-batch-size 32
```

## 3. Test retrieve

```bash
python retrieve.py \
  --index-dir "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --query "khong co giay phep lai xe bi xu phat the nao" \
  --top-k 10
```

Gợi ý thử thêm:

```bash
python retrieve.py \
  --index-dir "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --query "xe mo to khong doi mu bao hiem" \
  --top-k 10

python retrieve.py \
  --index-dir "../data/preprocessed/linearrag_index_hybrid_v2_bm25_graph" \
  --gazetteer-root "../data/preprocessed/gazetteers_v2" \
  --query "o to vuot den do bi phat bao nhieu" \
  --top-k 10
```
