# Legal Graph Builder Baseline

Build graph baseline cho traffic-bot từ các output đã tiền xử lý:

```text
data/preprocessed/passages/
data/preprocessed/entity_links_v1_pruned/
data/preprocessed/gazetteers_v1_pruned/
```

Module tạo graph ở mức document, passage và entity. Passage có thể là phần văn bản chính hoặc phần đính kèm đã được passage builder sinh ra.

## Node Types

```text
DOCUMENT
PASSAGE
ENTITY
```

## Edge Types

```text
DOCUMENT_CONTAINS_PASSAGE
PASSAGE_MENTIONS_ENTITY
PASSAGE_REFERS_TO_PASSAGE
PASSAGE_REFERS_TO_DOCUMENT
```

`PASSAGE_REFERS_TO_PASSAGE` và `PASSAGE_REFERS_TO_DOCUMENT` được lấy best-effort từ `outgoing_refs` trong passage. Builder chỉ tạo cạnh khi tham chiếu đã `resolved` và target có node tương ứng trong graph.

## Entity Edge Modes

```text
strong = link match_mode=keep
weak   = link match_mode=downweight
```

Weight:

```text
strong edge weight = 1.0
weak edge weight   = graph_weight, tối đa 0.5
reference weight   = score/confidence từ resolver, fallback 1.0
```

## Run

Chạy đầy đủ cả strong và weak entity links:

```powershell
python build_graph.py `
  --passages-root "../data/preprocessed/passages" `
  --entity-links-root "../data/preprocessed/entity_links_v1_pruned" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/legal_graph_v1"
```

Chạy baseline sạch hơn, chỉ dùng strong entity links:

```powershell
python build_graph.py `
  --passages-root "../data/preprocessed/passages" `
  --entity-links-root "../data/preprocessed/entity_links_v1_pruned" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/legal_graph_v1_strong" `
  --strong-only
```

Nếu muốn bỏ toàn bộ reference edges:

```powershell
python build_graph.py `
  --passages-root "../data/preprocessed/passages" `
  --entity-links-root "../data/preprocessed/entity_links_v1_pruned" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1_pruned" `
  --output "../data/preprocessed/legal_graph_v1_no_refs" `
  --no-reference-edges
```

## Output

```text
data/preprocessed/legal_graph_v1/
  nodes.jsonl
  edges.jsonl
  document_nodes.jsonl
  passage_nodes.jsonl
  entity_nodes.jsonl
  document_edges.jsonl
  mention_edges.jsonl
  reference_edges.jsonl
  skipped_reference_edges.jsonl
  passage_to_entities.json
  entity_to_passages.json
  graph_summary.json
```

`skipped_reference_edges.jsonl` dùng để audit những tham chiếu không được đưa vào graph. Các lý do thường gặp:

```text
status_not_resolved       # unresolved/ambiguous/non_reference từ resolver
missing_passage_node      # target đã resolved nhưng passage builder chưa có node tương ứng
missing_document_node     # target document không có trong corpus graph
missing_target_id         # ref không có target id
```

## Dùng Cho Retrieval Baseline

- `entity_to_passages.json`: tra từ entity canonical/query anchor sang passages.
- `passage_to_entities.json`: expansion hoặc reranking bằng entity trong passage.
- `mention_edges.jsonl`: build graph DB hoặc NetworkX.
- `reference_edges.jsonl`: expansion theo tham chiếu pháp lý đã resolved.
- `nodes.jsonl` + `edges.jsonl`: import vào Neo4j/NetworkX/GraphML nếu cần.

## Debug

Sau khi build, xem:

```text
graph_summary.json
```

Cần kiểm tra nhanh:

```text
by_node_type
by_edge_type
by_mention_edge_mode
by_entity_label
references.skipped_by_reason
```

Nếu weak entity edges gây nhiễu, chạy lại với `--strong-only`.
