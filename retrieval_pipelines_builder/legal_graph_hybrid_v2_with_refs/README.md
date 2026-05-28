# legal_graph_hybrid_v2_with_refs

Build graph từ:

```text
entities_gazetteer_v2 + entities_gliner_v2_th070 + resolved_references
→ entity_links_hybrid_v2
→ legal_graph_hybrid_v2
```

## 0. Nếu chưa có resolved references

Repo đã có module `reference_resolver`. Chạy trước:

```bash
cd reference_resolver

python resolve_references.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/resolved_references"
```

Output cần cho graph:

```text
../data/preprocessed/resolved_references/all_resolved_references.jsonl
```

## 1. Build hybrid entity links

Không cần GPU.

```bash
cd legal_graph_hybrid_v2_with_refs

python build_entity_links_hybrid_v2.py \
  --gazetteer-entities-root "../data/preprocessed/entities_gazetteer_v2" \
  --gliner-entities-root "../data/preprocessed/entities_gliner_v2_th070" \
  --output "../data/preprocessed/entity_links_hybrid_v2"
```

## 2. Build graph có reference edges

Không cần GPU.

```bash
python build_graph_hybrid_v2.py \
  --entity-links-dir "../data/preprocessed/entity_links_hybrid_v2" \
  --references-file "../data/preprocessed/resolved_references/all_resolved_references.jsonl" \
  --output "../data/graphs/legal_graph_hybrid_v2" \
  --min-cooccur-weight 0.05 \
  --max-entity-degree-for-cooccur 2500
```

Nếu muốn build graph không có reference edge:

```bash
python build_graph_hybrid_v2.py \
  --entity-links-dir "../data/preprocessed/entity_links_hybrid_v2" \
  --output "../data/graphs/legal_graph_hybrid_v2_no_refs"
```

## 3. Graph schema

Nodes:
- `document`
- `passage`
- `entity`

Edges:
- `document --CONTAINS--> passage`
- `passage --HAS_ENTITY--> entity`
- `entity --COOCCURS--> entity`
- `passage --REFERENCES--> passage`

Reference edge lấy từ `resolved_references`:

```text
source_unit_id      → source passage
selected_target_id  → target passage/legal unit
status == resolved  → được nối edge
```

Weight tham chiếu:

```text
point:          1.00 * confidence
clause:         0.90 * confidence
article:        0.80 * confidence
form:           0.75 * confidence
appendix:       0.70 * confidence
legal_document: 0.50 * confidence
```

## 4. Inspect

```bash
python inspect_graph_hybrid_v2.py \
  --graph-dir "../data/graphs/legal_graph_hybrid_v2" \
  --top-k 40
```

Gửi lại:

```text
../data/preprocessed/entity_links_hybrid_v2/entity_links_summary.json
../data/graphs/legal_graph_hybrid_v2/graph_summary.json
```
