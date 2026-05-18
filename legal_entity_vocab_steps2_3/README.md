# legal_entity_vocab — Steps 2 and 3

## Step 2: build gazetteer

```powershell
python build_gazetteer.py `
  --reviewed "../data/preprocessed/entity_vocab_v1/reviewed_surface_forms.csv" `
  --output "../data/preprocessed/gazetteers_v1" `
  --min-count 8
```

By default, accepted rows with `label_conflict=True` are skipped so a surface form
does not map to multiple labels. The default `--min-count 8` keeps only rows with
`count > 7`. Use `--include-conflicts` only for diagnostics.

Output:

```text
data/preprocessed/gazetteers_v1/
  aliases.jsonl
  canonical_entities.jsonl
  skipped_conflicts.csv
  match_blocklist.txt
  gazetteer_terms.csv
  gazetteer_summary.json
  behavior.txt
  vehicle.txt
  actor.txt
  infrastructure.txt
  document.txt
  vehicle_condition_or_equipment.txt
  condition.txt
```

Optional: create `match_blocklist.txt` in the gazetteer directory, one term per
line, to skip noisy canonical/surface forms during matching. Example:

```text
sử dụng
cầu
hàng hóa
```

## Step 3: match all sentences

```powershell
python gazetteer_matcher.py `
  --sentences-root "../data/preprocessed/sentences" `
  --gazetteer-root "../data/preprocessed/gazetteers_v1" `
  --output "../data/preprocessed/entity_links_v1"
```

Output:

```text
data/preprocessed/entity_links_v1/
  all_sentence_entity_links.jsonl
  entity_link_summary.json

data/preprocessed/entity_links_v1/<PACKAGE_ID>/
  sentence_entity_links.jsonl
  sentences_with_entity_links.jsonl
  entity_link_summary.json
```
