# legal_entity_vocab — Step 1: aggregate vocabulary

Bước này gom `all_entity_mentions.jsonl` thành danh sách surface forms để review.

## Input

```text
data/preprocessed/entities_llm_v2/all_entity_mentions.jsonl
```

## Chạy

```powershell
python aggregate_vocab.py `
  --entity-mentions "../data/preprocessed/entities_llm_v2/all_entity_mentions.jsonl" `
  --output "../data/preprocessed/entity_vocab_v1"
```

## Output

```text
data/preprocessed/entity_vocab_v1/
  surface_forms.jsonl
  surface_summary.csv
  label_conflicts.csv
  reviewed_surface_forms.csv
  vocab_summary.json
```

## Bạn cần review file nào?

Review/sửa file:

```text
reviewed_surface_forms.csv
```

Các cột quan trọng:

```text
surface
label
count
status
canonical
label_final
reason
example_text
```

Giá trị `status`:

```text
accept  = giữ surface form
reject  = bỏ surface form
review  = cần bạn xem lại
```

Khi review:
- đổi `status` thành `accept` hoặc `reject`
- sửa `canonical` nếu muốn gom alias
- sửa `label_final` nếu nhãn sai

Ví dụ:

```csv
surface,label,count,status,canonical,label_final
xe máy,VEHICLE,45,accept,"xe mô tô, xe gắn máy",VEHICLE
phương tiện,VEHICLE,80,reject,,
vượt đèn đỏ,BEHAVIOR,37,accept,vượt đèn đỏ,BEHAVIOR
```

Sau khi review xong, bước tiếp theo sẽ là `build_gazetteer`.
