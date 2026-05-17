# Reference Resolver v2

Rule-based resolver cho `all_ref_mentions.jsonl`, không dùng LLM.

## Sửa lỗi so với bản trước

1. `parse_selector()` anchor theo `span`/mention hiện tại, không lấy match đầu tiên trong context.
2. `khoản` chỉ resolve tới node khoản exact; không trả các điểm con cùng score.
3. Form resolver dùng numeric normalization (`Mẫu số 1` = `Mẫu số 01`) và boost theo context Phụ lục/title/source_file.
4. Relative references:
   - `Điều này`
   - `khoản này`
   - `điểm a`
   - `điểm b khoản này`
   suy từ `source_path_text`.
5. Normalize số hiệu văn bản không greedy: `35/2024/TT-BGTVT ngày...` -> `35/2024/TT-BGTVT`.

## Chạy

```bash
python resolve_references.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/resolved_references"
```

Một package:

```bash
python resolve_references.py \
  -i "../data/preprocessed/parsed/12_2025_TTBCA" \
  -o "../data/preprocessed/resolved_references"
```

## Output

```text
resolved_references/<PACKAGE_ID>/
  resolved_references.jsonl
  reference_resolution_summary.json

resolved_references/
  all_resolved_references.jsonl
  reference_resolution_summary.json
```
