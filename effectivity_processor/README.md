\
# Effectivity Processor

Module hậu xử lý cho parser pháp luật.

Input chuẩn hiện tại là output package parser:

```text
data/preprocessed/parsed/<PACKAGE_ID>/main/units.jsonl
```

Nếu truyền `data/preprocessed/parsed`, module sẽ tự tìm đúng `main/units.jsonl`
của từng package. Các `attachments/<slug>/units.jsonl` không được scan mặc định,
vì ngày hiệu lực là metadata của văn bản chính.

Module vẫn hỗ trợ layout parser cũ:

```text
data/preprocessed/parsed/<DOC_ID>/units.jsonl
```

Output:

```text
data/preprocessed/effectivity/<PACKAGE_ID>/effectivity_events.jsonl
data/preprocessed/effectivity/<PACKAGE_ID>/effectivity_summary.json
data/preprocessed/effectivity/effectivity_index.csv
data/preprocessed/effectivity/effectivity_unit_overrides.csv
data/preprocessed/effectivity/effectivity_unresolved.csv
```

`effectivity_index.csv` là bảng trạng thái tổng hợp cho toàn bộ input, mỗi dòng là một văn bản:

```text
document_id,document_number,effective_from,effective_to,...
```

Nếu chưa xác định được ngày bắt đầu hoặc ngày kết thúc hiệu lực thì giá trị là `null`.
Các event bãi bỏ toàn văn bản sẽ cập nhật `effective_to` của văn bản bị bãi bỏ.

`effectivity_unit_overrides.csv` lưu các trường hợp điều/khoản/điểm/phụ lục có ngày bắt đầu hiệu lực riêng, khác hoặc chi tiết hơn ngày hiệu lực chung của văn bản:

```text
document_id,document_number,target_selector_raw,target_article,target_clause,target_point,target_appendix,effective_from,...
```

`effectivity_unresolved.csv` lưu các trường hợp có hiệu lực riêng nhưng ngày chỉ được dẫn gián tiếp, ví dụ "có hiệu lực theo quy định của pháp luật ...". Các dòng này có `date_inference` và `raw_text`, nhưng không có ngày ISO cụ thể.

## Chạy

```bash
python extract_effectivity.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/effectivity"
```

Quét tất cả unit:

```bash
python extract_effectivity.py \
  -i "../data/preprocessed/parsed" \
  -o "../data/preprocessed/effectivity" \
  --scan-all-units
```

## Event types

```text
effective_from
repeal_document
repeal_unit
repeal_unspecified
```

Các event chỉ là candidate. Sau này dùng resolver để gắn với document/chunk thật.
