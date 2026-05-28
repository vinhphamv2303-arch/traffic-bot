# legal_parser_modular

Module parser văn bản pháp luật Việt Nam từ `.docx`/`.doc` sang dữ liệu cấu trúc.

Module có 3 entrypoint chính:

- `parse_main_body.py`: parse phần thân văn bản chính.
- `parse_attachments.py`: parse riêng phụ lục/mẫu/QCVN đính kèm.
- `parse_package.py`: parse một package gồm văn bản chính và các file đính kèm.

## Cấu trúc thư mục

```text
legal_parser_modular/
  parse_main_body.py          # chạy body parser
  parse_attachments.py        # chạy attachment parser
  parse_package.py            # chạy package parser
  requirements.txt
  README.md

  legal_parser/
    __init__.py

    body/
      parser.py               # LegalBodyParser
      cli.py                  # CLI cho văn bản chính
      docx_iter.py
      mentions.py
      tables.py

    attachments/
      cli.py                  # CLI cho phụ lục/mẫu/QCVN
      base.py
      classifier.py
      common.py
      appendix_parser.py
      form_parser.py
      qcvn_parser.py

    package/
      parser.py               # LegalPackageParser
      cli.py                  # CLI cho package

    common/
      models.py
      utils.py
      file_classifier.py
      doc_converter.py
      logging_utils.py

```

## Cài đặt

```bash
cd legal_parser_modular
pip install -r requirements.txt
```

Parser đọc `.docx` bằng `python-docx`/`mammoth`. Với file `.doc`, CLI sẽ tự chuyển sang `.docx` trước khi parse.

## Chạy parse văn bản chính

Parse một file hoặc một folder:

```bash
python parse_main_body.py \
  -i "../data/dataset/119_2024_NDCP" \
  -o "../data/preprocessed/parsed"
```

Quét đệ quy toàn bộ dataset, mỗi folder lấy văn bản chính và bỏ qua phụ lục/mẫu:

```bash
python parse_main_body.py \
  -i "../data/dataset" \
  -o "../data/preprocessed/parsed" \
  --recursive
```

Nếu muốn body parser parse cả file đính kèm như văn bản thường:

```bash
python parse_main_body.py \
  -i "../data/dataset/119_2024_NDCP" \
  -o "../data/preprocessed/parsed_body_all" \
  --include-attachments
```

## Chạy parse file đính kèm

Parse các file phụ lục/mẫu/QCVN trong một folder:

```bash
python parse_attachments.py \
  -i "../data/dataset/119_2024_NDCP" \
  -o "../data/preprocessed/parsed_attachments/119_2024_NDCP" \
  --package-id "119_2024_NDCP"
```

Nếu tên file đính kèm không có dấu hiệu rõ như `Phu luc`, `Mau`, `QCVN`, dùng thêm:

```bash
python parse_attachments.py \
  -i "../data/dataset/119_2024_NDCP" \
  -o "../data/preprocessed/parsed_attachments/119_2024_NDCP" \
  --package-id "119_2024_NDCP" \
  --include-unknown
```

## Chạy parse package

Đây là lệnh nên dùng cho dataset chuẩn. Parser sẽ tự chọn văn bản chính, parse phụ lục/mẫu/QCVN, rồi gộp output.

Parse toàn bộ dataset:

```bash
python parse_package.py \
  -i "../data/dataset" \
  -o "../data/preprocessed/parsed"
```

Parse một package:

```bash
python parse_package.py \
  -i "../data/dataset/119_2024_NDCP" \
  -o "../data/preprocessed/parsed" \
  --single-package
```

## Chuyển `.doc` sang `.docx`

Mặc định các CLI sẽ chuyển `.doc` legacy sang `.docx` trước khi parse:

- Trên Windows ưu tiên Microsoft Word qua `pywin32`.
- Nếu có LibreOffice/`soffice`, dùng làm fallback.
- Khi `.docx` được tạo hợp lệ, file `.doc` gốc sẽ bị xoá.

Tuỳ chọn:

```bash
--keep-doc        # giữ lại file .doc sau khi chuyển thành công
--no-convert-doc  # tắt chuyển đổi .doc
--log-file NAME   # đổi tên file log trong output folder
```

## Output body parser

Mỗi văn bản chính tạo một folder theo số hiệu văn bản:

```text
<OUTPUT>/<DOC_NUMBER>/
  <DOC_NUMBER>.tree.json
  units.jsonl
  tables.jsonl
  ref_mentions.jsonl
  amendment_mentions.jsonl
```

`ref_mentions.jsonl` và `amendment_mentions.jsonl` là danh sách mention/candidate, chưa phải kết quả resolve cuối cùng.

## Output attachment parser

```text
<OUTPUT>/
  attachments_inventory.json
  all_units.jsonl
  all_tables.jsonl
  all_ref_mentions.jsonl

  <attachment_slug>/
    attachment.json
    units.jsonl
    tables.jsonl
    table_rows.jsonl
    form_fields.jsonl
    ref_mentions.jsonl
```

## Output package parser

```text
<OUTPUT>/<PACKAGE_ID>/
  package_inventory.json

  main/
    tree.json
    units.jsonl
    tables.jsonl
    ref_mentions.jsonl
    amendment_mentions.jsonl

  attachments/
    <attachment_slug>/
      attachment.json
      units.jsonl
      tables.jsonl
      table_rows.jsonl
      form_fields.jsonl
      ref_mentions.jsonl

  all_units.jsonl
  all_tables.jsonl
  all_ref_mentions.jsonl
```

Downstream nên đọc các file gộp `all_units.jsonl`, `all_tables.jsonl`, `all_ref_mentions.jsonl` khi xử lý theo package.

### Metadata liên kết mẫu số - phụ lục

Khi parse package, parser chạy thêm bước `appendix_form_linker`. Bước này đọc các câu trong văn bản chính kiểu `Mẫu số 01, Mẫu số 02 Phụ lục II`, sau đó ghi metadata vào `package_inventory.json`:

- `attachments[].parent_appendix_labels`: danh sách phụ lục cha suy luận được cho file mẫu.
- `attachments[].parent_appendix_inferences`: bằng chứng, confidence và các ứng viên đã so sánh.
- `inferred_appendix_groups`: nhóm phụ lục cha được tạo từ các file mẫu con, dùng cho resolver khi dataset không có file phụ lục cha riêng.
- `appendix_form_linking`: thống kê số cặp đã link và các quyết định.

Resolver ưu tiên metadata này trước khi dùng scoring theo context/title.

## Dùng như package

```python
from data_preprocessing.legal_parser_modular.legal_parser.body import LegalBodyParser
from data_preprocessing.legal_parser_modular.legal_parser.package import LegalPackageParser
from data_preprocessing.legal_parser_modular.legal_parser.common import ParserConfig
```
