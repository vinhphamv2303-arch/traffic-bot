# Inference Demo

Module này phục vụ demo chương trình RAG chính: nhập câu hỏi pháp luật giao thông, truy xuất top-k passage từ index đã build, rồi dùng GPT-4o mini qua OpenAI API hoặc OpenRouter API để sinh câu trả lời.

## Chuẩn bị

Chạy trong env `kltn`:

```powershell
conda run -n kltn pip install streamlit openai
```

API key có thể đặt trong `.env` ở repo root.

Nếu dùng OpenAI:

```text
OPENAI_API_KEY=...
```

Nếu dùng OpenRouter:

```text
OPENROUTER_API_KEY=...
```

Biến cũ `OPEN_ROUTER_API=...` cũng được hỗ trợ. Hoặc nhập trực tiếp API key trong sidebar của giao diện.

## Chạy giao diện

Từ repo root:

```powershell
conda run -n kltn streamlit run inference\app.py
```

Mặc định app dùng pipeline `Hybrid CPU: BM25 + Graph + Reference` để demo nhanh và không cần load embedding model. Có thể chọn BGE-M3 hoặc MiniLM ở sidebar nếu muốn chạy pipeline dense.

Trong sidebar:

- Chọn `OpenAI API` và model `gpt-4o-mini` nếu dùng key OpenAI.
- Chọn `OpenRouter API` và model `openai/gpt-4o-mini` nếu dùng key OpenRouter.
- Bật `Mở rộng truy vấn pháp lý` để hệ thống tự rewrite/mở rộng câu hỏi đời thường trước retrieval.

## Query expansion

Demo có một tầng mở rộng truy vấn nhẹ trước retrieval. Mục đích là nối cách hỏi tự nhiên của người dùng với thuật ngữ pháp lý trong văn bản. Ví dụ:

```text
say rượu, uống rượu, rượu bia
  -> trong máu hoặc hơi thở có nồng độ cồn
  -> có nồng độ cồn trong máu hoặc hơi thở
  -> có sử dụng rượu bia

vượt đèn đỏ
  -> không chấp hành hiệu lệnh của đèn tín hiệu giao thông

không đội mũ
  -> không đội mũ bảo hiểm
```

Khi câu hỏi có dạng `đối với từng loại phương tiện`, hệ thống sinh thêm query theo nhóm phương tiện như `xe ô tô`, `xe mô tô xe gắn máy`, `xe máy chuyên dùng`, `xe đạp xe đạp điện`, sau đó merge kết quả retrieval. Cách này giúp các câu hỏi thiếu đúng cụm pháp lý, như `say rượu`, vẫn kéo được passage chứa `nồng độ cồn trong máu hoặc hơi thở`.

## Các artifact được dùng

```text
data/preprocessed/linearrag_index_hybrid_v2_bm25_graph_v2
data/preprocessed/linearrag_index_hybrid_v2_full_dense
data/preprocessed/linearrag_index_hybrid_v2_minilm_cpu
ner_finetuning/data/preprocessed/expanded_gazetteer
```
