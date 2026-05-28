# Demo hỏi đáp pháp luật giao thông

Thư mục này chứa giao diện Streamlit để demo pipeline chính: người dùng nhập câu hỏi, hệ thống tự phân loại câu hỏi, rewrite truy vấn nếu cần, truy xuất passage từ index và sinh câu trả lời bằng LLM.

## Chuẩn bị

Chạy trong env `kltn`:

```powershell
conda activate kltn
pip install streamlit
```

API key có thể đặt trong file `.env` ở repo root:

```text
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

Giao diện cũng cho phép nhập API key trực tiếp ở sidebar.

## Chạy demo

Từ repo root:

```powershell
conda activate kltn
streamlit run interface\app.py
```

Backend mặc định là `OpenAI API` với model `gpt-4o-mini`. Nếu dùng OpenRouter, chọn `OpenRouter API` và dùng model dạng `openai/gpt-4o-mini`.

## Cơ chế xử lý câu hỏi

Trước khi retrieval, hệ thống gọi LLM ở chế độ route/rewrite:

- Nếu câu hỏi thuộc phạm vi pháp luật giao thông đường bộ Việt Nam, LLM viết lại câu hỏi sang thuật ngữ gần với văn bản pháp luật. Ví dụ `say rượu` được rewrite thành `trong máu hoặc hơi thở có nồng độ cồn`, `vượt đèn đỏ` thành `không chấp hành hiệu lệnh của đèn tín hiệu giao thông`.
- Nếu câu hỏi là chào hỏi hoặc không liên quan tới phạm vi RAG, hệ thống trả lời như chatbot thông thường và không truy xuất corpus.

Có thể tắt bước này bằng checkbox `Bật route/rewrite câu hỏi`.

## Artifact đang dùng

```text
data/retrieval/index_bm25_graph
data/retrieval/index_bge_m3_hybrid
data/retrieval/index_minilm_hybrid
ner_finetuning/data/preprocessed/expanded_gazetteer
retrieval_pipelines_builder/legal_linearrag_retriever/retrieve.py
```
