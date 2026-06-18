# Traffic Bot - RAG hỏi đáp pháp luật giao thông đường bộ Việt Nam

Traffic Bot là hệ thống RAG tôi xây dựng để hỏi đáp trên tập văn bản pháp luật giao thông đường bộ Việt Nam. Dự án tập trung vào việc biến văn bản pháp luật dạng DOC/DOCX thành dữ liệu có cấu trúc, xây dựng chỉ mục truy xuất theo passage, kết hợp BM25, dense embedding, graph pháp lý, gazetteer/entity và sinh câu trả lời có căn cứ từ LLM.

Mục tiêu của project là tạo một pipeline có thể kiểm soát được từ dữ liệu thô đến demo hỏi đáp cuối cùng, thay vì chỉ gọi LLM trực tiếp. Câu trả lời được giới hạn trong các passage đã truy xuất và có citation để kiểm tra lại nguồn.

## Tính năng chính

- Parse văn bản pháp luật và phụ lục từ DOC/DOCX sang JSONL có cấu trúc.
- Trích xuất hiệu lực văn bản, tham chiếu pháp lý và các đơn vị điều/khoản/điểm.
- Xây dựng passage theo hướng phù hợp cho RAG pháp luật.
- Tách câu, tạo gazetteer, fine-tune GLiNER và kết hợp NER với gazetteer.
- Xây dựng graph pháp lý gồm document, passage, entity và reference edge.
- Truy xuất hybrid bằng BM25, dense embedding, graph propagation và reference expansion.
- Sinh câu trả lời bằng OpenAI API, OpenRouter API hoặc model local Hugging Face.
- Demo bằng Streamlit với ba chế độ: hỏi đáp, kiểm tra retriever và kiểm tra NER.
- Đánh giá retrieval và answer generation trên benchmark nội bộ.

## Kiến trúc tổng quát

```text
DOC/DOCX pháp luật
  -> legal_parser_modular
  -> effectivity_processor
  -> reference_resolver
  -> legal_passage_builder
  -> legal_sentence_splitter
  -> NER/gazetteer/GLiNER
  -> legal graph
  -> retrieval index
  -> answer_generation
  -> Streamlit interface
```

## Cấu trúc thư mục

```text
answer_generation/              Sinh câu trả lời, query routing, rewrite và gọi LLM
conversation_memory/             Bộ nhớ hội thoại cho câu hỏi nhiều lượt
data_preprocessing/              Pipeline tiền xử lý văn bản pháp luật
evaluation/                      Script chạy benchmark và tính metric
interface/                       Giao diện Streamlit
ner_finetuning/                  Pipeline gazetteer, GLiNER và đánh giá NER
retrieval_pipelines_builder/     Build graph, build index và chạy retriever
utils/                           Script hỗ trợ chuẩn hóa/tải dữ liệu
requirements.txt                 Phụ thuộc tối thiểu để chạy demo/API
```

Thư mục `data/`, `ner_finetuning/data/` và file `.env` không được commit lên GitHub. Đây là nơi chứa dữ liệu, model, index và API key.

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Windows, Linux hoặc macOS. Nếu cần chuyển file `.doc` legacy sang `.docx` trên Windows, project có hỗ trợ `pywin32`.
- API key của OpenAI hoặc OpenRouter nếu chạy chế độ API.
- GPU là tùy chọn, nhưng nên có nếu build embedding lớn hoặc fine-tune GLiNER.

## Cài đặt

Tạo môi trường ảo và cài phụ thuộc tối thiểu:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu chạy retriever có dense embedding hoặc build lại index:

```powershell
python -m pip install -r retrieval_pipelines_builder\legal_linearrag_retriever\requirements.txt
```

Nếu chạy GLiNER/NER:

```powershell
python -m pip install -r ner_finetuning\gliner_finetuning\requirements.txt
```

## Cấu hình API key

Tạo file `.env` ở thư mục root của project:

```text
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Các biến tùy chọn:

```text
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Giao diện Streamlit cũng cho phép nhập API key trực tiếp ở sidebar, nên không bắt buộc phải lưu key vào `.env` khi chỉ demo nhanh.

## Artifact cần có để chạy demo

Do dữ liệu và index khá lớn, repo GitHub chỉ nên chứa mã nguồn. Để chạy demo đầy đủ, cần chuẩn bị các artifact sau:

```text
data/retrieval/index_minilm_hybrid/
data/retrieval/index_bge_m3_hybrid/
ner_finetuning/data/preprocessed/expanded_gazetteer/
```

Nếu dùng chế độ kiểm tra GLiNER trong giao diện, cần thêm:

```text
ner_finetuning/data/models/gliner_traffic_ner/final_model/
```

Nếu hỏi các câu liên quan hiệu lực văn bản, nên có thêm:

```text
data/preprocessed/effectivity/
```

Artifact hiện tại của tôi được build từ khoảng 100 nghìn passage và hơn 11 nghìn entity. Khi clone repo mới, cần tải artifact đã build sẵn hoặc chạy lại pipeline ở phần bên dưới.

## Chạy giao diện Streamlit

Từ thư mục root:

```powershell
streamlit run interface\app.py
```

Các pipeline có sẵn trong giao diện:

| Pipeline | Mục đích |
| --- | --- |
| `hybrid_minilm` | Dense MiniLM + BM25 + graph + reference |
| `hybrid_bge_m3` | Dense BGE-M3 + BM25 + graph + reference |
| `bm25` | Baseline BM25 |
| `dense_minilm` | Baseline dense MiniLM |
| `dense_bge_m3` | Baseline dense BGE-M3 |

Backend mặc định là OpenAI API với model `gpt-4o-mini`. Có thể chuyển sang OpenRouter hoặc local Hugging Face trong sidebar.

## Chạy hỏi đáp bằng CLI

Ví dụ chạy một câu hỏi với OpenAI API và pipeline MiniLM:

```powershell
python answer_generation\run_llm.py `
  --query "Người lái xe ô tô vượt đèn đỏ bị xử phạt như thế nào?" `
  --mode openai `
  --pipeline minilm `
  --top-k 5
```

Chạy với OpenRouter:

```powershell
python answer_generation\run_llm.py `
  --query "Xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?" `
  --mode openrouter `
  --model qwen/qwen-2.5-7b-instruct `
  --pipeline bge_m3
```

Kiểm tra retrieval và prompt mà không gọi LLM:

```powershell
python answer_generation\run_llm.py `
  --query "Thời gian lái xe liên tục tối đa là bao lâu?" `
  --pipeline minilm `
  --dry-run
```

## Chạy retriever trực tiếp

```powershell
python retrieval_pipelines_builder\legal_linearrag_retriever\retrieve.py `
  --index-dir data\retrieval\index_minilm_hybrid `
  --gazetteer-root ner_finetuning\data\preprocessed\expanded_gazetteer `
  --query "Thời gian lái xe liên tục tối đa là bao lâu?" `
  --top-k 5 `
  --candidate-k 300 `
  --dense-weight 0.15 `
  --bm25-weight 0.25 `
  --graph-weight 0.15 `
  --reference-weight 0.30
```

Output là JSON, trong đó `results` chứa các passage được truy xuất cùng điểm tổng hợp và điểm thành phần.

## Tái tạo dữ liệu và index

Các lệnh dưới đây chạy từ thư mục root. Tùy kích thước dữ liệu và model embedding, quá trình build có thể tốn nhiều thời gian.

### 1. Parse văn bản pháp luật

```powershell
python data_preprocessing\legal_parser_modular\parse_package.py `
  -i data\dataset `
  -o data\preprocessed\parsed
```

### 2. Trích xuất hiệu lực

```powershell
python data_preprocessing\effectivity_processor\extract_effectivity.py `
  -i data\preprocessed\parsed `
  -o data\preprocessed\effectivity
```

### 3. Resolve tham chiếu pháp lý

```powershell
python data_preprocessing\reference_resolver\resolve_references.py `
  -i data\preprocessed\parsed `
  -o data\preprocessed\resolved_references
```

### 4. Build passage

```powershell
python data_preprocessing\legal_passage_builder\build_passages.py `
  -i data\preprocessed\parsed `
  -o data\preprocessed\passages `
  --effectivity-root data\preprocessed\effectivity `
  --resolved-refs-root data\preprocessed\resolved_references
```

### 5. Tách câu cho NER

```powershell
python data_preprocessing\legal_sentence_splitter\split_sentences.py `
  -i data\preprocessed\passages `
  -o data\preprocessed\sentences
```

### 6. Build hoặc cập nhật NER artifact

Pipeline NER đầy đủ nằm trong `ner_finetuning/README.md`. Artifact quan trọng nhất cho retriever là:

```text
ner_finetuning/data/preprocessed/expanded_gazetteer/
```

Nếu đã có expanded gazetteer, có thể match lại lên corpus:

```powershell
python ner_finetuning\gazetteer_building\match_gazetteer_to_corpus.py `
  --sentences-root data\preprocessed\sentences `
  --gazetteer-root ner_finetuning\data\preprocessed\expanded_gazetteer `
  --output ner_finetuning\data\preprocessed\gazetteer_pseudo_labels
```

### 7. Build retrieval index

Khi đã có graph tương thích retriever ở `data/retrieval/retriever_graph`, build index MiniLM:

```powershell
python retrieval_pipelines_builder\legal_linearrag_retriever\build_index.py `
  --graph-root data\retrieval\retriever_graph `
  --gazetteer-root ner_finetuning\data\preprocessed\expanded_gazetteer `
  --output data\retrieval\index_minilm_hybrid `
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Build index BGE-M3:

```powershell
python retrieval_pipelines_builder\legal_linearrag_retriever\build_index.py `
  --graph-root data\retrieval\retriever_graph `
  --gazetteer-root ner_finetuning\data\preprocessed\expanded_gazetteer `
  --output data\retrieval\index_bge_m3_hybrid `
  --embedding-model BAAI/bge-m3
```

## Benchmark

Chạy benchmark retrieval và answer generation:

```powershell
python evaluation\run_final_rag_benchmark.py `
  --mode openai `
  --pipelines naive_bm25 naive_dense bge_m3 `
  --models gpt-4o-mini `
  --top-k 5
```

Kết quả được ghi vào:

```text
data/benchmark/traffic_rag_final_retrieval_answer_benchmark_v1/
```

Có thể dùng `--dry-run`, `--limit`, `--skip-generation` hoặc `--skip-eval` để kiểm tra nhanh từng phần.
