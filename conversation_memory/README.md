# conversation_memory module

Module memory nhẹ cho hệ thống RAG pháp luật giao thông Việt Nam. Mục tiêu là cải tiến memory/rewrite mà **không động vào RAG chính**.

## Ý tưởng

Module này đứng trước RAG hiện tại:

```text
user_question + session_id
  -> ConversationMemoryManager.prepare()
  -> Conversation Resolver / QueryPlan: relation, route, memory_score, primary_query, retrieval_queries
  -> RAG hiện tại
  -> answer + retrieved_passages
  -> ConversationMemoryManager.commit()
```

Memory chỉ dùng để hiểu câu hỏi nối tiếp, không được dùng làm căn cứ pháp lý. Với lượt hỏi mơ hồ, `resolver.py` có thể gọi LLM để xuất JSON theo schema chặt, gồm `relation`, `use_memory`, `standalone_question`, `retrieval_query`, `route` và `confidence`.

## Cài đặt

Copy thư mục `conversation_memory/` vào project của bạn.

Không cần dependency ngoài Python standard library.

## Dùng nhanh

```python
from conversation_memory import ConversationMemoryManager

memory_manager = ConversationMemoryManager(llm_call=your_llm_call)

plan = memory_manager.prepare(session_id="demo", question=user_question)

retrieved = retriever.search(plan.primary_query)

answer = generate_answer(
    question=plan.answer_question,
    retrieved_passages=retrieved,
    conversation_context=plan.answer_memory_context,
)

memory_manager.commit(
    session_id="demo",
    plan=plan,
    answer=answer,
    retrieved_passages=retrieved,
)
```

## Nếu chưa muốn dùng LLM rewrite

```python
memory_manager = ConversationMemoryManager(llm_call=None)
```

Khi đó module vẫn tự xử lý:

- intent
- route
- memory_score
- use_memory
- topic_action

và giữ nguyên query gốc.

## QueryPlan quan trọng

```python
plan.intent              # definition / effectivity / penalty / comparison / roadmap / ...
plan.route               # traffic_law / effectivity_index / normal_chat
plan.use_memory          # True/False
plan.memory_score        # 0.0 - 1.0
plan.new_topic_score     # 0.0 - 1.0
plan.primary_query       # query sạch để đưa vào RAG hiện tại
plan.retrieval_queries   # raw + standalone, nếu sau này muốn multi-query
plan.answer_memory_context
plan.debug
```

## Rule quan trọng

- `definition` không được route sang `effectivity_index`.
- Câu hỏi chủ đề mới không được kéo memory cũ vào query.
- Rewrite không được tự thêm số Điều/Khoản/Điểm nếu chưa có căn cứ.
- Memory state chỉ lưu JSON ngắn, không lưu prompt/rewrite dài.
- State chỉ update từ retrieved passages + final answer.

## Chạy test

Từ thư mục chứa package:

```bash
pip install pytest
pytest tests/
```

## Gợi ý sửa prompt answer

Trong prompt sinh answer, tách memory khỏi evidence:

```text
Memory hội thoại, chỉ dùng để hiểu đại từ/câu nối tiếp, không dùng làm căn cứ pháp lý:
{conversation_context}

Căn cứ pháp lý được truy xuất:
{retrieved_context}

Câu hỏi:
{question}

Yêu cầu:
- Chỉ trả lời dựa trên căn cứ pháp lý được truy xuất.
- Không dùng memory hội thoại như nguồn luật.
- Nếu memory mâu thuẫn với căn cứ truy xuất, bỏ qua memory.
```

## Case đã test

- Chủ đề mới sau topic nồng độ cồn.
- Follow-up định nghĩa: “vùng phát thải thấp nghĩa là gì”.
- Follow-up hiệu lực thật: “văn bản này có hiệu lực từ ngày nào”.
- Follow-up mức phạt: “nếu vượt quá thì có bị phạt không”.
- Câu hỏi mới nhưng có memory cũ: “Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu”.
- Correction: “ý tôi là xe máy”.
