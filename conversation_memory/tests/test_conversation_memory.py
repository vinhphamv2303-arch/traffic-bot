from conversation_memory.models import ConversationState, MemoryDocument, MemoryEntity
from conversation_memory.planner import build_plan


def make_state(topic, entities=None, docs=None):
    return ConversationState(
        session_id="test",
        active_topic=topic,
        last_user_question=topic,
        last_standalone_question=topic,
        focus_entities=[MemoryEntity(text=e) for e in (entities or [])],
        focus_docs=[MemoryDocument(doc_id=d) for d in (docs or [])],
        turn_count=1,
    )


def test_new_topic_after_alcohol():
    state = make_state(
        topic="So sánh quy định nồng độ cồn giữa Luật 2008 và Luật 2024",
        entities=["nồng độ cồn"],
        docs=["23/2008/QH12", "36/2024/QH15"],
    )

    plan = build_plan(
        session_id="test",
        question="Lộ trình vùng phát thải thấp ở Hà Nội gồm những giai đoạn nào?",
        state=state,
    )

    assert plan.use_memory is False
    assert plan.intent == "roadmap"
    assert plan.route == "traffic_law"
    assert "nồng độ cồn" not in plan.primary_query.lower()


def test_definition_followup_low_emission_zone():
    state = make_state(
        topic="Lộ trình vùng phát thải thấp ở Hà Nội",
        entities=["vùng phát thải thấp", "Hà Nội"],
        docs=["57/2025/NQ-HDND"],
    )

    plan = build_plan(
        session_id="test",
        question="vùng phát thải thấp nghĩa là gì",
        state=state,
    )

    assert plan.use_memory is True
    assert plan.intent == "definition"
    assert plan.route == "traffic_law"


def test_effectivity_followup():
    state = make_state(
        topic="Lộ trình vùng phát thải thấp ở Hà Nội",
        entities=["vùng phát thải thấp"],
        docs=["57/2025/NQ-HDND"],
    )

    plan = build_plan(
        session_id="test",
        question="văn bản này có hiệu lực từ ngày nào?",
        state=state,
    )

    assert plan.use_memory is True
    assert plan.intent == "effectivity"
    assert plan.route == "effectivity_index"


def test_penalty_followup():
    state = make_state(
        topic="Thời gian lái xe liên tục tối đa",
        entities=["thời gian lái xe liên tục"],
        docs=["36/2024/QH15"],
    )

    plan = build_plan(
        session_id="test",
        question="nếu vượt quá thì có bị phạt không?",
        state=state,
    )

    assert plan.use_memory is True
    assert plan.intent == "penalty"


def test_specific_new_question_should_not_use_memory():
    state = make_state(
        topic="Vùng phát thải thấp tại Hà Nội",
        entities=["vùng phát thải thấp", "Hà Nội"],
        docs=["57/2025/NQ-HDND"],
    )

    plan = build_plan(
        session_id="test",
        question="Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?",
        state=state,
    )

    assert plan.use_memory is False
    assert plan.intent == "penalty"
    assert "vùng phát thải thấp" not in plan.primary_query.lower()


def test_correction_followup():
    state = make_state(
        topic="Ô tô vượt đèn đỏ bị phạt bao nhiêu?",
        entities=["ô tô", "vượt đèn đỏ"],
        docs=["168/2024/ND-CP"],
    )

    plan = build_plan(
        session_id="test",
        question="ý tôi là xe máy",
        state=state,
    )

    assert plan.use_memory is True
    assert plan.topic_action == "clarify_previous"


def test_answer_generation_adapter_rewrites_vehicle_followup_cleanly():
    from answer_generation.conversation_memory import prepare_memory_plan

    state = make_state(
        topic="Phạt người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông như thế nào?",
        entities=[
            "người điều khiển xe ô tô",
            "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        ],
        docs=["168/2024/ND-CP"],
    )

    plan = prepare_memory_plan("vậy còn xe máy thì sao", state)

    assert plan.use_memory is True
    assert "xe máy" in plan.primary_query
    assert "xe mô tô" in plan.primary_query
    assert "xe ô tô" not in plan.primary_query
    assert "Thực thể liên quan" not in plan.primary_query
    assert "Văn bản liên quan" not in plan.primary_query


def test_llm_conversation_resolver_uses_clean_retrieval_query():
    import json

    from answer_generation.conversation_memory import resolve_query_with_memory

    state = make_state(
        topic="Phạt người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông như thế nào?",
        entities=["người điều khiển xe ô tô", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"],
        docs=["168/2024/NĐ-CP"],
    )
    state.last_answer_summary = "Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với xe ô tô."

    def fake_llm_call(_messages):
        return json.dumps(
            {
                "relation": "replace_constraint",
                "use_memory": True,
                "reason": "đổi phương tiện",
                "current_focus": "xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
                "dropped_answered_content": ["mức phạt xe ô tô đã được trả lời"],
                "changed_constraints": {"vehicle": {"from": "xe ô tô", "to": "xe máy"}},
                "standalone_question": "Người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào?",
                "retrieval_query": "người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào 168/2024/NĐ-CP",
                "route": "traffic_law",
                "confidence": 0.93,
            },
            ensure_ascii=False,
        )

    query, memory_context, resolution = resolve_query_with_memory(
        "vậy còn xe máy thì sao",
        state,
        llm_call=fake_llm_call,
        enable_llm=True,
    )

    assert resolution["accepted"] is True
    assert resolution["relation"] == "replace_constraint"
    assert query == "người điều khiển xe máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị xử phạt như thế nào 168/2024/NĐ-CP"
    assert "Thực thể liên quan" not in query
    assert "xe ô tô" not in query
    assert "Trọng tâm lượt hiện tại" in memory_context
