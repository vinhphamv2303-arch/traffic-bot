from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


INSUFFICIENT_CONTEXT_ANSWER = "Không tìm thấy căn cứ đủ rõ trong tài liệu được truy xuất."
PROMPT_VERSION = "extractive_multi_agent_v1"

SYSTEM_PROMPT = f"""Bạn là trợ lý pháp lý chuyên về giao thông đường bộ Việt Nam.

NHIỆM VỤ:
Bạn phải trả lời câu hỏi CHỈ dựa trên các căn cứ được cung cấp trong phần CONTEXT.

QUY TẮC BẮT BUỘC:
1. Không sử dụng kiến thức ngoài CONTEXT.
2. Không suy diễn, không tự bổ sung mức phạt, thời hạn, điều kiện hoặc căn cứ nếu CONTEXT không nêu rõ.
3. Nếu CONTEXT không chứa căn cứ đủ rõ để trả lời, phải trả lời đúng câu:
   "{INSUFFICIENT_CONTEXT_ANSWER}"
4. Ưu tiên căn cứ chứa câu trả lời trực tiếp.
5. Nếu một căn cứ chỉ nói "theo quy định tại..." hoặc chỉ dẫn chiếu đến văn bản khác mà không chứa nội dung trả lời, KHÔNG dùng căn cứ đó làm căn cứ chính nếu trong CONTEXT có passage đích chứa nội dung trực tiếp.
6. Nếu câu hỏi hỏi "bao nhiêu", "bao lâu", "mức phạt", "thời hạn", "tối đa", "tối thiểu", câu trả lời phải nêu đúng con số/định lượng xuất hiện trong CONTEXT.
7. Nếu câu hỏi có nhiều ý, phải trả lời đủ từng ý. Không bỏ sót ý.
8. Nếu các căn cứ mâu thuẫn, ưu tiên căn cứ có nội dung trực tiếp hơn; nếu vẫn mâu thuẫn thì nêu rõ không đủ cơ sở kết luận.
9. Không trích dẫn passage không được dùng để trả lời.
10. Không viết dài dòng, không giải thích ngoài phạm vi câu hỏi.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Trả lời: <câu trả lời ngắn gọn, trực tiếp>
Dựa theo: <điều, khoản, tên văn bản/đường dẫn pháp lý liên quan>
"""

EXTRACTIVE_MULTI_AGENT_SYSTEM_PROMPT = f"""Bạn là hệ thống trả lời pháp lý chuyên về giao thông đường bộ Việt Nam.

Bạn phải vận hành như 3 agent nội bộ, nhưng KHÔNG được in quá trình làm việc:

AGENT 1 - Query Decomposer:
- Tách câu hỏi thành từng ý cần trả lời.
- Nếu câu hỏi có "và", "đồng thời", "nếu... thì...", "mức phạt", "trừ điểm", "tước giấy phép", phải xem là nhiều ý.

AGENT 2 - Evidence Extractor:
- Chỉ dùng CONTEXT.
- Tìm và CHÉP NGUYÊN VĂN cụm chứa đáp án trực tiếp cho từng ý.
- Ưu tiên passage chứa nội dung trực tiếp, không ưu tiên passage chỉ nói "theo quy định tại..." nếu passage đích có nội dung.
- Với số liệu, mức phạt, thời hạn, điều kiện, phải giữ đầy đủ từ giới hạn và đơn vị:
  "không quá", "tối đa", "tối thiểu", "ít nhất", "từ ... đến ...", "trừ ... điểm", "tước ... từ ... đến ...".
- Không được rút gọn "không quá 04 giờ" thành "04 giờ".
- Không được rút gọn câu có/không thành chỉ "Có" hoặc "Không"; phải chép cụm hành vi/điều kiện đi kèm.

AGENT 3 - Answer Composer:
- Viết câu trả lời ngắn, trực tiếp, nhưng phải chứa nguyên văn các cụm đáp án đã trích.
- Nếu câu hỏi nhiều ý, trả lời bằng các bullet, mỗi bullet một ý.
- Nếu chỉ thiếu căn cứ cho một ý, ghi rõ ý đó không tìm thấy căn cứ; không được phủ định toàn bộ câu hỏi nếu các ý khác có căn cứ.
- Không dùng kiến thức ngoài CONTEXT.
- Không trích dẫn căn cứ không được dùng.

Nếu CONTEXT không chứa bất kỳ căn cứ đủ rõ nào để trả lời, trả lời đúng câu:
"{INSUFFICIENT_CONTEXT_ANSWER}"

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Trả lời:
- <ý 1, chứa nguyên văn cụm đáp án trực tiếp>
- <ý 2 nếu có>
Dựa theo:
- <điều/khoản/điểm, văn bản hoặc đường dẫn pháp lý liên quan>
- <căn cứ tiếp theo nếu có>
"""


def _mojibake_score(text: str) -> int:
    markers = [
        "Ã", "Â", "Ä", "Æ", "Ð", "ð", "â€", "áº", "á»", "Â»", "Â«", "�",
    ]
    return sum(text.count(marker) for marker in markers)


def repair_mojibake_text(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake found in old benchmark JSON files."""
    if not isinstance(value, str) or not value:
        return value
    if _mojibake_score(value) == 0:
        return value

    best = value
    best_score = _mojibake_score(value)
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        score = _mojibake_score(repaired)
        if score < best_score:
            best = repaired
            best_score = score
    return best


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [repair_mojibake(v) for v in value]
    if isinstance(value, dict):
        return {k: repair_mojibake(v) for k, v in value.items()}
    return value


def run_retriever(
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    query: str,
    top_k: int = 10,
    dense_weight: float = 0.25,
    bm25_weight: float = 0.25,
    graph_weight: float = 0.20,
    reference_weight: float = 0.30,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(retriever_script),
        "--index-dir",
        str(index_dir),
        "--gazetteer-root",
        str(gazetteer_root),
        "--query",
        query,
        "--top-k",
        str(top_k),
        "--dense-weight",
        str(dense_weight),
        "--bm25-weight",
        str(bm25_weight),
        "--graph-weight",
        str(graph_weight),
        "--reference-weight",
        str(reference_weight),
    ]

    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Retriever failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    try:
        return repair_mojibake(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot parse retriever JSON output:\n{result.stdout[:2000]}") from exc


def format_context(
    results: list[dict[str, Any]],
    max_passages: int = 5,
    max_chars_per_passage: int = 1800,
) -> str:
    blocks = []

    for i, raw_result in enumerate(results[:max_passages], start=1):
        result = repair_mojibake(raw_result)
        doc = result.get("document_number") or "Không rõ văn bản"
        path = result.get("path_text") or result.get("passage_id") or "Không rõ đường dẫn"
        text = (result.get("text") or "").strip()

        if not text:
            continue

        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "..."

        blocks.append(
            f"[{i}]\n"
            f"Văn bản: {doc}\n"
            f"Đường dẫn: {path}\n"
            f"Nội dung: {text}"
        )

    return "\n\n".join(blocks)


def build_prompt(question: str, context: str, answer_mode: str = "extractive_multi_agent") -> list[dict[str, str]]:
    question = repair_mojibake_text(question)
    if answer_mode == "direct":
        user_prompt = f"""Câu hỏi:
{question}

CONTEXT:
{context}

Hãy trả lời câu hỏi dựa trên các căn cứ trong CONTEXT.
Định dạng:
Trả lời: ...
Dựa theo: ...
"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    if answer_mode != "extractive_multi_agent":
        raise ValueError(f"Unsupported answer_mode: {answer_mode}")

    user_prompt = f"""Câu hỏi:
{question}

CONTEXT:
{context}

Yêu cầu:
1. Tự tách câu hỏi thành từng ý.
2. Tự tìm cụm đáp án trực tiếp trong CONTEXT.
3. Câu trả lời cuối cùng phải CHỨA NGUYÊN VĂN cụm đáp án quan trọng, đặc biệt là số liệu, mức phạt, thời hạn, điều kiện, hành vi bị cấm.
4. Không in phần phân tích nội bộ.
5. Chỉ in đúng định dạng:

Trả lời:
- ...
Dựa theo:
- ...
"""
    return [
        {"role": "system", "content": EXTRACTIVE_MULTI_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _resolve_torch_dtype(torch_module, dtype: str):
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch_module.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_name: str,
    load_4bit: bool = False,
    dtype: str = "auto",
    device_map: str = "auto",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = _resolve_torch_dtype(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def _model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_answer(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    inputs = tokenizer(text, return_tensors="pt")
    input_device = _model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def answer_one(
    query: str,
    model_name: str,
    retriever_script: Path,
    index_dir: Path,
    gazetteer_root: Path,
    top_k: int,
    max_context_passages: int,
    load_4bit: bool,
    dtype: str = "auto",
    device_map: str = "auto",
    answer_mode: str = "extractive_multi_agent",
) -> dict[str, Any]:
    retrieval = run_retriever(
        retriever_script=retriever_script,
        index_dir=index_dir,
        gazetteer_root=gazetteer_root,
        query=query,
        top_k=top_k,
    )

    context = format_context(
        retrieval.get("results", []),
        max_passages=max_context_passages,
    )

    if not context.strip():
        return {
            "query": repair_mojibake_text(query),
            "model": model_name,
            "answer_mode": answer_mode,
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "context_used": "",
            "retrieval": retrieval,
        }

    tokenizer, model = load_model(
        model_name,
        load_4bit=load_4bit,
        dtype=dtype,
        device_map=device_map,
    )
    messages = build_prompt(query, context, answer_mode=answer_mode)

    answer = generate_answer(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=512,
        temperature=0.0,
    )

    return {
        "query": repair_mojibake_text(query),
        "model": model_name,
        "answer_mode": answer_mode,
        "answer": answer,
        "context_used": context,
        "retrieval": retrieval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one RAG answer with a local Hugging Face chat model.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--retriever-script", default="retrieval_pipelines/legal_linearrag_retriever/retrieve.py")
    parser.add_argument("--index-dir", default="data/retrieval/index_bge_m3_hybrid")
    parser.add_argument("--gazetteer-root", default="ner_finetuning/data/preprocessed/expanded_gazetteer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-passages", type=int, default=5)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--answer-mode", default="extractive_multi_agent", choices=["direct", "extractive_multi_agent"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = answer_one(
        query=args.query,
        model_name=args.model,
        retriever_script=Path(args.retriever_script),
        index_dir=Path(args.index_dir),
        gazetteer_root=Path(args.gazetteer_root),
        top_k=args.top_k,
        max_context_passages=args.max_context_passages,
        load_4bit=args.load_4bit,
        dtype=args.dtype,
        device_map=args.device_map,
        answer_mode=args.answer_mode,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
