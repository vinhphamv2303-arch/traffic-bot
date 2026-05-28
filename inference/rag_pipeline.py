from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_pipelines.legal_linearrag_retriever.legal_linearrag_retriever import LinearRAGRetriever  # noqa: E402


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    key: str
    display_name: str
    index_dir: Path
    weights: dict[str, float]
    use_reference_expansion: bool = True
    graph_only_penalty: float = 0.65
    description: str = ""


GAZETTEER_ROOT = ROOT / "ner_finetuning" / "data" / "preprocessed" / "expanded_gazetteer"

PIPELINES: dict[str, RetrievalPipelineConfig] = {
    "hybrid_cpu": RetrievalPipelineConfig(
        key="hybrid_cpu",
        display_name="Hybrid CPU: BM25 + Graph + Reference",
        index_dir=ROOT / "data" / "retrieval" / "index_bm25_graph",
        weights={"dense": 0.0, "bm25": 0.25, "graph": 0.15, "reference": 0.60},
        description="Nhanh, không cần load embedding model; phù hợp demo local.",
    ),
    "hybrid_bge_m3": RetrievalPipelineConfig(
        key="hybrid_bge_m3",
        display_name="Hybrid BGE-M3: Dense + BM25 + Graph + Reference",
        index_dir=ROOT / "data" / "retrieval" / "index_bge_m3_hybrid",
        weights={"dense": 0.25, "bm25": 0.25, "graph": 0.20, "reference": 0.30},
        description="Pipeline đầy đủ nhất, nhưng lần đầu load BGE-M3 có thể chậm trên CPU.",
    ),
    "hybrid_minilm": RetrievalPipelineConfig(
        key="hybrid_minilm",
        display_name="Hybrid MiniLM CPU",
        index_dir=ROOT / "data" / "retrieval" / "index_minilm_hybrid",
        weights={"dense": 0.20, "bm25": 0.30, "graph": 0.20, "reference": 0.30},
        description="Có dense retrieval nhẹ hơn BGE-M3.",
    ),
    "bm25": RetrievalPipelineConfig(
        key="bm25",
        display_name="Naive BM25",
        index_dir=ROOT / "data" / "retrieval" / "index_bm25_graph",
        weights={"dense": 0.0, "bm25": 1.0, "graph": 0.0, "reference": 0.0},
        use_reference_expansion=False,
        description="Baseline từ khóa.",
    ),
}


DOMAIN_QUERY_EXPANSIONS: list[tuple[str, list[str]]] = [
    (
        r"\b(say\s*rượu|uống\s*rượu|rượu\s*bia|bia\s*rượu|có\s*cồn|hơi\s*men|nồng\s*độ\s*cồn)\b",
        [
            "trong máu hoặc hơi thở có nồng độ cồn",
            "có nồng độ cồn trong máu hoặc hơi thở",
            "có sử dụng rượu bia",
            "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
        ],
    ),
    (
        r"\b(vượt\s*đèn\s*đỏ|đèn\s*đỏ|đèn\s*tín\s*hiệu)\b",
        [
            "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
            "không chấp hành hiệu lệnh đèn tín hiệu giao thông",
        ],
    ),
    (
        r"\b(không\s*đội\s*mũ|mũ\s*bảo\s*hiểm)\b",
        [
            "không đội mũ bảo hiểm",
            "không đội mũ bảo hiểm cho người đi mô tô xe máy",
        ],
    ),
    (
        r"\b(quá\s*tốc\s*độ|chạy\s*quá\s*tốc\s*độ|vượt\s*tốc\s*độ)\b",
        [
            "điều khiển xe chạy quá tốc độ quy định",
            "tốc độ tối đa cho phép",
        ],
    ),
    (
        r"\b(bằng\s*lái|bằng\s*lái\s*xe|gplx)\b",
        [
            "giấy phép lái xe",
            "người lái xe không có giấy phép lái xe",
        ],
    ),
    (
        r"\b(đăng\s*kiểm|kiểm\s*định)\b",
        [
            "giấy chứng nhận kiểm định",
            "kiểm định an toàn kỹ thuật và bảo vệ môi trường",
        ],
    ),
]


VEHICLE_QUERY_EXPANSIONS: list[tuple[str, str]] = [
    (r"\b(ô\s*tô|xe\s*hơi|xe\s*con|xe\s*tải|xe\s*khách)\b", "xe ô tô"),
    (r"\b(mô\s*tô|xe\s*máy|xe\s*gắn\s*máy)\b", "xe mô tô xe gắn máy"),
    (r"\b(xe\s*đạp|xe\s*đạp\s*điện)\b", "xe đạp xe đạp điện"),
]

DEFAULT_VEHICLE_BREAKDOWN = [
    "xe ô tô",
    "xe mô tô xe gắn máy",
    "xe máy chuyên dùng",
    "xe đạp xe đạp điện",
]

PENALTY_HINT_PATTERN = r"\b(mức\s*xử\s*phạt|xử\s*phạt|bị\s*phạt|phạt\s*bao\s*nhiêu|mức\s*phạt|phạt\s*tiền)\b"


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(provider: str = "openai", explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    load_dotenv()
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def validate_paths(config: RetrievalPipelineConfig, gazetteer_root: Path = GAZETTEER_ROOT) -> None:
    if not config.index_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy index: {config.index_dir}")
    if not gazetteer_root.exists():
        raise FileNotFoundError(f"Không tìm thấy gazetteer: {gazetteer_root}")


def load_retriever(config: RetrievalPipelineConfig, gazetteer_root: Path = GAZETTEER_ROOT) -> LinearRAGRetriever:
    validate_paths(config, gazetteer_root=gazetteer_root)
    return LinearRAGRetriever.from_index(config.index_dir, gazetteer_root)


def expand_query(question: str, max_queries: int = 8) -> list[str]:
    """Create a small set of legal-domain query variants for retrieval."""
    question = " ".join((question or "").split())
    if not question:
        return []

    lowered = question.lower()
    queries = [question]
    expansions: list[str] = []
    penalty_hint = bool(re.search(PENALTY_HINT_PATTERN, lowered, flags=re.IGNORECASE))

    for pattern, phrases in DOMAIN_QUERY_EXPANSIONS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            expansions.extend(phrases)

    asks_each_vehicle = any(token in lowered for token in ["từng loại phương tiện", "các loại phương tiện", "từng loại xe", "đối với từng loại"])

    if expansions:
        legal_prefix = "xử phạt vi phạm hành chính phạt tiền " if penalty_hint else ""
        queries.append(f"{legal_prefix}{question} {' '.join(expansions[:2])}")
        if penalty_hint:
            queries.append(f"Nghị định 168/2024/NĐ-CP xử phạt vi phạm hành chính {expansions[0]}")

        if not asks_each_vehicle:
            for phrase in expansions[:3]:
                queries.append(f"{legal_prefix}{phrase}".strip())

    vehicle_hits = []
    for pattern, phrase in VEHICLE_QUERY_EXPANSIONS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            vehicle_hits.append(phrase)

    vehicles = vehicle_hits or (DEFAULT_VEHICLE_BREAKDOWN if asks_each_vehicle and expansions else [])
    for vehicle in vehicles[:4]:
        if expansions:
            if penalty_hint:
                queries.append(f"mức phạt phạt tiền {vehicle} điều khiển xe trên đường mà {expansions[0]}")
            else:
                queries.append(f"{vehicle} {expansions[0]}")
        else:
            queries.append(f"{question} {vehicle}")

    deduped = []
    seen = set()
    for query in queries:
        norm = query.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(query)
    return deduped[:max_queries]


def retrieve(
    retriever: LinearRAGRetriever,
    question: str,
    config: RetrievalPipelineConfig,
    top_k: int = 5,
    candidate_k: int = 300,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
) -> dict[str, Any]:
    return retriever.retrieve(
        query=question,
        top_k=top_k,
        candidate_k=candidate_k,
        semantic_entity_top_k=semantic_entity_top_k,
        semantic_entity_min_score=semantic_entity_min_score,
        weights=config.weights,
        use_reference_expansion=config.use_reference_expansion,
        graph_only_penalty=config.graph_only_penalty,
    )


def retrieve_multi_query(
    retriever: LinearRAGRetriever,
    question: str,
    config: RetrievalPipelineConfig,
    top_k: int = 5,
    candidate_k: int = 300,
    semantic_entity_top_k: int = 20,
    semantic_entity_min_score: float = 0.45,
    enable_expansion: bool = True,
    max_queries: int = 8,
) -> dict[str, Any]:
    queries = expand_query(question, max_queries=max_queries) if enable_expansion else [question]
    queries = queries or [question]

    per_query = []
    by_passage: dict[str, dict[str, Any]] = {}
    query_count = max(len(queries), 1)

    for q_idx, query in enumerate(queries):
        result = retrieve(
            retriever=retriever,
            question=query,
            config=config,
            top_k=max(top_k, min(candidate_k, top_k * 4)),
            candidate_k=candidate_k,
            semantic_entity_top_k=semantic_entity_top_k,
            semantic_entity_min_score=semantic_entity_min_score,
        )
        per_query.append(result)

        query_weight = 0.55 if q_idx == 0 and len(queries) > 1 else 1.0
        for rank, item in enumerate(result.get("results") or [], start=1):
            pid = item.get("passage_id")
            if not pid:
                continue
            rrf = query_weight / (60.0 + rank)
            weighted_score = query_weight * float(item.get("score") or 0.0)
            entry = by_passage.setdefault(
                pid,
                {
                    **item,
                    "score": 0.0,
                    "merged_score": 0.0,
                    "source_queries": [],
                    "best_rank": rank,
                    "best_original_score": float(item.get("score") or 0.0),
                },
            )
            entry["merged_score"] += rrf + weighted_score / query_count
            entry["score"] = round(float(entry["merged_score"]), 6)
            entry["best_rank"] = min(int(entry.get("best_rank") or rank), rank)
            entry["best_original_score"] = max(float(entry.get("best_original_score") or 0.0), float(item.get("score") or 0.0))
            entry["source_queries"].append({"query": query, "rank": rank, "score": item.get("score")})

    merged = sorted(
        by_passage.values(),
        key=lambda x: (float(x.get("merged_score") or 0.0), float(x.get("best_original_score") or 0.0)),
        reverse=True,
    )[:top_k]

    activated_entities = []
    for result in per_query:
        activated_entities.extend(result.get("activated_entities") or [])

    return {
        "query": question,
        "expanded_queries": queries,
        "results": merged,
        "activated_entities": activated_entities[:50],
        "per_query_debug": [
            {
                "query": r.get("query"),
                "result_count": len(r.get("results") or []),
                "top_passage_id": (r.get("results") or [{}])[0].get("passage_id"),
            }
            for r in per_query
        ],
        "debug": {
            "multi_query": True,
            "query_count": len(queries),
            "pipeline": config.key,
        },
    }


def format_context(results: list[dict[str, Any]], max_chars_per_passage: int = 1800) -> str:
    blocks = []
    for idx, item in enumerate(results, start=1):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "..."

        doc = item.get("document_number") or item.get("document_id") or "Không rõ văn bản"
        path = item.get("path_text") or item.get("passage_id") or "Không rõ đường dẫn"
        blocks.append(
            f"[{idx}]\n"
            f"Văn bản: {doc}\n"
            f"Đường dẫn: {path}\n"
            f"Nội dung: {text}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về giao thông đường bộ Việt Nam.

Nhiệm vụ của bạn là trả lời câu hỏi chỉ dựa trên CONTEXT được cung cấp.

Quy tắc:
- Không dùng kiến thức ngoài CONTEXT.
- Nếu CONTEXT không đủ căn cứ, hãy nói rõ không tìm thấy căn cứ đủ rõ trong tài liệu được truy xuất.
- Trả lời ngắn gọn, trực tiếp, nhưng giữ nguyên các con số, thời hạn, mức phạt, điều kiện quan trọng nếu có.
- Nếu câu hỏi có nhiều ý, trả lời theo gạch đầu dòng.
- Luôn nêu căn cứ ở phần "Dựa theo".

Định dạng:
Trả lời:
- ...
Dựa theo:
- ...
"""


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    user_prompt = f"""Câu hỏi:
{question}

CONTEXT:
{context}

Hãy trả lời câu hỏi dựa trên CONTEXT."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer_openai(
    question: str,
    retrieval_result: dict[str, Any],
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    max_passages: int = 5,
    max_chars_per_passage: int = 1800,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> dict[str, Any]:
    from openai import OpenAI

    provider = provider.lower().strip()
    if provider not in {"openai", "openrouter"}:
        raise ValueError("provider phải là 'openai' hoặc 'openrouter'.")

    key = get_api_key(provider=provider, explicit=api_key)
    if not key:
        if provider == "openrouter":
            raise RuntimeError("Thiếu OPENROUTER_API_KEY hoặc OPEN_ROUTER_API. Hãy thêm vào .env hoặc nhập API key trong giao diện.")
        raise RuntimeError("Thiếu OPENAI_API_KEY. Hãy thêm vào .env hoặc nhập API key trong giao diện.")

    results = retrieval_result.get("results") or []
    context = format_context(results[:max_passages], max_chars_per_passage=max_chars_per_passage)
    if not context.strip():
        return {
            "answer": "Không tìm thấy căn cứ đủ rõ trong tài liệu được truy xuất.",
            "context": context,
            "model": model,
        }

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
        default_headers={
            "HTTP-Referer": "http://localhost/traffic-law-rag-demo",
            "X-Title": "Traffic Law RAG Demo",
        } if provider == "openrouter" else None,
    )
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, context),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer.strip(),
        "context": context,
        "model": model,
        "provider": provider,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def answer_question(
    question: str,
    pipeline_key: str = "hybrid_cpu",
    top_k: int = 5,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> dict[str, Any]:
    config = PIPELINES[pipeline_key]
    retriever = load_retriever(config)
    retrieval_result = retrieve_multi_query(retriever, question=question, config=config, top_k=top_k)
    generation = generate_answer_openai(
        question=question,
        retrieval_result=retrieval_result,
        api_key=api_key,
        model=model,
        provider=provider,
        max_passages=top_k,
    )
    return {
        "question": question,
        "pipeline": config.display_name,
        "retrieval": retrieval_result,
        "generation": generation,
    }
