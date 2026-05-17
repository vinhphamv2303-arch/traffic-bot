import json
import urllib.request
import urllib.error

class BaseProvider:
    def generate(self, system_prompt, user_prompt):
        raise NotImplementedError

class OllamaProvider(BaseProvider):
    def __init__(self, endpoint="http://localhost:11434", model="qwen3:8b", temperature=0.0, timeout_seconds=120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, system_prompt, user_prompt):
        url = f"{self.endpoint}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "legal-llm-ner/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out.get("message", {}).get("content") or out.get("response") or ""

class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, api_base="http://localhost:8000/v1", api_key=None, model="qwen3:8b", temperature=0.0, max_tokens=2048, timeout_seconds=120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, system_prompt, user_prompt):
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "legal-llm-ner/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]

class MockProvider(BaseProvider):
    """
    For tests/dry-run. Uses simple regex-like heuristics, not a real LLM.
    """
    def generate(self, system_prompt, user_prompt):
        import re
        import json

        m = re.search(r"Danh sách câu:\s*(\[.*\])", user_prompt, flags=re.S)
        items = json.loads(m.group(1)) if m else []
        results = []
        for item in items:
            text = item.get("text") or ""
            ents = []
            for pat, label in [
                (r"xe mô tô|xe gắn máy|xe ô tô|xe máy", "VEHICLE_TYPE"),
                (r"không đội mũ bảo hiểm|vượt đèn đỏ|đi ngược chiều", "VIOLATION_OR_BEHAVIOR"),
                (r"phạt tiền|tước quyền sử dụng|trừ điểm|tạm giữ", "SANCTION"),
                (r"\d{1,3}(?:\.\d{3})+\s*đồng(?:\s*đến\s*\d{1,3}(?:\.\d{3})+\s*đồng)?", "FINE_AMOUNT"),
                (r"giấy phép lái xe|giấy đăng ký xe|chứng nhận kiểm định", "DOCUMENT_OR_PERMIT"),
                (r"người đi bộ|người điều khiển xe|chủ xe", "REGULATED_SUBJECT"),
                (r"biển số xe", "VEHICLE_IDENTIFIER"),
            ]:
                for mm in re.finditer(pat, text, flags=re.I):
                    ents.append({"text": mm.group(0), "label": label, "confidence": 0.75})
            results.append({"id": item.get("id"), "entities": ents})
        return json.dumps({"results": results}, ensure_ascii=False)

def make_provider(config):
    if config.provider == "ollama":
        return OllamaProvider(
            endpoint=config.endpoint,
            model=config.model,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )
    if config.provider == "openai_compatible":
        return OpenAICompatibleProvider(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )
    if config.provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported provider: {config.provider}")
