from __future__ import annotations
import json
import time
import urllib.error
import urllib.request

RESULT_SCHEMA={"type":"object","additionalProperties":False,"properties":{"results":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"id":{"type":"string"},"entities":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"text":{"type":"string"},"label":{"type":"string","enum":["BEHAVIOR","VEHICLE","ACTOR","INFRASTRUCTURE","DOCUMENT","VEHICLE_CONDITION_OR_EQUIPMENT","CONDITION"]}},"required":["text","label"]}}},"required":["id","entities"]}}},"required":["results"]}


class OpenAIChatProvider:
    def __init__(self, api_key, api_base="https://openrouter.ai/api/v1", temperature=0.0, max_tokens=2048, timeout_seconds=120, use_json_schema=True, max_retries=5, retry_base_seconds=2.0):
        if not api_key:
            raise ValueError("API key is required. Set OPEN_ROUTER_API, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
        self.api_key=api_key; self.api_base=api_base.rstrip("/"); self.temperature=temperature; self.max_tokens=max_tokens; self.timeout_seconds=timeout_seconds; self.use_json_schema=use_json_schema; self.max_retries=max_retries; self.retry_base_seconds=retry_base_seconds

    def generate(self, model, system_prompt, user_prompt):
        payload={"model":model,"messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],"temperature":self.temperature,"max_tokens":self.max_tokens}
        if self.use_json_schema:
            payload["response_format"]={"type":"json_schema","json_schema":{"name":"legal_ner_results","strict":True,"schema":RESULT_SCHEMA}}
        try:
            return self._post_chat(payload)
        except urllib.error.HTTPError as exc:
            body=exc.read().decode("utf-8", errors="replace")
            if self.use_json_schema and exc.code in {400, 404, 422}:
                payload.pop("response_format", None)
                return self._post_chat(payload)
            raise RuntimeError(f"Chat completion failed: HTTP {exc.code}: {body[:1000]}") from exc

    def _post_chat(self, payload):
        headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {self.api_key}",
            "HTTP-Referer":"http://localhost/legal-llm-ner-v2",
            "X-Title":"legal-llm-ner-v2",
        }
        req=urllib.request.Request(f"{self.api_base}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers)
        last_error=None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    out=json.loads(resp.read().decode("utf-8"))
                return out["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                body=exc.read().decode("utf-8", errors="replace")
                last_error=RuntimeError(f"Chat completion failed: HTTP {exc.code}: {body[:1000]}")
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise last_error from exc
                retry_after=exc.headers.get("Retry-After")
                delay=float(retry_after) if retry_after and retry_after.isdigit() else self.retry_base_seconds * (2 ** attempt)
                time.sleep(min(delay, 60))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error=RuntimeError(f"Chat completion failed: {exc}")
                if attempt >= self.max_retries:
                    raise last_error from exc
                time.sleep(min(self.retry_base_seconds * (2 ** attempt), 60))
        raise last_error or RuntimeError("Chat completion failed")
