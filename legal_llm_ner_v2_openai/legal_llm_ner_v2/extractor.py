from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .config import LLMNERv2Config
from .provider_openai import OpenAIChatProvider
from .env import get_api_key
from .prompt import build_system_prompt, build_user_prompt
from .schema import ALLOWED_LABELS
from .selector import select_candidates
from .utils import append_jsonl, chunks, collapse_ws, dedupe_entities, ensure_dir, extract_json_object, find_offsets, find_sentence_package_dirs, is_reference_like, is_too_generic, md5_text, read_jsonl, summarize_mentions, write_json, write_jsonl

class LLMNERv2:
    def __init__(self, config: LLMNERv2Config):
        self.config=config; self.package_dirs=find_sentence_package_dirs(config.sentences_root)
        self.provider=OpenAIChatProvider(api_key=get_api_key(config.api_key), api_base=config.api_base, temperature=config.temperature, max_tokens=config.max_tokens, timeout_seconds=config.timeout_seconds, use_json_schema=config.use_json_schema, max_retries=config.max_retries, retry_base_seconds=config.retry_base_seconds)
        self.system_prompt=build_system_prompt()
    def run_all(self):
        root=ensure_dir(self.config.output_root); summary={"package_count":0,"total_input_sentences":0,"total_selected_sentences":0,"total_annotated_sentences":0,"total_entity_mentions":0,"packages":{},"models":{"gold_model":self.config.gold_model,"silver_model":self.config.silver_model,"quality_tier":self.config.quality_tier}}
        all_mentions=[]
        remaining=self.config.limit
        for package_dir in self.package_dirs:
            if remaining is not None and remaining <= 0:
                break
            pkg=self.run_package(package_dir, sentence_limit=remaining); summary["packages"][package_dir.name]=pkg["summary"]
            summary["package_count"]+=1
            summary["total_input_sentences"]+=pkg["summary"]["input_sentence_count"]; summary["total_selected_sentences"]+=pkg["summary"]["selected_sentence_count"]; summary["total_annotated_sentences"]+=pkg["summary"]["annotated_sentence_count"]; summary["total_entity_mentions"]+=pkg["summary"]["entity_mention_count"]
            all_mentions.extend(pkg["mentions"])
            if remaining is not None:
                remaining-=pkg["summary"]["input_sentence_count"]
        write_jsonl(root/"all_entity_mentions.jsonl", all_mentions); write_json(root/"entity_summary.json", summary); return summary
    def run_package(self, package_dir: Path, sentence_limit: int | None = None):
        out_dir=ensure_dir(self.config.output_root/package_dir.name); sent_path=out_dir/"sentence_entities.jsonl"; mention_path=out_dir/"entity_mentions.jsonl"; selected_path=out_dir/"selected_candidates.jsonl"
        if not self.config.resume:
            for path in [sent_path, mention_path, selected_path]:
                if path.exists():
                    path.unlink()
        all_sentences=list(read_jsonl(package_dir/"sentences.jsonl"));
        if sentence_limit is not None: all_sentences=all_sentences[:sentence_limit]
        selected=select_candidates(all_sentences,self.config.min_candidate_score) if self.config.candidate_only else list(all_sentences); write_jsonl(selected_path, selected)
        processed=set()
        if self.config.resume and sent_path.exists():
            for r in read_jsonl(sent_path):
                if r.get("sentence_id"): processed.add(r["sentence_id"])
        selected=[s for s in selected if s.get("sentence_id") not in processed]
        gold_rows, silver_rows = self.route_quality(selected)
        if gold_rows: self.annotate_rows(gold_rows, self.config.gold_model, "gold", sent_path, mention_path)
        if silver_rows: self.annotate_rows(silver_rows, self.config.silver_model, "silver", sent_path, mention_path)
        sent_rows=list(read_jsonl(sent_path)) if sent_path.exists() else []; mentions=list(read_jsonl(mention_path)) if mention_path.exists() else []
        summ={"package_id":package_dir.name,"input_sentence_count":len(all_sentences),"selected_sentence_count":len(list(read_jsonl(selected_path))) if selected_path.exists() else 0,"annotated_sentence_count":len(sent_rows),"sentence_with_entity_count":sum(1 for r in sent_rows if r.get("entities")),"entity_mention_count":len(mentions),"by_label":summarize_mentions(mentions)}
        write_json(out_dir/"entity_summary.json", summ); return {"summary":summ,"mentions":mentions}
    def route_quality(self, rows):
        if self.config.quality_tier=="gold": return rows, []
        if self.config.quality_tier=="silver": return [], rows
        if self.config.quality_tier!="mixed": raise ValueError("Invalid quality_tier")
        rows=sorted(rows,key=lambda x:x.get("_candidate_score",0),reverse=True); return rows[:self.config.gold_limit], rows[self.config.gold_limit:]
    def annotate_rows(self, rows, model, quality, sent_path, mention_path):
        for batch in chunks(rows,self.config.batch_size):
            self.annotate_batch_resilient(batch, model, quality, sent_path, mention_path)
    def annotate_batch_resilient(self, batch, model, quality, sent_path, mention_path):
        try:
            srows, mentions = self.annotate_batch(batch, model, quality); append_jsonl(sent_path,srows); append_jsonl(mention_path,mentions)
            return
        except Exception as exc:
            if len(batch) > 1:
                mid=max(1,len(batch)//2)
                self.annotate_batch_resilient(batch[:mid], model, quality, sent_path, mention_path)
                self.annotate_batch_resilient(batch[mid:], model, quality, sent_path, mention_path)
                return
            failed=self.failed_sentence_rows(batch, model, quality, exc)
            append_jsonl(sent_path, failed)
            append_jsonl(self.config.output_root/"annotation_errors.jsonl", [{"sentence_id":batch[0].get("sentence_id"),"package_id":batch[0].get("package_id"),"document_id":batch[0].get("document_id"),"model":model,"quality":quality,"error":str(exc),"created_at_unix":int(time.time())}])
    def annotate_batch(self, batch, model, quality):
        items=[]
        for s in batch:
            items.append({"id":s.get("sentence_id"),"text":s.get(self.config.text_field) or s.get("text") or "","context":s.get(self.config.context_field) or s.get("path_text") or ""})
        user_prompt=build_user_prompt(items); parsed=None; last_error=None
        parse_retries=min(self.config.max_retries, 1)
        for attempt in range(parse_retries + 1):
            raw=self.provider.generate(model,self.system_prompt,user_prompt)
            try:
                parsed=extract_json_object(raw); break
            except Exception as exc:
                last_error=exc
                if attempt >= parse_retries:
                    raise RuntimeError(f"Could not parse LLM JSON after {attempt + 1} attempts; batch_size={len(batch)}; raw_prefix={raw[:300]!r}") from exc
                time.sleep(min(self.config.retry_base_seconds * (2 ** attempt), 60))
        if parsed is None:
            raise RuntimeError(f"Could not parse LLM JSON: {last_error}")
        by_id={r.get("id"):r for r in parsed.get("results",[])}
        sentence_rows=[]; mention_rows=[]; review_status="gold_llm_candidate" if quality=="gold" else "silver_llm_candidate"
        for s in batch:
            sid=s.get("sentence_id"); text=s.get(self.config.text_field) or s.get("text") or ""; ents=[]
            for ent in (by_id.get(sid,{"entities":[]}).get("entities") or []):
                norm=self.normalize_entity(ent,s,quality)
                if norm: ents.append(norm)
            ents=dedupe_entities(ents)
            srow={"sentence_id":sid,"passage_id":s.get("passage_id"),"source_unit_id":s.get("source_unit_id"),"package_id":s.get("package_id"),"document_id":s.get("document_id"),"document_number":s.get("document_number"),"text":text,"context_text":s.get("context_text"),"path_text":s.get("path_text"),"unit_type":s.get("unit_type"),"entities":ents,"entity_count":len(ents),"source":"llm_openai","quality":quality,"model":model,"review_status":review_status,"prompt_hash":md5_text(self.system_prompt+user_prompt),"created_at_unix":int(time.time()),"candidate_score":s.get("_candidate_score"),"candidate_labels":s.get("_candidate_labels")}
            sentence_rows.append(srow)
            for e in ents:
                mention_rows.append({**e,"sentence_id":sid,"passage_id":s.get("passage_id"),"source_unit_id":s.get("source_unit_id"),"package_id":s.get("package_id"),"document_id":s.get("document_id"),"document_number":s.get("document_number"),"source_type":s.get("source_type"),"attachment_id":s.get("attachment_id"),"unit_type":s.get("unit_type"),"path_text":s.get("path_text"),"source":"llm_openai","quality":quality,"model":model,"review_status":review_status})
        return sentence_rows, mention_rows
    def failed_sentence_rows(self, batch, model, quality, exc):
        review_status="llm_annotation_failed"
        rows=[]
        for s in batch:
            text=s.get(self.config.text_field) or s.get("text") or ""
            rows.append({"sentence_id":s.get("sentence_id"),"passage_id":s.get("passage_id"),"source_unit_id":s.get("source_unit_id"),"package_id":s.get("package_id"),"document_id":s.get("document_id"),"document_number":s.get("document_number"),"text":text,"context_text":s.get("context_text"),"path_text":s.get("path_text"),"unit_type":s.get("unit_type"),"entities":[],"entity_count":0,"source":"llm_openai","quality":quality,"model":model,"review_status":review_status,"annotation_error":str(exc),"created_at_unix":int(time.time()),"candidate_score":s.get("_candidate_score"),"candidate_labels":s.get("_candidate_labels")})
        return rows
    def normalize_entity(self, ent: Dict[str,Any], sentence: Dict[str,Any], quality: str):
        label=collapse_ws(ent.get("label") or ""); span=collapse_ws(ent.get("text") or "")
        if label not in ALLOWED_LABELS or not span or is_reference_like(span) or is_too_generic(span,label): return None
        text=sentence.get(self.config.text_field) or sentence.get("text") or ""; start,end=find_offsets(text,span)
        if start is None and self.config.reject_unaligned: return None
        return {"entity_id":"ent_"+md5_text(f"{sentence.get('sentence_id')}|{label}|{span}|{start}|{end}")[:16],"text":span if start is None else text[start:end],"label":label,"start":start,"end":end,"confidence":1.0 if quality=="gold" else 0.85,"alignment_status":"aligned" if start is not None else "unaligned"}
