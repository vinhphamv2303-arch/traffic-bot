from __future__ import annotations
from pathlib import Path
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForTokenClassification, AutoTokenizer
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl

def snap_to_word_boundaries(start, end, text):
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return start, end

def finalize_entity(cur, scores, text, min_confidence=0.0):
    if not cur:
        return None
    confidence = round(sum(scores)/max(len(scores),1), 4)
    if confidence < min_confidence:
        return None
    start, end = snap_to_word_boundaries(cur["start"], cur["end"], text)
    span = text[start:end].strip()
    if not span:
        return None
    return {"label": cur["label"], "start": start, "end": end, "confidence": confidence, "text": span}

def merge_bio(labels, probs, offsets, text, min_confidence=0.0):
    ents=[]; cur=None; scores=[]
    for lab,prob,(start,end) in zip(labels,probs,offsets):
        if start == 0 and end == 0: continue
        if lab == "O":
            if cur:
                ent = finalize_entity(cur, scores, text, min_confidence=min_confidence)
                if ent: ents.append(ent)
                cur=None; scores=[]
            continue
        prefix, typ = lab.split("-",1) if "-" in lab else ("B", lab)
        if prefix == "B" or cur is None or cur["label"] != typ:
            if cur:
                ent = finalize_entity(cur, scores, text, min_confidence=min_confidence)
                if ent: ents.append(ent)
            cur = {"label": typ, "start": int(start), "end": int(end)}; scores=[float(prob)]
        else:
            cur["end"] = int(end); scores.append(float(prob))
    if cur:
        ent = finalize_entity(cur, scores, text, min_confidence=min_confidence)
        if ent: ents.append(ent)
    return ents

class Predictor:
    def __init__(self, model_dir, device=None, max_length=256):
        self.model_dir = str(model_dir); self.max_length=max_length
        self.tok = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device); self.model.eval()
    def predict_rows(self, rows, text_field="text", batch_size=16, min_confidence=0.0):
        out=[]
        for i in range(0, len(rows), batch_size):
            batch=rows[i:i+batch_size]; texts=[r.get(text_field) or r.get("text") or "" for r in batch]
            enc=self.tok(texts, return_offsets_mapping=True, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            offsets=enc.pop("offset_mapping").tolist(); enc={k:v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                logits=self.model(**enc).logits; probs=softmax(logits, dim=-1); ids=logits.argmax(dim=-1).cpu().tolist(); pmax=probs.max(dim=-1).values.cpu().tolist()
            for row,text,idseq,pseq,offs in zip(batch,texts,ids,pmax,offsets):
                labs=[self.model.config.id2label[int(x)] for x in idseq]
                ents=merge_bio(labs,pseq,offs,text,min_confidence=min_confidence)
                out.append({**row, "predicted_entities": ents, "predicted_entity_count": len(ents)})
        return out

def sentence_dirs(root):
    p=Path(root)
    if (p/"sentences.jsonl").exists(): return [p]
    return sorted([x for x in p.iterdir() if x.is_dir() and (x/"sentences.jsonl").exists()])

def summarize(rows):
    by={}; total=0; sw=0
    for r in rows:
        ents=r.get("predicted_entities") or []; sw += bool(ents); total += len(ents)
        for e in ents: by[e["label"]]=by.get(e["label"],0)+1
    return {"sentence_count": len(rows), "sentence_with_entity_count": sw, "entity_count": total, "by_label": dict(sorted(by.items()))}

def predict_all(model_dir, sentences_root, output_root, batch_size=16, max_length=256, device=None, min_confidence=0.0):
    output_root=ensure_dir(output_root); pred=Predictor(model_dir, device=device, max_length=max_length)
    all_mentions=[]; glob={"packages":{}, "sentence_count":0, "sentence_with_entity_count":0, "entity_count":0, "by_label":{}}
    for d in sentence_dirs(sentences_root):
        out_dir=ensure_dir(output_root/d.name); rows=list(read_jsonl(d/"sentences.jsonl")); pr=pred.predict_rows(rows, batch_size=batch_size, min_confidence=min_confidence)
        write_jsonl(out_dir/"predicted_sentence_entities.jsonl", pr)
        mentions=[]
        for row in pr:
            for i,e in enumerate(row.get("predicted_entities") or [],1):
                mentions.append({"entity_id": f"{row.get('sentence_id')}.pred_{i:03d}", "sentence_id": row.get("sentence_id"), "passage_id": row.get("passage_id"), "source_unit_id": row.get("source_unit_id"), "package_id": row.get("package_id"), "document_id": row.get("document_id"), "document_number": row.get("document_number"), "path_text": row.get("path_text"), "text": e["text"], "label": e["label"], "start": e["start"], "end": e["end"], "confidence": e["confidence"], "source": "finetuned_ner_v1", "model": str(model_dir)})
        write_jsonl(out_dir/"predicted_entity_mentions.jsonl", mentions); summ=summarize(pr); write_json(out_dir/"prediction_summary.json", summ)
        glob["packages"][d.name]=summ; glob["sentence_count"]+=summ["sentence_count"]; glob["sentence_with_entity_count"]+=summ["sentence_with_entity_count"]; glob["entity_count"]+=summ["entity_count"]
        for k,v in summ["by_label"].items(): glob["by_label"][k]=glob["by_label"].get(k,0)+v
        all_mentions.extend(mentions)
    glob["by_label"]=dict(sorted(glob["by_label"].items())); write_jsonl(output_root/"all_predicted_entity_mentions.jsonl", all_mentions); write_json(output_root/"prediction_summary.json", glob); return glob
