import numpy as np

def bio_to_spans(seq):
    spans, cur = [], None
    for i, lab in enumerate(seq):
        if lab == "O":
            if cur: spans.append(cur); cur = None
            continue
        prefix, typ = lab.split("-", 1) if "-" in lab else ("B", lab)
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur: spans.append(cur)
            cur = [i, i+1, typ]
        else:
            cur[1] = i+1
    if cur: spans.append(cur)
    return {(s,e,t) for s,e,t in spans}

def make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp=fp=fn=tok_ok=tok_total=0
        for pseq, lseq in zip(preds, labels):
            glabs, plabs = [], []
            for p,l in zip(pseq,lseq):
                if int(l) == -100: continue
                g = id2label[int(l)]; pr = id2label[int(p)]
                glabs.append(g); plabs.append(pr); tok_total += 1; tok_ok += int(g == pr)
            gold, pred = bio_to_spans(glabs), bio_to_spans(plabs)
            tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        return {"entity_precision": prec, "entity_recall": rec, "entity_f1": f1, "token_accuracy": tok_ok/tok_total if tok_total else 0.0}
    return compute_metrics
