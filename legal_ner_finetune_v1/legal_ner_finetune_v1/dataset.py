from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import Dataset

@dataclass
class TokenDataset(Dataset):
    encodings: Dict[str, Any]
    labels: List[List[int]]
    raw_rows: List[Dict[str, Any]]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def encode_rows(rows, tokenizer, label2id, max_length=256, text_field="text"):
    texts = [r.get(text_field) or "" for r in rows]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_offsets_mapping=True)
    offset_batches = enc.pop("offset_mapping")
    all_labels = []
    for row, offsets in zip(rows, offset_batches):
        ents = row.get("entities") or []
        labs = []
        for start, end in offsets:
            if start == 0 and end == 0:
                labs.append(-100); continue
            assigned = "O"
            for e in ents:
                es, ee, lab = e["start"], e["end"], e["label"]
                if end <= es or start >= ee: continue
                assigned = f"B-{lab}" if (start <= es < end or start == es) else f"I-{lab}"
                break
            labs.append(label2id[assigned])
        all_labels.append(labs)
    return enc, all_labels
