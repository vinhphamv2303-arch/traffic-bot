from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

from .utils import simple_tokenize


class BM25Index:
    def __init__(self, corpus_tokens: List[List[str]], doc_ids: List[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(x) for x in corpus_tokens) / max(self.N, 1)
        self.doc_lens = [len(x) for x in corpus_tokens]
        self.tfs = [Counter(tokens) for tokens in corpus_tokens]
        self.df = defaultdict(int)
        for tf in self.tfs:
            for term in tf:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    @classmethod
    def from_texts(cls, texts: List[str], doc_ids: List[str]):
        return cls([simple_tokenize(t) for t in texts], doc_ids)

    def search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        q_terms = simple_tokenize(query)
        if not q_terms:
            return {}
        q_counts = Counter(q_terms)
        scores = [0.0] * self.N

        for term, qf in q_counts.items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        pairs = [(self.doc_ids[i], s) for i, s in enumerate(scores) if s > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return dict(pairs[:top_k])
