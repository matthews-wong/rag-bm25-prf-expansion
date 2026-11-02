"""BM25 over an in-memory corpus, accepting weighted query terms."""

import math
from collections import Counter

from .tokenize import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, docs: dict[str, str]) -> None:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        self.tf = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        self.lengths = {doc_id: sum(tf.values()) for doc_id, tf in self.tf.items()}
        self.avg_length = sum(self.lengths.values()) / len(docs) or 1.0
        df: Counter[str] = Counter()
        for tf in self.tf.values():
            df.update(tf.keys())
        total = len(docs)
        # The +1 inside the log keeps idf positive for terms in most documents.
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, doc_id: str, query: dict[str, float]) -> float:
        tf = self.tf[doc_id]
        norm = K1 * (1 - B + B * self.lengths[doc_id] / self.avg_length)
        total = 0.0
        for term, weight in query.items():
            freq = tf.get(term, 0)
            if freq:
                total += weight * self.idf[term] * freq * (K1 + 1) / (freq + norm)
        return total

    def search(self, query: dict[str, float], limit: int = 10) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(doc_id, query)) for doc_id in self.tf]
        # Ties break on id so rankings are reproducible.
        ranked = sorted((s for s in scored if s[1] > 0), key=lambda s: (-s[1], s[0]))
        return ranked[:limit]


def query_weights(text: str) -> dict[str, float]:
    """Plain query: each distinct term weighs 1.0."""
    return {term: 1.0 for term in dict.fromkeys(tokenize(text))}
