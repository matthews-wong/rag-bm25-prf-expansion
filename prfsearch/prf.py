"""Pseudo-relevance feedback: expand a query with terms from its top results."""

from collections import defaultdict

from .bm25 import BM25Index

DEFAULT_FEEDBACK_DOCS = 3
DEFAULT_EXPANSION_TERMS = 5
DEFAULT_EXPANSION_WEIGHT = 0.5


def expand_query(
    index: BM25Index,
    query: dict[str, float],
    *,
    feedback_docs: int = DEFAULT_FEEDBACK_DOCS,
    expansion_terms: int = DEFAULT_EXPANSION_TERMS,
    expansion_weight: float = DEFAULT_EXPANSION_WEIGHT,
) -> dict[str, float]:
    """Return the query plus up to `expansion_terms` new terms.

    Candidate terms are scored by length-normalised frequency in the feedback
    documents times idf, so a term common to the whole corpus does not win.
    Expansion terms share `expansion_weight` in total, which keeps the
    original terms dominant and limits query drift.
    """
    if feedback_docs < 1 or expansion_terms < 0 or not 0 <= expansion_weight <= 1:
        raise ValueError(
            f"invalid PRF settings: feedback_docs={feedback_docs}, "
            f"expansion_terms={expansion_terms}, expansion_weight={expansion_weight}"
        )
    top = index.search(query, limit=feedback_docs)
    candidates: dict[str, float] = defaultdict(float)
    for doc_id, _ in top:
        length = index.lengths[doc_id] or 1
        for term, freq in index.tf[doc_id].items():
            if term not in query:
                candidates[term] += freq / length * index.idf[term]
    best = sorted(candidates.items(), key=lambda c: (-c[1], c[0]))[:expansion_terms]
    total = sum(score for _, score in best)
    if not best or total == 0:
        return dict(query)
    expanded = dict(query)
    for term, score in best:
        expanded[term] = expansion_weight * score / total
    return expanded
