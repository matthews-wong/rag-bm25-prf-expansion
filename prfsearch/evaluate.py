"""Recall@k and MRR for plain BM25 versus BM25 with PRF expansion."""

import argparse
import sys
from pathlib import Path

from .bm25 import BM25Index, query_weights
from .corpus import CorpusError, load_corpus, read_jsonl
from .prf import expand_query

DEFAULT_K = 3


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / position
    return 0.0


def evaluate(index: BM25Index, queries: list[dict], k: int) -> dict[str, tuple[float, float]]:
    totals = {"bm25": [0.0, 0.0], "bm25+prf": [0.0, 0.0]}
    for row in queries:
        relevant = set(row["relevant"])
        base_query = query_weights(row["query"])
        for name, query in (("bm25", base_query), ("bm25+prf", expand_query(index, base_query))):
            ranked = [doc_id for doc_id, _ in index.search(query, limit=len(index.tf))]
            totals[name][0] += recall_at_k(ranked, relevant, k)
            totals[name][1] += reciprocal_rank(ranked, relevant)
    n = len(queries)
    return {name: (recall / n, rr / n) for name, (recall, rr) in totals.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prfsearch.evaluate", description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("fixtures/corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("fixtures/queries.jsonl"))
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)
    try:
        docs = load_corpus(args.corpus)
        queries = read_jsonl(args.queries)
        if not queries:
            raise CorpusError(f"{args.queries}: no queries")
        for row in queries:
            unknown = sorted(set(row["relevant"]) - set(docs))
            if unknown:
                raise CorpusError(f"query {row['query']!r} names unknown documents: {', '.join(unknown)}")
        scores = evaluate(BM25Index(docs), queries, args.k)
    except (CorpusError, KeyError, OSError) as exc:
        print(f"error: {exc!r}" if isinstance(exc, KeyError) else f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{len(queries)} queries over {len(docs)} documents ({args.corpus})")
    print(f"{'method':<10} recall@{args.k}  MRR")
    for name, (recall, mrr) in scores.items():
        print(f"{name:<10} {recall:8.3f}  {mrr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
