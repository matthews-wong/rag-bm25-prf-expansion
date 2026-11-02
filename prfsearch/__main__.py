"""Search a JSONL corpus: python3 -m prfsearch "query" --corpus fixtures/corpus.jsonl."""

import argparse
import sys
from pathlib import Path

from .bm25 import BM25Index, query_weights
from .corpus import CorpusError, load_corpus
from .prf import DEFAULT_EXPANSION_TERMS, DEFAULT_FEEDBACK_DOCS, expand_query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prfsearch", description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--corpus", type=Path, default=Path("fixtures/corpus.jsonl"))
    parser.add_argument("--prf", action="store_true", help="expand the query from the top results")
    parser.add_argument("--feedback-docs", type=int, default=DEFAULT_FEEDBACK_DOCS)
    parser.add_argument("--expansion-terms", type=int, default=DEFAULT_EXPANSION_TERMS)
    parser.add_argument("-n", "--limit", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        index = BM25Index(load_corpus(args.corpus))
        query = query_weights(args.query)
        if args.prf:
            query = expand_query(
                index, query, feedback_docs=args.feedback_docs, expansion_terms=args.expansion_terms
            )
    except (CorpusError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("query: " + " ".join(f"{t}^{w:.2f}" if w != 1.0 else t for t, w in query.items()))
    for doc_id, score in index.search(query, limit=args.limit):
        print(f"{score:6.3f}  {doc_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
