# rag-bm25-prf-expansion

BM25 retrieval with pseudo-relevance feedback (PRF) query expansion, and a
small check that compares it with plain BM25. Pure standard-library Python,
no model downloads, runs offline.

Lexical search misses documents that use different words than the query.
PRF takes the top few results, picks the terms that characterise them
(frequency in the document times idf) and adds those to the query with a
reduced weight.

## Usage

```bash
python3 -m prfsearch "failed rollout" --prf
python3 -m prfsearch "direct debit" --corpus fixtures/corpus.jsonl -n 3
```

Corpus format is one JSON object per line: `{"id": "dep-1", "text": "..."}`.
Options: `--feedback-docs` (default 3), `--expansion-terms` (default 5).

## Check

```bash
make validate   # unit tests + the fixture eval
```

Fixture result (8 hand-written queries over 16 documents in `fixtures/`,
not a real-world benchmark):

```text
method     recall@3  MRR
bm25          0.438  0.542
bm25+prf      0.500  0.542
```

With this few queries the difference is anecdotal; the point is the harness.

## Design notes

- Expansion terms share a fixed total weight (default 0.5), so the original
  query terms stay dominant.
- PRF needs at least one hit to learn from. A query with no word in common
  with the corpus is returned unchanged.
- PRF can drift: `failed rollout` pulls in `card` from a payments document
  that also says "failed". Fewer feedback documents or terms reduces this.
- Ties in ranking break on document id, so output is reproducible.
