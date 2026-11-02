import unittest

from prfsearch.bm25 import BM25Index, query_weights


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(
            {"a": "refund policy for damaged goods", "b": "shipping time estimates", "c": "refund refund refund"}
        )

    def test_matching_documents_rank_first(self):
        ranked = self.index.search(query_weights("refund policy"))
        self.assertEqual(ranked[0][0], "a")

    def test_non_matching_documents_are_omitted(self):
        self.assertEqual([d for d, _ in self.index.search(query_weights("refund"))], ["c", "a"])

    def test_limit_applies(self):
        self.assertEqual(len(self.index.search(query_weights("refund"), limit=1)), 1)

    def test_weights_scale_the_score(self):
        one = self.index.score("a", {"refund": 1.0})
        half = self.index.score("a", {"refund": 0.5})
        self.assertAlmostEqual(half * 2, one)

    def test_empty_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty corpus"):
            BM25Index({})

    def test_stopwords_dropped_from_query(self):
        self.assertEqual(query_weights("what is the refund"), {"refund": 1.0})


if __name__ == "__main__":
    unittest.main()
