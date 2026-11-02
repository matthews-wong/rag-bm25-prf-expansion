import unittest

from prfsearch.bm25 import BM25Index, query_weights
from prfsearch.prf import expand_query

DOCS = {
    "d1": "invoice dispute credit memo accounting",
    "d2": "invoice credit memo refund accounting ledger",
    "d3": "holiday calendar office closed",
    "d4": "credit memo ledger posting accounting",
}


class ExpandQueryTest(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(DOCS)
        self.query = query_weights("invoice dispute")

    def test_adds_terms_from_top_documents(self):
        got = expand_query(self.index, self.query, expansion_terms=3)
        self.assertEqual(set(self.query), set(got) & set(self.query))
        self.assertTrue({"credit", "memo", "accounting"} & set(got))
        self.assertNotIn("holiday", got)

    def test_original_terms_keep_full_weight(self):
        got = expand_query(self.index, self.query)
        for term, weight in self.query.items():
            self.assertEqual(got[term], weight)

    def test_expansion_weight_is_shared_across_new_terms(self):
        got = expand_query(self.index, self.query, expansion_weight=0.4)
        added = sum(w for t, w in got.items() if t not in self.query)
        self.assertAlmostEqual(added, 0.4)

    def test_no_hits_leaves_query_unchanged(self):
        query = query_weights("zebra")
        self.assertEqual(expand_query(self.index, query), query)

    def test_zero_expansion_terms_is_identity(self):
        self.assertEqual(expand_query(self.index, self.query, expansion_terms=0), self.query)

    def test_invalid_settings_name_the_values(self):
        with self.assertRaisesRegex(ValueError, "feedback_docs=0"):
            expand_query(self.index, self.query, feedback_docs=0)


if __name__ == "__main__":
    unittest.main()
