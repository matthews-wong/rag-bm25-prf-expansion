import contextlib
import io
import unittest
from pathlib import Path

from prfsearch.evaluate import main, recall_at_k, reciprocal_rank

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class MetricsTest(unittest.TestCase):
    def test_recall_counts_relevant_in_top_k(self):
        self.assertEqual(recall_at_k(["x", "a", "b"], {"a", "b"}, 2), 0.5)

    def test_reciprocal_rank_uses_first_hit(self):
        self.assertEqual(reciprocal_rank(["x", "y", "a"], {"a", "b"}), 1 / 3)

    def test_reciprocal_rank_zero_without_hit(self):
        self.assertEqual(reciprocal_rank(["x"], {"a"}), 0.0)


class MainTest(unittest.TestCase):
    def run_main(self, *args: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_fixture_report_states_dataset_size(self):
        code, out, _ = self.run_main(
            "--corpus", str(FIXTURES / "corpus.jsonl"), "--queries", str(FIXTURES / "queries.jsonl")
        )
        self.assertEqual(code, 0)
        self.assertIn("8 queries over 16 documents", out)
        self.assertIn("bm25+prf", out)

    def test_missing_file_is_an_error(self):
        code, _, err = self.run_main("--corpus", str(FIXTURES / "nope.jsonl"))
        self.assertEqual(code, 2)
        self.assertIn("nope.jsonl", err)


if __name__ == "__main__":
    unittest.main()
