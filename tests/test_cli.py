import contextlib
import io
import unittest
from pathlib import Path

from prfsearch.__main__ import main

CORPUS = str(Path(__file__).resolve().parent.parent / "fixtures" / "corpus.jsonl")


def run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_plain_search_lists_hits(self):
        code, out, _ = run("direct debit", "--corpus", CORPUS)
        self.assertEqual(code, 0)
        self.assertIn("pay-2", out)

    def test_prf_prints_weighted_expansion_terms(self):
        _, out, _ = run("failed rollout", "--corpus", CORPUS, "--prf")
        self.assertIn("^", out.splitlines()[0])

    def test_missing_corpus_is_an_error(self):
        code, _, err = run("x", "--corpus", CORPUS + ".missing")
        self.assertEqual(code, 2)
        self.assertIn("missing", err)

    def test_invalid_prf_settings_are_an_error(self):
        code, _, err = run("refund", "--corpus", CORPUS, "--prf", "--feedback-docs", "0")
        self.assertEqual(code, 2)
        self.assertIn("feedback_docs=0", err)


if __name__ == "__main__":
    unittest.main()
