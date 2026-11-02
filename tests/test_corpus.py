import tempfile
import unittest
from pathlib import Path

from prfsearch.corpus import CorpusError, load_corpus


def load(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.jsonl"
        path.write_text(text, encoding="utf-8")
        return load_corpus(path)


class LoadCorpusTest(unittest.TestCase):
    def test_loads_rows_and_skips_blank_lines(self):
        self.assertEqual(load('{"id":"a","text":"x"}\n\n'), {"a": "x"})

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(CorpusError, "duplicate document id 'a'"):
            load('{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n')

    def test_missing_text_rejected(self):
        with self.assertRaisesRegex(CorpusError, "string 'id' and 'text'"):
            load('{"id":"a"}\n')

    def test_bad_json_names_line(self):
        with self.assertRaisesRegex(CorpusError, ":2: invalid JSON"):
            load('{"id":"a","text":"x"}\nnope\n')


if __name__ == "__main__":
    unittest.main()
