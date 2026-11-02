"""Lowercase word tokenizer with a small stopword list."""

import re

WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it of on or that the to was we what when with you".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in WORD.findall(text.lower()) if t not in STOPWORDS]
