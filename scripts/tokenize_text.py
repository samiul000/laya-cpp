"""Encode raw text to token IDs (pipe into laya inputs)."""

import sys

from tokenizers import Tokenizer  # pip install tokenizers

text = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else "models/tokenizer/tokenizer.json"

print(" ".join(map(str, Tokenizer.from_file(path).encode(text).ids)))
