"""Fetch the ONNX bundle (laya.onnx + sidecar) from Hugging Face into models/."""

import sys

from huggingface_hub import snapshot_download  # pip install huggingface_hub

repo = sys.argv[1] if len(sys.argv) > 1 else "receptron/laya-onnx"
dest = sys.argv[2] if len(sys.argv) > 2 else "models"

print(snapshot_download(
    repo,
    local_dir=dest,
    allow_patterns=["laya.onnx", "laya.onnx.data", "laya_config.json"],
))
