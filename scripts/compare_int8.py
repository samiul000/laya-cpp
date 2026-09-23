import sys

import numpy as np
import onnxruntime as ort

fp32_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/laya.onnx"
int8_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/laya_int8.onnx"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

rng = np.random.default_rng(0)
s32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

max_logit = 0.0
max_prob = 0.0
sum_prob = 0.0
for _ in range(n):
    seq = 32
    feed = {
        "input_ids": rng.integers(0, 2000, size=(1, seq)).astype(np.int64),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "marker_pos": rng.integers(0, seq, size=(1, 2)).astype(np.int64),
        "marker_mask": np.array([[True, True]]),
        "qtype": np.array([int(rng.integers(0, 3))], dtype=np.int64),
    }
    l32, p32 = s32.run(None, feed)
    l8, p8 = s8.run(None, feed)
    max_logit = max(max_logit, float(np.max(np.abs(l32 - l8))))
    d = np.abs(p32 - p8)
    max_prob = max(max_prob, float(np.max(d)))
    sum_prob += float(np.mean(d))

print(f"cases={n} max|dlogit|={max_logit:.6f} max|dprob|={max_prob:.6f} mean|dprob|={sum_prob / n:.6f}")
