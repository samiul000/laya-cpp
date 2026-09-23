"""Time one predict-equivalent inference at several intra-op thread counts."""

import sys
import time

import numpy as np
import onnxruntime as ort

model = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/laya.onnx"
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15

feed = {
    "input_ids": np.full((1, 32), 101, dtype=np.int64),
    "attention_mask": np.ones((1, 32), dtype=np.int64),
    "marker_pos": np.array([[5, 10]], dtype=np.int64),
    "marker_mask": np.array([[True, True]]),
    "qtype": np.array([0], dtype=np.int64),
}

for threads in (1, 2, 3, 4):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    s = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
    s.run(None, feed)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        s.run(None, feed)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"threads={threads} avg={ms:.1f}ms ({1000 / ms:.1f}Hz)", flush=True)
