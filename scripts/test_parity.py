import sys
import numpy as np
import onnxruntime as ort

model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/laya.onnx"
session = ort.InferenceSession(model_path)

seq_len = 32
input_ids = np.full((1, seq_len), 101, dtype=np.int64)
attention_mask = np.ones((1, seq_len), dtype=np.int64)
marker_pos = np.array([[5, 10]], dtype=np.int64)
marker_mask = np.array([[True, True]], dtype=bool)
qtype = np.array([0], dtype=np.int64)

outputs = session.run(
    None,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    },
)

act_probs = outputs[1][0]
print(f"[PYTHON REFERENCE] Act Prob 0: {act_probs[0]:.6f}")
print(f"[PYTHON REFERENCE] Act Prob 1: {act_probs[1]:.6f}")