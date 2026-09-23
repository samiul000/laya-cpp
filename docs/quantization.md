# INT8 Quantization

Dynamic quantization (no calibration data needed) of `models/laya.onnx`:
1.6 GB → 403 MB, ~438 ms → ~171 ms per forward pass (i3-1215U, Docker).

## Run

```bash
python3 /workspace/scripts/quantize_int8.py   # -> models/laya_int8.onnx (+ .data)
/workspace/cpp/build/laya_cli /workspace/models/laya_int8.onnx
python3 /workspace/scripts/compare_int8.py    # deviation report
```

Requires the `onnx` package in the container (`pip install onnx`); it is
deliberately not in the `Dockerfile` since quantization is a one-time,
offline step.

## What the script works around

`quantize_dynamic` runs ONNX shape inference as preprocessing, which fails on
this export:

```
InferenceError: Inferred shape and existing shape differ in dimension 0: (1028) vs (256)
```

The file's baked `value_info` shapes are stale. The script loads the model
with `onnx`, deletes `value_info` (optional metadata the runtime ignores),
and passes the in-memory proto to the quantizer with
`use_external_data_format=True` so the 1.6 GB weights stay in a sidecar file.

## Accuracy assessment

`compare_int8.py` runs 20 fixed-seed randomized inputs through both models:

- max |Δlogit| ≈ 0.44, max |Δact_prob| = 0.0 (to 6 decimals)

Limits of that check: random token IDs are out-of-distribution, so outputs
saturate at 1.0/0.0 and hide movement. Near a real decision boundary the
probabilities will shift. Before trusting INT8 in production:

1. Compare on **real inputs**, especially near boundaries.
2. Note `models/laya_config.json` temperatures were fit on FP32 logits
   refit on INT8 logits if you operate near boundaries.
3. Judge by **task metrics** (success rate, episode return), never by logit
   distance alone.

## If you need more

Dynamic quantization is the no-data shortcut. Static quantization with real
calibration inputs preserves accuracy better and can be slightly faster, at
the cost of collecting a calibration set. Expect ~1.5–2.5x from quantization
in this family it will not take a 400M-param transformer to 15–30 Hz.
