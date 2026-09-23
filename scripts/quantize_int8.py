import sys

import onnx

from onnxruntime.quantization import QuantType, quantize_dynamic

src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/laya.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/laya_int8.onnx"

print(f"Loading {src} ...", flush=True)
model = onnx.load(src, load_external_data=True)
del model.graph.value_info[:]

print(f"Quantizing -> {dst} ...", flush=True)
quantize_dynamic(
    model_input=model,
    model_output=dst,
    weight_type=QuantType.QInt8,
    use_external_data_format=True,
)
print("Done.", flush=True)
