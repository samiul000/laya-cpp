# Integration

## Use as a C++ library (recommended)

The engine is two files with no project-specific dependencies copy them:

- `cpp/include/laya.hpp`
- `cpp/src/laya.cpp`

Requirements: C++17 and ONNX Runtime headers + `libonnxruntime.so` (1.23.x;
the version must be new enough for your model export.

```cmake
add_executable(my_robot main.cpp laya.cpp)
target_include_directories(my_robot PRIVATE . ${ORT_HOME}/include)
target_link_directories(my_robot PRIVATE ${ORT_HOME}/lib)
target_link_libraries(my_robot PRIVATE onnxruntime)
```

```cpp
#include "laya.hpp"

// Construct once: loads ~1.6 GB (FP32) / ~400 MB (INT8), takes seconds.
LayaEngine engine("/path/to/laya_int8.onnx", /*num_threads=*/4);

// Per decision tick (shapes: [1, seq], [1, seq], [1, opts], [1, opts], [1]):
std::vector<float> probs = engine.predict(
    input_ids, attention_mask, marker_pos, marker_mask, qtype);
// probs = [p0, p1]
```

Notes:

- FP32 and INT8 models are drop-in interchangeable (identical I/O).
- `Session::Run` is re-entrant: one shared engine across threads is fine.
- Full I/O contract (names, shapes, dtypes) is in the README's Model I/O
  table, match it exactly or `Run` throws.
- Keep `num_threads` modest (4); more threads did not help on heterogeneous
  laptop CPUs (see Rate budget below).

## Use as a CLI (testing / scripting only)

```bash
laya_cli /path/to/model.onnx
```

Parses fixed demo inputs and prints `act_probs` plus a latency report. Fine
for smoke tests and shell pipelines. Not for production loops: every
invocation reloads the model (seconds) on top of inference.

## Suggested architecture

`laya` is a high-level selector (query + options in, action probabilities
out), not a planner or controller. Deploy it one level above the fast loop:

```
perception → candidate modes/routes (classical planner: A*, RRT, state machine)
           → laya scores options (~4–7 Hz)
           → selected mode → low-level controller (PID/MPC, 10–50 Hz)
```

Planner proposes, laya disposes: generate N candidates classically, encode
each as markers/options, execute the highest `act_prob`.

## Rate budget (measured, i3-1215U)

| Model | Latency          | Fits                              |
| ----- | ---------------- | --------------------------------- |
| FP32  | ~438 ms / 2.3 Hz | 250 ms task selector, marginally  |
| INT8  | ~171 ms / 5.8 Hz | 250 ms task selector, comfortably |

Neither fits a 10–50 Hz reactive loop. Do not call the model per search node
(e.g. as an A\* heuristic) at ~6 Hz the search would stall. Replan
route/mode selection at a few Hz; run obstacle avoidance and tracking
underneath classically.

## ROS 2 sketch

Wrap the engine in a node: subscribe to perception/marker inputs, run
inference on a timer (~4 Hz), publish the selected macro-action/mode.
Keep inference off the subscription callback so a slow forward pass never
blocks message handling. Re-run `scripts/bench_threads.py` and `laya_cli` on
the target board (e.g. ARM edge computer) before fixing the timer rate,
laptop numbers do not transfer.

## Thread count

`main.cpp` constructs `LayaEngine(model_path, 4)`. A thread sweep
(`scripts/bench_threads.py`) on the reference laptop showed ±30% run-to-run
noise swamping any thread-count effect (heterogeneous P/E cores, thermals,
host contention), so 4 is kept as a sane default. Re-sweep on deployment
hardware if latency is marginal.
