#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class LayaEngine {
public:
    LayaEngine(const std::string& model_path, int num_threads = 4);
    ~LayaEngine() = default;

    // Returns calibrated action probabilities [p0, p1]
    std::vector<float> predict(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        const std::vector<int64_t>& marker_pos,
        const std::vector<bool>& marker_mask,
        int64_t qtype = 0
    );

private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
};