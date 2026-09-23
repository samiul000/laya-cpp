#include "laya.hpp"
#include <iostream>
#include <stdexcept>

LayaEngine::LayaEngine(const std::string& model_path, int num_threads)
    : env(ORT_LOGGING_LEVEL_WARNING, "LayaEngine"),
      session(nullptr),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(num_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    session = Ort::Session(env, model_path.c_str(), options);
}

std::vector<float> LayaEngine::predict(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    const std::vector<int64_t>& marker_pos,
    const std::vector<bool>& marker_mask,
    int64_t qtype_val) {

    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    int64_t num_markers = static_cast<int64_t>(marker_pos.size());

    std::vector<int64_t> seq_shape = {1, seq_len};
    std::vector<int64_t> marker_shape = {1, num_markers};
    std::vector<int64_t> qtype_shape = {1};
    std::vector<int64_t> qtype_tensor = {qtype_val};

    std::vector<uint8_t> marker_mask_bytes(marker_mask.begin(), marker_mask.end());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), seq_shape.data(), 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(attention_mask.data()), attention_mask.size(), seq_shape.data(), 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(marker_pos.data()), marker_pos.size(), marker_shape.data(), 2));
    input_tensors.push_back(Ort::Value::CreateTensor<bool>(
        memory_info, reinterpret_cast<bool*>(marker_mask_bytes.data()), marker_mask_bytes.size(), marker_shape.data(), 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, qtype_tensor.data(), 1, qtype_shape.data(), 1));

    const char* input_names[] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
    const char* output_names[] = {"logits", "act_probs"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, input_tensors.data(), input_tensors.size(), 
        output_names, 2
    );

    if (outputs.size() < 2)
        throw std::runtime_error("expected 2 model outputs (logits, act_probs)");

    float* act_ptr = outputs[1].GetTensorMutableData<float>();
    return {act_ptr[0], act_ptr[1]};
}