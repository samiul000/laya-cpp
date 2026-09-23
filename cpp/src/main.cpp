#include "laya.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    std::string model_path = (argc > 1) ? argv[1] : "/workspace/models/laya.onnx";
    std::cout << "[INFO] Loading model: " << model_path << std::endl;

    try {
        LayaEngine engine(model_path, 4);

        // Fixed test input: length 32, 2 marker positions
        int64_t seq_len = 32;
        std::vector<int64_t> input_ids(seq_len, 101);
        std::vector<int64_t> attention_mask(seq_len, 1);
        std::vector<int64_t> marker_pos = {5, 10};
        std::vector<bool> marker_mask = {true, true};

        // 1. Single inference test
        auto act_probs = engine.predict(input_ids, attention_mask, marker_pos, marker_mask, 0);
        std::cout << "[SUCCESS] Forward pass executed successfully!" << std::endl;
        std::cout << "Act Prob 0: " << act_probs[0] << std::endl;
        std::cout << "Act Prob 1: " << act_probs[1] << std::endl;

        // 2. Loop latency benchmark
        int iterations = (argc > 2) ? std::stoi(argv[2]) : 50;
        std::cout << "\n[BENCHMARK] Running " << iterations << " warm iterations for latency..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            engine.predict(input_ids, attention_mask, marker_pos, marker_mask, 0);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / iterations;

        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Average CPU Latency: " << avg_ms << " ms" << std::endl;
        std::cout << "Estimated Control Rate: " << (1000.0 / avg_ms) << " Hz" << std::endl;
        std::cout << "------------------------------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}