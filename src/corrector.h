// corrector.h — LLM text correction via llama.cpp (in-process, no HTTP)
#pragma once

#include <string>

struct llama_model;
struct llama_context;
struct llama_sampler;

struct Corrector {
    // Load the GGUF model and create context. Returns false on failure.
    bool init(const std::string& model_path, int n_gpu_layers = 99);

    // Correct a raw transcription. Returns corrected text, or the original
    // if the model output fails safety checks.
    std::string correct(const std::string& raw_text);

    void free();

    double last_correct_ms = 0;

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;

    bool is_safe_correction(const std::string& original, const std::string& corrected);
};
