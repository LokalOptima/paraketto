// corrector.cpp — LLM text correction via llama.cpp
//
// Embeds OLMoE-1B-7B as a text corrector for ASR output.
// Uses few-shot prompting (base model, no chat template needed).

#include "corrector.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "llama.h"
#include "ggml.h"

// Suppress verbose llama.cpp logging (model metadata, graph reservations, etc.)
static void llama_log_quiet(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN)
        fprintf(stderr, "%s", text);
}

// ---------------------------------------------------------------------------
// Few-shot correction prompt (from llm-server/server.py)
// ---------------------------------------------------------------------------

static const char* CORRECT_PROMPT =
    "Remove filler words (uh, um) and stuttered repetitions. "
    "Fix capitalization and punctuation. "
    "Do NOT change, replace, rephrase, or reorder any other words. "
    "Copy everything else exactly.\n\n"
    "IN: I I think we should uh probably go with the first option\n"
    "OUT: I think we should probably go with the first option.\n\n"
    "IN: it's uh it's basically um a wrapper around the the main API\n"
    "OUT: It's basically a wrapper around the main API.\n\n"
    "IN: the function takes a list of no sorry a dictionary of parameters\n"
    "OUT: The function takes a list of, no sorry, a dictionary of parameters.\n\n"
    "IN: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour fattened sauce,\n"
    "OUT: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour fattened sauce.\n\n"
    "IN: uh what\n"
    "OUT: What?\n\n"
    "IN: Please review the pull request before merging.\n"
    "OUT: Please review the pull request before merging.\n\n"
    "IN: Uh recently we started working to with ABB, one of the world's leading global technology companies.\n"
    "OUT: Recently, we started working to with ABB, one of the world's leading global technology companies.\n\n"
    "IN: um ok go ahead\n"
    "OUT: OK, go ahead.";

// ---------------------------------------------------------------------------
// Safety validation (port of _is_safe_correction from server.py)
// ---------------------------------------------------------------------------

static const std::unordered_set<std::string> FILLERS = {
    "um", "uh", "like", "you", "know", "i", "mean", "so", "basically", "right"
};

// Extract lowercase alphanumeric+apostrophe word tokens.
static std::vector<std::string> extract_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        char lc = (c >= 'A' && c <= 'Z') ? (c + 32) : c;
        if ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9') || lc == '\'') {
            current += lc;
        } else if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) words.push_back(std::move(current));
    return words;
}

// Detect repeated n-grams (sign of hallucination/looping).
static bool has_repetition(const std::vector<std::string>& words, int ngram_size = 4) {
    if ((int)words.size() < ngram_size * 2) return false;
    std::unordered_set<std::string> seen;
    for (int i = 0; i <= (int)words.size() - ngram_size; i++) {
        std::string key;
        for (int j = 0; j < ngram_size; j++) {
            if (j > 0) key += ' ';
            key += words[i + j];
        }
        if (!seen.insert(key).second) return true;
    }
    return false;
}

bool Corrector::is_safe_correction(const std::string& original, const std::string& corrected) {
    auto orig_words = extract_words(original);
    auto corr_words = extract_words(corrected);

    if (corr_words.empty()) return false;

    std::unordered_set<std::string> orig_set(orig_words.begin(), orig_words.end());
    std::unordered_set<std::string> corr_set(corr_words.begin(), corr_words.end());

    // No new words allowed — the corrector should only remove, not invent
    int added = 0;
    for (auto& w : corr_set)
        if (orig_set.find(w) == orig_set.end()) added++;

    if (added > 0) {
        fprintf(stderr, "corrector: safety: model added %d new word(s)\n", added);
        return false;
    }

    // Word count ratio: output should be 60-105% of input length
    // (only fillers/stutters should be removed, nothing else)
    double ratio = (double)corr_words.size() / orig_words.size();
    if (ratio < 0.6 || ratio > 1.05) {
        fprintf(stderr, "corrector: safety: word count ratio %.2f (orig=%zu, corr=%zu)\n",
                ratio, orig_words.size(), corr_words.size());
        return false;
    }

    // Unique words dropped (excluding fillers) — tighter threshold
    std::unordered_set<std::string> meaningful_orig;
    for (auto& w : orig_set)
        if (FILLERS.find(w) == FILLERS.end()) meaningful_orig.insert(w);

    int dropped = 0;
    for (auto& w : meaningful_orig)
        if (corr_set.find(w) == corr_set.end()) dropped++;

    if (!meaningful_orig.empty() &&
        (double)dropped / meaningful_orig.size() > 0.15) {
        fprintf(stderr, "corrector: safety: model dropped %.0f%% of unique words\n",
                100.0 * dropped / meaningful_orig.size());
        return false;
    }

    // Repetition detection (hallucination/looping)
    if (has_repetition(corr_words)) {
        fprintf(stderr, "corrector: safety: repeated n-gram detected\n");
        return false;
    }

    // Output shouldn't be longer than input
    if (corrected.size() > original.size() + 20) {
        fprintf(stderr, "corrector: safety: output length %zu > input length %zu\n",
                corrected.size(), original.size());
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Init / Free
// ---------------------------------------------------------------------------

bool Corrector::init(const std::string& model_path, int n_gpu_layers) {
    ggml_log_set(llama_log_quiet, nullptr);
    llama_log_set(llama_log_quiet, nullptr);
    ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "corrector: failed to load model: %s\n", model_path.c_str());
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    cparams.no_perf = true;

    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "corrector: failed to create context\n");
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    // Greedy sampler (temperature = 0)
    sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // Warmup: run a trivial correction to trigger CUDA kernel compilation
    fprintf(stderr, "corrector: warming up...\n");
    correct("hello");

    return true;
}

void Corrector::free() {
    if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
}

// ---------------------------------------------------------------------------
// Correction
// ---------------------------------------------------------------------------

std::string Corrector::correct(const std::string& raw_text) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Skip correction for long texts — they already have excellent WER
    // and the model tends to truncate or rearrange them
    auto input_words = extract_words(raw_text);
    if (input_words.size() > 100) {
        auto t1 = std::chrono::high_resolution_clock::now();
        last_correct_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return raw_text;
    }

    // Build the full prompt
    std::string prompt = std::string(CORRECT_PROMPT) + "\n\nIN: " + raw_text + "\nOUT:";

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize
    int n_prompt_max = prompt.size() + 128;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   tokens.data(), n_prompt_max,
                                   true,   // add_bos
                                   false); // parse_special
    if (n_tokens < 0) {
        fprintf(stderr, "corrector: tokenization failed\n");
        return raw_text;
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Prefill: decode the prompt in one batch
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "corrector: prefill failed\n");
        return raw_text;
    }

    // Generate
    std::string generated;
    int max_tokens = std::max(256, (int)raw_text.size());

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        // Check EOS/EOG
        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Detokenize the new token
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            generated.append(buf, n);
        }

        // Check stop sequences
        if (generated.size() >= 4) {
            if (generated.rfind("\nIN:") == generated.size() - 4) {
                generated.resize(generated.size() - 4);
                break;
            }
        }
        if (generated.size() >= 2) {
            if (generated.rfind("\n\n") == generated.size() - 2) {
                generated.resize(generated.size() - 2);
                break;
            }
        }

        // Decode the new token (pos auto-assigned from KV cache state)
        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1)) != 0) {
            fprintf(stderr, "corrector: decode failed at token %d\n", i);
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_correct_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Trim whitespace
    size_t start = generated.find_first_not_of(" \t\n\r");
    size_t end = generated.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) {
        fprintf(stderr, "corrector: empty output, using original\n");
        return raw_text;
    }
    std::string corrected = generated.substr(start, end - start + 1);

    // Safety check
    if (corrected != raw_text && !is_safe_correction(raw_text, corrected)) {
        fprintf(stderr, "corrector: safety check failed, using original\n");
        return raw_text;
    }

    return corrected;
}
