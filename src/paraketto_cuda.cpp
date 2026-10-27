// paraketto_cuda.cpp — CUDA runtime for Parakeet V2, V3, and Ultra
//
// Build: make paraketto.cuda
// Usage: ./paraketto.cuda [--weights FILE] audio.wav

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "paraketto.h"
#include "server.h"

#ifdef WITH_CORRECTOR
#include "corrector.h"
#endif

using namespace paraketto;

// Compute backend name, injected per-binary by the Makefile link recipe
// (CUTLASS FP16 / cuBLAS FP16). Falls back for ad-hoc builds.
#ifndef BACKEND_NAME
#define BACKEND_NAME "unknown backend"
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto usage = [&]() {
        fprintf(stderr,
            "Usage: %s [OPTIONS] <wav_file>...\n"
            "\n"
            "Speech-to-text using Paraketto (CUDA/cuBLAS backend).\n"
            "Accepts one or more 16kHz or 24kHz mono WAV files (int16 or float32).\n"
            "Long files (>120s) are automatically chunked at silence boundaries.\n"
            "Weights are auto-downloaded from HuggingFace on first run.\n"
            "\n"
            "Options:\n"
            "  --model v2|v3              Model version [default: v2 English, v3 multilingual]\n"
            "  --weights FILE             Model weights [default: ~/.cache/paraketto/]\n"
            "  --server [[host]:port]     Start HTTP server [default: 0.0.0.0:8080]\n"
#ifdef WITH_CORRECTOR
            "  --correct                  Enable LLM text correction\n"
            "  --llm-model FILE           LLM weights [default: ~/.cache/paraketto/olmoe-1b-7b-0924-q4_k_m.gguf]\n"
#endif
            "  -h, --help                 Show this help\n",
            argv[0]);
    };

    if (argc < 2) { usage(); return 1; }

    std::string dir = cache_dir();
    bool custom_weights = false;
    int model_version = 2;  // default: V2 English
    std::string weights_path;
    std::vector<std::string> wav_files;
    bool server_mode = false;
    bool warmup = false;
    std::string server_host = "0.0.0.0";
    int server_port = 8080;
#ifdef WITH_CORRECTOR
    bool enable_correct = false;
    std::string llm_model_path = dir + "/olmoe-1b-7b-0924-q4_k_m.gguf";
#endif

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        if (arg == "--model" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "v2" || v == "2") model_version = 2;
            else if (v == "v3" || v == "3") model_version = 3;
            else { fprintf(stderr, "Unknown model version: %s (use v2 or v3)\n", v.c_str()); return 1; }
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
            custom_weights = true;
#ifdef WITH_CORRECTOR
        } else if (arg == "--correct") {
            enable_correct = true;
        } else if (arg == "--llm-model" && i + 1 < argc) {
            llm_model_path = argv[++i];
            enable_correct = true;
#endif
        } else if (arg == "--warmup") {
            warmup = true;
        } else if (arg == "--server") {
            server_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string addr = argv[++i];
                auto colon = addr.rfind(':');
                if (colon != std::string::npos) {
                    if (colon > 0) server_host = addr.substr(0, colon);
                    server_port = std::stoi(addr.substr(colon + 1));
                }
            }
        } else {
            wav_files.push_back(arg);
        }
    }

    if (!server_mode && wav_files.empty()) {
        fprintf(stderr, "No WAV files specified.\n");
        return 1;
    }

    // Set default weight paths based on model version (if not overridden by --weights)
    const char* hf_base = "https://huggingface.co/localoptima/paraketto/resolve/main";
    if (!custom_weights) {
        if (model_version == 3) {
            weights_path = dir + "/paraketto-v3-fp16.bin";
        } else {
            weights_path = dir + "/paraketto-fp16.bin";
        }
    }

#ifdef WITH_CORRECTOR
    if (server_mode) enable_correct = true;
#endif

    using clk = std::chrono::high_resolution_clock;
    auto t_main_start = clk::now();

    // Download weights if missing
    if (!custom_weights) {
        const char* fname;
        fname = (model_version == 3) ? "paraketto-v3-fp16.bin" : "paraketto-fp16.bin";
        std::string url = std::string(hf_base) + "/" + fname;
        ensure_file(weights_path, url.c_str());
    }

    Weights prefetched;
    // Prefetch in background while CUDA context inits.
    std::thread prefetch_thread([&]() {
        prefetched = Weights::prefetch(weights_path);
    });

    // Force eager CUDA context initialization (overlaps with weight prefetch)
    cudaFree(0);
    auto t_cuda_init = clk::now();

    if (!server_mode) {
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    prefetch_thread.join();

    auto t_prefetch = clk::now();

    Pipeline pipeline;
    pipeline.init(std::move(prefetched));
    auto t_init_done = clk::now();

#ifdef WITH_CORRECTOR
    Corrector corrector;
    if (enable_correct) {
        ensure_file(llm_model_path,
            "https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF/resolve/main/"
            "olmoe-1b-7b-0924-q4_k_m.gguf");
        if (!corrector.init(llm_model_path)) {
            fprintf(stderr, "error: LLM correction model failed to load\n");
            return 1;
        }
    }
#endif

    // Warmup: trigger cublasLt lazy init before timed inference
    {
        std::vector<float> silence(16000, 0.0f);
        pipeline.transcribe(silence.data(), silence.size());
    }

    if (warmup && !server_mode) {
        WavData warmup_wav = read_wav(wav_files[0]);
        pipeline.transcribe(warmup_wav.samples.data(), warmup_wav.samples.size());
    }

    if (server_mode) {
        auto t_warmup_done = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t vram_free, vram_total;
        cudaMemGetInfo(&vram_free, &vram_total);

        auto file_size_mb = [](const std::string& path) -> double {
            struct stat st;
            return (stat(path.c_str(), &st) == 0) ? st.st_size / (1024.0 * 1024.0) : 0;
        };
        double weights_mb = file_size_mb(weights_path);
        int    version    = pipeline.weights.config.version;
        double startup_ms = ms(t_main_start, t_warmup_done);
        double vram_total_mb = vram_total / (1024.0 * 1024.0);
        double vram_free_mb  = vram_free  / (1024.0 * 1024.0);

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     parakeet-tdt-0.6b-v%d  ·  %s\n", version, BACKEND_NAME);
        fprintf(stderr, "weights:   %s (%.0f MB)\n", weights_path.c_str(), weights_mb);
        fprintf(stderr, "device:    %s (compute %d.%d, %.0f MB VRAM, %.0f MB free)\n",
                prop.name, prop.major, prop.minor, vram_total_mb, vram_free_mb);
        fprintf(stderr, "startup:   %.0f ms (cuda=%.0f prefetch=%.0f load=%.0f warmup=%.0f)\n",
                startup_ms, ms(t_main_start, t_cuda_init), ms(t_main_start, t_prefetch),
                ms(t_prefetch, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | GET /info | POST /transcribe\n");
#ifdef WITH_CORRECTOR
        if (enable_correct)
            fprintf(stderr, "corrector: enabled (%.0fms warmup)\n", corrector.last_correct_ms);
#endif
        fprintf(stderr, "\n");

        // Build the JSON served at GET /info so the web UI (and any client) can
        // show exactly which model version and backend are running.
        char nb[64];
        std::string info = "{";
        info += "\"model\":\"parakeet-tdt-0.6b-v" + std::to_string(version) + "\",";
        info += "\"version\":" + std::to_string(version) + ",";
        info += "\"backend\":\"" + json_escape(BACKEND_NAME) + "\",";
        info += "\"weights\":\"" + json_escape(weights_path) + "\",";
        snprintf(nb, sizeof nb, "%.0f", weights_mb);      info += "\"weights_mb\":" + std::string(nb) + ",";
        info += "\"device\":\"" + json_escape(prop.name) + "\",";
        info += "\"compute\":\"" + std::to_string(prop.major) + "." + std::to_string(prop.minor) + "\",";
        snprintf(nb, sizeof nb, "%.0f", vram_total_mb);   info += "\"vram_total_mb\":" + std::string(nb) + ",";
        snprintf(nb, sizeof nb, "%.0f", vram_free_mb);    info += "\"vram_free_mb\":" + std::string(nb) + ",";
        snprintf(nb, sizeof nb, "%.0f", startup_ms);      info += "\"startup_ms\":" + std::string(nb);
#ifdef WITH_CORRECTOR
        info += ",\"corrector\":";
        info += (enable_correct ? "true" : "false");
#endif
        info += "}";

#ifdef WITH_CORRECTOR
        run_server(pipeline, enable_correct ? &corrector : nullptr, info, server_host, server_port);
#else
        run_server(pipeline, info, server_host, server_port);
#endif
        return 0;
    }

    auto t_ready = clk::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    fprintf(stderr, "startup: %.0fms (cuda=%.0f prefetch=%.0f load=%.0f)\n",
            ms(t_main_start, t_ready), ms(t_main_start, t_cuda_init),
            ms(t_main_start, t_prefetch),
            ms(t_prefetch, t_init_done));

    // Process all files
    for (size_t fi = 0; fi < wav_files.size(); fi++) {
        WavData wav = read_wav(wav_files[fi]);
        double audio_dur = (double)wav.samples.size() / wav.sample_rate;

        auto t0 = clk::now();
        auto progress = [](int i, int n) {
            int w = 30;
            int filled = (i + 1) * w / n;
            fprintf(stderr, "\r  [");
            for (int j = 0; j < w; j++) fputc(j < filled ? '#' : '.', stderr);
            fprintf(stderr, "] %d/%d", i + 1, n);
            if (i + 1 == n) fputc('\n', stderr);
        };
        std::string text = pipeline.transcribe(wav.samples.data(), wav.samples.size(), progress);
        auto t1 = clk::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

#ifdef WITH_CORRECTOR
        if (enable_correct) {
            std::string corrected = corrector.correct(text);
            printf("%s\n", corrected.c_str());
            fprintf(stderr, "%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f correct=%.1f), %.0fx RTFx\n",
                    audio_dur, elapsed * 1000 + corrector.last_correct_ms,
                    pipeline.last_mel_ms, pipeline.last_enc_ms, pipeline.last_dec_ms,
                    corrector.last_correct_ms, audio_dur / elapsed);
        } else
#endif
        {
            printf("%s\n", text.c_str());
            fprintf(stderr, "%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f), %.0fx RTFx\n",
                    audio_dur, elapsed * 1000,
                    pipeline.last_mel_ms, pipeline.last_enc_ms, pipeline.last_dec_ms,
                    audio_dur / elapsed);
        }
    }

    return 0;
}
