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
#include "gpu_status.h"
#include "model_selection.h"


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
            "V2/V3 weights download on first run; Ultra weights need one-time conversion.\n"
            "\n"
            "Options:\n"
            "  --model v2|v3|ultra        Model preset [default: v2 English]\n"
            "                             v3 and ultra are multilingual (V3 architecture)\n"
            "  --weights FILE             Explicit weights; without --model, detect architecture\n"
            "                             Default: beside executable, then ~/.cache/paraketto/\n"
            "  --server [[host]:port]     Start HTTP server [default: 0.0.0.0:8080]\n"
            "  -h, --help                 Show this help\n",
            argv[0]);
    };

    if (argc < 2) { usage(); return 1; }

    std::string dir = cache_dir();
    bool custom_weights = false;
    bool explicit_model = false;
    const models::Preset* selected_model = &models::preset("v2");
    std::string weights_path;
    std::vector<std::string> wav_files;
    bool server_mode = false;
    bool warmup = false;
    std::string server_host = "0.0.0.0";
    int server_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        if (arg == "--model") {
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(stderr, "--model requires v2, v3, or ultra.\n");
                return 1;
            }
            try { selected_model = &models::preset(argv[++i]); }
            catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
            explicit_model = true;
        } else if (arg == "--weights") {
            if (i + 1 >= argc || !argv[i + 1][0] || argv[i + 1][0] == '-') {
                fprintf(stderr, "--weights requires a file path.\n");
                return 1;
            }
            weights_path = argv[++i];
            custom_weights = true;
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
        } else if (!arg.empty() && arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 1;
        } else {
            wav_files.push_back(arg);
        }
    }

    if (!server_mode && wav_files.empty()) {
        fprintf(stderr, "No WAV files specified.\n");
        return 1;
    }

    // Custom weights alone declare no checkpoint identity. V3 and Ultra have
    // the same header; only an explicit preset can distinguish their labels.
    if (custom_weights && !explicit_model) selected_model = nullptr;
    if (selected_model) {
        std::error_code error;
        auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
        if (error) executable = std::filesystem::absolute(argv[0]);
        weights_path = models::weights_path(*selected_model,
            executable.parent_path().string(), dir, weights_path);
    }

    using clk = std::chrono::high_resolution_clock;
    auto t_main_start = clk::now();

    // Only built-in V2/V3 paths can trigger the CLI's existing auto-download.
    // Explicit weights are never downloaded or replaced.
    if (!custom_weights) {
        if (selected_model->download_url[0]) {
            ensure_file(weights_path, selected_model->download_url);
        } else if (access(weights_path.c_str(), F_OK) != 0) {
            fprintf(stderr,
                "Ultra weights were not found at %s. No converted Ultra download is published.\n"
                "Prepare them once from the repository:\n"
                "  uvx hf download moondream/parakeet-ultra --revision "
                "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b --local-dir scratch/parakeet-ultra\n"
                "  uv run python scripts/export_ultra_weights.py scratch/parakeet-ultra\n"
                "Place paraketto-ultra-fp16.bin beside this executable or at the path above,\n"
                "or supply --model ultra --weights FILE.\n", weights_path.c_str());
            return 1;
        }
    }
    // Reject bad files and architecture mismatches before any CUDA context or
    // background prefetch starts, including when --weights precedes --model.
    try {
        const int actual = models::read_architecture(weights_path);
        if (selected_model) models::validate_architecture(*selected_model, actual);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
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

    // Validate the file actually mapped by prefetch as well, in case it was
    // replaced between the small header read and the mapping.
    if (selected_model) {
        try { models::validate_architecture(*selected_model, prefetched.config.version); }
        catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
    }

    auto t_prefetch = clk::now();

    Pipeline pipeline;
    pipeline.init(std::move(prefetched));
    auto t_init_done = clk::now();


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
        const std::string model_label = models::label(selected_model, version);
        double startup_ms = ms(t_main_start, t_warmup_done);
        double vram_total_mb = vram_total / (1024.0 * 1024.0);
        double vram_free_mb  = vram_free  / (1024.0 * 1024.0);

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     %s (architecture v%d)  ·  %s\n",
                model_label.c_str(), version, BACKEND_NAME);
        fprintf(stderr, "weights:   %s (%.0f MB)\n", weights_path.c_str(), weights_mb);
        fprintf(stderr, "device:    %s (compute %d.%d, %.0f MB VRAM, %.0f MB free)\n",
                prop.name, prop.major, prop.minor, vram_total_mb, vram_free_mb);
        fprintf(stderr, "startup:   %.0f ms (cuda=%.0f prefetch=%.0f load=%.0f warmup=%.0f)\n",
                startup_ms, ms(t_main_start, t_cuda_init), ms(t_main_start, t_prefetch),
                ms(t_prefetch, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | GET /info | POST /transcribe\n");
        fprintf(stderr, "\n");

        // Build the JSON served at GET /info so the web UI (and any client) can
        // show exactly which model version and backend are running.
        char nb[64];
        std::string info = "{";
        info += "\"model\":\"" + json_escape(model_label) + "\",";
        info += "\"model_preset\":\"" + std::string(selected_model ? selected_model->key : "custom") + "\",";
        info += "\"model_architecture\":\"v" + std::to_string(version) + "\",";
        info += "\"version\":" + std::to_string(version) + ",";
        info += "\"backend\":\"" + json_escape(BACKEND_NAME) + "\",";
        info += "\"weights\":\"" + json_escape(weights_path) + "\",";
        snprintf(nb, sizeof nb, "%.0f", weights_mb);      info += "\"weights_mb\":" + std::string(nb) + ",";
        info += "\"device\":\"" + json_escape(prop.name) + "\",";
        info += "\"compute\":\"" + std::to_string(prop.major) + "." + std::to_string(prop.minor) + "\",";
        snprintf(nb, sizeof nb, "%.0f", vram_total_mb);   info += "\"vram_total_mb\":" + std::string(nb) + ",";
        snprintf(nb, sizeof nb, "%.0f", vram_free_mb);    info += "\"vram_free_mb\":" + std::string(nb) + ",";
        snprintf(nb, sizeof nb, "%.0f", startup_ms);      info += "\"startup_ms\":" + std::string(nb);
        info += "}";

        GpuStatus gpu_status;
        auto live_status = [&gpu_status] { return gpu_status.json(); };
        run_server(pipeline, info, server_host, server_port, live_status);
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
