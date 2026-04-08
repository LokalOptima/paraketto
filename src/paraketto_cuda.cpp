// paraketto_cuda.cpp — Custom CUDA/cuBLAS runtime for Parakeet TDT 0.6B V2
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
#include <sys/mman.h>
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
#ifdef PARAKETTO_FP8
            weights_path = dir + "/paraketto-v3-fp8.bin";
#else
            weights_path = dir + "/paraketto-v3-fp16.bin";
#endif
        } else {
#ifdef PARAKETTO_FP8
            weights_path = dir + "/paraketto-fp8.bin";
#else
            weights_path = dir + "/paraketto-fp16.bin";
#endif
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
#ifdef PARAKETTO_FP8
        fname = (model_version == 3) ? "paraketto-v3-fp8.bin" : "paraketto-fp8.bin";
#else
        fname = (model_version == 3) ? "paraketto-v3-fp16.bin" : "paraketto-fp16.bin";
#endif
        std::string url = std::string(hf_base) + "/" + fname;
        ensure_file(weights_path, url.c_str());
    }

    Weights prefetched;
    std::thread prefetch_thread;

    // FP8: read model version from header so config is set before pool allocation
    void* fp8_prefetch_ptr = nullptr;
    size_t fp8_prefetch_size = 0;
#ifdef PARAKETTO_FP8
    {
        int fd = open(weights_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            char hdr[16];
            if (read(fd, hdr, 16) == 16
                    && memcmp(hdr, "PRKTFP8", 7) == 0) {
                uint32_t model_ver; memcpy(&model_ver, hdr + 12, 4);
                if (model_ver == 3) {
                    prefetched.config.version  = 3;
                    prefetched.config.n_vocab  = 8193;
                    prefetched.config.d_output = 8198;
                    prefetched.config.blank_id = 8192;
                }
            }
            close(fd);
        }
    }
#endif

    // Prefetch in background while CUDA context inits.
    prefetch_thread = std::thread([&]() {
#ifdef PARAKETTO_FP8
        // Pre-populate FP8 file pages while CUDA init runs
        int fd = open(weights_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st; fstat(fd, &st);
            fp8_prefetch_size = (size_t)st.st_size;
            fp8_prefetch_ptr = mmap(nullptr, st.st_size, PROT_READ,
                                    MAP_PRIVATE | MAP_POPULATE, fd, 0);
            close(fd);
            if (fp8_prefetch_ptr == MAP_FAILED) {
                fp8_prefetch_ptr = nullptr;
                fp8_prefetch_size = 0;
            } else {
                madvise(fp8_prefetch_ptr, st.st_size, MADV_SEQUENTIAL);
            }
        }
#else
        prefetched = Weights::prefetch(weights_path);
#endif
    });

    // Force eager CUDA context initialization (overlaps with weight prefetch)
    cudaFree(0);
    auto t_cuda_init = clk::now();

    if (!server_mode) {
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    if (prefetch_thread.joinable()) prefetch_thread.join();
    auto t_prefetch = clk::now();

    Pipeline pipeline;
    pipeline.init(std::move(prefetched), weights_path,
                  fp8_prefetch_ptr, fp8_prefetch_size);
    auto t_init_done = clk::now();

    // Clean up FP8 prefetch mapping (data is on GPU now)
    if (fp8_prefetch_ptr) {
        munmap(fp8_prefetch_ptr, fp8_prefetch_size);
        fp8_prefetch_ptr = nullptr;
    }

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

    if (server_mode) {
        // Warmup with 1s of silence
        std::vector<float> silence(16000, 0.0f);
        pipeline.transcribe(silence.data(), silence.size());
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

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     parakeet-tdt-0.6b-v%d (custom CUDA)\n",
                pipeline.weights.config.version);
        fprintf(stderr, "weights:   %s (%.0f MB)\n", weights_path.c_str(), weights_mb);
        fprintf(stderr, "device:    %s (compute %d.%d, %.0f MB VRAM, %.0f MB free)\n",
                prop.name, prop.major, prop.minor,
                vram_total / (1024.0 * 1024.0), vram_free / (1024.0 * 1024.0));
        fprintf(stderr, "startup:   %.0f ms (cuda=%.0f prefetch=%.0f load=%.0f warmup=%.0f)\n",
                ms(t_main_start, t_warmup_done), ms(t_main_start, t_cuda_init),
                ms(t_main_start, t_prefetch),
                ms(t_prefetch, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | POST /transcribe\n");
#ifdef WITH_CORRECTOR
        if (enable_correct)
            fprintf(stderr, "corrector: enabled (%.0fms warmup)\n", corrector.last_correct_ms);
#endif
        fprintf(stderr, "\n");

#ifdef WITH_CORRECTOR
        run_server(pipeline, enable_correct ? &corrector : nullptr, server_host, server_port);
#else
        run_server(pipeline, server_host, server_port);
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
