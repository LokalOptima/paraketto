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
#include "conformer.h"

// ---------------------------------------------------------------------------
// Auto-download helpers
// ---------------------------------------------------------------------------

static std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/paraketto";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/paraketto";
    return ".";
}

static void mkdirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/')
            mkdir(cur.c_str(), 0755);  // ignores EEXIST
    }
    if (!cur.empty() && cur.back() != '/')
        mkdir(cur.c_str(), 0755);
}

static void ensure_file(const std::string& path, const char* url) {
    if (access(path.c_str(), F_OK) == 0) return;
    std::string dir = path.substr(0, path.rfind('/'));
    if (!dir.empty()) mkdirs(dir);
    fprintf(stderr, "Downloading %s\n", path.c_str());
    pid_t pid = fork();
    if (pid == 0) {
        execlp("curl", "curl", "-#", "-fL", "-o", path.c_str(), url, nullptr);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        unlink(path.c_str());
        fprintf(stderr, "Download failed. Get weights from https://huggingface.co/localoptima/paraketto\n");
        exit(1);
    }
}

#include "common.h"
#include "wav.h"
#include "mel.h"
#include "vocab.h"
#include "server.h"

#ifdef WITH_CORRECTOR
#include "corrector.h"
#endif


// ---------------------------------------------------------------------------
// Inference pipeline (CUDA backend)
// ---------------------------------------------------------------------------

struct DecodeResult {
    std::vector<int> tokens;
    std::vector<int> frames;  // encoder frame index per token
};

struct Pipeline {
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Weights weights;
    CudaModel cuda_model;
    MelSpec mel;
    cudaStream_t stream = nullptr;

    void init(Weights&& prefetched,
              const std::string& weights_path = "",
              const void* fp8_prefetch = nullptr,
              size_t fp8_prefetch_size = 0) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        weights = std::move(prefetched);

#ifdef PARAKETTO_FP8
        weights.allocate_nongemm_only();
#else
        weights.upload(stream);
#endif

        mel.init();

#ifdef PARAKETTO_FP8
        cuda_model.init(weights, stream, MAX_MEL_FRAMES,
                        weights_path.c_str(), fp8_prefetch, fp8_prefetch_size);
#else
        cuda_model.init(weights, stream, MAX_MEL_FRAMES);
#endif
    }

    ~Pipeline() {
        cuda_model.free();
        weights.free();
        if (stream) cudaStreamDestroy(stream);
    }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    // Per-token timestamps from last transcribe() call
    std::vector<int> last_token_ids;
    std::vector<int> last_token_ms;  // absolute time in ms per token

    // Encoder: 3x stride-2 conv = 8x downsampling → 80ms per encoder frame
    static constexpr int MS_PER_ENC_FRAME = 80;

    // on_chunk: called with (chunk_index, n_chunks) after each chunk completes
    using ChunkCallback = std::function<void(int, int)>;

    std::string transcribe(const float* samples, int num_samples,
                           ChunkCallback on_chunk = nullptr) {
        const char* const* vocab = (weights.config.version == 3) ? VOCAB_V3 : VOCAB_V2;
        const int vocab_size = weights.config.n_vocab;
        const int max_chunk_samples = MAX_MEL_FRAMES * HOP;

        if (num_samples <= max_chunk_samples) {
            auto dr = transcribe_chunk(samples, num_samples);
            last_token_ids = std::move(dr.tokens);
            last_token_ms.resize(last_token_ids.size());
            for (size_t i = 0; i < last_token_ids.size(); i++)
                last_token_ms[i] = dr.frames[i] * MS_PER_ENC_FRAME;
            return detokenize(last_token_ids, vocab, vocab_size);
        }

        // Long audio: split at silence boundaries, transcribe each chunk
        auto splits = find_silence_splits(samples, num_samples);
        last_token_ids.clear();
        last_token_ms.clear();

        std::string result;
        double mel_ms = 0, enc_ms = 0, dec_ms = 0;
        int pos = 0;
        int n_chunks = (int)splits.size() + 1;

        for (int i = 0; i < n_chunks; i++) {
            int end = (i < (int)splits.size()) ? splits[i] : num_samples;
            int len = end - pos;
            if (len <= 0) { pos = end; continue; }

            auto dr = transcribe_chunk(samples + pos, len);
            mel_ms += last_mel_ms;
            enc_ms += last_enc_ms;
            dec_ms += last_dec_ms;

            // Accumulate tokens with absolute timestamps
            int offset_ms = (int)((int64_t)pos * 1000 / 16000);
            for (size_t j = 0; j < dr.tokens.size(); j++) {
                last_token_ids.push_back(dr.tokens[j]);
                last_token_ms.push_back(offset_ms + dr.frames[j] * MS_PER_ENC_FRAME);
            }

            std::string chunk_text = detokenize(dr.tokens, vocab, vocab_size);
            if (!chunk_text.empty()) {
                if (!result.empty()) result += ' ';
                result += chunk_text;
            }
            if (on_chunk) on_chunk(i, n_chunks);
            pos = end;
        }

        last_mel_ms = mel_ms;
        last_enc_ms = enc_ms;
        last_dec_ms = dec_ms;
        fprintf(stderr, "chunked: %d chunks from %.1fs audio\n",
                n_chunks, (double)num_samples / 16000.0);
        return result;
    }

private:
    // Find split points at silence boundaries for audio exceeding 120s.
    // Returns sample indices where the audio should be cut.
    std::vector<int> find_silence_splits(const float* samples, int num_samples) {
        const int max_chunk    = MAX_MEL_FRAMES * HOP;
        const int target_chunk = 16000 * 100;   // aim for ~100s
        const int search_radius = 16000 * 15;   // search ±15s around target
        const int min_chunk    = 16000 * 20;    // never shorter than 20s
        const int energy_win   = 800;           // 50ms window for energy

        std::vector<int> splits;
        int pos = 0;

        while (num_samples - pos > max_chunk) {
            int target = pos + target_chunk;
            int lo = std::max(pos + min_chunk, target - search_radius);
            int hi = std::min(num_samples - min_chunk, target + search_radius);
            hi = std::min(hi, pos + max_chunk - energy_win);

            if (lo >= hi) {
                // No room to search — hard cut at max
                splits.push_back(pos + max_chunk);
                pos += max_chunk;
                continue;
            }

            // Find lowest-energy window in [lo, hi)
            float min_energy = 1e30f;
            int best = target;
            for (int i = lo; i + energy_win <= hi; i += energy_win / 2) {
                float e = 0;
                for (int j = 0; j < energy_win; j++)
                    e += samples[i + j] * samples[i + j];
                if (e < min_energy) {
                    min_energy = e;
                    best = i + energy_win / 2;
                }
            }
            splits.push_back(best);
            pos = best;
        }
        return splits;
    }

    DecodeResult transcribe_chunk(const float* samples, int num_samples) {
        // Need at least 10 mel frames (100ms) for a valid encoder pass
        if (num_samples < HOP * 10) return {};

        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 1. Mel spectrogram (fused GPU pipeline) ---
        int n_frames, n_valid;
        mel.compute(samples, num_samples, cuda_model.mel_fp32, n_frames, n_valid, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_mel = std::chrono::high_resolution_clock::now();

        // --- 2. Encoder (mel_fp32 already on GPU) ---
        int T = cuda_model.encode_gpu(n_valid);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_enc = std::chrono::high_resolution_clock::now();

        // --- 3. TDT greedy decode ---
        auto result = decode(T);
        auto t_dec = std::chrono::high_resolution_clock::now();

        last_mel_ms = std::chrono::duration<double, std::milli>(t_mel - t_start).count();
        last_enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
        last_dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
        return result;
    }

    DecodeResult decode(int enc_len) {
        cuda_model.decoder_reset();

        const int blank_id = weights.config.blank_id;
        const int n_vocab  = weights.config.n_vocab;
        const int d_output = weights.config.d_output;

        DecodeResult result;
        int last_token = blank_id;
        int t = 0, emitted = 0;

        // Host buffer for argmax results (just 2 ints: token, step)
        int argmax_host[2];

        while (t < enc_len) {
            half* joint_out = cuda_model.decode_step(t, last_token);

            // GPU argmax: finds token and step on device, transfers just 2 ints
            dual_argmax_fp16(joint_out, cuda_model.argmax_out,
                              n_vocab, d_output, stream);
            CUDA_CHECK(cudaMemcpyAsync(argmax_host, cuda_model.argmax_out,
                                        2 * sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int token = argmax_host[0];
            int step = argmax_host[1];

            if (token != blank_id) {
                cuda_model.decoder_commit();
                result.tokens.push_back(token);
                result.frames.push_back(t);
                last_token = token;
                emitted++;
            }

            if (step > 0) { t += step; emitted = 0; }
            else if (token == blank_id || emitted >= 2) { t++; emitted = 0; }
        }

        return result;
    }
};

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
            "  --timestamps               Output word-level timestamps (tab-delimited)\n"
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
    bool timestamps = false;
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
        } else if (arg == "--timestamps") {
            timestamps = true;
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
            if (timestamps) {
                const char* const* vocab = (pipeline.weights.config.version == 3) ? VOCAB_V3 : VOCAB_V2;
                auto words = words_with_timestamps(pipeline.last_token_ids, pipeline.last_token_ms,
                                                   vocab, pipeline.weights.config.n_vocab);
                for (auto& w : words)
                    printf("%d\t%s\n", w.start_ms, w.word.c_str());
            } else {
                printf("%s\n", text.c_str());
            }
            fprintf(stderr, "%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f), %.0fx RTFx\n",
                    audio_dur, elapsed * 1000,
                    pipeline.last_mel_ms, pipeline.last_enc_ms, pipeline.last_dec_ms,
                    audio_dur / elapsed);
        }
    }

    return 0;
}
