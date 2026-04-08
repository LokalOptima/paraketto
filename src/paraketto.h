// paraketto.h — Public API for paraketto STT library
//
// Usage:
//   #include "paraketto.h"
//   paraketto::Pipeline stt;
//   auto w = paraketto::Weights::prefetch("~/.cache/paraketto/paraketto-fp8.bin");
//   stt.init(std::move(w));
//   std::string text = stt.transcribe(samples, n_samples);

#pragma once

#ifdef PARAKETTO_FP8
#include "conformer_fp8.h"  // CudaModel (FP8 variant), Weights, model constants
#else
#include "conformer.h"      // CudaModel, Weights, model constants
#endif
#include "common.h"      // CUDA_CHECK, mel constants
#include "wav.h"         // read_wav, WavData
#include "mel.h"         // MelSpec
#include "vocab.h"       // VOCAB_V2, VOCAB_V3, detokenize

#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

namespace paraketto {

// ---------------------------------------------------------------------------
// Auto-download helpers
// ---------------------------------------------------------------------------

static inline std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/paraketto";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/paraketto";
    return ".";
}

static inline void mkdirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/')
            mkdir(cur.c_str(), 0755);
    }
    if (!cur.empty() && cur.back() != '/')
        mkdir(cur.c_str(), 0755);
}

static inline void ensure_file(const std::string& path, const char* url) {
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

// ---------------------------------------------------------------------------
// Inference pipeline (CUDA backend)
// ---------------------------------------------------------------------------

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

    using ChunkCallback = std::function<void(int, int)>;

    std::string transcribe(const float* samples, int num_samples,
                           ChunkCallback on_chunk = nullptr) {
        const int max_chunk_samples = MAX_MEL_FRAMES * HOP;
        if (num_samples <= max_chunk_samples)
            return transcribe_single(samples, num_samples);

        auto splits = find_silence_splits(samples, num_samples);

        std::string result;
        double mel_ms = 0, enc_ms = 0, dec_ms = 0;
        int pos = 0;
        int n_chunks = (int)splits.size() + 1;

        for (int i = 0; i < n_chunks; i++) {
            int end = (i < (int)splits.size()) ? splits[i] : num_samples;
            int len = end - pos;
            if (len <= 0) { pos = end; continue; }

            std::string chunk_text = transcribe_single(samples + pos, len);
            mel_ms += last_mel_ms;
            enc_ms += last_enc_ms;
            dec_ms += last_dec_ms;

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
    std::vector<int> find_silence_splits(const float* samples, int num_samples) {
        const int max_chunk    = MAX_MEL_FRAMES * HOP;
        const int target_chunk = 16000 * 100;
        const int search_radius = 16000 * 15;
        const int min_chunk    = 16000 * 20;
        const int energy_win   = 800;

        std::vector<int> splits;
        int pos = 0;

        while (num_samples - pos > max_chunk) {
            int target = pos + target_chunk;
            int lo = std::max(pos + min_chunk, target - search_radius);
            int hi = std::min(num_samples - min_chunk, target + search_radius);
            hi = std::min(hi, pos + max_chunk - energy_win);

            if (lo >= hi) {
                splits.push_back(pos + max_chunk);
                pos += max_chunk;
                continue;
            }

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

    std::string transcribe_single(const float* samples, int num_samples) {
        if (num_samples < HOP * 10) return "";

        auto t_start = std::chrono::high_resolution_clock::now();

        int n_frames, n_valid;
        mel.compute(samples, num_samples, cuda_model.mel_fp32, n_frames, n_valid, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_mel = std::chrono::high_resolution_clock::now();

        int T = cuda_model.encode_gpu(n_valid);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_enc = std::chrono::high_resolution_clock::now();

        auto text = decode(T);
        auto t_dec = std::chrono::high_resolution_clock::now();

        last_mel_ms = std::chrono::duration<double, std::milli>(t_mel - t_start).count();
        last_enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
        last_dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
        return text;
    }

    std::string decode(int enc_len) {
        cuda_model.decoder_reset();

        const int blank_id = weights.config.blank_id;
        const int n_vocab  = weights.config.n_vocab;
        const int d_output = weights.config.d_output;

        std::vector<int> tokens;
        int last_token = blank_id;
        int t = 0, emitted = 0;

        int argmax_host[2];

        while (t < enc_len) {
            half* joint_out = cuda_model.decode_step(t, last_token);

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
                tokens.push_back(token);
                last_token = token;
                emitted++;
            }

            if (step > 0) { t += step; emitted = 0; }
            else if (token == blank_id || emitted >= 2) { t++; emitted = 0; }
        }

        const char* const* vocab = (weights.config.version == 3) ? VOCAB_V3 : VOCAB_V2;
        return detokenize(tokens, vocab, weights.config.n_vocab);
    }
};

} // namespace paraketto
