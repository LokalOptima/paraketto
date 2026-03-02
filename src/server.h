// server.h — HTTP server for Parakeet backends (header-only, templated)
//
// PipelineT must expose:
//   std::string transcribe(const float* samples, int num_samples)
//   double last_mel_ms, last_enc_ms, last_dec_ms
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "cpp-httplib/httplib.h"
#include "wav.h"

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static thread_local std::string t_log_detail;

static inline void log_request(const httplib::Request& req, const httplib::Response& res) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char ts[20];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    fprintf(stderr, "%s  %s %s  %d\n", ts, req.method.c_str(), req.path.c_str(), res.status);

    if (!t_log_detail.empty()) {
        fprintf(stderr, "         %s\n", t_log_detail.c_str());
        t_log_detail.clear();
    }
}

template<typename PipelineT>
static void run_server(PipelineT& pipeline, const std::string& host, int port) {
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/transcribe", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'file' field\"}", "application/json");
            return;
        }
        const auto& file = req.get_file_value("file");
        auto wav = read_wav_from_memory(file.content.data(), file.content.size());
        if (wav.samples.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid WAV (need 16kHz mono, int16/float32)\"}", "application/json");
            return;
        }

        double audio_dur = (double)wav.samples.size() / wav.sample_rate;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text;
        {
            std::lock_guard<std::mutex> lock(mtx);
            text = pipeline.transcribe(wav.samples.data(), wav.samples.size());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Stash detail for the logger callback (same thread)
        std::string preview = text.substr(0, 80);
        if (text.size() > 80) preview += "...";
        char detail[256];
        snprintf(detail, sizeof(detail), "audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                 audio_dur, elapsed * 1000, audio_dur / elapsed, preview.c_str());
        t_log_detail = detail;

        std::string body = "{\"text\":\"" + json_escape(text) +
            "\",\"audio_duration_s\":" + std::to_string(audio_dur) +
            ",\"inference_time_s\":" + std::to_string(elapsed) +
            ",\"mel_ms\":" + std::to_string(pipeline.last_mel_ms) +
            ",\"enc_ms\":" + std::to_string(pipeline.last_enc_ms) +
            ",\"dec_ms\":" + std::to_string(pipeline.last_dec_ms) + "}";
        res.set_content(body, "application/json");
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}
