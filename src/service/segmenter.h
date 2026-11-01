#pragma once
#include "session_store.h"
#include <functional>
#include <memory>
#include <string>

namespace paraketto::service {
class Segmenter {
public:
    using Plan = std::function<void(int64_t, int64_t, const std::vector<SourceSpan>&)>;
    Segmenter(int64_t start, std::string weights, bool filter, Plan plan);
    ~Segmenter();
    void process(const int16_t* samples, size_t count, bool final = false);
    void warmup(const int16_t* samples, size_t count);
    void flush();
    int64_t position() const;
    bool speech_active() const;
    bool filtering() const;
    const std::string& status() const;
    static std::string default_weights();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
