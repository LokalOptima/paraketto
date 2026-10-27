#pragma once

#include <gst/app/gstappsink.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace desktop {

class Recorder {
public:
    ~Recorder();
    // Source is a GStreamer audio source description; normally pulsesrc.
    bool start(const std::string& source, std::string& error);
    std::string stop();
    std::string error();
    double seconds() const { return sample_count_.load() / 16000.0; }
    float rms() const { return rms_.load(); } // Linear amplitude relative to full scale.
    bool full() const { return sample_count_.load() >= max_samples; }
    static constexpr size_t max_samples = 16000 * 60 * 30;
private:
    GstElement* pipeline_ = nullptr;
    std::mutex mutex_;
    std::vector<int16_t> samples_;
    std::atomic<size_t> sample_count_{0};
    std::atomic<float> rms_{0};
    static GstFlowReturn sample(GstAppSink*, gpointer);
};

class Player {
public:
    ~Player();
    bool play(const std::string& path, std::string& error);
    void stop();
    bool playing();
private:
    GstElement* pipeline_ = nullptr;
};

std::string pcm_wav(const std::vector<int16_t>& samples);

} // namespace desktop
