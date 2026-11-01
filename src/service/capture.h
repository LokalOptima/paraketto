#pragma once

#include "session_store.h"
#include <gst/app/gstappsink.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace paraketto::service {
inline constexpr int audio_rate = 16000;
inline constexpr int64_t audio_block_frames = 30 * audio_rate;

void sync_directory(const std::string& path);
std::string wav_header(int64_t frames);
std::vector<int16_t> read_audio(SessionStore& store, const std::string& id,
                              int64_t start, int64_t end);
void recover_audio(SessionStore& store, const SessionRecord& session);

struct CaptureState {
    int64_t captured = 0, written = 0, durable = 0, discontinuities = 0;
    double rms = 0;
    bool finished = false, paused = false;
    std::string error;
};

// The streaming callback only copies into a bounded queue. A dedicated writer
// owns file I/O and durable checkpoint callbacks. No model runs on either thread.
class AudioCapture {
public:
    using Checkpoint = std::function<void(const AudioBlock&)>;
    using Gap = std::function<void(int64_t, int64_t, const std::string&)>;
    AudioCapture(std::string directory, std::string session, std::string source,
                 bool live, Checkpoint checkpoint, Gap gap = {});
    ~AudioCapture();
    void start();
    void stop();
    // Pause drains and synchronizes all accepted PCM without sealing the block.
    void pause();
    void resume();
    CaptureState state();
private:
    struct Buffer { std::vector<int16_t> samples; bool discontinuity = false; int64_t missing = 0; };
    std::string directory_, session_, source_, error_;
    bool live_;
    Checkpoint checkpoint_;
    Gap gap_;
    GstElement* pipeline_ = nullptr;
    std::thread writer_;
    std::mutex mutex_, control_;
    std::condition_variable wake_;
    std::deque<Buffer> buffers_;
    size_t queued_ = 0;
    bool stopping_ = false, paused_ = false, flush_requested_ = false, flush_completed_ = false;
    bool reset_timestamp_ = false;
    std::atomic<bool> finished_{false};
    std::atomic<int64_t> captured_{0}, written_{0}, durable_{0}, discontinuities_{0};
    std::atomic<double> rms_{0};
    GstClockTime expected_pts_ = GST_CLOCK_TIME_NONE;
    static GstFlowReturn sample(GstAppSink*, gpointer);
    void write();
    void fail(const std::string& error);
    void poll_bus();
};
} // namespace paraketto::service
