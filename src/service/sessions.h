#pragma once
#include "session_store.h"
#include "capture.h"
#include "segmenter.h"
#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <thread>

namespace httplib { class Server; }
namespace paraketto::service {
struct ServiceReply { int status; std::string body; std::string model_revision{}; };
struct SessionOptions {
    std::string data_directory, asr_revision, vad_weights;
};

class Sessions {
public:
    using Transcribe = std::function<std::future<ServiceReply>(std::string)>;
    using SpeechActivity = std::function<void(bool)>;
    using AsrIdentity = std::function<std::string()>;
    Sessions(SessionOptions options, Transcribe transcribe, SpeechActivity activity = {}, AsrIdentity identity = {});
    ~Sessions();
    void routes(httplib::Server& server);
    void stop();
    std::string status();
    SessionStore& store() { return store_; }
private:
    SessionOptions options_;
    SessionStore store_;
    Transcribe transcribe_;
    SpeechActivity activity_;
    AsrIdentity asr_identity_;
    std::mutex mutex_, capture_control_;
    std::shared_ptr<AudioCapture> capture_;
    std::string capture_id_, capture_key_, vad_status_ = "Waiting for audio", engine_error_;
    bool speech_active_ = false;
    int64_t pause_planned_ = -1;
    std::condition_variable pause_ready_;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
    struct Planner { int64_t generation; std::unique_ptr<Segmenter> detector; bool reset_on_resume = false; };
    std::map<std::string, Planner> planners_;
    std::optional<AsrJob> asr_job_;
    std::optional<std::future<ServiceReply>> asr_result_;
    void run();
    void plan(const SessionRecord& session);
    std::string start(const std::string& body, bool file);
    std::string pause_capture(const std::string& id, bool paused);
    void finish_capture(const std::string& id, bool cancel = false, bool interrupted = false);
    std::string levels(const std::string& id);
    void serve_audio(httplib::Server& server);
};
}
