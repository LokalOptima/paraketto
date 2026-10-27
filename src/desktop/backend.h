#pragma once

#include "model_settings.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace httplib { class Client; }
namespace desktop {

struct ServerState {
    bool connected = false, ready = false, starting = false, load_failed = false, busy = false;
    std::string status = "connecting", message, model, device, weights, log_path, url;
    double server_vram_mb = -1, used_vram_mb = -1, total_vram_mb = -1, idle_remaining = 0;
    paraketto::app::Settings settings;
};
struct Result {
    std::string id, text, error;
    double duration = 0, inference_ms = 0;
};

// HTTP client only. The service owns the inference process and residency policy.
class Backend {
public:
    Backend(std::string url, std::string data);
    ~Backend();
    ServerState state();
    void load();
    void unload();
    void hold_recording(bool hold);
    void settings(const paraketto::app::Settings& settings);
    std::string action_error();
    bool transcribe(std::string id, std::string&& wav);
    std::optional<Result> result();
private:
    struct Command { std::string method, path, body; };
    struct Endpoint { std::string url; int pid = 0; };
    std::string url_, endpoint_file_, lease_id_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread monitor_, transcriber_;
    std::atomic<bool> stopping_{false};
    ServerState state_;
    std::deque<Command> commands_;
    std::optional<std::pair<std::string, std::string>> job_;
    std::optional<Result> result_;
    bool busy_ = false, hold_ = false;
    std::string action_error_;
    std::shared_ptr<httplib::Client> control_, request_;
    Endpoint endpoint() const;
    void command(std::string method, std::string path, std::string body = "{}");
    void monitor();
    void transcribe();
};

} // namespace desktop
