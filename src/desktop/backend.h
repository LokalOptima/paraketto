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
#include <vector>

namespace httplib { class Client; }
namespace desktop {

struct ServerState {
    bool connected = false, ready = false, starting = false, load_failed = false, busy = false;
    std::string status = "connecting", message, model, device, weights, log_path, url;
    std::string active_session_id, vad_status, session_error;
    bool sessions_available = false;
    int service_pid = 0;
    double server_vram_mb = -1, used_vram_mb = -1, total_vram_mb = -1, idle_remaining = 0;
    paraketto::app::Settings settings;
};
struct Result {
    std::string id, text, error;
    double duration = 0, inference_ms = 0;
};

// Session responses are consumed once on the GTK thread. They contain bounded
// deltas, not an ever-growing copy of every recording on each UI frame.
struct SessionEvent {
    std::string operation, id, body, error;
    int status = 0;
};
struct SessionLevels {
    std::string id;
    double rms = 0, elapsed_seconds = 0;
    uint64_t captured_samples = 0, durable_samples = 0;
    bool speech_active = false;
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
    void session_request(std::string operation, std::string method, std::string path,
                         std::string body = "{}", std::string id = {});
    void watch_session(std::string id);
    std::vector<SessionEvent> session_events();
    SessionLevels levels();
    bool commands_pending();
private:
    struct Command { std::string method, path, body; };
    struct Endpoint { std::string url; int pid = 0; };
    std::string url_, endpoint_file_, lease_id_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread monitor_, transcriber_;
    std::thread sessions_;
    std::atomic<bool> stopping_{false};
    ServerState state_;
    std::deque<Command> commands_;
    std::optional<std::pair<std::string, std::string>> job_;
    std::optional<Result> result_;
    bool busy_ = false, hold_ = false;
    std::string action_error_;
    std::shared_ptr<httplib::Client> control_, request_;
    struct SessionCommand { std::string operation, method, path, body, id; };
    std::deque<SessionCommand> session_commands_;
    std::deque<SessionEvent> session_events_;
    std::shared_ptr<httplib::Client> session_client_;
    std::string watched_session_;
    SessionLevels levels_;
    bool session_busy_ = false;
    Endpoint endpoint() const;
    void command(std::string method, std::string path, std::string body = "{}");
    void monitor();
    void transcribe();
    void sessions();
};

} // namespace desktop
