#pragma once

#include <gio/gio.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace httplib { class Client; }
namespace desktop {

// Read capacity without starting the model or allocating a CUDA context.
double query_gpu_total_mb();

struct ServerState {
    bool ready = false;
    bool starting = true;
    bool load_failed = false;
    std::string message = "Starting server…", model, device;
    double server_vram_mb = -1, used_vram_mb = -1, total_vram_mb = -1;
};
struct Result {
    std::string id, text, error;
    double duration = 0, inference_ms = 0;
};

class Backend {
public:
    Backend(std::string executable, std::string weights, std::string log_path);
    ~Backend();
    ServerState state();
    bool transcribe(std::string id, std::string&& wav);
    std::optional<Result> result();
private:
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
    ServerState state_;
    std::optional<std::pair<std::string, std::string>> job_;
    std::optional<Result> result_;
    bool busy_ = false;
    GSubprocess* child_ = nullptr;
    GCancellable* cancel_ = g_cancellable_new();
    std::shared_ptr<httplib::Client> request_;
    int port_ = 0;
    int pid_ = 0;
    void run(const std::string& executable, const std::string& weights, const std::string& log_path);
    void poll();
};

} // namespace desktop
