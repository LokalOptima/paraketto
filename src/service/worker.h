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
namespace paraketto::service {

// Read capacity without starting the model or allocating a CUDA context.
double query_gpu_total_mb();

struct WorkerState {
    int pid = 0;
    bool ready = false;
    bool starting = true;
    bool load_failed = false;
    std::string message = "Starting server…", model, device, info_json;
    double server_vram_mb = -1, used_vram_mb = -1, total_vram_mb = -1;
};
struct WorkerResult {
    std::string body;
    int status = 502;
    bool transport_failed = false;
};

class Worker {
public:
    Worker(std::string executable, std::string weights, std::string log_path);
    ~Worker();
    WorkerState state();
    bool transcribe(std::string&& wav);
    std::optional<WorkerResult> result();
private:
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
    WorkerState state_;
    std::optional<std::string> job_;
    std::optional<WorkerResult> result_;
    bool busy_ = false;
    GSubprocess* child_ = nullptr;
    GCancellable* cancel_ = g_cancellable_new();
    std::shared_ptr<httplib::Client> request_;
    int port_ = 0;
    int pid_ = 0;
    void run(const std::string& executable, const std::string& weights, const std::string& log_path);
    void poll();
};

} // namespace paraketto::service
