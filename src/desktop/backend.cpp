#include "backend.h"
#include "common.h"
#include "cpp-httplib/httplib.h"
#include <fcntl.h>
#include <cstdio>
#include <cmath>
#include <glib/gstdio.h>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

namespace desktop {

double query_gpu_total_mb() {
    std::string device = "0";
    if (const char* visible = g_getenv("CUDA_VISIBLE_DEVICES")) {
        device = std::string(visible).substr(0, std::string(visible).find(','));
        if (device.empty() || device == "-1") return -1;
    }
    auto flags = static_cast<GSubprocessFlags>(G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE);
    auto* process = g_subprocess_new(flags, nullptr, "nvidia-smi", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits", "--id", device.c_str(), nullptr);
    if (!process) return -1;
    gchar* output = nullptr;
    bool ok = g_subprocess_communicate_utf8(process, nullptr, nullptr, &output, nullptr, nullptr);
    double total = ok && g_subprocess_get_successful(process) && output ? g_ascii_strtod(output, nullptr) : -1;
    g_free(output);
    g_object_unref(process);
    return std::isfinite(total) && total > 0 ? total : -1;
}

Backend::Backend(std::string executable, std::string weights, std::string log_path) {
    worker_ = std::thread([this, executable, weights, log_path] {
        try { run(executable, weights, log_path); }
        catch (const std::exception& e) {
            int fd = g_open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
            if (fd >= 0) {
                if (FILE* log = fdopen(fd, "a")) { fprintf(log, "Startup error: %s\n", e.what()); fclose(log); }
                else close(fd);
            }
            std::lock_guard<std::mutex> lock(mutex_);
            state_.ready = false;
            state_.starting = false;
            state_.load_failed = true;
            state_.message = std::string(e.what()) + " · See server.log";
        }
    });
}

Backend::~Backend() {
    stopping_ = true;
    g_cancellable_cancel(cancel_);
    std::shared_ptr<httplib::Client> request;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request = request_;
        if (child_) g_subprocess_send_signal(child_, SIGTERM);
    }
    if (request) request->stop();
    wake_.notify_all();
    worker_.join();
    if (child_) {
        // The child is exclusively ours. Never stop a server found on a fixed port.
        g_subprocess_send_signal(child_, SIGTERM);
        g_subprocess_wait(child_, nullptr, nullptr);
        g_object_unref(child_);
    }
    g_object_unref(cancel_);
}

ServerState Backend::state() {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}
bool Backend::transcribe(std::string id, std::string&& wav) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_.ready || busy_) return false;
    busy_ = true;
    job_ = std::make_pair(std::move(id), std::move(wav));
    wake_.notify_all();
    return true;
}
std::optional<Result> Backend::result() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = std::move(result_);
    result_.reset();
    return result;
}

void Backend::run(const std::string& executable, const std::string& weights, const std::string& log_path) {
    if (!g_file_test(weights.c_str(), G_FILE_TEST_IS_REGULAR))
        throw std::runtime_error("Weights not found: " + weights);
    auto* launcher = g_subprocess_launcher_new(G_SUBPROCESS_FLAGS_STDOUT_PIPE);
    int log_fd = g_open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (log_fd < 0) { g_object_unref(launcher); throw std::runtime_error("Could not open server log"); }
    g_subprocess_launcher_take_stderr_fd(launcher, log_fd);
    g_subprocess_launcher_set_child_setup(launcher, [](gpointer) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() == 1) _exit(1);
    }, nullptr, nullptr);
    const char* args[] = {executable.c_str(), "--weights", weights.c_str(), "--server", "127.0.0.1:0", nullptr};
    GError* error = nullptr;
    auto* child = g_subprocess_launcher_spawnv(launcher, args, &error);
    g_object_unref(launcher);
    if (!child) {
        std::string message = error->message;
        g_error_free(error);
        throw std::runtime_error(message);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        child_ = child;
        const char* identifier = g_subprocess_get_identifier(child);
        if (!identifier) throw std::runtime_error("Server exited before becoming ready");
        pid_ = std::stoi(identifier);
        if (stopping_) { g_subprocess_send_signal(child_, SIGTERM); return; }
    }
    auto* input = g_data_input_stream_new(g_subprocess_get_stdout_pipe(child_));
    g_filter_input_stream_set_close_base_stream(G_FILTER_INPUT_STREAM(input), FALSE);
    while (!stopping_ && !port_) {
        gchar* line = g_data_input_stream_read_line(input, nullptr, cancel_, nullptr);
        if (!line) break;
        try {
            Json json(take_string(line));
            if (json.text("event") == "ready") {
                int port = static_cast<int>(json.number("port"));
                auto pid = static_cast<int>(json.number("pid"));
                if (port > 0 && port <= 65535 && pid == pid_)
                    port_ = port;
            }
        } catch (...) { /* Ignore ordinary startup output. */ }
    }
    g_object_unref(input);
    if (stopping_) return;
    if (!port_) throw std::runtime_error("Server exited before becoming ready");
    while (!stopping_) {
        poll();
        std::optional<std::pair<std::string, std::string>> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait_for(lock, std::chrono::seconds(2), [this] { return stopping_ || job_.has_value(); });
            job = std::move(job_);
            job_.reset();
        }
        if (!job || stopping_) continue;
        Result result;
        result.id = job->first;
        auto client = std::make_shared<httplib::Client>("127.0.0.1", port_);
        client->set_connection_timeout(1);
        client->set_read_timeout(300);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            request_ = client;
        }
        try {
            httplib::MultipartFormDataItems items{{"file", job->second, "recording.wav", "audio/wav"}};
            auto response = client->Post("/transcribe", items);
            if (!response) throw std::runtime_error("Could not reach transcription server");
            Json json(response->body);
            if (response->status != 200) throw std::runtime_error(json.text("error", "Transcription failed"));
            result.text = json.text("corrected_text");
            if (result.text.empty()) result.text = json.text("text");
            result.duration = json.number("audio_duration_s");
            result.inference_ms = json.number("inference_time_s") * 1000;
        } catch (const std::exception& e) { result.error = e.what(); }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            request_.reset();
            result_ = std::move(result);
            busy_ = false;
        }
    }
}

void Backend::poll() {
    ServerState next;
    next.starting = false;
    next.message = "Server unavailable";
    try {
        httplib::Client client("127.0.0.1", port_);
        client.set_connection_timeout(0, 300000);
        client.set_read_timeout(0, 500000);
        auto response = client.Get("/status");
        if (!response || response->status != 200) throw std::runtime_error(next.message);
        Json status(response->body);
        if (status.text("status") != "ready" ||
            static_cast<int>(status.number("pid")) != pid_)
            throw std::runtime_error("Unexpected server response");
        next.ready = true; next.message = "Ready";
        next.server_vram_mb = status.number("server_vram_mb", -1);
        next.used_vram_mb = status.number("vram_used_mb", -1);
        next.total_vram_mb = status.number("vram_total_mb", -1);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            next.model = state_.model; next.device = state_.device;
        }
        if (next.model.empty()) {
            auto info = client.Get("/info");
            if (info && info->status == 200) {
                Json json(info->body);
                next.model = json.text("weights").find("ultra") != std::string::npos ? "Parakeet Ultra" : json.text("model");
                next.model += " · " + json.text("backend");
                next.device = json.text("device");
            }
        }
    } catch (const std::exception& e) { next.message = e.what(); }
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = std::move(next);
}

} // namespace desktop
