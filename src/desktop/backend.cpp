#include "backend.h"
#include "app_common.h"
#include "service/api.h"
#include "cpp-httplib/httplib.h"
#include <cmath>
#include <limits>

namespace desktop {
using namespace paraketto::app;

namespace {
void check_identity(const Json& json, int pid = 0) {
    if (json.text("service") != "paraketto" || json.number("api_version") != paraketto::service::api_version)
        throw std::runtime_error("Incompatible transcription service");
    if (pid && json.number("pid") != pid) throw std::runtime_error("Service address is out of date");
}
void check_health(httplib::Client& client, int pid) {
    auto response = client.Get("/health");
    if (!response || response->status != 200) throw std::runtime_error("Service unavailable");
    check_identity(Json(response->body), pid);
}
}

Backend::Backend(std::string url, std::string data)
    : url_(std::move(url)), endpoint_file_(data + "/" + paraketto::service::endpoint_filename),
      lease_id_(take_string(g_uuid_string_random())) {
    if (!url_.empty() && url_.compare(0, 7, "http://") != 0)
        throw std::runtime_error("The local service URL must start with http://");
    monitor_ = std::thread([this] { monitor(); });
    transcriber_ = std::thread([this] { transcribe(); });
}
Backend::~Backend() {
    stopping_ = true;
    wake_.notify_all();
    std::shared_ptr<httplib::Client> control, request;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        control = control_; request = request_;
    }
    if (control) control->stop();
    if (request) request->stop();
    monitor_.join(); transcriber_.join();
    // Release our recording lease, never another client's model or service.
    try {
        auto target = endpoint();
        httplib::Client client(target.url);
        client.set_connection_timeout(0, 200000); client.set_read_timeout(0, 300000);
        check_health(client, target.pid);
        client.Post("/model/release", "{\"lease_id\":\"" + lease_id_ + "\"}", "application/json");
    } catch (...) { /* A crashed client's lease expires automatically. */ }
}
Backend::Endpoint Backend::endpoint() const {
    if (!url_.empty()) return {url_, 0};
    Json published(read_file(endpoint_file_));
    check_identity(published);
    double port = published.number("port"), pid = published.number("pid");
    if (!std::isfinite(port) || port < 1 || port > 65535 || std::floor(port) != port ||
        !std::isfinite(pid) || pid < 1 || pid > std::numeric_limits<int>::max() || std::floor(pid) != pid)
        throw std::runtime_error("Invalid service address");
    return {"http://127.0.0.1:" + std::to_string(static_cast<int>(port)), static_cast<int>(pid)};
}
ServerState Backend::state() { std::lock_guard<std::mutex> lock(mutex_); return state_; }
void Backend::command(std::string method, std::string path, std::string body) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (path == "/model/load" && (state_.status == "offline" || state_.status == "error")) {
        state_.status = "connecting"; state_.starting = false; state_.load_failed = false;
    }
    commands_.push_back({std::move(method), std::move(path), std::move(body)});
    wake_.notify_all();
}
void Backend::load() { command("POST", "/model/load"); }
void Backend::unload() { command("POST", "/model/unload"); }
void Backend::hold_recording(bool hold) {
    std::lock_guard<std::mutex> lock(mutex_);
    hold_ = hold;
    if (hold && (state_.status == "offline" || state_.status == "error")) {
        state_.status = "connecting"; state_.starting = false; state_.load_failed = false;
    }
    wake_.notify_all();
}
void Backend::settings(const Settings& settings) {
    command("PATCH", "/settings", "{\"idle_seconds\":" + std::to_string(settings.idle_seconds) +
        ",\"keep_loaded\":" + (settings.keep_loaded ? "true" : "false") + "}");
}
std::string Backend::action_error() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto error = std::move(action_error_); action_error_.clear(); return error;
}
bool Backend::transcribe(std::string id, std::string&& wav) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_) return false;
    busy_ = true; job_ = std::make_pair(std::move(id), std::move(wav));
    wake_.notify_all(); return true;
}
std::optional<Result> Backend::result() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = std::move(result_); result_.reset(); return result;
}
void Backend::monitor() {
    bool leased = false;
    auto renewed = std::chrono::steady_clock::time_point{};
    Endpoint current;
    std::shared_ptr<httplib::Client> control;
    while (!stopping_) {
        std::deque<Command> commands;
        bool hold;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            commands.swap(commands_); hold = hold_;
        }
        ServerState next;
        next.starting = false;
        next.status = "offline";
        next.message = "Service unavailable";
        try {
            auto target = endpoint();
            bool changed = !control || target.url != current.url || target.pid != current.pid;
            if (changed) {
                current = target;
                control = std::make_shared<httplib::Client>(target.url);
                control->set_connection_timeout(0, 300000); control->set_read_timeout(1);
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) break;
                control_ = control;
                leased = false;
            }
            next.url = target.url;
            auto now = std::chrono::steady_clock::now();
            bool reserve = hold && (!leased || now - renewed >= std::chrono::seconds(2));
            bool release = !hold && leased;
            if (changed || !commands.empty() || reserve || release) check_health(*control, target.pid);
            for (const auto& command : commands) {
                if (stopping_) break;
                auto response = command.method == "PATCH" ? control->Patch(command.path, command.body, "application/json") :
                    control->Post(command.path, command.body, "application/json");
                if (!response || response->status >= 300) {
                    std::string error = "Service unavailable";
                    if (response) {
                        try { error = Json(response->body).text("error", "Service request failed"); } catch (...) {}
                    }
                    std::lock_guard<std::mutex> lock(mutex_); action_error_ = error;
                }
            }
            if (stopping_) break;
            if (reserve) {
                auto response = control->Post("/model/load", "{\"lease_id\":\"" + lease_id_ + "\"}", "application/json");
                leased = response && response->status < 300;
                renewed = now;
            } else if (release) {
                control->Post("/model/release", "{\"lease_id\":\"" + lease_id_ + "\"}", "application/json");
                leased = false;
            }
            auto response = control->Get("/status");
            if (!response || response->status != 200) throw std::runtime_error(next.message);
            Json status(response->body);
            check_identity(status, target.pid);
            next.connected = true;
            next.status = status.text("status");
            next.ready = next.status == "ready" || next.status == "busy";
            next.starting = next.status == "loading";
            next.busy = status.number("pending_requests") > 0 || status.number("recording_leases") > 0;
            next.load_failed = status.boolean("load_failed");
            next.message = status.text("error");
            next.model = status.text("model"); next.device = status.text("device");
            next.weights = status.text("weights"); next.log_path = status.text("log_path");
            next.server_vram_mb = status.number("server_vram_mb", -1);
            next.used_vram_mb = status.number("vram_used_mb", -1);
            next.total_vram_mb = status.number("vram_total_mb", -1);
            next.idle_remaining = status.number("idle_remaining_s");
            next.settings.idle_seconds = static_cast<int>(status.number("idle_seconds", 300));
            next.settings.keep_loaded = status.boolean("keep_loaded");
        } catch (const std::exception& error) {
            next.message = error.what();
            if (!commands.empty()) {
                std::lock_guard<std::mutex> lock(mutex_); action_error_ = next.message;
            }
        }
        std::unique_lock<std::mutex> lock(mutex_);
        state_ = std::move(next);
        wake_.wait_for(lock, std::chrono::milliseconds(250), [&] {
            return stopping_ || !commands_.empty() || hold_ != hold;
        });
    }
}
void Backend::transcribe() {
    while (!stopping_) {
        std::optional<std::pair<std::string, std::string>> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || job_.has_value(); });
            if (stopping_) break;
            job = std::move(job_); job_.reset();
        }
        Result result; result.id = job->first;
        try {
            auto target = endpoint();
            auto client = std::make_shared<httplib::Client>(target.url);
            client->set_connection_timeout(1); client->set_read_timeout(600);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) break;
                request_ = client;
            }
            check_health(*client, target.pid);
            auto response = client->Post("/transcribe", httplib::MultipartFormDataItems{{"file", job->second, "recording.wav", "audio/wav"}});
            if (!response) throw std::runtime_error("Service unavailable");
            Json json(response->body);
            if (response->status != 200) throw std::runtime_error(json.text("error", "Transcription failed"));
            result.text = json.text("corrected_text");
            if (result.text.empty()) result.text = json.text("text");
            result.duration = json.number("audio_duration_s");
            result.inference_ms = json.number("inference_time_s") * 1000;
        } catch (const std::exception& error) { result.error = error.what(); }
        std::lock_guard<std::mutex> lock(mutex_);
        request_.reset(); result_ = std::move(result); busy_ = false;
    }
}
} // namespace desktop
