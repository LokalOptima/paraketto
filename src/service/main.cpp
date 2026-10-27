#include "api.h"
#include "worker.h"
#include "app_common.h"
#include "model_settings.h"
#include "web_ui.h"
#include "cpp-httplib/httplib.h"
#include <glib-unix.h>
#include <glib/gstdio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <future>
#include <map>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>

namespace paraketto::service {
using namespace paraketto::app;
using Clock = std::chrono::steady_clock;

struct Reply { int status; std::string body; };
struct Job {
    std::string wav;
    std::promise<Reply> completion;
};

class PublishedEndpoint {
    std::string path_;
public:
    PublishedEndpoint(const std::string& data, int port) : path_(data + "/" + endpoint_filename) {
        write_file(path_, "{\"service\":\"paraketto\",\"api_version\":" + std::to_string(api_version) +
            ",\"pid\":" + std::to_string(getpid()) + ",\"port\":" + std::to_string(port) + "}\n");
    }
    ~PublishedEndpoint() { g_unlink(path_.c_str()); }
};

class Service {
public:
    Service(std::string binary, std::string weights, std::string data, int load_timeout)
        : binary_(std::move(binary)), weights_(std::move(weights)), data_(std::move(data)), load_timeout_(load_timeout) {
        settings_.load(data_ + "/preferences.ini");
        capacity_ = query_gpu_total_mb();
        controller_ = std::thread([this] { run(); });
    }
    ~Service() { stop(); if (controller_.joinable()) controller_.join(); }
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        wake_.notify_all();
    }
    std::string status() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* b = json_builder_new();
        json_builder_begin_object(b);
        json_field(b, "service", std::string("paraketto"));
        json_int(b, "api_version", api_version);
        json_int(b, "pid", getpid());
        json_field(b, "status", phase_);
        json_field(b, "error", error_);
        json_bool(b, "load_failed", load_failed_);
        json_int(b, "worker_pid", worker_pid_);
        json_field(b, "model", model_);
        json_field(b, "device", device_);
        json_field(b, "weights", weights_);
        json_field(b, "log_path", data_ + "/server.log");
        json_field(b, "server_vram_mb", memory_);
        json_field(b, "vram_used_mb", used_memory_);
        json_field(b, "vram_total_mb", capacity_);
        json_int(b, "pending_requests", jobs_.size() + (active_ ? 1 : 0));
        json_int(b, "recording_leases", leases_.size());
        json_int(b, "idle_seconds", settings_.idle_seconds);
        json_bool(b, "keep_loaded", settings_.keep_loaded);
        double remaining = settings_.idle_seconds - std::chrono::duration<double>(Clock::now() - idle_since_).count();
        json_field(b, "idle_remaining_s", std::max(0.0, remaining));
        json_builder_end_object(b);
        auto result = json_string(b);
        g_object_unref(b);
        return result;
    }
    std::string info() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!info_.empty()) return info_;
        auto* b = json_builder_new();
        json_builder_begin_object(b);
        json_field(b, "model", std::string("Parakeet"));
        json_field(b, "backend", std::string("CUDA · model unloaded"));
        json_field(b, "weights", weights_);
        json_builder_end_object(b);
        auto result = json_string(b); g_object_unref(b);
        return result;
    }
    Reply load(const std::string& body) {
        Json request(body.empty() ? "{}" : body);
        auto lease = request.text("lease_id");
        if (json_object_has_member(request.object(), "lease_id") && !g_uuid_string_is_valid(lease.c_str()))
            return fail(400, "Invalid recording lease");
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return fail(503, "Service is stopping");
        if (!lease.empty()) {
            if (!leases_.count(lease) && leases_.size() >= 32) return fail(429, "Too many recording sessions");
            leases_[lease] = Clock::now() + std::chrono::seconds(lease_seconds);
        }
        want_load_ = true;
        want_unload_ = false;
        if (phase_ == "sleeping" || phase_ == "error") { phase_ = "loading"; error_.clear(); load_failed_ = false; }
        idle_since_ = Clock::now();
        wake_.notify_one();
        return {202, "{\"accepted\":true}"};
    }
    Reply release(const std::string& body) {
        Json request(body);
        auto lease = request.text("lease_id");
        if (!g_uuid_string_is_valid(lease.c_str())) return fail(400, "Invalid recording lease");
        std::lock_guard<std::mutex> lock(mutex_);
        if (leases_.erase(lease)) idle_since_ = Clock::now();
        wake_.notify_one();
        return {200, "{}"};
    }
    Reply unload() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ || !jobs_.empty() || !leases_.empty()) return fail(409, "Model is in use");
        want_unload_ = true;
        want_load_ = false;
        wake_.notify_one();
        return {202, "{\"accepted\":true}"};
    }
    Reply settings(const std::string& body) {
        Json request(body);
        std::lock_guard<std::mutex> lock(mutex_);
        auto next = settings_;
        if (json_object_has_member(request.object(), "idle_seconds")) {
            double seconds = request.number("idle_seconds", -1);
            if (!std::isfinite(seconds) || seconds < 1 || seconds > 86400 || std::floor(seconds) != seconds)
                return fail(400, "idle_seconds must be an integer from 1 to 86400");
            next.idle_seconds = static_cast<int>(seconds);
        }
        if (json_object_has_member(request.object(), "keep_loaded")) {
            if (json_node_get_value_type(json_object_get_member(request.object(), "keep_loaded")) != G_TYPE_BOOLEAN)
                return fail(400, "keep_loaded must be a boolean");
            next.keep_loaded = request.boolean("keep_loaded");
        }
        next.save(data_ + "/preferences.ini");
        settings_ = next;
        idle_since_ = Clock::now();
        wake_.notify_one();
        return {200, "{}"};
    }
    Reply transcribe(std::string wav) {
        auto job = std::make_shared<Job>();
        job->wav = std::move(wav);
        auto future = job->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return fail(503, "Service is stopping");
            if (jobs_.size() + (active_ ? 1 : 0) >= 4) return fail(429, "Transcription queue is full");
            jobs_.push_back(job);
            want_unload_ = false;
            want_load_ = true;
            if (phase_ == "sleeping" || phase_ == "error") { phase_ = "loading"; error_.clear(); load_failed_ = false; }
            wake_.notify_one();
        }
        return future.get();
    }
private:
    std::string binary_, weights_, data_;
    int load_timeout_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread controller_;
    bool stopping_ = false, want_load_ = false, want_unload_ = false, load_failed_ = false;
    std::string phase_ = "sleeping", error_, model_, device_, info_;
    int worker_pid_ = 0;
    double capacity_ = -1, memory_ = 0, used_memory_ = -1;
    Settings settings_;
    Clock::time_point idle_since_ = Clock::now();
    std::map<std::string, Clock::time_point> leases_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::shared_ptr<Job> active_;
    static Reply fail(int status, const std::string& message) { return {status, json_error(message)}; }
    void fail_jobs(const std::string& error) {
        if (active_) { active_->completion.set_value(fail(503, error)); active_.reset(); }
        for (auto& job : jobs_) job->completion.set_value(fail(503, error));
        jobs_.clear();
    }
    void run() {
        std::unique_ptr<Worker> worker;
        Clock::time_point loading_since;
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            auto now = Clock::now();
            for (auto it = leases_.begin(); it != leases_.end();) {
                if (it->second <= now) { it = leases_.erase(it); idle_since_ = now; }
                else ++it;
            }
            bool busy = active_ || !jobs_.empty() || !leases_.empty();
            if (busy) idle_since_ = now;
            bool expired = worker && phase_ == "ready" && !busy && !settings_.keep_loaded &&
                now - idle_since_ >= std::chrono::seconds(settings_.idle_seconds);
            if (want_unload_ || expired) {
                want_unload_ = false;
                phase_ = "unloading";
                lock.unlock(); worker.reset(); lock.lock();
                phase_ = "sleeping"; error_.clear(); load_failed_ = false;
                worker_pid_ = 0; memory_ = 0; used_memory_ = -1;
            }
            if (want_load_) {
                want_load_ = false;
                if (!worker) {
                    phase_ = "loading"; error_.clear(); load_failed_ = false; memory_ = -1;
                    loading_since = now;
                    lock.unlock();
                    worker = std::make_unique<Worker>(binary_, weights_, data_ + "/server.log");
                    lock.lock();
                }
            }
            if (worker) {
                auto state = worker->state();
                bool timeout = state.starting && now - loading_since >= std::chrono::seconds(load_timeout_);
                if (timeout || (!state.starting && !state.ready)) {
                    load_failed_ = timeout || state.load_failed;
                    error_ = timeout ? "Model loading timed out" : state.message;
                    if (timeout) {
                        int fd = g_open((data_ + "/server.log").c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
                        if (fd >= 0) {
                            if (FILE* log = fdopen(fd, "a")) { fprintf(log, "%s\n", error_.c_str()); fclose(log); }
                            else close(fd);
                        }
                    }
                    phase_ = "error";
                    fail_jobs(load_failed_ ? "Couldn’t load model" : "Transcription worker unavailable");
                    want_load_ = false;
                    lock.unlock(); worker.reset(); lock.lock();
                    worker_pid_ = 0; memory_ = 0; used_memory_ = -1;
                } else if (state.ready) {
                    if (phase_ == "loading") idle_since_ = now;
                    memory_ = state.server_vram_mb;
                    used_memory_ = state.used_vram_mb;
                    if (state.total_vram_mb > 0) capacity_ = state.total_vram_mb;
                    model_ = state.model; device_ = state.device; info_ = state.info_json;
                    worker_pid_ = state.pid;
                    if (auto result = worker->result()) {
                        if (active_) {
                            active_->completion.set_value({result->status, std::move(result->body)});
                            active_.reset(); idle_since_ = now;
                        }
                        if (result->transport_failed) {
                            phase_ = "error"; error_ = "Transcription worker unavailable"; load_failed_ = false;
                            fail_jobs(error_); want_load_ = false;
                            lock.unlock(); worker.reset(); lock.lock();
                            worker_pid_ = 0; memory_ = 0; used_memory_ = -1;
                            continue;
                        }
                    }
                    if (!active_ && !jobs_.empty()) {
                        auto job = jobs_.front();
                        if (worker->transcribe(std::move(job->wav))) {
                            active_ = job; jobs_.pop_front();
                        }
                    }
                    phase_ = active_ ? "busy" : "ready";
                }
            }
            wake_.wait_for(lock, std::chrono::milliseconds(50));
        }
        phase_ = "stopping";
        fail_jobs("Service is stopping");
        lock.unlock();
        worker.reset();
    }
};
} // namespace paraketto::service

int main(int argc, char** argv) {
    using namespace paraketto::app;
    using namespace paraketto::service;
    gchar *weights_arg = nullptr, *binary_arg = nullptr, *data_arg = nullptr;
    gint port = -1, load_timeout = 120; // Omitted port prefers 8080, then a free port.
    GOptionEntry options[] = {
        {"port", 0, 0, G_OPTION_ARG_INT, &port, "Fixed API port (default: 8080 with automatic fallback; 0: free port)", "PORT"},
        {"weights", 0, 0, G_OPTION_ARG_FILENAME, &weights_arg, "FP16 weights (default: Ultra)", "FILE"},
        {"worker-binary", 0, 0, G_OPTION_ARG_FILENAME, &binary_arg, "Inference worker executable", "FILE"},
        {"data-dir", 0, 0, G_OPTION_ARG_FILENAME, &data_arg, "Settings and worker log directory", "DIRECTORY"},
        {"load-timeout", 0, 0, G_OPTION_ARG_INT, &load_timeout, "Model loading deadline in seconds", "SECONDS"},
        {nullptr, 0, 0, G_OPTION_ARG_NONE, nullptr, nullptr, nullptr}
    };
    auto* context = g_option_context_new("— persistent local Paraketto API");
    g_option_context_add_main_entries(context, options, nullptr);
    GError* error = nullptr;
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printerr("%s\n", error->message); g_error_free(error); g_option_context_free(context); return 1;
    }
    g_option_context_free(context);
    try {
        if (port < -1 || port > 65535 || load_timeout < 1) throw std::runtime_error("Invalid port or load timeout");
        auto executable = take_string(g_file_read_link("/proc/self/exe", nullptr));
        auto base = take_string(g_path_get_dirname(executable.c_str()));
        std::string binary = binary_arg ? binary_arg : base + "/paraketto.cuda";
        if (!binary_arg && !g_file_test(binary.c_str(), G_FILE_TEST_IS_EXECUTABLE)) binary = base + "/paraketto";
        std::string weights = weights_arg ? weights_arg : base + "/paraketto-ultra-fp16.bin";
        if (!weights_arg && !g_file_test(weights.c_str(), G_FILE_TEST_IS_REGULAR))
            weights = std::string(g_get_user_cache_dir()) + "/paraketto/paraketto-ultra-fp16.bin";
        std::string data = data_arg ? data_arg : std::string(g_get_user_data_dir()) + "/paraketto";
        binary = take_string(g_canonicalize_filename(binary.c_str(), nullptr));
        weights = take_string(g_canonicalize_filename(weights.c_str(), nullptr));
        data = take_string(g_canonicalize_filename(data.c_str(), nullptr));
        g_free(weights_arg); g_free(binary_arg); g_free(data_arg);
        if (g_mkdir_with_parents(data.c_str(), 0700) != 0) throw std::runtime_error("Could not create service data directory");
        int lock_fd = g_open((data + "/service.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0)
            throw std::runtime_error("Another service owns this data directory");
        auto make_server = [] {
            auto server = std::make_unique<httplib::Server>();
            server->set_socket_options([](socket_t socket) {
                // httplib defaults to SO_REUSEPORT, which would silently share
                // traffic with another listener instead of detecting a conflict.
                int enabled = 1;
                setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
            });
            return server;
        };
        auto server = make_server();
        if (port == -1) {
            port = default_port;
            if (!server->bind_to_port("127.0.0.1", port)) {
                // A failed bind permanently decommissions this httplib instance.
                server = make_server();
                port = server->bind_to_any_port("127.0.0.1");
            }
        } else if (port == 0) port = server->bind_to_any_port("127.0.0.1");
        else if (!server->bind_to_port("127.0.0.1", port)) port = -1;
        if (port < 0) throw std::runtime_error("Could not bind local service port");
        auto& http = *server;
        http.new_task_queue = [] { return new httplib::ThreadPool(8, 16); };
        http.set_payload_max_length(64 * 1024 * 1024);
        Service service(binary, weights, data, load_timeout);
        PublishedEndpoint endpoint(data, port);
        auto reply = [](httplib::Response& response, const Reply& result) {
            response.status = result.status; response.set_content(result.body, "application/json");
        };
        http.set_pre_routing_handler([port](const httplib::Request& request, httplib::Response& response) {
            auto origin = request.get_header_value("Origin");
            if ((!origin.empty() && origin != "http://127.0.0.1:" + std::to_string(port) &&
                    origin != "http://localhost:" + std::to_string(port)) || request.get_header_value("Sec-Fetch-Site") == "cross-site") {
                response.status = 403; response.set_content(json_error("Cross-origin access is disabled"), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        http.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr exception) {
            std::string message = "Service request failed";
            response.status = 500;
            try { if (exception) std::rethrow_exception(exception); }
            catch (const std::invalid_argument& e) { message = e.what(); response.status = 400; }
            catch (const std::exception& e) { message = e.what(); }
            catch (...) {}
            response.set_content(json_error(message), "application/json");
        });
        http.Get("/health", [](const auto&, auto& res) {
            res.set_content("{\"service\":\"paraketto\",\"api_version\":" + std::to_string(api_version) +
                ",\"pid\":" + std::to_string(getpid()) + "}", "application/json");
        });
        http.Get("/status", [&](const auto&, auto& res) { res.set_content(service.status(), "application/json"); });
        http.Get("/info", [&](const auto&, auto& res) { res.set_content(service.info(), "application/json"); });
        http.Get("/", [](const auto&, auto& res) { res.set_content(paraketto::web_ui, "text/html"); });
        http.Post("/model/load", [&](const auto& req, auto& res) { reply(res, service.load(req.body)); });
        http.Post("/model/unload", [&](const auto&, auto& res) { reply(res, service.unload()); });
        http.Post("/model/release", [&](const auto& req, auto& res) { reply(res, service.release(req.body)); });
        http.Patch("/settings", [&](const auto& req, auto& res) { reply(res, service.settings(req.body)); });
        http.Post("/transcribe", [&](const auto& req, auto& res) {
            if (!req.has_file("file") || req.get_file_value("file").content.empty()) {
                reply(res, {400, json_error("Missing audio in multipart 'file' field")}); return;
            }
            reply(res, service.transcribe(req.get_file_value("file").content));
        });
        auto* loop = g_main_loop_new(nullptr, FALSE);
        struct Shutdown { Service* service; httplib::Server* http; GMainLoop* loop; } shutdown{&service, &http, loop};
        auto stop = [](gpointer p) -> gboolean {
            auto* shutdown = static_cast<Shutdown*>(p);
            shutdown->service->stop(); shutdown->http->stop();
            return G_SOURCE_CONTINUE;
        };
        guint sigint = g_unix_signal_add(SIGINT, stop, &shutdown);
        guint sigterm = g_unix_signal_add(SIGTERM, stop, &shutdown);
        std::atomic<bool> listener_finished{false};
        bool listener_ok = false;
        std::thread listener([&] {
            listener_ok = http.listen_after_bind();
            listener_finished = true;
            g_idle_add([](gpointer p) -> gboolean { g_main_loop_quit(static_cast<GMainLoop*>(p)); return G_SOURCE_REMOVE; }, loop);
        });
        // Signal callbacks run only after the listener starts (or has failed),
        // so a stop during startup cannot race past httplib's is_running guard.
        while (!http.is_running() && !listener_finished) g_usleep(1000);
        if (http.is_running()) {
            printf("{\"event\":\"ready\",\"port\":%d,\"pid\":%ld}\n", port, static_cast<long>(getpid())); fflush(stdout);
        }
        g_main_loop_run(loop);
        service.stop(); http.stop(); listener.join();
        g_source_remove(sigint); g_source_remove(sigterm); g_main_loop_unref(loop);
        // Keep the process-lifetime lock until exit, after Service has reaped
        // its worker. Releasing it here would let a second service overlap cleanup.
        return listener_ok ? 0 : 1;
    } catch (const std::exception& e) { g_printerr("%s\n", e.what()); return 1; }
}
