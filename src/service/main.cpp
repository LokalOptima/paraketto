#include "api.h"
#include "worker.h"
#include "sessions.h"
#include "app_common.h"
#include "model_settings.h"
#include "model_selection.h"
#include "web_ui.h"
#include "cpp-httplib/httplib.h"
#include <glib-unix.h>
#include <glib/gstdio.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <future>
#include <functional>
#include <map>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace paraketto::service {
using namespace paraketto::app;
using Clock = std::chrono::steady_clock;

using Reply = ServiceReply;
struct Job {
    std::string wav;
    std::promise<Reply> completion;
};

struct FileFingerprint {
    std::string status, sha256, error;
    int64_t bytes = 0;
    struct stat identity{};
};
// Read large weights with bounded memory. Missing assets remain a usable
// service configuration and are recorded explicitly rather than hashed as an
// empty file. Reject a hash if the file changed while it was being read.
static FileFingerprint fingerprint_file(const std::string& path, const std::function<bool()>& cancelled = {}) {
    FileFingerprint result;
    if (cancelled && cancelled()) { result.status = "cancelled"; return result; }
    int fd = g_open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK, 0);
    if (fd < 0) {
        result.status = errno == ENOENT || errno == ENOTDIR ? "missing" : "unreadable";
        result.error = g_strerror(errno);
        return result;
    }
    struct stat before{}, after{}, named{};
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode)) {
        result.status = "unreadable"; result.error = "Not a readable regular file";
        close(fd); return result;
    }
    result.bytes = before.st_size;
    auto* checksum = g_checksum_new(G_CHECKSUM_SHA256);
    std::array<guchar, 128 * 1024> buffer{};
    int64_t read_bytes = 0;
    for (;;) {
        if (cancelled && cancelled()) { result.status = "cancelled"; break; }
        ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) { result.error = g_strerror(errno); break; }
        if (count == 0) break;
        g_checksum_update(checksum, buffer.data(), count);
        read_bytes += count;
    }
    bool unchanged = fstat(fd, &after) == 0 && g_stat(path.c_str(), &named) == 0 &&
        before.st_dev == named.st_dev && before.st_ino == named.st_ino &&
        before.st_size == after.st_size && read_bytes == after.st_size &&
        before.st_mtim.tv_sec == after.st_mtim.tv_sec && before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
        before.st_ctim.tv_sec == after.st_ctim.tv_sec && before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
    close(fd);
    result.identity = after;
    if (result.status == "cancelled") { /* Cooperative cancellation is not a fingerprint. */ }
    else if (!result.error.empty()) result.status = "unreadable";
    else if (!unchanged) { result.status = "changed"; result.error = "File changed during fingerprinting"; }
    else { result.status = "verified"; result.sha256 = g_checksum_get_string(checksum); }
    g_checksum_free(checksum);
    return result;
}

static void append_fingerprint(JsonBuilder* builder, const std::string& name, const FileFingerprint& identity) {
    json_field(builder, (name + "_status").c_str(), identity.status);
    json_field(builder, (name + "_sha256").c_str(), identity.sha256);
    json_int(builder, (name + "_bytes").c_str(), identity.bytes);
    if (!identity.error.empty()) json_field(builder, (name + "_error").c_str(), identity.error);
}

static bool same_file(const std::string& path, const FileFingerprint& fingerprint) {
    struct stat current{};
    const auto& expected = fingerprint.identity;
    return fingerprint.status == "verified" && g_stat(path.c_str(), &current) == 0 &&
        current.st_dev == expected.st_dev && current.st_ino == expected.st_ino &&
        current.st_size == expected.st_size && current.st_mtim.tv_sec == expected.st_mtim.tv_sec &&
        current.st_mtim.tv_nsec == expected.st_mtim.tv_nsec && current.st_ctim.tv_sec == expected.st_ctim.tv_sec &&
        current.st_ctim.tv_nsec == expected.st_ctim.tv_nsec;
}
static std::string model_identity(const std::string& binary, const std::string& weights,
                                  const FileFingerprint& runtime, const FileFingerprint& model,
                                  const std::string& selected_preset = {}) {
    auto* b = json_builder_new(); json_builder_begin_object(b);
    json_field(b,"weights",weights); append_fingerprint(b,"weights",model);
    json_field(b,"runtime",binary); append_fingerprint(b,"runtime",runtime);
    json_field(b,"model_preset",selected_preset.empty() ? std::string("custom") : selected_preset);
    json_field(b,"precision",std::string("FP16")); json_builder_end_object(b);
    auto result = json_string(b); g_object_unref(b); return result;
}

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
    Service(std::string binary, std::string weights, std::string data, int load_timeout, std::string configured_revision,
            std::string selected_preset = {})
        : binary_(std::move(binary)), weights_(std::move(weights)), data_(std::move(data)), configured_revision_(std::move(configured_revision)),
          selected_preset_(std::move(selected_preset)), load_timeout_(load_timeout) {
        refresh_architecture();
        model_ = paraketto::models::label(selected_preset_.empty() ? nullptr : &paraketto::models::preset(selected_preset_), architecture_);
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
        append_selection(b);
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
    std::string model_revision() {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_revision_.empty() ? configured_revision_ : loaded_revision_;
    }
    std::string info() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* b = json_builder_new(); json_builder_begin_object(b);
        if (info_.empty()) {
            json_field(b, "model", model_);
            json_field(b, "backend", std::string("CUDA · model unloaded"));
            json_field(b, "weights", weights_);
        } else {
            Json previous(info_); auto* members = json_object_get_members(previous.object());
            for (auto* item=members; item; item=item->next) {
                auto* name = static_cast<const char*>(item->data);
                if (std::string(name)=="model_revision" || std::string(name)=="model_preset" || std::string(name)=="model_architecture") continue;
                json_builder_set_member_name(b,name);
                json_builder_add_value(b,json_node_copy(json_object_get_member(previous.object(),name)));
            }
            g_list_free(members);
        }
        Json identity(loaded_revision_.empty() ? configured_revision_ : loaded_revision_);
        append_selection(b);
        json_builder_set_member_name(b,"model_revision");
        json_builder_add_value(b,json_node_init_object(json_node_alloc(),identity.object()));
        json_builder_end_object(b);auto result=json_string(b);g_object_unref(b);return result;
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
        if (active_ || !jobs_.empty() || !leases_.empty() || session_speech_) return fail(409, "Model is in use");
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
    std::future<Reply> enqueue(std::string wav) {
        auto job = std::make_shared<Job>();
        job->wav = std::move(wav);
        auto future = job->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { job->completion.set_value(fail(503, "Service is stopping")); return future; }
            if (jobs_.size() + (active_ ? 1 : 0) >= 4) { job->completion.set_value(fail(429, "Transcription queue is full")); return future; }
            jobs_.push_back(job);
            want_unload_ = false;
            want_load_ = true;
            if (phase_ == "sleeping" || phase_ == "error") { phase_ = "loading"; error_.clear(); load_failed_ = false; }
            wake_.notify_one();
        }
        return future;
    }
    Reply transcribe(std::string wav) { return enqueue(std::move(wav)).get(); }
    void speech(bool active) {
        std::lock_guard<std::mutex> lock(mutex_);
        session_speech_ = active;
        if (active && phase_ == "sleeping" && !load_failed_) {
            want_load_ = true; want_unload_ = false; wake_.notify_one();
        }
    }
private:
    std::string binary_, weights_, data_, configured_revision_, loaded_revision_, selected_preset_;
    int architecture_ = 0;
    void refresh_architecture() {
        architecture_ = 0;
        try { architecture_ = paraketto::models::read_architecture(weights_); }
        catch (const std::exception&) { /* Missing/invalid assets fail on load, preserving service availability and retry. */ }
    }
    void append_selection(JsonBuilder* b) const {
        json_field(b,"model_preset",selected_preset_.empty() ? std::string("custom") : selected_preset_);
        json_field(b,"model_architecture",std::string(architecture_ == 2 ? "v2" : architecture_ == 3 ? "v3" : "unknown"));
    }
    int load_timeout_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread controller_;
    bool stopping_ = false, want_load_ = false, want_unload_ = false, load_failed_ = false, session_speech_ = false;
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
        FileFingerprint loading_model, loading_runtime;
        std::string loading_revision;
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            auto now = Clock::now();
            for (auto it = leases_.begin(); it != leases_.end();) {
                if (it->second <= now) { it = leases_.erase(it); idle_since_ = now; }
                else ++it;
            }
            bool busy = active_ || !jobs_.empty() || !leases_.empty() || session_speech_;
            if (busy) idle_since_ = now;
            bool expired = worker && phase_ == "ready" && !busy && !settings_.keep_loaded &&
                now - idle_since_ >= std::chrono::seconds(settings_.idle_seconds);
            if (want_unload_ || expired) {
                want_unload_ = false;
                phase_ = "unloading";
                lock.unlock(); worker.reset(); lock.lock();
                phase_ = "sleeping"; error_.clear(); load_failed_ = false; loaded_revision_.clear();
                worker_pid_ = 0; memory_ = 0; used_memory_ = -1;
            }
            if (want_load_) {
                want_load_ = false;
                if (!worker) {
                    phase_ = "loading"; error_.clear(); load_failed_ = false; memory_ = -1;
                    info_.clear();
                    loading_since = Clock::now();
                    lock.unlock();
                    auto cancelled = [this, loading_since] {
                        std::lock_guard<std::mutex> guard(mutex_);
                        return stopping_ || want_unload_ || Clock::now()-loading_since >= std::chrono::seconds(load_timeout_);
                    };
                    loading_model = fingerprint_file(weights_,cancelled);
                    loading_runtime = fingerprint_file(binary_,cancelled);
                    lock.lock();
                    if (stopping_ || want_unload_) continue;
                    refresh_architecture();
                    model_ = paraketto::models::label(selected_preset_.empty() ? nullptr : &paraketto::models::preset(selected_preset_), architecture_);
                    configured_revision_ = model_identity(binary_,weights_,loading_runtime,loading_model,selected_preset_);
                    if (loading_model.status!="verified" || loading_runtime.status!="verified") {
                        error_ = Clock::now()-loading_since >= std::chrono::seconds(load_timeout_) ? "Model loading timed out during asset fingerprinting" :
                            loading_model.status=="missing" ? "Weights not found: " + weights_ +
                                (selected_preset_ == "ultra" ? " · Convert Ultra with scripts/export_ultra_weights.py SOURCE --output FILE using the weights path shown above, then retry. If preparing weights beside the executable instead, restart the service with --weights pointing to that file." : " · Prepare the selected weights before loading the service model.") :
                            loading_runtime.status=="missing" ? "Inference worker not found: " + binary_ : "Could not verify inference assets before loading";
                        phase_="error"; load_failed_=true; want_load_=false; memory_=0; loaded_revision_.clear();
                        int log=g_open((data_+"/server.log").c_str(),O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC,0600);
                        if(log>=0){auto message=error_+"\n";auto written=write(log,message.data(),message.size());(void)written;close(log);}
                        fail_jobs("Couldn’t load model");continue;
                    }
                    loading_revision=configured_revision_;
                    lock.unlock();
                    worker = std::make_unique<Worker>(binary_, weights_, data_ + "/server.log", selected_preset_);
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
                    worker_pid_ = 0; memory_ = 0; used_memory_ = -1; loaded_revision_.clear();
                } else if (state.ready) {
                    if (phase_ == "loading") {
                        if (!same_file(weights_,loading_model) || !same_file(binary_,loading_runtime)) {
                            phase_="error";error_="Inference assets changed while the model was loading";load_failed_=true;
                            fail_jobs(error_);want_load_=false;loaded_revision_.clear();
                            lock.unlock();worker.reset();lock.lock();worker_pid_=0;memory_=0;used_memory_=-1;
                            continue;
                        }
                        loaded_revision_=loading_revision;idle_since_=Clock::now();
                    }
                    memory_ = state.server_vram_mb;
                    used_memory_ = state.used_vram_mb;
                    if (state.total_vram_mb > 0) capacity_ = state.total_vram_mb;
                    model_ = state.model; device_ = state.device; info_ = state.info_json;
                    worker_pid_ = state.pid;
                    if (auto result = worker->result()) {
                        if (active_) {
                            active_->completion.set_value({result->status, std::move(result->body), loaded_revision_});
                            active_.reset(); idle_since_ = now;
                        }
                        if (result->transport_failed) {
                            phase_ = "error"; error_ = "Transcription worker unavailable"; load_failed_ = false;
                            fail_jobs(error_); want_load_ = false;
                            lock.unlock(); worker.reset(); lock.lock();
                            worker_pid_ = 0; memory_ = 0; used_memory_ = -1; loaded_revision_.clear();
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
    gchar *weights_arg = nullptr, *binary_arg = nullptr, *data_arg = nullptr, *model_arg = nullptr;
    gchar *vad_arg = nullptr;
    gint port = -1, load_timeout = 120; // Omitted port prefers 8080, then a free port.
    GOptionEntry options[] = {
        {"port", 0, 0, G_OPTION_ARG_INT, &port, "Fixed API port (default: 8080 with automatic fallback; 0: free port)", "PORT"},
        {"weights", 0, 0, G_OPTION_ARG_FILENAME, &weights_arg, "FP16 weights (default: Ultra)", "FILE"},
        {"model", 0, 0, G_OPTION_ARG_STRING, &model_arg, "Model preset (default: ultra; custom --weights alone auto-detects architecture)", "v2|v3|ultra"},
        {"worker-binary", 0, 0, G_OPTION_ARG_FILENAME, &binary_arg, "Inference worker executable", "FILE"},
        {"data-dir", 0, 0, G_OPTION_ARG_FILENAME, &data_arg, "Settings and worker log directory", "DIRECTORY"},
        {"load-timeout", 0, 0, G_OPTION_ARG_INT, &load_timeout, "Model loading deadline in seconds", "SECONDS"},
        {"vad-model", 0, 0, G_OPTION_ARG_FILENAME, &vad_arg, "Pinned Silero v5 model", "FILE"},
        {nullptr, 0, 0, G_OPTION_ARG_NONE, nullptr, nullptr, nullptr}
    };
    auto* context = g_option_context_new("— persistent local Paraketto API");
    g_option_context_add_main_entries(context, options, nullptr);
    GError* error = nullptr;
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printerr("%s\n", error->message); g_error_free(error); g_option_context_free(context); return 1;
    }
    g_option_context_free(context);
    gst_init(nullptr, nullptr);
    try {
        if (port < -1 || port > 65535 || load_timeout < 1) throw std::runtime_error("Invalid port or load timeout");
        auto executable = take_string(g_file_read_link("/proc/self/exe", nullptr));
        auto base = take_string(g_path_get_dirname(executable.c_str()));
        std::string binary = binary_arg ? binary_arg : base + "/paraketto.cuda";
        if (!binary_arg && !g_file_test(binary.c_str(), G_FILE_TEST_IS_EXECUTABLE)) binary = base + "/paraketto";
        const auto* selected = model_arg || !weights_arg ? &paraketto::models::preset(model_arg ? model_arg : "ultra") : nullptr;
        std::string selected_preset = selected ? selected->key : "";
        std::string weights = weights_arg ? weights_arg : paraketto::models::weights_path(*selected, base,
            std::string(g_get_user_cache_dir()) + "/paraketto");
        std::string data = data_arg ? data_arg : std::string(g_get_user_data_dir()) + "/paraketto";
        binary = take_string(g_canonicalize_filename(binary.c_str(), nullptr));
        weights = take_string(g_canonicalize_filename(weights.c_str(), nullptr));
        data = take_string(g_canonicalize_filename(data.c_str(), nullptr));
        g_free(weights_arg); g_free(binary_arg); g_free(data_arg); g_free(model_arg);
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
        auto asr_revision=model_identity(binary,weights,fingerprint_file(binary),fingerprint_file(weights),selected_preset);
        Service service(binary,weights,data,load_timeout,asr_revision,selected_preset);
        SessionOptions session_options{data,asr_revision,vad_arg ? vad_arg : Segmenter::default_weights()};
        g_free(vad_arg);
        Sessions sessions(session_options,[&service](std::string wav) { return service.enqueue(std::move(wav)); },
            [&service](bool active) { service.speech(active); }, [&service] { return service.model_revision(); });
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
        http.Get("/status", [&](const auto&, auto& res) {
            Json asr(service.status()), extra(sessions.status());
            auto* object = asr.object(); auto* e = extra.object();
            for (const char* key : {"active_session_id","vad_status","session_error","queues"})
                if (json_object_has_member(e,key)) json_object_set_member(object,key,json_node_copy(json_object_get_member(e,key)));
            auto* capabilities = json_object_new(); json_object_set_boolean_member(capabilities,"sessions",true);
            json_object_set_object_member(object,"capabilities",capabilities);
            auto* models = json_object_new(); auto* asr_model = json_object_new();
            for (const char* key : {"status","worker_pid","server_vram_mb","vram_used_mb","vram_total_mb","error","model","model_preset","model_architecture"})
                json_object_set_member(asr_model,key,json_node_copy(json_object_get_member(object,key)));
            json_object_set_object_member(models,"asr",asr_model);
            json_object_set_object_member(object,"models",models);
            auto* node = json_node_init_object(json_node_alloc(),object);
            auto* generator = json_generator_new(); json_generator_set_root(generator,node);
            res.set_content(take_string(json_generator_to_data(generator,nullptr)),"application/json");
            json_node_free(node); g_object_unref(generator);
        });
        http.Get("/info", [&](const auto&, auto& res) { res.set_content(service.info(), "application/json"); });
        http.Get("/", [](const auto&, auto& res) { res.set_content(paraketto::web_ui, "text/html"); });
        http.Post("/model/load", [&](const auto& req, auto& res) { reply(res, service.load(req.body)); });
        http.Post("/model/unload", [&](const auto&, auto& res) { reply(res, service.unload()); });
        http.Post("/model/release", [&](const auto& req, auto& res) { reply(res, service.release(req.body)); });
        http.Patch("/settings", [&](const auto& req, auto& res) {
            auto result = service.settings(req.body);
            reply(res,result);
        });
        http.Post("/models/asr/load", [&](const auto& req, auto& res) { reply(res,service.load(req.body)); });
        http.Post("/models/asr/unload", [&](const auto&, auto& res) { reply(res,service.unload()); });
        sessions.routes(http);
        http.Post("/transcribe", [&](const auto& req, auto& res) {
            if (!req.has_file("file") || req.get_file_value("file").content.empty()) {
                reply(res, {400, json_error("Missing audio in multipart 'file' field")}); return;
            }
            reply(res, service.transcribe(req.get_file_value("file").content));
        });
        auto* loop = g_main_loop_new(nullptr, FALSE);
        struct Shutdown { Service* service; Sessions* sessions; httplib::Server* http; GMainLoop* loop; } shutdown{&service, &sessions, &http, loop};
        auto stop = [](gpointer p) -> gboolean {
            auto* shutdown = static_cast<Shutdown*>(p);
            shutdown->sessions->stop(); shutdown->service->stop(); shutdown->http->stop();
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
        // its worker. Releasing it here could admit a second inference worker.
        return listener_ok ? 0 : 1;
    } catch (const std::exception& e) { g_printerr("%s\n", e.what()); return 1; }
}
