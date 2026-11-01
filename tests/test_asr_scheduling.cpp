// Exercise the actual service controller with its normal subprocess Worker.
// The renamed executable entrypoint is discarded by the test link; production
// code and HTTP routes have no fixture switches or identity-check bypasses.
#define main paraketto_service_entry_for_test
#include "../src/service/main.cpp"
#undef main
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/wait.h>

using namespace paraketto::service;
using namespace paraketto::app;
namespace fs = std::filesystem;

static void wait_for(const std::function<bool()>& predicate) {
    auto deadline = Clock::now() + std::chrono::seconds(8);
    while (Clock::now() < deadline) {
        if (predicate()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("ASR scheduling fixture timed out");
}

static bool exited(pid_t pid) {
    if (kill(pid, 0) != 0) return true;
    // A container init may leave the reparented process as a zombie; it is
    // already dead and no longer owns execution resources.
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    std::string id, name;
    char state = 0;
    stat >> id >> name >> state;
    return state == 'Z';
}

struct Fixture {
    std::string directory = take_string(g_dir_make_tmp("paraketto-asr-scheduler-XXXXXX", nullptr));
    std::string data = directory + "/data", weights = directory + "/weights";
    Fixture() {
        fs::create_directory(data);
        write_file(weights, "fixture");
        write_file(data + "/preferences.ini", "[Model]\nidle-seconds=60\nkeep-loaded=false\n");
        write_file(directory + "/release-transcription", "yes");
        g_setenv("PARAKETTO_FIXTURE_DIR", directory.c_str(), true);
        g_setenv("PARAKETTO_FIXTURE_DELAY", "0.01", true);
    }
    ~Fixture() { fs::remove_all(directory); }
    int launches() const {
        int count = 0;
        if (fs::exists(directory + "/spawned"))
            for ([[maybe_unused]] const auto& entry : fs::directory_iterator(directory + "/spawned")) ++count;
        return count;
    }
    std::string revision(const std::string& binary) const {
        return model_identity(binary, weights, fingerprint_file(binary), fingerprint_file(weights));
    }
};

// A private concurrent HTTP worker makes serialization observable: an accidental
// second request can enter this server while the first is deliberately blocked.
// The normal Worker launches it through exactly the production argv/protocol.
static int fixture_worker() {
    const std::string directory = g_getenv("PARAKETTO_FIXTURE_DIR");
    fs::create_directories(directory + "/spawned");
    write_file(directory + "/spawned/" + std::to_string(getpid()), "worker");
    if (fs::exists(directory + "/fail-load")) return 1;
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(4); };
    server.Get("/status", [](const auto&, auto& response) {
        response.set_content("{\"status\":\"ready\",\"pid\":" + std::to_string(getpid()) + "}", "application/json");
    });
    server.Get("/info", [](const auto&, auto& response) {
        response.set_content("{\"model\":\"Fixture\",\"backend\":\"CPU fixture\",\"device\":\"No GPU\"}", "application/json");
    });
    std::mutex log_mutex;
    std::atomic<int> active{0};
    server.Post("/transcribe", [&](const auto& request, auto& response) {
        auto name = request.get_file_value("file").content;
        if (active.fetch_add(1) != 0) write_file(directory + "/overlap", "concurrent inference");
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::ofstream log(directory + "/requests", std::ios::app);
            log << name << '\n';
        }
        write_file(directory + "/started-" + name, "yes");
        wait_for([&] { return fs::exists(directory + "/release-transcription") || fs::exists(directory + "/release-" + name); });
        --active;
        response.set_content("{\"text\":\"" + name + "\",\"audio_duration_s\":1,\"inference_time_s\":0.01}", "application/json");
    });
    int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    std::cout << "{\"event\":\"ready\",\"pid\":" << getpid() << ",\"port\":" << port << "}" << std::endl;
    return server.listen_after_bind() ? 0 : 1;
}

static Reply await(std::future<Reply>& future) {
    wait_for([&] { return future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; });
    return future.get();
}

static void shared_queue(const std::string& binary) {
    Fixture fixture;
    fs::remove(fixture.directory + "/release-transcription");
    Service service(binary, fixture.weights, fixture.data, 2, fixture.revision(binary));
    // A legacy HTTP caller blocks in transcribe(), while session work keeps a
    // future from enqueue(). Exercise both adapters against the same controller.
    auto first = std::async(std::launch::async, [&] { return service.transcribe("legacy-first"); });
    wait_for([&] { return fs::exists(fixture.directory + "/started-legacy-first"); });
    auto second = service.enqueue("session-second");
    auto third = std::async(std::launch::async, [&] { return service.transcribe("legacy-third"); });
    wait_for([&] { return Json(service.status()).number("pending_requests") == 3; });
    auto fourth = service.enqueue("session-fourth");
    auto overflow = service.enqueue("overflow");
    assert(await(overflow).status == 429);
    assert(service.unload().status == 409);
    assert(service.settings("{\"keep_loaded\":true}").status == 200);
    assert(read_file(fixture.directory + "/requests") == "legacy-first\n");
    assert(second.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready);
    assert(!fs::exists(fixture.directory + "/overlap"));
    write_file(fixture.directory + "/release-legacy-first", "yes");
    auto reply = await(first);
    assert(reply.status == 200 && Json(reply.body).text("text") == "legacy-first");
    assert(reply.model_revision == service.model_revision());
    wait_for([&] { return fs::exists(fixture.directory + "/started-session-second"); });
    assert(read_file(fixture.directory + "/requests") == "legacy-first\nsession-second\n");
    assert(third.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready);
    write_file(fixture.directory + "/release-transcription", "yes");
    for (auto* future : {&second, &third, &fourth}) assert(await(*future).status == 200);
    assert(read_file(fixture.directory + "/requests") == "legacy-first\nsession-second\nlegacy-third\nsession-fourth\n");
    assert(!fs::exists(fixture.directory + "/overlap") && fixture.launches() == 1);
    wait_for([&] { return Json(service.status()).text("status") == "ready"; });
    int pid = Json(service.status()).number("worker_pid");
    assert(service.unload().status == 202);
    wait_for([&] { return Json(service.status()).text("status") == "sleeping"; });
    assert(exited(pid));
    std::cout << "PASS: concurrent legacy/session callers share bounded FIFO ASR with one worker and responsive controls\n";
}

static void failure_and_retry(const std::string& binary) {
    Fixture fixture;
    auto identity = fingerprint_file(fixture.weights);
    assert(identity.status == "verified" && identity.bytes == 7);
    assert(identity.sha256 == take_string(g_compute_checksum_for_string(G_CHECKSUM_SHA256, "fixture", -1)));
    auto missing = fingerprint_file(fixture.directory + "/missing");
    assert(missing.status == "missing" && missing.sha256.empty());
    assert(fingerprint_file(fixture.directory).status == "unreadable");
    write_file(fixture.directory + "/fail-load", "failed");
    Service service(binary, fixture.weights, fixture.data, 2, fixture.revision(binary));
    auto first = service.enqueue("first"), second = service.enqueue("second");
    assert(await(first).status == 503 && await(second).status == 503);
    assert(fixture.launches() == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(fixture.launches() == 1);
    fs::remove(fixture.directory + "/fail-load");
    auto retry = service.enqueue("retry");
    assert(await(retry).status == 200 && fixture.launches() == 2);
    std::cout << "PASS: verified fingerprints, failed load resolves the queue once, explicit new request retries\n";
}

static void stop_and_crash(const std::string& binary, bool crash) {
    Fixture fixture;
    fs::remove(fixture.directory + "/release-transcription");
    auto service = std::make_unique<Service>(binary, fixture.weights, fixture.data, 2, fixture.revision(binary));
    auto first = service->enqueue("active");
    wait_for([&] { return fs::exists(fixture.directory + "/started-active"); });
    auto second = service->enqueue("queued");
    int pid = Json(service->status()).number("worker_pid");
    auto started = Clock::now();
    if (crash) assert(kill(pid, SIGKILL) == 0);
    else service->stop();
    assert(await(first).status == 503 && await(second).status == 503);
    service.reset();
    assert(exited(pid));
    assert(Clock::now() - started < std::chrono::seconds(3));
    assert(!fs::exists(fixture.directory + "/started-queued"));
    std::cout << "PASS: " << (crash ? "worker crash" : "service shutdown") << " resolves active/queued ASR and reaps the owned child\n";
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--weights") return fixture_worker();
    if (argc == 4 && std::string(argv[1]) == "--owner") {
        const std::string directory = argv[2];
        g_setenv("PARAKETTO_FIXTURE_DIR", directory.c_str(), true);
        g_setenv("PARAKETTO_FIXTURE_IGNORE_TERM", "1", true);
        Worker worker(argv[3], directory + "/weights", directory + "/owner.log");
        wait_for([&] { return worker.state().ready; });
        write_file(directory + "/owned-pid", std::to_string(worker.state().pid));
        _exit(0); // No C++ destructors: exercise parent-death behavior.
    }
    assert(argc == 2);
    const auto binary = fs::absolute(argv[0]).string();
    shared_queue(binary);
    failure_and_retry(binary);
    stop_and_crash(binary, false);
    stop_and_crash(binary, true);
    {
        Fixture fixture;
        const auto external = fs::absolute(argv[1]).string();
        pid_t owner = fork(); assert(owner >= 0);
        if (owner == 0) {
            // GLib may own threads; exec a fresh owner rather than reusing its
            // inherited state after fork.
            execl(argv[0], argv[0], "--owner", fixture.directory.c_str(), external.c_str(), nullptr);
            _exit(127);
        }
        int status = 0;
        assert(waitpid(owner, &status, 0) == owner && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        int child = std::stoi(read_file(fixture.directory + "/owned-pid"));
        wait_for([&] { return exited(child); });
        std::cout << "PASS: abrupt service-owner death kills a TERM-ignoring ASR child\n";
    }
}
