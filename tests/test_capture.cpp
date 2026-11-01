#include "service/capture.h"
#include "app_common.h"
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sys/resource.h>
#include <thread>

using namespace paraketto::service;
using namespace paraketto::app;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void wait(const std::function<bool()>& condition, double seconds = 10) {
    auto deadline = Clock::now() + std::chrono::duration<double>(seconds);
    while (Clock::now() < deadline) {
        if (condition()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("Capture fixture timed out");
}
struct Fixture {
    std::string directory = take_string(g_dir_make_tmp("paraketto-capture-test-XXXXXX", nullptr));
    std::unique_ptr<SessionStore> store = std::make_unique<SessionStore>(directory);
    SessionRecord session = store->create_session("test", "import");
    ~Fixture() { store.reset(); fs::remove_all(directory); }
    fs::path audio() { return fs::path(directory) / "sessions" / session.id / "audio"; }
    void block(const AudioBlock& block) { store->put_audio_block(block); }
};

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    {
        Fixture fixture;
        std::mutex mutex; std::condition_variable wake;
        bool entered = false, release = false;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc samplesperbuffer=512 num-buffers=32 ! identity sync=true", true,
            [&](const AudioBlock& block) {
                { std::unique_lock<std::mutex> lock(mutex);
                    entered = true; wake.notify_all();
                    wake.wait(lock, [&] { return release; }); }
                fixture.block(block);
            });
        capture.start();
        wait([&] { return capture.state().captured >= 1536; });
        auto pausing = std::async(std::launch::async, [&] { capture.pause(); });
        { std::unique_lock<std::mutex> lock(mutex); assert(wake.wait_for(lock, std::chrono::seconds(3), [&] { return entered; })); }
        assert(pausing.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready);
        assert(capture.state().paused && capture.state().durable == 0);
        { std::lock_guard<std::mutex> lock(mutex); release = true; } wake.notify_all();
        pausing.get();
        auto paused = capture.state();
        assert(paused.paused && !paused.finished && paused.error.empty());
        assert(paused.captured == paused.written && paused.written == paused.durable);
        assert(paused.rms == 0 && paused.discontinuities == 0);
        capture.pause();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        assert(capture.state().captured == paused.captured);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == paused.captured);
        assert(!fixture.store->audio_blocks(fixture.session.id).front().finalized);
        capture.resume(); capture.resume();
        wait([&] { return capture.state().captured >= paused.captured + 1536; });
        capture.pause();
        auto second = capture.state();
        assert(second.captured > paused.captured && second.durable == second.captured);
        capture.resume();
        wait([&] { return capture.state().finished; });
        capture.stop();
        auto finished = capture.state();
        assert(finished.error.empty() && finished.discontinuities == 0);
        assert(finished.captured == 32 * 512 && finished.durable == finished.captured);
        assert(read_audio(*fixture.store, fixture.session.id, 0, finished.captured).size() == 32 * 512);
        assert(fixture.store->audio_blocks(fixture.session.id).size() == 1);
        std::cout << "PASS: pause waits for durable PCM and paced audio resumes with exact source frames and no timestamp gaps\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc is-live=true samplesperbuffer=512", true,
            [&](const AudioBlock& block) { fixture.block(block); });
        capture.start(); wait([&] { return capture.state().captured >= 1536; });
        capture.pause();
        auto first = capture.state();
        assert(first.paused && first.captured == first.durable && first.written == first.durable);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        assert(capture.state().captured == first.captured);
        capture.resume(); wait([&] { return capture.state().captured >= first.captured + 1536; });
        capture.pause();
        auto second = capture.state();
        assert(second.paused && second.captured == second.durable && second.discontinuities == 0);
        capture.stop();
        auto stopped = capture.state();
        assert(stopped.finished && stopped.captured == second.captured && stopped.durable == second.captured);
        assert(stopped.error.empty() && stopped.discontinuities == 0);
        assert(read_audio(*fixture.store, fixture.session.id, 0, stopped.durable).size() == size_t(stopped.captured));
        std::cout << "PASS: live-source Pause freezes captured time, Resume appends saved samples, Stop preserves the exact paused boundary\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc is-live=true samplesperbuffer=512", true,
            [](const AudioBlock&) { throw std::runtime_error("pause checkpoint rejected"); });
        capture.start(); wait([&] { return capture.state().captured >= 512; });
        bool failed = false;
        try { capture.pause(); } catch (const std::runtime_error&) { failed = true; }
        capture.stop();
        assert(failed && capture.state().error.find("pause checkpoint rejected") != std::string::npos);
        assert(capture.state().durable == 0 && capture.state().written > 0);
        recover_audio(*fixture.store, fixture.session);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == capture.state().written);
        std::cout << "PASS: a failed pause checkpoint never acknowledges unsaved audio and its prefix is recoverable\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc wave=sine freq=440 volume=0.25 samplesperbuffer=512 num-buffers=2000",
            false, [&](const AudioBlock& block) { fixture.block(block); });
        capture.start();
        wait([&] {
            auto state = capture.state();
            // At most the 4 s queue plus the one buffer currently being written.
            assert(state.captured - state.written <= 8 * audio_rate);
            return state.finished;
        });
        capture.stop();
        auto state = capture.state();
        assert(state.error.empty());
        assert(state.captured == 2000 * 512 && state.written == state.captured && state.durable == state.captured);
        auto blocks = fixture.store->audio_blocks(fixture.session.id);
        assert(blocks.size() == 3);
        for (size_t i = 0; i < blocks.size(); ++i) {
            assert(blocks[i].sequence == static_cast<int64_t>(i));
            assert(blocks[i].start_sample == static_cast<int64_t>(i) * audio_block_frames);
            assert(blocks[i].finalized && blocks[i].path.find(".part") == std::string::npos);
            assert(fs::file_size(fs::path(fixture.directory) / blocks[i].path) == static_cast<uint64_t>(blocks[i].durable_frames * 2));
        }
        assert(blocks[0].durable_frames == audio_block_frames && blocks[1].durable_frames == audio_block_frames);
        assert(blocks[2].durable_frames == 4 * audio_rate);
        auto crossing = read_audio(*fixture.store, fixture.session.id, audio_block_frames - 16, audio_block_frames + 16);
        assert(crossing.size() == 32);
        std::vector<int16_t> expected(32);
        std::ifstream first(fs::path(fixture.directory) / blocks[0].path, std::ios::binary);
        first.seekg((audio_block_frames - 16) * 2); first.read(reinterpret_cast<char*>(expected.data()), 32);
        std::ifstream second(fs::path(fixture.directory) / blocks[1].path, std::ios::binary);
        second.read(reinterpret_cast<char*>(expected.data() + 16), 32);
        assert(crossing == expected);
        bool rejected = false;
        try { read_audio(*fixture.store, fixture.session.id, 0, 121 * audio_rate); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
        fixture.store->seal_capture(fixture.session.id);
        fixture.store.reset(); fixture.store = std::make_unique<SessionStore>(fixture.directory);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == state.captured);
        assert(read_audio(*fixture.store, fixture.session.id, audio_block_frames - 16, audio_block_frames + 16) == expected);
        std::cout << "PASS: 30 s rotation, bounded import queue, cross-block read and persisted coverage\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc wave=sine samplesperbuffer=512 num-buffers=3", false,
            [](const AudioBlock&) { throw std::runtime_error("injected metadata commit failure"); });
        capture.start(); wait([&] { return capture.state().finished; }); capture.stop();
        auto state = capture.state();
        assert(state.error.find("injected") != std::string::npos && state.written == 1536 && state.durable == 0);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == 0);
        recover_audio(*fixture.store, fixture.session);
        recover_audio(*fixture.store, fixture.session);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == 1536);
        assert(fixture.store->audio_blocks(fixture.session.id).size() == 1);
        assert(read_audio(*fixture.store, fixture.session.id, 0, 1536).size() == 1536);
        std::cout << "PASS: rename-before-metadata failure is visible and recovery is idempotent\n";
    }
    {
        Fixture fixture;
        fs::create_directories(fixture.audio());
        auto path = fixture.audio() / "000000.pcm.part";
        { std::ofstream file(path, std::ios::binary); int16_t samples[] = {1, -2, 3, -4, 5, -6, 7}; file.write(reinterpret_cast<char*>(samples), sizeof(samples)); file.put('x'); }
        fixture.block({fixture.session.id, fs::relative(path, fixture.directory).string(), 0, 0, 3, false});
        recover_audio(*fixture.store, fixture.session);
        assert(!fs::exists(path) && fs::file_size(fixture.audio() / "000000.pcm") == 14);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == 7);
        assert((read_audio(*fixture.store, fixture.session.id, 0, 7) == std::vector<int16_t>{1, -2, 3, -4, 5, -6, 7}));
        std::cout << "PASS: interrupted .part tail recovered with incomplete sample truncated\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc samplesperbuffer=64001 num-buffers=1", true,
            [&](const AudioBlock& block) { fixture.block(block); });
        capture.start(); wait([&] { return capture.state().finished; }); capture.stop();
        auto state = capture.state();
        assert(state.error.find("capacity") != std::string::npos && state.durable == 0);
        std::cout << "PASS: oversized microphone buffer rejected without growing memory\n";
    }
    {
        Fixture fixture;
        std::mutex mutex; std::condition_variable wake;
        bool entered = false, release = false;
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc is-live=true samplesperbuffer=512", true,
            [&](const AudioBlock& block) {
                { std::unique_lock<std::mutex> lock(mutex);
                    entered = true; wake.notify_all();
                    wake.wait(lock, [&] { return release; }); }
                fixture.block(block);
            });
        capture.start();
        { std::unique_lock<std::mutex> lock(mutex); assert(wake.wait_for(lock, std::chrono::seconds(3), [&] { return entered; })); }
        wait([&] { return !capture.state().error.empty(); }, 6);
        auto failed = capture.state();
        assert(failed.error.find("overflow") != std::string::npos);
        assert(failed.captured - failed.written <= 4 * audio_rate);
        { std::lock_guard<std::mutex> lock(mutex); release = true; } wake.notify_all();
        capture.stop();
        assert(capture.state().durable == capture.state().written);
        assert(capture.state().durable <= capture.state().captured);
        std::cout << "PASS: a blocked writer causes visible bounded-queue failure and preserves its prefix\n";
    }
    {
        Fixture fixture;
        struct rlimit saved{}; assert(getrlimit(RLIMIT_FSIZE, &saved) == 0);
        auto limited = saved; limited.rlim_cur = 4096;
        auto old_handler = signal(SIGXFSZ, SIG_IGN);
        assert(setrlimit(RLIMIT_FSIZE, &limited) == 0);
        AudioCapture capture(fixture.directory, fixture.session.id,
            "audiotestsrc samplesperbuffer=512 num-buffers=500", false,
            [&](const AudioBlock& block) { fixture.block(block); });
        capture.start(); wait([&] { return capture.state().finished; }); capture.stop();
        assert(setrlimit(RLIMIT_FSIZE, &saved) == 0); signal(SIGXFSZ, old_handler);
        auto state = capture.state();
        assert(state.error.find("Writing recording") != std::string::npos && state.durable == 0);
        recover_audio(*fixture.store, fixture.session);
        auto recovered = fixture.store->get_session(fixture.session.id)->durable_samples;
        assert(recovered > 0 && recovered <= 2048);
        assert(read_audio(*fixture.store, fixture.session.id, 0, recovered).size() == static_cast<size_t>(recovered));
        std::cout << "PASS: actual file write failure preserves a recoverable prefix without false durability\n";
    }
    {
        Fixture fixture;
        AudioCapture capture(fixture.directory, fixture.session.id, "nonexistent-paraketto-test-source", true,
            [&](const AudioBlock& block) { fixture.block(block); });
        bool failed = false;
        try { capture.start(); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        assert(fixture.store->get_session(fixture.session.id)->durable_samples == 0);
        std::cout << "PASS: unavailable capture source fails without claiming saved audio\n";
    }
}
