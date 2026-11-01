#include "capture.h"
#include "app_common.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace paraketto::service {
namespace fs = std::filesystem;
namespace {
constexpr size_t queue_limit = 4 * audio_rate;
void io_error(const char* operation) { throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno)); }
void sync_fd(int fd) { if (fsync(fd) != 0) io_error("Synchronizing recording"); }
void write_all(int fd, const void* data, size_t length) {
    auto* bytes = static_cast<const char*>(data);
    while (length) {
        auto count = ::write(fd, bytes, length);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) io_error("Writing recording");
        bytes += count; length -= count;
    }
}
std::string block_name(int64_t sequence) {
    std::ostringstream out; out << std::setfill('0') << std::setw(6) << sequence << ".pcm"; return out.str();
}
uint32_t le32(const unsigned char* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
// Legacy history was saved as 16 kHz S16 WAV. Locate its data chunk without
// materializing the file, retaining old entries' recorded text provenance.
int64_t wav_data_offset(int fd) {
    unsigned char header[12];
    if (pread(fd, header, 12, 0) != 12 || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4))
        throw std::runtime_error("Invalid legacy WAV");
    int64_t offset = 12;
    for (int i = 0; i < 1024; ++i) {
        unsigned char chunk[8];
        if (pread(fd, chunk, 8, offset) != 8) break;
        uint32_t size = le32(chunk + 4);
        if (!std::memcmp(chunk, "data", 4)) return offset + 8;
        offset += 8 + int64_t(size) + (size & 1);
    }
    throw std::runtime_error("Legacy WAV has no audio data");
}
}

void sync_directory(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) io_error("Opening recording directory");
    int result = fsync(fd), saved = errno; close(fd); errno = saved;
    if (result != 0) io_error("Synchronizing recording directory");
}

std::string wav_header(int64_t frames) {
    if (frames < 0 || frames > (UINT32_MAX - 36) / 2) throw std::invalid_argument("Requested WAV range is too large");
    std::string h("RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0", 44);
    auto put = [&](int at, uint32_t value) { for (int i = 0; i < 4; ++i) h[at + i] = char(value >> (8*i)); };
    put(4, uint32_t(frames * 2 + 36)); put(40, uint32_t(frames * 2)); return h;
}

std::vector<int16_t> read_audio(SessionStore& store, const std::string& id, int64_t start, int64_t end) {
    if (start < 0 || end < start || end - start > 120 * audio_rate) throw std::invalid_argument("Audio read exceeds bounded range");
    std::vector<int16_t> result(static_cast<size_t>(end - start));
    int64_t position = start;
    for (const auto& b : store.audio_blocks(id, start, end)) {
        int64_t lo = std::max(start, b.start_sample), hi = std::min(end, b.start_sample + b.durable_frames);
        if (hi <= lo) continue;
        if (lo != position) throw std::runtime_error("Saved audio contains a gap");
        auto path = fs::path(b.path).is_absolute() ? b.path : store.data_directory() + "/" + b.path;
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0 && path.size() >= 5 && path.substr(path.size()-5) == ".part") {
            path.resize(path.size()-5); fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        }
        if (fd < 0) io_error("Opening saved audio");
        try {
            int64_t offset = (lo - b.start_sample) * 2;
            if (fs::path(path).extension() == ".wav") offset += wav_data_offset(fd);
            auto* out = reinterpret_cast<char*>(result.data() + lo - start);
            size_t remaining = static_cast<size_t>((hi - lo) * 2);
            while (remaining) {
                auto n = pread(fd, out, remaining, offset);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) throw std::runtime_error("Saved audio is shorter than its durable metadata");
                out += n; offset += n; remaining -= n;
            }
        } catch (...) { close(fd); throw; }
        close(fd); position = hi;
    }
    if (position != end) throw std::runtime_error("Requested audio has not been saved");
    return result;
}

void recover_audio(SessionStore& store, const SessionRecord& session) {
    fs::path relative = fs::path("sessions") / session.id / "audio";
    fs::path directory = fs::path(store.data_directory()) / relative;
    if (!fs::exists(directory)) return;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(directory)) {
        auto name = e.path().filename().string();
        auto dot = name.find('.');
        if (dot == std::string::npos || dot < 6 || !std::all_of(name.begin(), name.begin()+dot, ::isdigit)) continue;
        if (name.substr(dot) != ".pcm" && name.substr(dot) != ".pcm.part") continue;
        if (e.is_symlink() || !e.is_regular_file()) throw std::runtime_error("Unexpected recording block type");
        files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    for (auto path : files) {
        auto sequence = std::stoll(path.filename().string());
        auto bytes = fs::file_size(path);
        if (bytes > uint64_t(audio_block_frames * 2)) throw std::runtime_error("Recording block exceeds size limit");
        int fd = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) io_error("Recovering recording block");
        if (bytes % 2 && ftruncate(fd, bytes - 1) != 0) { close(fd); io_error("Recovering partial sample"); }
        try { sync_fd(fd); } catch (...) { close(fd); throw; }
        close(fd);
        if (path.extension() == ".part") { auto dest = path; dest.replace_extension(""); fs::rename(path, dest); path = dest; }
        sync_directory(directory.string());
        store.put_audio_block({session.id, (relative/path.filename()).string(), sequence, sequence*audio_block_frames, int64_t(bytes/2), true});
    }
}

AudioCapture::AudioCapture(std::string directory, std::string session, std::string source, bool live,
                           Checkpoint checkpoint, Gap gap)
    : directory_(std::move(directory)), session_(std::move(session)), source_(std::move(source)),
      live_(live), checkpoint_(std::move(checkpoint)), gap_(std::move(gap)) {}
AudioCapture::~AudioCapture() { stop(); }
void AudioCapture::start() {
    auto audio = fs::path(directory_)/"sessions"/session_/"audio";
    fs::create_directories(audio);
    for (auto p = audio; p != fs::path(directory_).parent_path(); p = p.parent_path()) {
        chmod(p.c_str(), 0700); sync_directory(p.string());
    }
    GError* error = nullptr;
    auto description = source_ + " ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! appsink name=capture emit-signals=true sync=false max-buffers=8 drop=false";
    pipeline_ = gst_parse_launch(description.c_str(), &error);
    if (error || !pipeline_) {
        std::string message = error ? error->message : "Could not create microphone pipeline";
        if (error) g_error_free(error);
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        throw std::runtime_error(message);
    }
    auto* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "capture");
    if (!sink) throw std::runtime_error("Missing capture sink");
    g_signal_connect(sink, "new-sample", G_CALLBACK(sample), this); gst_object_unref(sink);
    writer_ = std::thread([this] { write(); });
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        fail("Could not open audio source"); stop(); throw std::runtime_error(state().error);
    }
}

void AudioCapture::fail(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_.empty()) error_ = error;
    stopping_ = true; wake_.notify_all();
}
void AudioCapture::stop() {
    std::lock_guard<std::mutex> control(control_);
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; paused_ = false; }
    wake_.notify_all();
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (writer_.joinable()) writer_.join();
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
    rms_ = 0;
}
void AudioCapture::pause() {
    std::lock_guard<std::mutex> control(control_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || finished_ || !pipeline_) throw std::runtime_error("Recording is no longer active");
        if (paused_) return;
    }
    // Live sources stop producing data in PAUSED; a non-live fixture may need
    // one preroll buffer before the state change completes. No device is reopened.
    auto transition = gst_element_set_state(pipeline_, GST_STATE_PAUSED);
    GstState current = GST_STATE_VOID_PENDING;
    if (transition != GST_STATE_CHANGE_FAILURE)
        transition = gst_element_get_state(pipeline_, &current, nullptr, 2 * GST_SECOND);
    if (transition == GST_STATE_CHANGE_FAILURE || transition == GST_STATE_CHANGE_ASYNC || current != GST_STATE_PAUSED) {
        fail("Could not pause audio source");
        throw std::runtime_error("Could not pause audio source");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_ || finished_) throw std::runtime_error(error_.empty() ? "Recording ended while pausing" : error_);
    paused_ = true; rms_ = 0;
    flush_completed_ = false; flush_requested_ = true; wake_.notify_all();
    wake_.wait(lock, [&] { return flush_completed_ || finished_ || !error_.empty(); });
    if (!flush_completed_ || !error_.empty())
        throw std::runtime_error(error_.empty() ? "Recording ended before its pause checkpoint" : error_);
}
void AudioCapture::resume() {
    std::lock_guard<std::mutex> control(control_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || finished_ || !pipeline_) throw std::runtime_error("Recording is no longer active");
        if (!paused_) return;
        // A device may report a new PTS origin or DISCONT on resume. It is an
        // intentional boundary, not missing recorded source audio.
        expected_pts_ = GST_CLOCK_TIME_NONE; reset_timestamp_ = true; paused_ = false;
    }
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        fail("Could not resume audio source");
        throw std::runtime_error("Could not resume audio source");
    }
}
CaptureState AudioCapture::state() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {captured_, written_, durable_, discontinuities_, rms_, finished_, paused_, error_};
}
GstFlowReturn AudioCapture::sample(GstAppSink* sink, gpointer data) {
    auto& self = *static_cast<AudioCapture*>(data);
    auto* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_EOS;
    auto* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map{}; GstFlowReturn result = GST_FLOW_OK;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        Buffer item;
        const size_t count = map.size / 2;
        // Reject malformed/oversized buffers before allocating callback memory.
        if (map.size % 2 || count > queue_limit) { self.fail("Audio buffer exceeds capture queue capacity"); result = GST_FLOW_ERROR; }
        else {
            item.samples.resize(count);
            double sum = 0;
            for (size_t i = 0; i < count; ++i) {
                item.samples[i] = int16_t(uint16_t(map.data[2*i]) | uint16_t(map.data[2*i+1]) << 8);
                double x = item.samples[i] / 32768.0; sum += x*x;
            }
            std::unique_lock<std::mutex> lock(self.mutex_);
            if (!self.live_) self.wake_.wait(lock, [&] { return self.stopping_ || self.paused_ || self.queued_ + count <= queue_limit; });
            if (self.stopping_) result = GST_FLOW_FLUSHING;
            else if (self.paused_) { /* An in-flight buffer cannot extend an acknowledged pause. */ }
            else if (self.queued_ + count > queue_limit) {
                self.error_ = "Audio capture queue overflow; recording stopped"; self.stopping_ = true; result = GST_FLOW_ERROR;
            } else {
                auto pts = GST_BUFFER_PTS(buffer);
                item.discontinuity = !self.reset_timestamp_ && GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DISCONT) && self.captured_ > 0;
                if (GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(self.expected_pts_) && pts > self.expected_pts_ + 5*GST_MSECOND) {
                    item.discontinuity = true;
                    item.missing = gst_util_uint64_scale(pts - self.expected_pts_, audio_rate, GST_SECOND);
                }
                if (GST_CLOCK_TIME_IS_VALID(pts)) self.expected_pts_ = pts + gst_util_uint64_scale(count, GST_SECOND, audio_rate);
                self.reset_timestamp_ = false;
                self.queued_ += count; self.captured_ += count;
                self.rms_ = count ? std::sqrt(sum/count) : 0;
                self.buffers_.push_back(std::move(item));
            }
            self.wake_.notify_all();
        }
        gst_buffer_unmap(buffer, &map);
    } else { self.fail("Could not read microphone buffer"); result = GST_FLOW_ERROR; }
    gst_sample_unref(sample); return result;
}
void AudioCapture::poll_bus() {
    auto* bus = gst_element_get_bus(pipeline_);
    auto* message = gst_bus_pop_filtered(bus, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);
    if (!message) return;
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* error = nullptr; gst_message_parse_error(message, &error, nullptr);
        fail(error ? error->message : "Audio source failed"); if (error) g_error_free(error);
    } else { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; wake_.notify_all(); }
    gst_message_unref(message);
}
void AudioCapture::write() {
    int fd = -1; int64_t sequence = 0, frames = 0;
    auto relative = fs::path("sessions")/session_/"audio";
    auto directory = fs::path(directory_)/relative;
    auto last_sync = std::chrono::steady_clock::now();
    auto checkpoint = [&](bool final) {
        if (fd < 0) return;
        sync_fd(fd);
        auto name = block_name(sequence);
        if (final) { fs::rename(directory/(name+".part"), directory/name); sync_directory(directory.string()); }
        checkpoint_({session_, (relative/(name+(final ? "" : ".part"))).string(), sequence, sequence*audio_block_frames, frames, final});
        durable_ = written_.load(); last_sync = std::chrono::steady_clock::now();
        if (final) { close(fd); fd = -1; frames = 0; ++sequence; }
    };
    try {
        while (true) {
            poll_bus();
            Buffer buffer; bool flush = false;
            { std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait_for(lock, std::chrono::milliseconds(100), [&] { return stopping_ || !buffers_.empty() || flush_requested_; });
                if (!buffers_.empty()) { buffer = std::move(buffers_.front()); buffers_.pop_front(); queued_ -= buffer.samples.size(); wake_.notify_all(); }
                else if (stopping_) break;
                else if (flush_requested_) { flush_requested_ = false; flush = true; }
            }
            if (buffer.discontinuity) {
                ++discontinuities_;
                if (gap_) gap_(written_, buffer.missing, "Audio source timestamp discontinuity");
            }
            size_t pos = 0;
            while (pos < buffer.samples.size()) {
                if (fd < 0) {
                    fd = open((directory/(block_name(sequence)+".part")).c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
                    if (fd < 0) io_error("Creating recording block");
                    sync_directory(directory.string());
                }
                size_t n = std::min<size_t>(buffer.samples.size()-pos, audio_block_frames-frames);
                write_all(fd, buffer.samples.data()+pos, n*2); pos += n; frames += n; written_ += n;
                if (frames == audio_block_frames) checkpoint(true);
            }
            if (fd >= 0 && written_ > durable_ && (flush || std::chrono::steady_clock::now()-last_sync >= std::chrono::seconds(1))) checkpoint(false);
            if (flush) { std::lock_guard<std::mutex> lock(mutex_); flush_completed_ = true; wake_.notify_all(); }
        }
        checkpoint(true);
    } catch (const std::exception& e) { fail(e.what()); }
    if (fd >= 0) close(fd);
    { std::lock_guard<std::mutex> lock(mutex_); rms_ = 0; finished_ = true; wake_.notify_all(); }
}
} // namespace paraketto::service
