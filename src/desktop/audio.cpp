#include "audio.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace desktop {

std::string pcm_wav(const std::vector<int16_t>& samples) {
    std::string wav("RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0", 44);
    auto put32 = [&](size_t offset, uint32_t n) {
        for (int i = 0; i < 4; ++i) wav[offset + i] = static_cast<char>(n >> (i * 8));
    };
    put32(4, 36 + samples.size() * 2);
    put32(40, samples.size() * 2);
    for (int16_t sample : samples) {
        wav.push_back(static_cast<char>(sample));
        wav.push_back(static_cast<char>(static_cast<uint16_t>(sample) >> 8));
    }
    return wav;
}

Recorder::~Recorder() { stop(); }

bool Recorder::start(const std::string& source, std::string& error) {
    stop();
    samples_.clear();
    sample_count_ = 0;
    rms_ = 0;
    GError* e = nullptr;
    const auto description = source + " ! audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,channels=1,rate=16000 ! "
        "appsink name=capture emit-signals=true sync=false max-buffers=8 drop=false";
    pipeline_ = gst_parse_launch(description.c_str(), &e);
    if (e || !pipeline_) {
        error = e ? e->message : "Could not create microphone pipeline";
        if (e) g_error_free(e);
        stop();
        return false;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "capture");
    g_signal_connect(sink, "new-sample", G_CALLBACK(sample), this);
    gst_object_unref(sink);
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        error = this->error();
        if (error.empty()) error = "Could not open microphone";
        stop();
        return false;
    }
    return true;
}

GstFlowReturn Recorder::sample(GstAppSink* sink, gpointer data) {
    auto& self = *static_cast<Recorder*>(data);
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_EOS;
    GstMapInfo map{};
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        double sum = 0;
        size_t count = map.size / 2;
        std::lock_guard<std::mutex> lock(self.mutex_);
        count = std::min(count, max_samples - self.samples_.size());
        for (size_t i = 0; i < count; ++i) {
            int16_t value = static_cast<int16_t>(map.data[2*i] | map.data[2*i+1] << 8);
            self.samples_.push_back(value);
            double normalized = value / 32768.0;
            sum += normalized * normalized;
        }
        self.sample_count_ = self.samples_.size();
        self.rms_ = count ? std::sqrt(sum / count) : 0;
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

std::string Recorder::error() {
    if (!pipeline_) return {};
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
    gst_object_unref(bus);
    if (!msg) return {};
    GError* e = nullptr;
    gst_message_parse_error(msg, &e, nullptr);
    std::string result = e ? e->message : "Microphone error";
    if (e) g_error_free(e);
    gst_message_unref(msg);
    return result;
}

std::string Recorder::stop() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    rms_ = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    auto wav = samples_.empty() ? std::string() : pcm_wav(samples_);
    samples_.clear();
    return wav;
}

Player::~Player() { stop(); }
void Player::stop() {
    if (!pipeline_) return;
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
}
bool Player::play(const std::string& path, std::string& error) {
    stop();
    pipeline_ = gst_element_factory_make("playbin", nullptr);
    if (!pipeline_) { error = "GStreamer playbin is unavailable"; return false; }
    gchar* uri = g_filename_to_uri(path.c_str(), nullptr, nullptr);
    if (!uri) { error = "Invalid recording path"; stop(); return false; }
    g_object_set(pipeline_, "uri", uri, nullptr);
    g_free(uri);
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        error = "Could not play recording";
        stop();
        return false;
    }
    return true;
}
bool Player::playing() {
    if (!pipeline_) return false;
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    gst_object_unref(bus);
    if (msg) { gst_message_unref(msg); stop(); }
    return pipeline_ != nullptr;
}

} // namespace desktop
