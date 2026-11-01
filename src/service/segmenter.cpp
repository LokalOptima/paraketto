#include "segmenter.h"
#include "capture.h"
#include "vad/vad_ggml.h"
#include "app_common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>

namespace paraketto::service {
namespace {
constexpr const char* model_sha = "0776449541b2b34cea5479bda89f6028e304ba0728fd4a1f053e17169f74a3eb";
bool checksum_matches(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    GChecksum* checksum = g_checksum_new(G_CHECKSUM_SHA256);
    std::array<char, 65536> buffer;
    while (input) { input.read(buffer.data(), buffer.size()); g_checksum_update(checksum, reinterpret_cast<const guchar*>(buffer.data()), input.gcount()); }
    bool valid = input.eof() && std::string(g_checksum_get_string(checksum)) == model_sha;
    g_checksum_free(checksum); return valid;
}
}
struct Segmenter::Impl {
    int64_t position, covered, speech_start = -1, last_voice = 0;
    bool filtered = false, speaking = false;
    std::string message;
    std::unique_ptr<SileroVad> vad;
    Plan plan;
    std::deque<std::pair<int64_t, double>> energies;
    Impl(int64_t start, const std::string& weights, bool filter, Plan p)
        : position(start), covered(start), plan(std::move(p)) {
        if (filter && checksum_matches(weights)) {
            vad = std::make_unique<SileroVad>();
            if (!vad->load(weights)) vad.reset();
        }
        filtered = bool(vad);
        message = filtered ? "Silero v5" : filter ? "VAD unavailable; silence filtering disabled" : "Silence filtering disabled";
    }
    void emit(int64_t end, const std::vector<SourceSpan>& spans) {
        if (end <= covered) return;
        plan(covered, end, spans); covered = end;
    }
    int64_t boundary() const {
        auto best = position;
        double energy = INFINITY;
        for (const auto& e : energies) if (e.first >= position - 2*audio_rate && e.first > covered && e.second < energy) {
            best = e.first; energy = e.second;
        }
        return best;
    }
    float probability(const int16_t* samples, size_t n) {
        std::array<float, 512> f{};
        double energy = 0;
        for (size_t i = 0; i < n; ++i) { f[i] = samples[i] / 32768.f; energy += f[i]*f[i]; }
        if (!vad) return 1;
        // Quiet short words can have low confidence despite being clear to
        // ASR, especially after extended silence. Apply bounded gain only to
        // the detector input; stored audio and source spans remain unchanged.
        const double rms = std::sqrt(energy/std::max<size_t>(n,1));
        const float gain = rms > 0 ? std::clamp(0.01/rms,1.0,16.0) : 1;
        for (size_t i = 0; i < n; ++i) f[i] *= gain;
        float p = vad->process(f.data());
        if (!std::isfinite(p) || p < 0 || p > 1) {
            vad.reset(); filtered = false; message = "VAD failed; silence filtering disabled"; return 1;
        }
        return p;
    }
    void frame(const int16_t* samples, size_t n) {
        double energy = 0;
        for (size_t i=0; i<n; ++i) { double s = samples[i]/32768.; energy += s*s; }
        energy /= std::max<size_t>(n,1);
        energies.emplace_back(position, energy);
        float p = probability(samples,n);
        const int64_t begin = position;
        position += n;
        while (!energies.empty() && energies.front().first < position-2*audio_rate) energies.pop_front();
        if (!filtered) {
            speaking = energy > 0.00001;
            if (position-covered >= 60*audio_rate) { auto end = boundary(); emit(end, {{covered,end}}); }
            return;
        }
        // Retain uncertain scores. Hysteresis avoids chopping quiet syllables.
        bool voice = p >= (speech_start >= 0 ? 0.10f : 0.15f);
        if (voice) {
            if (speech_start < 0) speech_start = std::max(covered, begin-4800);
            last_voice = position;
        }
        speaking = speech_start >= 0 && position-last_voice < 8000;
        if (speech_start >= 0) {
            int64_t quiet = position-last_voice;
            if (quiet >= 2*audio_rate || (position-speech_start >= 30*audio_rate && quiet >= 11200)) {
                auto end = std::min(position, last_voice+8000);
                emit(end, {{speech_start,end}}); speech_start = -1;
            } else if (position-speech_start >= 90*audio_rate) {
                auto end = boundary(); emit(end, {{speech_start,end}}); speech_start = end;
            }
        }
        if (speech_start < 0 && position-covered >= audio_rate) emit(position-4800, {});
    }
};
Segmenter::Segmenter(int64_t start, std::string weights, bool filter, Plan plan)
    : impl_(std::make_unique<Impl>(start, weights, filter, std::move(plan))) {}
Segmenter::~Segmenter() = default;
void Segmenter::process(const int16_t* samples, size_t count, bool final) {
    if (count % 512 && !final) throw std::invalid_argument("VAD requires complete 512-sample frames");
    for (size_t i=0; i<count; i+=512) impl_->frame(samples+i, std::min<size_t>(512,count-i));
    if (final) flush();
}
void Segmenter::warmup(const int16_t* samples, size_t count) {
    // The saved boundary need not be frame aligned because speech padding is
    // measured in samples. Keep the final context immediately adjacent to the
    // resumed audio; dropping the trailing remainder leaves a context gap.
    for (size_t i=count%512; i+512<=count; i+=512) impl_->probability(samples+i,512);
}
void Segmenter::flush() {
    auto& s = *impl_;
    if (!s.filtered) s.emit(s.position, {{s.covered,s.position}});
    else if (s.speech_start >= 0) s.emit(s.position, {{s.speech_start,s.position}});
    else s.emit(s.position, {});
    s.speech_start = -1; s.speaking = false;
}
int64_t Segmenter::position() const { return impl_->position; }
bool Segmenter::speech_active() const { return impl_->speaking; }
bool Segmenter::filtering() const { return impl_->filtered; }
const std::string& Segmenter::status() const { return impl_->message; }
std::string Segmenter::default_weights() {
    std::string path = std::string(g_get_user_cache_dir()) + "/paraketto/vad/silero_vad.bin";
    if (g_file_test(path.c_str(), G_FILE_TEST_IS_REGULAR)) return path;
    return std::string(g_get_user_cache_dir()) + "/jarvis/silero_vad.bin";
}
}
