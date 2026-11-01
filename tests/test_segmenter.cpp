#include "service/segmenter.h"
#include "service/vad/vad_ggml.h"
#include "app_common.h"
#include "wav.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace paraketto::service;
using namespace paraketto::app;
constexpr int sample_rate = 16000;

struct Group { int64_t start, end; std::vector<SourceSpan> spans; };
static std::vector<int16_t> deterministic_audio() {
    uint32_t noise = 0x12345678;
    std::vector<int16_t> samples(64000);
    for (int i = 0; i < 64000; ++i) {
        noise = noise * 1664525u + 1013904223u;
        int region = i / 16000;
        if (region == 1) samples[i] = int((noise >> 16) & 0xffff) % 101 - 50;
        else if (region == 2) samples[i] = ((i % 160) - 80) * 80 + int((noise >> 16) & 0xffff) % 401 - 200;
    }
    return samples;
}
static std::string hash_file(const std::string& path) {
    auto content = read_file(path);
    return take_string(g_compute_checksum_for_data(G_CHECKSUM_SHA256,
        reinterpret_cast<const guchar*>(content.data()), content.size()));
}
static void coverage(const std::vector<Group>& groups, int64_t begin, int64_t end) {
    auto previous = begin;
    for (const auto& group : groups) {
        assert(group.start == previous && group.end > group.start);
        int64_t retained = 0, span_end = group.start;
        for (const auto& span : group.spans) {
            assert(span.start >= span_end && span.end > span.start && span.end <= group.end);
            retained += span.end - span.start; span_end = span.end;
        }
        assert(retained <= 90 * sample_rate);
        previous = group.end;
    }
    assert(previous == end);
}
static std::vector<Group> plan_speech(const std::vector<int16_t>& samples, const std::string& weights, int64_t crash = -1) {
    std::vector<Group> groups;
    auto plan = [&](auto start, auto end, const auto& spans) { groups.push_back({start,end,spans}); };
    auto detector = std::make_unique<Segmenter>(0,weights,true,plan);
    if (crash >= 0) {
        crash = crash / 512 * 512;
        detector->process(samples.data(),crash);
        const auto restart = groups.empty() ? 0 : groups.back().end;
        detector = std::make_unique<Segmenter>(restart,weights,true,plan);
        // Deliberately not a multiple of 512: warmup must end at restart.
        const auto from = std::max<int64_t>(0,restart-sample_rate);
        detector->warmup(samples.data()+from,restart-from);
        detector->process(samples.data()+restart,samples.size()-restart,true);
    } else detector->process(samples.data(),samples.size(),true);
    coverage(groups,0,samples.size());
    return groups;
}
static std::vector<bool> retained_samples(const std::vector<Group>& groups, size_t count) {
    std::vector<bool> mask(count);
    for (const auto& group : groups) for (auto span : group.spans)
        for (auto i=span.start; i<span.end; ++i) { assert(!mask[i]); mask[i]=true; }
    return mask;
}

int main(int argc, char** argv) {
    assert(argc >= 1 && argc <= 3);
    std::string weights = argc >= 2 ? argv[1] : Segmenter::default_weights();
    std::string fixture = argc == 3 ? argv[2] : "tests/vadfixtures/reference.json";
    Json reference(read_file(fixture));
    assert(hash_file(weights) == reference.text("model_sha256"));
    auto* cases = json_object_get_array_member(reference.object(), "cases");
    for (guint i = 0; i < json_array_get_length(cases); ++i) {
        auto* item = json_array_get_object_element(cases, i);
        std::string name = json_object_get_string_member(item, "name");
        std::vector<float> samples;
        if (name == "deterministic-v1") {
            for (auto value : deterministic_audio()) samples.push_back(value / 32768.f);
        } else {
            std::string path = json_object_get_string_member(item, "path");
            if (!std::filesystem::exists(path)) {
                std::cout << "SKIP: optional local speech fixture " << path << '\n';
                continue;
            }
            assert(hash_file(path) == json_object_get_string_member(item, "wav_sha256"));
            samples = paraketto::read_wav(path).samples;
        }
        auto* expected = json_object_get_array_member(item, "probabilities");
        const size_t frames = samples.size() / 512;
        assert(frames == json_array_get_length(expected));
        SileroVad vad;
        assert(vad.load(weights));
        std::vector<float> actual;
        double worst = 0;
        for (size_t frame = 0; frame < frames; ++frame) {
            float p = vad.process(samples.data() + frame * 512);
            assert(std::isfinite(p) && p >= 0 && p <= 1);
            actual.push_back(p);
            worst = std::max(worst, std::abs(p - json_array_get_double_element(expected, frame)));
        }
        assert(worst < 0.0001);
        vad.reset();
        for (size_t frame = 0; frame < frames; ++frame)
            assert(std::abs(actual[frame] - vad.process(samples.data() + frame * 512)) < 0.000001);
        std::cout << "PASS: " << name << " reference probabilities and reset; max error " << worst << '\n';
    }
    {
        std::vector<Group> groups;
        Segmenter segmenter(0, weights, false, [&](auto start, auto end, const auto& spans) { groups.push_back({start, end, spans}); });
        std::array<int16_t, 512> frame{};
        for (int i = 0; i < 125 * sample_rate / 512; ++i) segmenter.process(frame.data(), frame.size());
        segmenter.flush(); segmenter.flush();
        assert(!segmenter.filtering());
        coverage(groups, 0, 125 * sample_rate / 512 * 512);
        assert(groups.size() >= 3);
        for (const auto& group : groups) assert(group.spans.size() == 1 && group.spans[0].start == group.start && group.spans[0].end == group.end);
        std::cout << "PASS: disabled VAD retains all audio with bounded, contiguous groups\n";
    }
    {
        auto directory = take_string(g_dir_make_tmp("paraketto-vad-test-XXXXXX", nullptr));
        auto invalid = directory + "/invalid.bin";
        { std::ofstream file(invalid); file << "SVAD\1\0\0\0"; }
        std::vector<Group> groups;
        Segmenter segmenter(4096, invalid, true, [&](auto start, auto end, const auto& spans) { groups.push_back({start, end, spans}); });
        assert(!segmenter.filtering() && segmenter.status().find("unavailable") != std::string::npos);
        std::array<int16_t, 17> partial{};
        bool rejected = false;
        try { segmenter.process(partial.data(), partial.size()); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && segmenter.position() == 4096);
        segmenter.process(partial.data(), partial.size(), true);
        coverage(groups, 4096, 4113);
        assert(groups.size() == 1 && groups[0].spans.size() == 1);
        std::filesystem::remove_all(directory);
        std::cout << "PASS: checksum rejection before model parsing; short Stop tail preserved\n";
    }
    {
        std::vector<Group> groups;
        Segmenter segmenter(0, weights, true, [&](auto start, auto end, const auto& spans) { groups.push_back({start, end, spans}); });
        assert(segmenter.filtering());
        std::array<int16_t, 512> silence{};
        for (int i = 0; i < 320; ++i) segmenter.process(silence.data(), silence.size());
        segmenter.flush();
        coverage(groups, 0, 320 * 512);
        for (const auto& group : groups) assert(group.spans.empty());
        assert(!segmenter.speech_active());
        std::cout << "PASS: sustained silence advances coverage without ASR input\n";
    }
    for (const std::string kind : {"white","brown","hum","breath"}) for (double gain : {0.001,0.01,0.1}) {
        std::vector<int16_t> noise(12*sample_rate);
        uint32_t random=0x12345678; double previous=0;
        for (size_t i=0; i<noise.size(); ++i) {
            random=random*1664525u+1013904223u;
            double value=(int(random>>16)-32768)/32768.;
            if (kind=="brown" || kind=="breath") { previous=0.96*previous+0.04*value; value=previous*5; }
            if (kind=="hum") value=std::sin(2*3.141592653589793*60*i/sample_rate);
            if (kind=="breath") {
                double phase=double(i%(3*sample_rate))/(3*sample_rate);
                value *= phase<0.4 ? std::sin(3.141592653589793*phase/0.4) : 0;
            }
            noise[i]=std::clamp(int(std::lround(value*gain*32768)),-32768,32767);
        }
        int64_t retained=0;
        for (const auto& group : plan_speech(noise,weights)) for (auto span : group.spans) retained+=span.end-span.start;
        // Conservative VAD may keep a brief ambiguous breath; steady noise
        // must not become an ongoing inference workload.
        assert(kind=="breath" ? retained<=sample_rate : retained==0);
    }
    std::cout << "PASS: steady noise/hum skipped; ambiguous breath retention bounded below one second at three levels\n";
    const std::string short_speech_file="data/librispeech/7127-75946-0001.wav";
    if (std::filesystem::exists(short_speech_file)) {
        // 'AH VERY WELL': at this gain its highest VAD probability is ~0.228,
        // so the old 0.25 onset threshold erased the entire real utterance.
        auto wav=paraketto::read_wav(short_speech_file);
        for (double gain : {0.025,0.016}) for (int pause : {3,23}) {
            std::vector<int16_t> quiet(pause*sample_rate);
            for (auto value : wav.samples) quiet.push_back(std::lround(value*32768*gain));
            quiet.resize(quiet.size()+3*sample_rate);
            auto groups=plan_speech(quiet,weights);
            double all_energy=0,kept_energy=0;
            for (auto value : quiet) all_energy+=double(value)*value;
            for (const auto& group : groups) for (auto span : group.spans)
                for (auto i=span.start; i<span.end; ++i) kept_energy+=double(quiet[i])*quiet[i];
            assert(kept_energy/all_energy>0.995);
        }
        std::cout << "PASS: quiet short words retained after both short and extended pauses\n";
    } else std::cout << "SKIP: optional quiet short-speech fixture " << short_speech_file << '\n';
    const std::string speech_file = "data/librispeech/1089-134686-0000.wav";
    if (std::filesystem::exists(speech_file)) {
        auto wav = paraketto::read_wav(speech_file);
        std::vector<int16_t> samples(3 * sample_rate);
        for (auto value : wav.samples) samples.push_back(static_cast<int16_t>(std::clamp(value, -1.f, 1.f) * 32767));
        samples.resize(samples.size() + 5 * sample_rate);
        std::vector<Group> groups;
        Segmenter segmenter(0, weights, true, [&](auto start, auto end, const auto& spans) { groups.push_back({start, end, spans}); });
        segmenter.process(samples.data(), samples.size(), true);
        coverage(groups, 0, samples.size());
        int64_t retained = 0;
        for (const auto& group : groups) for (const auto& span : group.spans) retained += span.end - span.start;
        assert(retained > 5 * sample_rate && retained < static_cast<int64_t>(samples.size()) - 3 * sample_rate);
        assert(!segmenter.speech_active());
        std::cout << "PASS: real speech retained between extended pauses and final tail flushed\n";
        std::vector<int16_t> source;
        for (auto value : wav.samples) source.push_back(std::clamp(int(std::lround(value*32768)),-32768,32767));
        std::vector<int16_t> background;
        for (int i=0; i<40; ++i) background.insert(background.end(),source.begin(),source.begin()+4800);
        for (const auto& group : plan_speech(background,weights)) assert(group.spans.empty());
        for (double gain : {0.25,0.016}) for (int pause : {0,3,20}) {
            std::vector<int16_t> quiet(pause*sample_rate);
            for (auto value : source) quiet.push_back(std::lround(value*gain));
            quiet.resize(quiet.size()+3*sample_rate);
            auto groups = plan_speech(quiet,weights);
            double all_energy=0, kept_energy=0;
            for (auto value : quiet) all_energy += double(value)*value;
            for (const auto& group : groups) for (auto span : group.spans)
                for (auto i=span.start; i<span.end; ++i) kept_energy += double(quiet[i])*quiet[i];
            // At -12/-36 dB, a pause must not erase a word or clipped onset.
            // Energy is a retention check, not a substitute for ASR comparison.
            assert(kept_energy/all_energy > 0.995);
        }
        std::cout << "PASS: quiet real speech retained after 0/3/20 s pauses at -12/-36 dB\n";
        std::vector<int16_t> continuous;
        for (int i=0; i<13; ++i) continuous.insert(continuous.end(),source.begin()+10400,source.begin()+150400);
        auto baseline = plan_speech(continuous,weights);
        assert(baseline.size() >= 2);
        auto expected = retained_samples(baseline,continuous.size());
        assert(std::all_of(expected.begin(),expected.end(),[](bool kept){ return kept; }));
        for (int crash : {29,91,95}) {
            auto resumed = plan_speech(continuous,weights,int64_t(crash)*sample_rate);
            assert(retained_samples(resumed,continuous.size()) == expected);
        }
        std::cout << "PASS: 113.75 s continuous speech has bounded forced splits and unchanged retained audio after restart\n";
    }
}
