// wav.h — WAV file reader (16kHz mono, int16 or float32)
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};

static inline WavData read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open WAV: %s\n", path.c_str()); std::exit(1); }

    char riff[4]; f.read(riff, 4);
    if (memcmp(riff, "RIFF", 4)) { fprintf(stderr, "Not RIFF: %s\n", path.c_str()); std::exit(1); }
    uint32_t file_size; f.read((char*)&file_size, 4);
    char wave[4]; f.read(wave, 4);
    if (memcmp(wave, "WAVE", 4)) { fprintf(stderr, "Not WAVE: %s\n", path.c_str()); std::exit(1); }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    std::vector<char> raw_data;

    while (f) {
        char id[4]; uint32_t sz;
        if (!f.read(id, 4) || !f.read((char*)&sz, 4)) break;
        if (!memcmp(id, "fmt ", 4)) {
            f.read((char*)&audio_format, 2); f.read((char*)&num_channels, 2);
            f.read((char*)&sample_rate, 4);
            f.seekg(6, std::ios::cur);  // skip byte_rate + block_align
            f.read((char*)&bits_per_sample, 2);
            if (sz > 16) f.seekg(sz - 16, std::ios::cur);
        } else if (!memcmp(id, "data", 4)) {
            data_size = sz; raw_data.resize(sz); f.read(raw_data.data(), sz);
        } else {
            f.seekg(sz, std::ios::cur);
        }
    }

    if (num_channels != 1) { fprintf(stderr, "Need mono, got %d ch\n", num_channels); std::exit(1); }
    if (sample_rate != 16000) { fprintf(stderr, "Need 16kHz, got %dHz: %s\n(resample with: ffmpeg -i input.wav -ar 16000 output.wav)\n", sample_rate, path.c_str()); std::exit(1); }

    WavData wav;
    wav.sample_rate = sample_rate;
    if (audio_format == 1 && bits_per_sample == 16) {
        int n = data_size / 2; wav.samples.resize(n);
        auto* src = (const int16_t*)raw_data.data();
        for (int i = 0; i < n; i++) wav.samples[i] = src[i] / 32768.0f;
    } else if (audio_format == 3 && bits_per_sample == 32) {
        int n = data_size / 4; wav.samples.resize(n);
        memcpy(wav.samples.data(), raw_data.data(), data_size);
    } else {
        fprintf(stderr, "Unsupported fmt=%d bits=%d\n", audio_format, bits_per_sample); std::exit(1);
    }
    return wav;
}

static inline WavData read_wav_from_memory(const char* buf, size_t len) {
    if (len < 44) return {};
    if (memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) return {};

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    const char* raw_data = nullptr;

    size_t pos = 12;
    while (pos + 8 <= len) {
        const char* id = buf + pos;
        uint32_t sz; memcpy(&sz, buf + pos + 4, 4);
        pos += 8;
        if (!memcmp(id, "fmt ", 4) && pos + 16 <= len) {
            memcpy(&audio_format, buf + pos, 2);
            memcpy(&num_channels, buf + pos + 2, 2);
            memcpy(&sample_rate, buf + pos + 4, 4);
            memcpy(&bits_per_sample, buf + pos + 14, 2);
        } else if (!memcmp(id, "data", 4)) {
            data_size = sz; raw_data = buf + pos;
            if (pos + sz > len) data_size = len - pos;
        }
        pos += sz;
    }

    if (!raw_data || num_channels != 1 || sample_rate != 16000) return {};

    WavData wav;
    wav.sample_rate = sample_rate;
    if (audio_format == 1 && bits_per_sample == 16) {
        int n = data_size / 2; wav.samples.resize(n);
        auto* src = (const int16_t*)raw_data;
        for (int i = 0; i < n; i++) wav.samples[i] = src[i] / 32768.0f;
    } else if (audio_format == 3 && bits_per_sample == 32) {
        int n = data_size / 4; wav.samples.resize(n);
        memcpy(wav.samples.data(), raw_data, data_size);
    } else {
        return {};
    }
    return wav;
}
