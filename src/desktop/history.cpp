#include "history.h"
#include "common.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <glib/gstdio.h>
#include <pango/pango.h>

namespace desktop {

HistoryStats history_stats(const std::vector<Entry>& entries) {
    HistoryStats stats;
    size_t timed_words = 0;
    double seconds = 0;
    for (const auto& entry : entries) {
        if (entry.text.empty() || !entry.error.empty()) continue;
        std::vector<PangoLogAttr> boundaries(g_utf8_strlen(entry.text.c_str(), -1) + 1);
        pango_get_log_attrs(entry.text.c_str(), entry.text.size(), -1, pango_language_get_default(),
            boundaries.data(), boundaries.size());
        size_t words = 0;
        const char* character = entry.text.c_str();
        gunichar previous = 0, before_previous = 0;
        for (size_t i = 0; *character; ++i, character = g_utf8_next_char(character)) {
            // Pango splits contractions at apostrophes; count "don't" and "I’ll" as one word.
            bool contraction = (previous == '\'' || previous == 0x2019) && g_unichar_isalpha(before_previous);
            if (boundaries[i].is_word_start && !contraction) ++words;
            before_previous = previous;
            previous = g_utf8_get_char(character);
        }
        stats.words += words;
        if (words && std::isfinite(entry.duration) && entry.duration > 0) {
            timed_words += words;
            seconds += entry.duration;
        }
    }
    if (seconds > 0) stats.words_per_minute = std::llround(timed_words * 60.0 / seconds);
    return stats;
}

History::History(std::string directory) : directory_(std::move(directory)) {
    if (g_mkdir_with_parents(directory_.c_str(), 0700) != 0)
        throw std::runtime_error("Could not create history directory: " + std::string(g_strerror(errno)));
}

std::string History::audio_path(const Entry& entry) const { return directory_ + "/" + entry.id + ".wav"; }

Entry History::create(const std::string& wav, const std::string& destination) {
    Entry entry;
    GDateTime* now = g_date_time_new_now_local();
    std::string stamp = take_string(g_date_time_format(now, "%Y%m%dT%H%M%S%f"));
    entry.time = take_string(g_date_time_format(now, "%e %b · %H:%M"));
    g_date_time_unref(now);
    std::string uuid = take_string(g_uuid_string_random());
    entry.id = stamp + "-" + uuid;
    entry.destination = destination;
    entry.duration = wav.size() >= 44 ? (wav.size() - 44) / 32000.0 : 0;
    entry.delivery = "Transcribing…";
    write_file(audio_path(entry), wav);
    save(entry);
    return entry;
}

void History::save(const Entry& e) const {
    JsonBuilder* b = json_builder_new();
    json_builder_begin_object(b);
    json_field(b, "id", e.id);
    json_field(b, "time", e.time);
    json_field(b, "text", e.text);
    json_field(b, "destination", e.destination);
    json_field(b, "delivery", e.delivery);
    json_field(b, "error", e.error);
    json_field(b, "duration", e.duration);
    json_field(b, "inference_ms", e.inference_ms);
    json_builder_end_object(b);
    auto data = json_string(b);
    g_object_unref(b);
    write_file(directory_ + "/" + e.id + ".json", data);
}

std::vector<Entry> History::load() const {
    std::vector<Entry> entries;
    GDir* dir = g_dir_open(directory_.c_str(), 0, nullptr);
    if (!dir) return entries;
    while (const gchar* name = g_dir_read_name(dir)) {
        if (!g_str_has_suffix(name, ".json")) continue;
        try {
            Json json(read_file(directory_ + "/" + name));
            Entry e;
            e.id = std::string(name, strlen(name) - 5);
            e.time = json.text("time"); e.text = json.text("text");
            e.destination = json.text("destination"); e.delivery = json.text("delivery");
            e.error = json.text("error"); e.duration = json.number("duration");
            e.inference_ms = json.number("inference_ms");
            if (e.delivery == "Transcribing…") {
                e.delivery = "Interrupted";
                e.error = "Recording retained; transcription did not finish.";
            }
            entries.push_back(std::move(e));
        } catch (const std::exception& e) {
            g_warning("Skipping history entry %s: %s", name, e.what());
        }
    }
    g_dir_close(dir);
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.id > b.id; });
    return entries;
}

void History::remove(const Entry& e) const {
    // Remove metadata first: a failed audio deletion must not resurrect the entry.
    if (g_unlink((directory_ + "/" + e.id + ".json").c_str()) != 0 && errno != ENOENT)
        throw std::runtime_error("Could not delete transcription");
    if (g_unlink(audio_path(e).c_str()) != 0 && errno != ENOENT)
        throw std::runtime_error("Could not delete recording");
}

} // namespace desktop
