#pragma once
#include "app_common.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace desktop {
inline std::string json_text(JsonObject* object, const char* key, const char* fallback = "") {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_VALUE(n) && json_node_get_value_type(n) == G_TYPE_STRING ? json_node_get_string(n) : fallback;
}
inline int64_t json_number(JsonObject* object, const char* key, int64_t fallback = 0) {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_VALUE(n) && (json_node_get_value_type(n) == G_TYPE_INT64 || json_node_get_value_type(n) == G_TYPE_DOUBLE) ?
        json_node_get_int(n) : fallback;
}
inline double json_decimal(JsonObject* object, const char* key, double fallback = 0) {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_VALUE(n) && (json_node_get_value_type(n) == G_TYPE_INT64 || json_node_get_value_type(n) == G_TYPE_DOUBLE) ?
        json_node_get_double(n) : fallback;
}
inline bool json_flag(JsonObject* object, const char* key, bool fallback = false) {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_VALUE(n) && json_node_get_value_type(n) == G_TYPE_BOOLEAN ? json_node_get_boolean(n) : fallback;
}
inline JsonObject* json_object(JsonObject* object, const char* key) {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_OBJECT(n) ? json_node_get_object(n) : nullptr;
}
inline JsonArray* json_array(JsonObject* object, const char* key) {
    auto* n = object ? json_object_get_member(object, key) : nullptr;
    return n && JSON_NODE_HOLDS_ARRAY(n) ? json_node_get_array(n) : nullptr;
}
inline std::string json_body(std::initializer_list<std::pair<std::string, std::string>> fields) {
    auto* b = json_builder_new(); json_builder_begin_object(b);
    for (const auto& [key, value] : fields) paraketto::app::json_field(b, key.c_str(), value);
    json_builder_end_object(b); auto result = paraketto::app::json_string(b); g_object_unref(b); return result;
}
struct SessionEdit {
    std::string id, before, after, origin;
    bool enabled = false;
    int64_t decision_revision = 0;
    int64_t start = -1, end = -1;
};
struct SessionSegment {
    std::string id, raw, edited, state, error;
    int64_t sequence = 0, revision = 0, start = 0, end = 0;
    std::vector<SessionEdit> edits;
    static SessionSegment parse(JsonObject* o) {
        SessionSegment s;
        s.id = json_text(o, "id"); s.raw = json_text(o, "raw_text"); s.edited = json_text(o, "edited_text", s.raw.c_str());
        if(s.id.empty())s.id=std::to_string(json_number(o,"id"));
        s.state = json_text(o, "state"); s.error = json_text(o, "error");
        s.sequence = json_number(o, "sequence"); s.revision = json_number(o, "revision_id");
        if (auto* spans = json_array(o, "source_spans"); spans && json_array_get_length(spans)) {
            s.start = json_number(json_array_get_object_element(spans, 0), "start");
            s.end = json_number(json_array_get_object_element(spans, json_array_get_length(spans) - 1), "end");
        }
        if (auto* edits = json_array(o, "edits")) for (guint i = 0; i < json_array_get_length(edits); ++i) {
            auto* e = json_array_get_object_element(edits, i);
            if (json_text(e, "origin") != "dictionary") continue;
            auto id = json_text(e, "id");
            if (id.empty()) id = std::to_string(json_number(e, "id"));
            s.edits.push_back({id, json_text(e, "before"), json_text(e, "after"), json_text(e, "origin"),
                json_flag(e, "enabled"), json_number(e, "decision_revision"),
                json_number(e, "start", -1), json_number(e, "end", -1)});
        }
        return s;
    }
};
struct SessionView {
    std::string id, source, capture, transcription, error, time, preview, publication_id;
    int64_t revision = 0, generation = 0, delivery_epoch = 0, durable = 0, transcribed = 0, words = 0;
    double duration = 0;
    bool publication_complete = false, replacements_enabled = true;
    void update(JsonObject* o) {
        if (!o) return;
        id = json_text(o, "id", id.c_str()); source = json_text(o, "source", source.c_str());
        capture = json_text(o, "capture_state", capture.c_str()); transcription = json_text(o, "transcription_state", transcription.c_str());
        error = json_text(o, "error");
        replacements_enabled = json_flag(o, "replacements_enabled", replacements_enabled);
        if (transcription == "completed") transcription = "complete";
        revision = json_number(o, "revision", revision); generation = json_number(o, "generation", generation);
        delivery_epoch = json_number(o, "delivery_epoch", delivery_epoch);
        durable = json_number(o, "durable_samples", durable);
        transcribed = static_cast<int64_t>(json_decimal(o, "transcribed_through", transcribed / 16000.0) * 16000);
        duration = json_decimal(o, "duration_seconds", duration);
        words = json_number(o, "word_count", words); preview = json_text(o, "preview", preview.c_str());
        time = json_text(o, "created_at", time.c_str());
        if (auto* p = json_object(o, "publication")) {
            publication_id = json_text(p, "id"); if (publication_id.empty() && json_number(p, "id")) publication_id = std::to_string(json_number(p, "id"));
            publication_complete = json_flag(p, "complete");
        } else if (json_object_has_member(o, "publication")) {
            publication_id.clear(); publication_complete = false;
        }
    }
    std::string attention() const {
        if (!error.empty()) return error;
        if (capture == "interrupted") return "Recording interrupted. Review the transcript before using it.";
        if (transcription == "failed" || transcription == "incomplete") return "Transcription incomplete. Retry to finish it.";
        return {};
    }
};
} // namespace desktop
