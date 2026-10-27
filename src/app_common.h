#pragma once

#include <glib.h>
#include <json-glib/json-glib.h>
#include <string>
#include <stdexcept>

namespace paraketto::app {

inline std::string take_string(gchar* value) {
    std::string result = value ? value : "";
    g_free(value);
    return result;
}

class Json {
    JsonParser* parser_ = json_parser_new();
public:
    explicit Json(const std::string& data) {
        GError* error = nullptr;
        if (!json_parser_load_from_data(parser_, data.data(), data.size(), &error)) {
            std::string message = error->message;
            g_error_free(error);
            g_object_unref(parser_);
            throw std::invalid_argument(message);
        }
        if (!JSON_NODE_HOLDS_OBJECT(json_parser_get_root(parser_))) {
            g_object_unref(parser_);
            throw std::invalid_argument("Expected a JSON object");
        }
    }
    ~Json() { g_object_unref(parser_); }
    Json(const Json&) = delete;
    JsonObject* object() const { return json_node_get_object(json_parser_get_root(parser_)); }
    std::string text(const char* name, const char* fallback = "") const {
        JsonNode* n = json_object_get_member(object(), name);
        return n && json_node_get_value_type(n) == G_TYPE_STRING ? json_node_get_string(n) : fallback;
    }
    double number(const char* name, double fallback = 0) const {
        JsonNode* n = json_object_get_member(object(), name);
        if (!n || !JSON_NODE_HOLDS_VALUE(n)) return fallback;
        GType t = json_node_get_value_type(n);
        return t == G_TYPE_INT64 || t == G_TYPE_DOUBLE ? json_node_get_double(n) : fallback;
    }
    bool boolean(const char* name, bool fallback = false) const {
        JsonNode* n = json_object_get_member(object(), name);
        return n && json_node_get_value_type(n) == G_TYPE_BOOLEAN ? json_node_get_boolean(n) : fallback;
    }
};

inline std::string read_file(const std::string& path) {
    gchar* data = nullptr;
    gsize size = 0;
    GError* error = nullptr;
    if (!g_file_get_contents(path.c_str(), &data, &size, &error)) {
        std::string message = error->message;
        g_error_free(error);
        throw std::runtime_error(message);
    }
    std::string result(data, size);
    g_free(data);
    return result;
}

inline void write_file(const std::string& path, const std::string& data) {
    GError* error = nullptr;
    if (!g_file_set_contents_full(path.c_str(), data.data(), data.size(),
            G_FILE_SET_CONTENTS_CONSISTENT, 0600, &error)) {
        std::string message = error->message;
        g_error_free(error);
        throw std::runtime_error(message);
    }
}

inline std::string json_string(JsonBuilder* builder) {
    JsonNode* root = json_builder_get_root(builder);
    JsonGenerator* generator = json_generator_new();
    json_generator_set_root(generator, root);
    std::string result = take_string(json_generator_to_data(generator, nullptr));
    json_node_free(root);
    g_object_unref(generator);
    return result;
}

inline void json_field(JsonBuilder* b, const char* name, const std::string& value) {
    json_builder_set_member_name(b, name);
    json_builder_add_string_value(b, value.c_str());
}

inline void json_field(JsonBuilder* b, const char* name, double value) {
    json_builder_set_member_name(b, name);
    json_builder_add_double_value(b, value);
}

inline void json_bool(JsonBuilder* b, const char* name, bool value) {
    json_builder_set_member_name(b, name);
    json_builder_add_boolean_value(b, value);
}

inline void json_int(JsonBuilder* b, const char* name, gint64 value) {
    json_builder_set_member_name(b, name);
    json_builder_add_int_value(b, value);
}

inline std::string json_error(const std::string& message) {
    auto* b = json_builder_new();
    json_builder_begin_object(b);
    json_field(b, "error", message);
    json_builder_end_object(b);
    auto result = json_string(b);
    g_object_unref(b);
    return result;
}

} // namespace paraketto::app
