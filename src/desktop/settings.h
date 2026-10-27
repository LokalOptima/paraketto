#pragma once

#include "common.h"
#include <algorithm>

namespace desktop {

struct Settings {
    int idle_seconds = 300;
    bool keep_loaded = false;

    void load(const std::string& path) {
        if (!g_file_test(path.c_str(), G_FILE_TEST_IS_REGULAR)) return;
        GKeyFile* file = g_key_file_new();
        GError* error = nullptr;
        if (!g_key_file_load_from_file(file, path.c_str(), G_KEY_FILE_NONE, &error)) {
            std::string message = error->message;
            g_error_free(error);
            g_key_file_unref(file);
            throw std::runtime_error(message);
        }
        if (g_key_file_has_key(file, "Model", "idle-seconds", nullptr))
            idle_seconds = std::clamp(g_key_file_get_integer(file, "Model", "idle-seconds", nullptr), 1, 86400);
        keep_loaded = g_key_file_get_boolean(file, "Model", "keep-loaded", nullptr);
        g_key_file_unref(file);
    }

    void save(const std::string& path) const {
        GKeyFile* file = g_key_file_new();
        g_key_file_set_integer(file, "Model", "idle-seconds", idle_seconds);
        g_key_file_set_boolean(file, "Model", "keep-loaded", keep_loaded);
        auto data = take_string(g_key_file_to_data(file, nullptr, nullptr));
        g_key_file_unref(file);
        write_file(path, data);
    }
};

} // namespace desktop
