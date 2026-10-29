#include "input.h"
#include "app_common.h"
#include <gdk/gdkx.h>
#include <gio/gdesktopappinfo.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <algorithm>
#include <vector>

namespace {
constexpr const char* shortcut_path = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/paraketto/";

std::string application_name(const std::string& window_class, const std::string& instance = {}) {
    for (const auto& id : {window_class, instance}) {
        if (id.empty()) continue;
        if (auto* app = g_desktop_app_info_new((id + ".desktop").c_str())) {
            std::string name = g_app_info_get_display_name(G_APP_INFO(app));
            g_object_unref(app);
            return name;
        }
    }
    std::string name;
    GList* applications = g_app_info_get_all();
    for (auto* item = applications; item && name.empty(); item = item->next) {
        if (!G_IS_DESKTOP_APP_INFO(item->data)) continue;
        auto* app = G_DESKTOP_APP_INFO(item->data);
        const char* wm_class = g_desktop_app_info_get_startup_wm_class(app);
        const char* id = g_app_info_get_id(G_APP_INFO(app));
        std::string desktop_id = id ? id : "";
        if (g_str_has_suffix(desktop_id.c_str(), ".desktop")) desktop_id.resize(desktop_id.size() - 8);
        for (const auto& candidate : {window_class, instance}) {
            if (!candidate.empty() && ((wm_class && !g_ascii_strcasecmp(wm_class, candidate.c_str())) ||
                    !g_ascii_strcasecmp(desktop_id.c_str(), candidate.c_str()))) {
                name = g_app_info_get_display_name(G_APP_INFO(app));
                break;
            }
        }
    }
    g_list_free_full(applications, g_object_unref);
    return name.empty() ? (window_class.empty() ? instance : window_class) : name;
}
}

namespace desktop {
using namespace paraketto::app;

TextInput::TextInput() {
    if (GDK_IS_X11_DISPLAY(gdk_display_get_default())) xdo_ = xdo_new(nullptr);
}

TextInput::~TextInput() {
    if (xdo_) xdo_free(xdo_);
}

Window TextInput::active_window() {
    if (!xdo_) return 0;
    Window window = 0;
    Atom actual; int format; unsigned long count, after; unsigned char* data = nullptr;
    XGetWindowProperty(xdo_->xdpy, DefaultRootWindow(xdo_->xdpy),
        XInternAtom(xdo_->xdpy, "_NET_ACTIVE_WINDOW", False), 0, 1, False, XA_WINDOW,
        &actual, &format, &count, &after, &data);
    if (data && count && format == 32) window = *reinterpret_cast<Window*>(data);
    if (data) XFree(data);
    if (!window) xdo_get_focused_window_sane(xdo_, &window);
    return window;
}

void TextInput::begin(bool insert) {
    end();
    insert_ = insert;
    armed_ = true;
    invalid_ = false;
    target_ = active_window();
    destination_ = insert ? "Focused application" : "History";
    window_class_.clear();
    if (insert && target_ && xdo_) {
        XClassHint hint{};
        if (XGetClassHint(xdo_->xdpy, target_, &hint)) {
            window_class_ = hint.res_class ? hint.res_class : "";
            auto name = application_name(window_class_, hint.res_name ? hint.res_name : "");
            if (!name.empty()) destination_ = std::move(name);
            if (hint.res_name) XFree(hint.res_name);
            if (hint.res_class) XFree(hint.res_class);
        }
    }
}

void TextInput::poll() {
    if (armed_ && insert_ && target_ && active_window() != target_) invalid_ = true;
}

void TextInput::end() {
    armed_ = false;
}

std::string TextInput::deliver(const std::string& text) {
    poll();
    std::string result;
    if (!armed_ || !insert_) result = "Saved to history";
    else if (changed()) result = "Focus changed · copy to insert";
    else if (target_ && xdo_) {
        // Send complete transcripts through the application's normal paste path.
        // Keep the transcript on the clipboard; timed restoration races readers.
        gtk_clipboard_set_text(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), text.c_str(), text.size());
        std::string lower = window_class_;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return g_ascii_tolower(c); });
        bool terminal = lower.find("terminal") != std::string::npos ||
            lower.find("terminator") != std::string::npos || lower.find("ghostty") != std::string::npos;
        // Let released global-shortcut modifiers settle before delivery (the app
        // also delays delivery until modifiers are released).
        int rc = xdo_send_keysequence_window(xdo_, CURRENTWINDOW, terminal ? "ctrl+shift+v" : "ctrl+v", 12000);
        result = rc == 0 ? "Paste sent to " + destination_ : "Saved to history · copy to insert";
    } else result = "Saved to history · no active input";
    end();
    return result;
}

bool install_shortcut(const std::string& executable, const std::string& accelerator, std::string& error) {
    auto* source = g_settings_schema_source_get_default();
    auto* schema = g_settings_schema_source_lookup(source, "org.gnome.settings-daemon.plugins.media-keys", TRUE);
    if (!schema) { error = "Set a desktop shortcut to: " + executable + " --toggle"; return false; }
    g_settings_schema_unref(schema);
    GSettings* settings = g_settings_new("org.gnome.settings-daemon.plugins.media-keys");
    gchar** paths = g_settings_get_strv(settings, "custom-keybindings");
    std::vector<std::string> existing;
    bool found = false;
    for (size_t i = 0; paths[i]; ++i) {
        existing.emplace_back(paths[i]);
        if (existing.back() == shortcut_path) { found = true; continue; }
        auto* binding = g_settings_new_with_path("org.gnome.settings-daemon.plugins.media-keys.custom-keybinding", paths[i]);
        std::string used = take_string(g_settings_get_string(binding, "binding"));
        g_object_unref(binding);
        guint key1 = 0, key2 = 0; GdkModifierType mods1{}, mods2{};
        gtk_accelerator_parse(used.c_str(), &key1, &mods1);
        gtk_accelerator_parse(accelerator.c_str(), &key2, &mods2);
        if (key1 == key2 && mods1 == mods2) {
            g_strfreev(paths); g_object_unref(settings);
            error = "That shortcut is already assigned to another custom action";
            return false;
        }
    }
    g_strfreev(paths);
    if (!found) existing.emplace_back(shortcut_path);
    auto* binding = g_settings_new_with_path("org.gnome.settings-daemon.plugins.media-keys.custom-keybinding", shortcut_path);
    std::string quoted = take_string(g_shell_quote(executable.c_str()));
    bool ok = g_settings_set_string(binding, "name", "Paraketto dictation") &&
        g_settings_set_string(binding, "command", (quoted + " --toggle").c_str()) &&
        g_settings_set_string(binding, "binding", accelerator.c_str());
    std::vector<const char*> names;
    for (const auto& name : existing) names.push_back(name.c_str());
    names.push_back(nullptr);
    ok = ok && g_settings_set_strv(settings, "custom-keybindings", names.data());
    g_settings_sync();
    g_object_unref(binding); g_object_unref(settings);
    if (!ok) error = "Could not save the desktop shortcut";
    return ok;
}

std::string shortcut_label() {
    auto* source = g_settings_schema_source_get_default();
    auto* schema = g_settings_schema_source_lookup(source, "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding", TRUE);
    if (!schema) return "Shortcut not configured";
    g_settings_schema_unref(schema);
    auto* settings = g_settings_new_with_path("org.gnome.settings-daemon.plugins.media-keys.custom-keybinding", shortcut_path);
    std::string accel = take_string(g_settings_get_string(settings, "binding"));
    g_object_unref(settings);
    guint key = 0; GdkModifierType mods{};
    gtk_accelerator_parse(accel.c_str(), &key, &mods);
    if (key == GDK_KEY_space && mods == (GDK_CONTROL_MASK | GDK_MOD1_MASK)) return "Ctrl+Alt+Space";
    return key ? take_string(gtk_accelerator_get_label(key, mods)) : "Shortcut not configured";
}

} // namespace desktop
