#include "input.h"
#include "common.h"
#include <gdk/gdkx.h>
#include <gio/gdesktopappinfo.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <algorithm>

namespace {
struct DictationEngine { IBusEngine parent; };
struct DictationEngineClass { IBusEngineClass parent; };
G_DEFINE_TYPE(DictationEngine, dictation_engine, IBUS_TYPE_ENGINE)

desktop::TextInput* owner(IBusEngine* engine) {
    return static_cast<desktop::TextInput*>(g_object_get_data(G_OBJECT(engine), "owner"));
}
void dictation_engine_class_init(DictationEngineClass* klass) {
    auto* engine = IBUS_ENGINE_CLASS(klass);
    engine->process_key_event = [](IBusEngine*, guint, guint, guint) -> gboolean { return FALSE; };
    // IBus 1.5.29 can send the first FocusIn before its asynchronous FocusId
    // capability query completes. It is not proof of an editable input.
    engine->focus_in = [](IBusEngine* e) {
        if (auto* input = owner(e)) input->focus_legacy(e);
    };
    engine->focus_out = [](IBusEngine* e) {
        if (auto* input = owner(e)) input->focus_out(e);
    };
    engine->focus_in_id = [](IBusEngine* e, const gchar* path, const gchar* client) {
        if (auto* input = owner(e)) input->focus_in(e, path, client);
    };
    engine->focus_out_id = [](IBusEngine* e, const gchar*) {
        if (auto* input = owner(e)) input->focus_out(e);
    };
    engine->disable = [](IBusEngine* e) {
        if (auto* input = owner(e)) input->focus_out(e);
    };
}
void dictation_engine_init(DictationEngine*) {}
constexpr const char* engine_name = "paraketto-dictation";
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

TextInput::TextInput() {
    lifetime_ = std::make_shared<Lifetime>(Lifetime{this});
    if (GDK_IS_X11_DISPLAY(gdk_display_get_default())) xdo_ = xdo_new(nullptr);
    ibus_init();
    bus_ = ibus_bus_new();
    g_signal_connect(bus_, "disconnected", G_CALLBACK(+[](IBusBus*, gpointer p) {
        auto* self = static_cast<TextInput*>(p);
        if (self->armed_ && !self->context_.empty()) self->invalid_ = true;
        self->focused_ = nullptr;
        self->current_context_.clear();
        g_clear_object(&self->factory_);
    }), this);
    register_engine();
}

void TextInput::register_engine() {
    if (factory_) return;
    if (!ibus_bus_is_connected(bus_)) return;
    factory_ = ibus_factory_new(ibus_bus_get_connection(bus_));
    g_signal_connect(factory_, "create-engine", G_CALLBACK(create_engine), this);
    auto* component = ibus_component_new("org.freedesktop.IBus.Paraketto",
        "Paraketto dictation", "1.0", "MIT", "Paraketto contributors", "", "", "");
    g_object_ref_sink(component);
    auto* description = ibus_engine_desc_new(engine_name, "Paraketto", "Local speech input",
        "en", "MIT", "Paraketto contributors", "audio-input-microphone", "default");
    ibus_component_add_engine(component, description);
    if (!ibus_bus_register_component(bus_, component)) {
        g_warning("Could not register Paraketto with IBus; X11 paste remains available");
        g_clear_object(&factory_);
    }
    g_object_unref(component);
}

IBusEngine* TextInput::create_engine(IBusFactory*, const gchar* name, gpointer data) {
    g_debug("Paraketto IBus factory requested %s", name);
    if (g_strcmp0(name, engine_name)) return nullptr;
    auto* self = static_cast<TextInput*>(data);
    std::string path = "/org/freedesktop/IBus/Engine/Paraketto/" + std::to_string(++self->next_engine_id_);
    auto* engine = IBUS_ENGINE(g_object_new(dictation_engine_get_type(), "engine-name", name,
        "object-path", path.c_str(), "connection", ibus_bus_get_connection(self->bus_),
        "has-focus-id", TRUE, nullptr));
    g_object_set_data(G_OBJECT(engine), "owner", self);
    self->engines_.push_back(IBUS_ENGINE(g_object_ref(engine)));
    g_signal_connect(engine, "destroy", G_CALLBACK(+[](IBusEngine* engine, gpointer data) {
        auto* self = static_cast<TextInput*>(data);
        if (self->focused_ == engine) self->focus_out(engine);
        auto found = std::find(self->engines_.begin(), self->engines_.end(), engine);
        if (found != self->engines_.end()) {
            self->engines_.erase(found);
            g_object_set_data(G_OBJECT(engine), "owner", nullptr);
            g_object_unref(engine);
        }
    }), self);
    return engine;
}

TextInput::~TextInput() {
    end();
    lifetime_->owner = nullptr;
    for (auto* engine : engines_) {
        g_signal_handlers_disconnect_by_data(engine, this);
        g_object_set_data(G_OBJECT(engine), "owner", nullptr);
        g_object_unref(engine);
    }
    g_clear_object(&factory_);
    g_clear_object(&bus_);
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
    register_engine();
    insert_ = insert;
    started_ = g_get_monotonic_time();
    armed_ = true;
    invalid_ = false;
    context_.clear(); current_context_.clear();
    focused_ = nullptr;
    legacy_focus_ = false;
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
    if (!insert || !factory_ || !ibus_bus_is_connected(bus_)) return;
    if (auto* description = ibus_bus_get_global_engine(bus_)) {
        const char* previous = ibus_engine_desc_get_name(description);
        if (g_strcmp0(previous, engine_name)) previous_engine_ = previous;
        g_object_unref(description);
    } else return;
    if (previous_engine_.empty()) return;
    select_engine(engine_name, [this](bool ok) {
        if (!armed_) { end(); return; }
        if (ok && legacy_focus_ && context_.empty() && !previous_engine_.empty()) {
            // The daemon now has the FocusId property cached. Reactivate once
            // through public APIs so it supplies the real field ID/client type.
            // A second legacy response is treated as unsupported, never guessed.
            select_engine(previous_engine_, [this](bool restored) {
                if (!armed_ || !restored) { end(); return; }
                select_engine(engine_name, [this](bool) { if (!armed_) end(); });
            });
        }
    });
}

void TextInput::select_engine(const std::string& name, std::function<void(bool)> done) {
    struct Request { std::shared_ptr<Lifetime> lifetime; std::function<void(bool)> done; };
    ++activating_;
    ibus_bus_set_global_engine_async(bus_, name.c_str(), 2000, nullptr,
        [](GObject* object, GAsyncResult* result, gpointer data) {
            std::unique_ptr<Request> request(static_cast<Request*>(data));
            GError* error = nullptr;
            gboolean activated = ibus_bus_set_global_engine_async_finish(IBUS_BUS(object), result, &error);
            g_debug("Paraketto IBus activation finished: %d", activated);
            if (error) { g_debug("IBus activation: %s", error->message); g_error_free(error); }
            if (auto* self = request->lifetime->owner) {
                --self->activating_;
                request->done(activated);
                if (!self->armed_ && !self->activating_) self->end();
            }
        }, new Request{lifetime_, std::move(done)});
}

bool TextInput::settled() const {
    if (!insert_ || !factory_ || invalid_) return true;
    if (activating_) return false;
    // Global shortcut grabs can temporarily focus IBus's fake context. Wait
    // for the original field to regain focus before committing or pasting.
    if (!context_.empty()) return focused_ && current_context_ == context_ && revision_ == target_revision_;
    return g_get_monotonic_time() - started_ > 500000;
}

void TextInput::focus_in(IBusEngine* engine, const char* path, const char* client) {
    g_debug("Paraketto IBus focus in: %s (%s)", path, client);
    ++revision_;
    legacy_focus_ = false;
    focused_ = engine;
    current_context_ = path;
    if (!armed_ || !insert_ || invalid_ || !g_strcmp0(client, "fake")) return;
    poll();
    if (invalid_) return;
    if (!context_.empty() && context_ != path) { invalid_ = true; return; }
    context_ = path;
    target_revision_ = revision_;
    if (!target_) {
        const char* name = strchr(client, ':');
        if (name && name[1]) destination_ = application_name(name + 1);
    }
}

void TextInput::focus_legacy(IBusEngine* engine) {
    g_debug("Paraketto IBus legacy focus in");
    ++revision_;
    focused_ = engine;
    legacy_focus_ = true;
    current_context_ = "legacy-" + std::to_string(revision_);
}

void TextInput::focus_out(IBusEngine* engine) {
    g_debug("Paraketto IBus focus out");
    if (focused_ != engine) return;
    ++revision_;
    focused_ = nullptr;
    current_context_.clear();
    // Losing focus alone can be a shortcut grab. Keep the original context;
    // a different real context or application permanently invalidates it.
}

void TextInput::poll() {
    if (armed_ && insert_ && target_ && active_window() != target_) invalid_ = true;
}

void TextInput::end() {
    armed_ = false;
    if (!previous_engine_.empty() && bus_ && ibus_bus_is_connected(bus_)) {
        if (auto* description = ibus_bus_get_global_engine(bus_)) {
            bool ours = !g_strcmp0(ibus_engine_desc_get_name(description), engine_name);
            g_object_unref(description);
            if (ours) ibus_bus_set_global_engine_async(bus_, previous_engine_.c_str(), 2000, nullptr, nullptr, nullptr);
        }
        g_dbus_connection_flush_sync(ibus_bus_get_connection(bus_), nullptr, nullptr);
    }
    // Retain the last non-Paraketto engine across back-to-back activations:
    // restoring it is asynchronous and may still be in flight at the next begin.
    context_.clear();
    focused_ = nullptr;
}

std::string TextInput::deliver(const std::string& text) {
    poll();
    std::string result;
    if (!insert_) result = "Saved to history";
    else if (changed()) result = "Focus changed · copy to insert";
    else if (focused_ && !context_.empty() && context_ == current_context_ && revision_ == target_revision_) {
        guint purpose = 0, hints = 0;
        ibus_engine_get_content_type(focused_, &purpose, &hints);
        if (purpose == IBUS_INPUT_PURPOSE_PASSWORD || purpose == IBUS_INPUT_PURPOSE_PIN)
            result = "Saved to history · protected input";
        else {
            ibus_engine_commit_text(focused_, ibus_text_new_from_string(text.c_str()));
            g_dbus_connection_flush_sync(ibus_bus_get_connection(bus_), nullptr, nullptr);
            result = "Inserted into " + destination_;
        }
    } else if (!context_.empty()) result = "Input changed · copy to insert";
    else if (target_ && xdo_) {
        // Speech Note's established fallback: clipboard + an X11 paste shortcut.
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
    return key ? take_string(gtk_accelerator_get_label(key, mods)) : "Shortcut not configured";
}

} // namespace desktop
