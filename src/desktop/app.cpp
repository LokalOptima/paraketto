#include "audio.h"
#include "backend.h"
#include "common.h"
#include "history.h"
#include "input.h"
#include "settings.h"
#include "wav.h"
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include <glib-unix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <signal.h>

namespace desktop {
namespace {
const char* css = R"css(
.heading { font-weight: 600; }
.server-state { font-weight: 700; }
.state-sleeping { color: mix(#f5c211, @theme_fg_color, 0.25); }
.state-ready { color: mix(#26a269, @theme_fg_color, 0.25); }
.vram { font-family: monospace; }
.secondary { opacity: 0.75; }
.server-panel { background-color: mix(@theme_bg_color, @theme_fg_color, 0.06); }
.transcriptions, .transcriptions viewport, .history-list, .history-list row { background-color: @theme_base_color; }
.history-list row { padding: 0; }
.history-entry { padding: 12px 16px; }
.transcript-separator { min-height: 1px; background-color: alpha(@theme_fg_color, 0.22); margin: 0 16px; }
.stats { padding: 12px 16px; }
.transcript { font-size: 1.08em; }
.failure { color: @error_color; }
.overlay { background: rgba(23,25,31,0.97); border: 1px solid #454953; border-radius: 16px; }
.overlay label { color: #ececf2; }
.overlay label.failure { color: #ff7b83; }
.overlay button.abort { border-radius: 999px; padding: 0; min-width: 34px; min-height: 34px; }
)css";

GtkWidget* label(const std::string& text, const char* style = nullptr) {
    auto* widget = gtk_label_new(text.c_str());
    gtk_label_set_xalign(GTK_LABEL(widget), 0);
    if (style) gtk_style_context_add_class(gtk_widget_get_style_context(widget), style);
    return widget;
}
void set_label(GtkWidget* widget, const std::string& text) {
    if (text != gtk_label_get_text(GTK_LABEL(widget))) gtk_label_set_text(GTK_LABEL(widget), text.c_str());
}
void set_class(GtkStyleContext* style, const char* name, bool enabled) {
    if (enabled) gtk_style_context_add_class(style, name);
    else gtk_style_context_remove_class(style, name);
}
GtkWidget* icon_button(const char* icon, const char* tooltip) {
    auto* widget = gtk_button_new_from_icon_name(icon, GTK_ICON_SIZE_BUTTON);
    gtk_style_context_add_class(gtk_widget_get_style_context(widget), "flat");
    gtk_widget_set_tooltip_text(widget, tooltip);
    return widget;
}
void pack(GtkWidget* box, GtkWidget* child, bool expand = false) {
    gtk_box_pack_start(GTK_BOX(box), child, expand, expand, 0);
}
std::string duration(double seconds) {
    int s = std::max(0, static_cast<int>(seconds));
    return take_string(g_strdup_printf("%02d:%02d", s / 60, s % 60));
}
std::string executable_path() { return take_string(g_file_read_link("/proc/self/exe", nullptr)); }

void install_launcher() {
    std::string directory = std::string(g_get_user_data_dir()) + "/applications";
    if (g_mkdir_with_parents(directory.c_str(), 0755) != 0)
        throw std::runtime_error("Could not create applications directory");
    std::string quoted = "\"";
    for (char c : executable_path()) {
        if (c == '\\' || c == '"' || c == '`' || c == '$') quoted += '\\';
        quoted += c;
    }
    quoted += '"';
    write_file(directory + "/io.localoptima.Paraketto.desktop",
        "[Desktop Entry]\nType=Application\nName=Paraketto\nComment=Local speech dictation\nExec=" + quoted +
        "\nIcon=audio-input-microphone\nTerminal=false\nCategories=Utility;Audio;\nStartupNotify=true\n");
}

struct Options {
    std::string binary, weights, data, audio_source = "pulsesrc";
};

std::string option_string(GVariantDict* dict, const char* name) {
    GVariant* value = g_variant_dict_lookup_value(dict, name, nullptr);
    if (!value) return {};
    std::string result;
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_STRING)) result = g_variant_get_string(value, nullptr);
    else if (g_variant_is_of_type(value, G_VARIANT_TYPE_BYTESTRING)) result = g_variant_get_bytestring(value);
    g_variant_unref(value);
    return result;
}

Options options_from(GVariantDict* dict) {
    Options result;
    auto option = [&](const char* name) { return option_string(dict, name); };
    auto base = take_string(g_path_get_dirname(executable_path().c_str()));
    result.binary = option("server-binary");
    if (result.binary.empty()) result.binary = base + "/paraketto.cuda";
    if (!g_file_test(result.binary.c_str(), G_FILE_TEST_IS_EXECUTABLE) && option("server-binary").empty())
        result.binary = base + "/paraketto";
    result.weights = option("weights");
    if (result.weights.empty()) {
        const auto local = base + "/paraketto-ultra-fp16.bin";
        result.weights = g_file_test(local.c_str(), G_FILE_TEST_IS_REGULAR) ? local :
            std::string(g_get_user_cache_dir()) + "/paraketto/paraketto-ultra-fp16.bin";
    }
    result.data = option("data-dir");
    if (result.data.empty()) result.data = std::string(g_get_user_data_dir()) + "/paraketto";
    auto source = option("audio-source");
    if (!source.empty()) result.audio_source = source;
    result.binary = take_string(g_canonicalize_filename(result.binary.c_str(), nullptr));
    result.weights = take_string(g_canonicalize_filename(result.weights.c_str(), nullptr));
    result.data = take_string(g_canonicalize_filename(result.data.c_str(), nullptr));
    return result;
}
}

class App {
public:
    App(GtkApplication* application, Options options) : application_(application), options_(std::move(options)),
        history_(options_.data + "/history") {
        entries_ = history_.load();
        auto* provider = gtk_css_provider_new();
        gtk_css_provider_load_from_data(provider, css, -1, nullptr);
        gtk_style_context_add_provider_for_screen(gdk_screen_get_default(), GTK_STYLE_PROVIDER(provider),
            GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
        g_object_unref(provider);
        input_ = std::make_unique<TextInput>();
        shortcut_text_ = shortcut_label();
        total_vram_mb_ = query_gpu_total_mb();
        build_window();
        build_overlay();
        try { settings_.load(options_.data + "/preferences.ini"); }
        catch (const std::exception& error) { message("Could not read preferences: " + std::string(error.what())); }
        timer_ = g_timeout_add(40, [](gpointer p) -> gboolean { static_cast<App*>(p)->tick(); return G_SOURCE_CONTINUE; }, this);
        g_application_hold(G_APPLICATION(application_));
    }
    ~App() {
        g_source_remove(timer_);
        recorder_.stop();
        player_.stop();
        input_->end();
        backend_.reset();
        if (preferences_) gtk_widget_destroy(preferences_);
        gtk_widget_destroy(overlay_);
        gtk_widget_destroy(window_);
    }
    void show() { gtk_widget_show_all(window_); gtk_window_present(GTK_WINDOW(window_)); }
    void toggle(bool insert = true) {
        if (mode_ == Mode::Recording) { stop_recording(); return; }
        if (mode_ != Mode::Idle) return;
        ensure_server();
        player_.stop();
        playing_id_.clear();
        input_->begin(insert);
        std::string error;
        if (!recorder_.start(options_.audio_source, error)) {
            input_->end(); overlay_error(error); return;
        }
        mode_ = Mode::Recording;
        overlay_until_ = 0;
        bars_.fill(0);
        gtk_style_context_remove_class(gtk_widget_get_style_context(overlay_title_), "failure");
        set_label(overlay_title_, duration(0) + "  ·  " + input_->destination());
        gtk_widget_set_tooltip_text(overlay_cancel_, "Cancel recording");
        place_overlay();
        gtk_widget_show_all(overlay_);
        gtk_button_set_label(GTK_BUTTON(record_button_), "Stop recording");
        auto* style = gtk_widget_get_style_context(record_button_);
        gtk_style_context_remove_class(style, "suggested-action");
        gtk_style_context_add_class(style, "destructive-action");
    }
    void cancel() {
        if (mode_ == Mode::Recording) {
            recorder_.stop(); input_->end(); reset_recording_button();
            gtk_widget_hide(overlay_);
            mode_ = Mode::Idle;
        } else if (mode_ == Mode::Transcribing || mode_ == Mode::Delivering) {
            // Finish recognition into history, but cancel delivery to the target.
            cancelled_delivery_ = true;
            input_->end();
            gtk_widget_hide(overlay_);
        } else gtk_widget_hide(overlay_);
    }
    void transcribe_file(const std::string& path) {
        if (mode_ != Mode::Idle) { message("Finish the current recording first"); return; }
        try {
            auto bytes = read_file(path);
            auto wav = paraketto::read_wav_from_memory(bytes.data(), bytes.size());
            if (wav.samples.empty()) throw std::runtime_error("Choose a 16 kHz or 24 kHz mono WAV recording");
            std::vector<int16_t> samples;
            samples.reserve(wav.samples.size());
            for (float value : wav.samples) samples.push_back(static_cast<int16_t>(std::clamp(value, -1.0f, 1.0f) * 32767));
            ensure_server();
            input_->begin(false);
            submit(pcm_wav(samples));
        } catch (const std::exception& e) { message(e.what()); }
    }
    void configure_shortcut() {
        std::string error;
        if (install_shortcut(executable_path(), "<Control><Alt>space", error)) {
            shortcut_text_ = shortcut_label();
            set_label(shortcut_, shortcut_text_);
        } else message(error);
    }
private:
    enum class Mode { Idle, Recording, Transcribing, Delivering };
    GtkApplication* application_;
    Options options_;
    Settings settings_;
    History history_;
    std::vector<Entry> entries_;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<TextInput> input_;
    Recorder recorder_;
    Player player_;
    GtkWidget *window_, *overlay_, *history_box_, *status_, *vram_, *server_detail_, *record_button_;
    GtkWidget *stats_, *status_stack_, *loading_, *shortcut_, *overlay_title_, *overlay_cancel_, *wave_, *server_actions_;
    GtkWidget* preferences_ = nullptr;
    guint timer_ = 0;
    Mode mode_ = Mode::Idle;
    std::array<float, 48> bars_{};
    std::string active_id_, playing_id_;
    std::string shortcut_text_;
    std::optional<Result> pending_result_;
    std::string pending_wav_;
    gint64 delivery_time_ = 0, overlay_until_ = 0, idle_since_ = 0;
    double total_vram_mb_ = -1;
    bool cancelled_delivery_ = false;

    void message(const std::string& text) {
        if (text.empty()) return;
        auto* dialog = gtk_message_dialog_new(GTK_WINDOW(window_), GTK_DIALOG_DESTROY_WITH_PARENT,
            GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "%s", text.c_str());
        g_signal_connect(dialog, "response", G_CALLBACK(+[](GtkDialog* dialog, gint, gpointer) {
            gtk_widget_destroy(GTK_WIDGET(dialog));
        }), nullptr);
        gtk_widget_show(dialog);
    }
    void overlay_error(const std::string& text) {
        set_label(overlay_title_, text);
        gtk_style_context_add_class(gtk_widget_get_style_context(overlay_title_), "failure");
        gtk_widget_set_tooltip_text(overlay_cancel_, "Dismiss");
        bars_.fill(0);
        gtk_widget_queue_draw(wave_);
        overlay_until_ = 0;
        if (!gtk_widget_get_visible(overlay_)) { place_overlay(); gtk_widget_show_all(overlay_); }
    }
    ServerState server_state() {
        if (backend_) return backend_->state();
        ServerState state;
        state.starting = false;
        state.message = "Sleeping";
        state.server_vram_mb = 0;
        return state;
    }
    void ensure_server() {
        auto state = server_state();
        if (state.ready || state.starting) return;
        backend_.reset();
        backend_ = std::make_unique<Backend>(options_.binary, options_.weights, options_.data + "/server.log");
        idle_since_ = 0;
    }
    void server_action() {
        if (mode_ != Mode::Idle) return;
        auto state = server_state();
        if (state.ready || state.starting) { backend_.reset(); idle_since_ = 0; }
        else ensure_server();
    }
    void save_preferences() {
        idle_since_ = g_get_monotonic_time();
        try { settings_.save(options_.data + "/preferences.ini"); }
        catch (const std::exception& error) { message("Could not save preferences: " + std::string(error.what())); }
    }
    void open_server_log() {
        auto path = options_.data + "/server.log";
        if (!g_file_test(path.c_str(), G_FILE_TEST_IS_REGULAR)) { message("The server has not started yet."); return; }
        auto uri = take_string(g_filename_to_uri(path.c_str(), nullptr, nullptr));
        GError* error = nullptr;
        if (!gtk_show_uri_on_window(GTK_WINDOW(window_), uri.c_str(), GDK_CURRENT_TIME, &error)) {
            message(error->message); g_error_free(error);
        }
    }
    void add_action(const char* name, GCallback callback, const GVariantType* parameter = nullptr) {
        auto* action = g_simple_action_new(name, parameter);
        g_signal_connect(action, "activate", callback, this);
        g_action_map_add_action(G_ACTION_MAP(application_), G_ACTION(action));
        g_object_unref(action);
    }
    void build_window() {
        window_ = gtk_application_window_new(application_);
        gtk_window_set_title(GTK_WINDOW(window_), "Paraketto");
        gtk_window_set_default_size(GTK_WINDOW(window_), 680, 680);
        gtk_widget_set_size_request(window_, 480, 360);
        gtk_window_set_icon_name(GTK_WINDOW(window_), "audio-input-microphone");
        g_signal_connect(window_, "delete-event", G_CALLBACK(+[](GtkWidget* w, GdkEvent*, gpointer) -> gboolean {
            gtk_widget_hide(w); return TRUE;
        }), this);
        add_action("preferences", G_CALLBACK(+[](GSimpleAction*, GVariant*, gpointer p) { static_cast<App*>(p)->show_preferences(); }));
        add_action("import", G_CALLBACK(+[](GSimpleAction*, GVariant*, gpointer p) { static_cast<App*>(p)->choose_file(); }));
        add_action("server-log", G_CALLBACK(+[](GSimpleAction*, GVariant*, gpointer p) { static_cast<App*>(p)->open_server_log(); }));
        add_action("model", G_CALLBACK(+[](GSimpleAction*, GVariant*, gpointer p) { static_cast<App*>(p)->server_action(); }));
        add_action("quit", G_CALLBACK(+[](GSimpleAction*, GVariant*, gpointer p) {
            g_application_quit(G_APPLICATION(static_cast<App*>(p)->application_));
        }));
        add_action("delete-recording", G_CALLBACK(+[](GSimpleAction*, GVariant* value, gpointer p) {
            static_cast<App*>(p)->remove_entry(g_variant_get_string(value, nullptr));
        }), G_VARIANT_TYPE_STRING);
        auto* header = gtk_header_bar_new();
        gtk_header_bar_set_title(GTK_HEADER_BAR(header), "Paraketto");
        gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(header), TRUE);
        gtk_window_set_titlebar(GTK_WINDOW(window_), header);
        record_button_ = gtk_button_new_with_label("Record");
        gtk_style_context_add_class(gtk_widget_get_style_context(record_button_), "suggested-action");
        gtk_widget_set_tooltip_text(record_button_, "Record into history. Use the shortcut to dictate into another application.");
        g_signal_connect_swapped(record_button_, "clicked", G_CALLBACK(+[](App* a) { a->toggle(false); }), this);
        gtk_header_bar_pack_start(GTK_HEADER_BAR(header), record_button_);
        auto* menu_button = gtk_menu_button_new();
        gtk_button_set_image(GTK_BUTTON(menu_button), gtk_image_new_from_icon_name("open-menu-symbolic", GTK_ICON_SIZE_BUTTON));
        gtk_widget_set_tooltip_text(menu_button, "Main menu");
        GMenu* menu = g_menu_new();
        g_menu_append(menu, "Import Recording…", "app.import");
        g_menu_append(menu, "Preferences", "app.preferences");
        g_menu_append(menu, "View Server Log", "app.server-log");
        auto* quit_section = g_menu_new();
        g_menu_append(quit_section, "Quit Paraketto", "app.quit");
        g_menu_append_section(menu, nullptr, G_MENU_MODEL(quit_section));
        g_object_unref(quit_section);
        gtk_menu_button_set_use_popover(GTK_MENU_BUTTON(menu_button), TRUE);
        gtk_menu_button_set_menu_model(GTK_MENU_BUTTON(menu_button), G_MENU_MODEL(menu));
        g_object_unref(menu);
        gtk_header_bar_pack_end(GTK_HEADER_BAR(header), menu_button);

        auto* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_container_add(GTK_CONTAINER(window_), root);
        auto* server_panel = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_style_context_add_class(gtk_widget_get_style_context(server_panel), "server-panel");
        pack(root, server_panel);
        auto* server = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
        gtk_container_set_border_width(GTK_CONTAINER(server), 16);
        auto* info = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
        auto* first = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
        status_ = label("Sleeping", "server-state");
        gtk_style_context_add_class(gtk_widget_get_style_context(status_), "state-sleeping");
        gtk_label_set_ellipsize(GTK_LABEL(status_), PANGO_ELLIPSIZE_END);
        status_stack_ = gtk_stack_new();
        gtk_stack_add_named(GTK_STACK(status_stack_), status_, "state");
        loading_ = gtk_progress_bar_new();
        gtk_widget_set_valign(loading_, GTK_ALIGN_CENTER);
        gtk_widget_set_tooltip_text(loading_, "Loading model");
        atk_object_set_name(gtk_widget_get_accessible(loading_), "Loading model");
        gtk_stack_add_named(GTK_STACK(status_stack_), loading_, "loading");
        pack(first, status_stack_, true);
        vram_ = label("", "vram");
        gtk_label_set_width_chars(GTK_LABEL(vram_), 19);
        gtk_label_set_xalign(GTK_LABEL(vram_), 1);
        pack(first, vram_);
        pack(info, first);
        server_detail_ = label("Loads when you start recording", "secondary");
        gtk_label_set_ellipsize(GTK_LABEL(server_detail_), PANGO_ELLIPSIZE_END);
        pack(info, server_detail_);
        pack(server, info, true);
        // Homogeneous stacks reserve the same space for every state and action.
        server_actions_ = gtk_stack_new();
        gtk_widget_set_valign(server_actions_, GTK_ALIGN_CENTER);
        for (const char* text : {"Load now", "Cancel", "Unload now", "Retry"}) {
            auto* button = gtk_button_new_with_label(text);
            gtk_actionable_set_action_name(GTK_ACTIONABLE(button), "app.model");
            gtk_stack_add_named(GTK_STACK(server_actions_), button, text);
        }
        pack(server, server_actions_);
        pack(server_panel, server);
        stats_ = label("");
        gtk_style_context_add_class(gtk_widget_get_style_context(stats_), "stats");
        gtk_widget_set_tooltip_text(stats_, "Saved transcriptions. WPM = total words ÷ total recording time in minutes.");
        pack(server_panel, stats_);
        pack(root, gtk_separator_new(GTK_ORIENTATION_HORIZONTAL));
        auto* transcriptions = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_style_context_add_class(gtk_widget_get_style_context(transcriptions), "transcriptions");
        pack(root, transcriptions, true);
        auto* history_header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        gtk_container_set_border_width(GTK_CONTAINER(history_header), 16);
        pack(history_header, label("History", "heading"), true);
        shortcut_ = label(shortcut_text_, "secondary");
        gtk_widget_set_tooltip_text(shortcut_, "Start or stop dictation in the focused application");
        pack(history_header, shortcut_);
        pack(transcriptions, history_header);
        auto* scroll = gtk_scrolled_window_new(nullptr, nullptr);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        pack(transcriptions, scroll, true);
        history_box_ = gtk_list_box_new();
        gtk_list_box_set_selection_mode(GTK_LIST_BOX(history_box_), GTK_SELECTION_NONE);
        gtk_style_context_add_class(gtk_widget_get_style_context(history_box_), "history-list");
        gtk_list_box_set_header_func(GTK_LIST_BOX(history_box_), [](GtkListBoxRow* row, GtkListBoxRow* before, gpointer) {
            if (!before) { gtk_list_box_row_set_header(row, nullptr); return; }
            auto* separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
            gtk_style_context_add_class(gtk_widget_get_style_context(separator), "transcript-separator");
            gtk_list_box_row_set_header(row, separator);
            gtk_widget_show(separator);
        }, nullptr, nullptr);
        gtk_container_add(GTK_CONTAINER(scroll), history_box_);
        render_history();
    }

    void show_preferences() {
        if (preferences_) { gtk_window_present(GTK_WINDOW(preferences_)); return; }
        preferences_ = gtk_dialog_new_with_buttons("Preferences", GTK_WINDOW(window_),
            static_cast<GtkDialogFlags>(GTK_DIALOG_DESTROY_WITH_PARENT | GTK_DIALOG_USE_HEADER_BAR), nullptr, nullptr);
        gtk_window_set_default_size(GTK_WINDOW(preferences_), 500, -1);
        gtk_window_set_resizable(GTK_WINDOW(preferences_), FALSE);
        g_signal_connect(preferences_, "response", G_CALLBACK(+[](GtkDialog* dialog, gint, gpointer) {
            gtk_widget_destroy(GTK_WIDGET(dialog));
        }), nullptr);
        g_signal_connect(preferences_, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer p) {
            static_cast<App*>(p)->preferences_ = nullptr;
        }), this);
        auto* body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 16);
        gtk_container_set_border_width(GTK_CONTAINER(body), 24);
        gtk_container_add(GTK_CONTAINER(gtk_dialog_get_content_area(GTK_DIALOG(preferences_))), body);
        pack(body, label("GPU memory", "heading"));
        auto* keep_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
        auto* keep_label = label("Keep loaded between recordings");
        gtk_label_set_line_wrap(GTK_LABEL(keep_label), TRUE);
        pack(keep_row, keep_label, true);
        auto* keep = gtk_switch_new();
        gtk_widget_set_valign(keep, GTK_ALIGN_CENTER);
        gtk_switch_set_active(GTK_SWITCH(keep), settings_.keep_loaded);
        pack(keep_row, keep); pack(body, keep_row);
        auto* idle_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        pack(idle_row, label("Unload after inactivity"), true);
        auto* timeout = gtk_spin_button_new_with_range(1, 1440, 1);
        gtk_spin_button_set_numeric(GTK_SPIN_BUTTON(timeout), TRUE);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(timeout), settings_.idle_seconds / 60.0);
        gtk_widget_set_sensitive(idle_row, !settings_.keep_loaded);
        pack(idle_row, timeout); pack(idle_row, label("minutes")); pack(body, idle_row);
        g_object_set_data(G_OBJECT(keep), "idle-row", idle_row);
        g_signal_connect(keep, "notify::active", G_CALLBACK(+[](GObject* object, GParamSpec*, gpointer p) {
            auto* a = static_cast<App*>(p);
            a->settings_.keep_loaded = gtk_switch_get_active(GTK_SWITCH(object));
            gtk_widget_set_sensitive(GTK_WIDGET(g_object_get_data(object, "idle-row")), !a->settings_.keep_loaded);
            a->save_preferences();
        }), this);
        g_signal_connect(timeout, "value-changed", G_CALLBACK(+[](GtkSpinButton* spin, gpointer p) {
            auto* a = static_cast<App*>(p);
            a->settings_.idle_seconds = gtk_spin_button_get_value_as_int(spin) * 60;
            a->save_preferences();
        }), this);
        auto* explanation = label("The model loads when you start recording. History stays available while it sleeps.", "secondary");
        gtk_label_set_line_wrap(GTK_LABEL(explanation), TRUE);
        gtk_label_set_max_width_chars(GTK_LABEL(explanation), 48);
        pack(body, explanation);
        pack(body, gtk_separator_new(GTK_ORIENTATION_HORIZONTAL));
        pack(body, label("Dictation shortcut", "heading"));
        auto* shortcut_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        auto* binding = label(shortcut_label());
        pack(shortcut_row, binding, true);
        auto* setup = gtk_button_new_with_label("Use default");
        gtk_widget_set_tooltip_text(setup, "Enable Ctrl+Alt+Space for dictation");
        g_object_set_data(G_OBJECT(setup), "shortcut-label", binding);
        g_signal_connect(setup, "clicked", G_CALLBACK(+[](GtkButton* button, gpointer p) {
            auto* a = static_cast<App*>(p); a->configure_shortcut();
            set_label(GTK_WIDGET(g_object_get_data(G_OBJECT(button), "shortcut-label")), a->shortcut_text_);
        }), this);
        pack(shortcut_row, setup); pack(body, shortcut_row);
        auto* hint = label("You can change the binding in your system Keyboard settings.", "secondary");
        gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
        gtk_label_set_max_width_chars(GTK_LABEL(hint), 48);
        pack(body, hint);
        pack(body, gtk_separator_new(GTK_ORIENTATION_HORIZONTAL));
        pack(body, label("Model", "heading"));
        auto* weights = label(take_string(g_path_get_basename(options_.weights.c_str())), "secondary");
        gtk_label_set_ellipsize(GTK_LABEL(weights), PANGO_ELLIPSIZE_MIDDLE);
        gtk_widget_set_tooltip_text(weights, options_.weights.c_str());
        pack(body, weights);
        gtk_widget_show_all(preferences_);
    }
    void build_overlay() {
        overlay_ = gtk_window_new(GTK_WINDOW_POPUP);
        gtk_widget_set_name(overlay_, "paraketto-recording");
        gtk_window_set_title(GTK_WINDOW(overlay_), "Paraketto recording");
        gtk_window_set_accept_focus(GTK_WINDOW(overlay_), FALSE);
        gtk_window_set_focus_on_map(GTK_WINDOW(overlay_), FALSE);
        gtk_window_set_keep_above(GTK_WINDOW(overlay_), TRUE);
        gtk_window_set_skip_taskbar_hint(GTK_WINDOW(overlay_), TRUE);
        gtk_window_set_skip_pager_hint(GTK_WINDOW(overlay_), TRUE);
        if (auto* visual = gdk_screen_get_rgba_visual(gdk_screen_get_default())) gtk_widget_set_visual(overlay_, visual);
        gtk_style_context_add_class(gtk_widget_get_style_context(overlay_), "overlay");
        auto* body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 7);
        gtk_container_set_border_width(GTK_CONTAINER(body), 16);
        gtk_container_add(GTK_CONTAINER(overlay_), body);
        overlay_title_ = label("");
        gtk_label_set_max_width_chars(GTK_LABEL(overlay_title_), 32);
        gtk_label_set_ellipsize(GTK_LABEL(overlay_title_), PANGO_ELLIPSIZE_END);
        pack(body, overlay_title_);
        auto* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        wave_ = gtk_drawing_area_new();
        gtk_widget_set_size_request(wave_, 252, 34);
        pack(row, wave_, true);
        overlay_cancel_ = icon_button("window-close-symbolic", "Cancel recording");
        auto* cancel_style = gtk_widget_get_style_context(overlay_cancel_);
        gtk_style_context_remove_class(cancel_style, "flat");
        gtk_style_context_add_class(cancel_style, "abort");
        gtk_widget_set_can_focus(overlay_cancel_, FALSE);
        gtk_widget_set_valign(overlay_cancel_, GTK_ALIGN_CENTER);
        g_signal_connect_swapped(overlay_cancel_, "clicked", G_CALLBACK(+[](App* a) { a->cancel(); }), this);
        pack(row, overlay_cancel_);
        pack(body, row);
        g_signal_connect(wave_, "draw", G_CALLBACK(+[](GtkWidget* w, cairo_t* cr, gpointer p) -> gboolean {
            auto* a = static_cast<App*>(p);
            const double width = gtk_widget_get_allocated_width(w);
            const double height = gtk_widget_get_allocated_height(w);
            const double step = width / a->bars_.size();
            cairo_set_source_rgb(cr, 0.65, 0.89, 0.80);
            cairo_set_line_width(cr, 3);
            cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
            for (size_t i = 0; i < a->bars_.size(); ++i) {
                double size = 2 + a->bars_[i] * (height - 6);
                double x = step * (i + 0.5);
                cairo_move_to(cr, x, (height - size) / 2);
                cairo_line_to(cr, x, (height + size) / 2);
                cairo_stroke(cr);
            }
            return FALSE;
        }), this);
    }
    void place_overlay() {
        GdkDisplay* display = gdk_display_get_default();
        GdkMonitor* monitor = nullptr;
        if (GDK_IS_X11_DISPLAY(display) && input_->target_window()) {
            auto* target = gdk_x11_window_foreign_new_for_display(display, input_->target_window());
            if (target) { monitor = gdk_display_get_monitor_at_window(display, target); g_object_unref(target); }
        }
        if (!monitor) monitor = gdk_display_get_primary_monitor(display);
        if (!monitor) monitor = gdk_display_get_monitor(display, 0);
        GdkRectangle area;
        gdk_monitor_get_workarea(monitor, &area);
        GtkRequisition natural;
        gtk_widget_show_all(gtk_bin_get_child(GTK_BIN(overlay_)));
        gtk_widget_get_preferred_size(overlay_, nullptr, &natural);
        gtk_window_move(GTK_WINDOW(overlay_), area.x + (area.width - natural.width) / 2,
            area.y + area.height - natural.height - 32);
    }
    Entry* entry(const std::string& id) {
        auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; });
        return it == entries_.end() ? nullptr : &*it;
    }
    void save(Entry& e) {
        try { history_.save(e); }
        catch (const std::exception& error) { message("Could not save history: " + std::string(error.what())); }
    }
    void reset_recording_button() {
        gtk_button_set_label(GTK_BUTTON(record_button_), "Record");
        auto* style = gtk_widget_get_style_context(record_button_);
        gtk_style_context_remove_class(style, "destructive-action");
        gtk_style_context_add_class(style, "suggested-action");
    }
    void stop_recording() {
        auto wav = recorder_.stop();
        reset_recording_button();
        if (wav.size() < 44 + 3200) {
            mode_ = Mode::Idle; input_->end(); overlay_error("Recording was too short"); return;
        }
        submit(std::move(wav));
    }
    void submit(std::string wav) {
        try {
            if (!backend_) ensure_server();
            Entry e = history_.create(wav, input_->destination());
            active_id_ = e.id;
            entries_.insert(entries_.begin(), std::move(e));
            cancelled_delivery_ = false;
            mode_ = Mode::Transcribing;
            pending_wav_ = std::move(wav);
            gtk_style_context_remove_class(gtk_widget_get_style_context(overlay_title_), "failure");
            set_label(overlay_title_, (server_state().ready ? "Transcribing… · " : "Loading model… · ") + input_->destination());
            gtk_widget_set_tooltip_text(overlay_cancel_, "Cancel insertion");
            if (!gtk_widget_get_visible(overlay_)) {
                place_overlay(); gtk_widget_show_all(overlay_);
            }
            render_history();
        } catch (const std::exception& error) {
            if (auto* e = entry(active_id_)) { e->error = error.what(); e->delivery = "Not transcribed"; save(*e); }
            gtk_widget_hide(overlay_);
            mode_ = Mode::Idle; input_->end(); message(error.what()); render_history();
        }
    }
    void finish_result(Result result, bool allow_delivery) {
        const bool failed = !result.error.empty();
        if (auto* e = entry(result.id)) {
            e->text = std::move(result.text); e->error = std::move(result.error);
            if (result.duration > 0) e->duration = result.duration;
            e->inference_ms = result.inference_ms;
            if (!e->error.empty()) { e->delivery = "Not transcribed"; input_->end(); }
            else if (e->text.empty()) { e->delivery = "No speech detected"; input_->end(); }
            else if (cancelled_delivery_ || !allow_delivery) { e->delivery = "Saved to history"; input_->end(); }
            else e->delivery = input_->deliver(e->text);
            save(*e);
            set_label(overlay_title_, e->error.empty() ? e->delivery : e->error == "Couldn’t load model" ? e->error : "Transcription failed");
            auto* style = gtk_widget_get_style_context(overlay_title_);
            if (e->error.empty()) gtk_style_context_remove_class(style, "failure");
            else gtk_style_context_add_class(style, "failure");
            gtk_widget_set_tooltip_text(overlay_cancel_, "Dismiss");
        }
        mode_ = Mode::Idle;
        active_id_.clear();
        overlay_until_ = failed ? 0 : g_get_monotonic_time() + 1500000;
        if (failed) { bars_.fill(0); gtk_widget_queue_draw(wave_); }
        render_history();
    }
    void tick() {
        auto state = server_state();
        auto now = g_get_monotonic_time();
        if (mode_ != Mode::Idle || state.starting) idle_since_ = 0;
        else if (backend_) {
            if (!idle_since_) idle_since_ = now;
            if (!settings_.keep_loaded && now - idle_since_ >= static_cast<gint64>(settings_.idle_seconds) * G_USEC_PER_SEC) {
                backend_.reset(); idle_since_ = 0; state = server_state();
            }
        }
        bool transcribing = mode_ == Mode::Transcribing || mode_ == Mode::Delivering;
        bool failed = backend_ && !state.ready && !state.starting;
        set_label(status_, !backend_ ? "Sleeping" : state.starting ? "Loading model…" :
            failed ? (state.load_failed ? "Couldn’t load model" : "Server unavailable") : transcribing ? "Transcribing…" : "Ready");
        auto* status_style = gtk_widget_get_style_context(status_);
        set_class(status_style, "state-sleeping", !backend_);
        set_class(status_style, "state-ready", state.ready);
        if (failed) gtk_style_context_add_class(status_style, "failure");
        else gtk_style_context_remove_class(status_style, "failure");
        gtk_stack_set_visible_child_name(GTK_STACK(status_stack_), state.starting ? "loading" : "state");
        if (state.starting) gtk_progress_bar_pulse(GTK_PROGRESS_BAR(loading_));
        std::string detail;
        if (!backend_) detail = "Loads when you start recording";
        else if (state.starting) detail = "Recording can start immediately";
        else if (!state.ready) detail = "Retry, or view the server log for details";
        else if (mode_ != Mode::Idle) detail = "Model in use";
        else if (settings_.keep_loaded) detail = "Automatic unloading disabled";
        else detail = "Unloads in " + duration(std::ceil(settings_.idle_seconds - (now - idle_since_) / 1000000.0));
        set_label(server_detail_, detail);
        gtk_widget_set_tooltip_text(server_detail_, detail.c_str());
        std::string model = state.model.empty() ? take_string(g_path_get_basename(options_.weights.c_str())) : state.model;
        if (!state.device.empty()) model += "\n" + state.device;
        gtk_widget_set_tooltip_text(status_, model.c_str());
        if (state.total_vram_mb > 0) total_vram_mb_ = state.total_vram_mb;
        auto memory_field = [](double mb) {
            return mb >= 0 ? take_string(g_strdup_printf("%6.2f", mb / 1024)) : std::string("    --");
        };
        set_label(vram_, memory_field(state.server_vram_mb) + " / " + memory_field(total_vram_mb_) + " GiB");
        std::string memory = !backend_ ? "The model is unloaded" : state.used_vram_mb >= 0 ? take_string(g_strdup_printf(
            "Model process memory. Entire GPU: %.2f / %.2f GiB used.", state.used_vram_mb / 1024, state.total_vram_mb / 1024)) : "Model VRAM usage is unavailable";
        gtk_widget_set_tooltip_text(vram_, memory.c_str());
        gtk_stack_set_visible_child_name(GTK_STACK(server_actions_),
            !backend_ ? "Load now" : state.starting ? "Cancel" : state.ready ? "Unload now" : "Retry");
        auto* model_action = g_action_map_lookup_action(G_ACTION_MAP(application_), "model");
        g_simple_action_set_enabled(G_SIMPLE_ACTION(model_action), mode_ == Mode::Idle);
        gtk_widget_set_sensitive(record_button_, mode_ == Mode::Recording || mode_ == Mode::Idle);
        if (mode_ == Mode::Recording) {
            input_->poll();
            std::move(bars_.begin() + 1, bars_.end(), bars_.begin());
            bars_.back() = recorder_.level();
            gtk_widget_queue_draw(wave_);
            set_label(overlay_title_, duration(recorder_.seconds()) + "  ·  " + input_->destination());
            auto error = recorder_.error();
            if (failed) {
                stop_recording();
                if (mode_ == Mode::Idle) {
                    overlay_error(state.load_failed ? "Couldn’t load model" : "Server unavailable");
                }
            } else if (!error.empty()) { cancel(); overlay_error(error); }
            else if (recorder_.full()) stop_recording();
        }
        if (mode_ == Mode::Transcribing || mode_ == Mode::Delivering) input_->poll();
        if (!pending_wav_.empty() && state.ready) {
            if (backend_->transcribe(active_id_, std::move(pending_wav_))) {
                pending_wav_.clear();
                set_label(overlay_title_, "Transcribing… · " + input_->destination());
            }
        } else if (!pending_wav_.empty() && !state.starting) {
            pending_wav_.clear();
            Result result; result.id = active_id_;
            result.error = state.load_failed ? "Couldn’t load model" : "Server unavailable";
            finish_result(std::move(result), false);
        }
        if (backend_) {
            if (auto result = backend_->result()) {
                pending_result_ = std::move(result);
                mode_ = Mode::Delivering;
                delivery_time_ = g_get_monotonic_time();
            }
        }
        if (pending_result_) {
            guint modifiers = gdk_keymap_get_modifier_state(gdk_keymap_get_for_display(gdk_display_get_default()));
            bool released = !(modifiers & (GDK_CONTROL_MASK | GDK_SHIFT_MASK | GDK_MOD1_MASK | GDK_SUPER_MASK));
            auto elapsed = g_get_monotonic_time() - delivery_time_;
            if ((released && input_->settled() && elapsed > 120000) || elapsed > 5000000) {
                finish_result(std::move(*pending_result_), released);
                pending_result_.reset();
            }
        }
        if (overlay_until_ && mode_ == Mode::Idle && g_get_monotonic_time() > overlay_until_) {
            gtk_widget_hide(overlay_); overlay_until_ = 0;
        }
        if (!playing_id_.empty() && !player_.playing()) { playing_id_.clear(); render_history(); }
    }
    struct Row { App* app; std::string id; };
    void remove_entry(const std::string& id) {
        if (id == active_id_) return;
        if (auto* e = entry(id)) {
            try { history_.remove(*e); }
            catch (const std::exception& error) { message(error.what()); return; }
            if (playing_id_ == id) { player_.stop(); playing_id_.clear(); }
            entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; }), entries_.end());
            render_history();
        }
    }
    void add_history_row(GtkWidget* content) {
        gtk_container_add(GTK_CONTAINER(history_box_), content);
        auto* row = GTK_LIST_BOX_ROW(gtk_widget_get_parent(content));
        gtk_list_box_row_set_activatable(row, FALSE);
        gtk_list_box_row_set_selectable(row, FALSE);
    }
    void render_history() {
        auto stats = history_stats(entries_);
        auto markup = take_string(g_strdup_printf("<b>%zu</b> words transcribed   ·   <b>%lld</b> WPM",
            stats.words, stats.words_per_minute));
        gtk_label_set_markup(GTK_LABEL(stats_), markup.c_str());
        auto* children = gtk_container_get_children(GTK_CONTAINER(history_box_));
        for (auto* it = children; it; it = it->next) gtk_widget_destroy(GTK_WIDGET(it->data));
        g_list_free(children);
        if (entries_.empty()) {
            auto* empty = gtk_box_new(GTK_ORIENTATION_VERTICAL, 16);
            gtk_container_set_border_width(GTK_CONTAINER(empty), 48);
            auto* icon = gtk_image_new_from_icon_name("audio-input-microphone-symbolic", GTK_ICON_SIZE_DIALOG);
            pack(empty, icon);
            auto* text = label("Your transcriptions will appear here.", "secondary");
            gtk_label_set_xalign(GTK_LABEL(text), 0.5);
            pack(empty, text);
            add_history_row(empty);
        }
        for (const auto& e : entries_) {
            auto* card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
            gtk_style_context_add_class(gtk_widget_get_style_context(card), "history-entry");
            auto* row = new Row{this, e.id};
            g_object_set_data_full(G_OBJECT(card), "row", row, [](gpointer p) { delete static_cast<Row*>(p); });
            const std::string text = !e.text.empty() ? e.text : !e.error.empty() ?
                (e.error == "Couldn’t load model" ? e.error : "Couldn’t transcribe recording") : e.delivery;
            auto* transcript = label(text, e.error.empty() ? "transcript" : "failure");
            if (!e.error.empty()) gtk_widget_set_tooltip_text(transcript, e.error.c_str());
            gtk_label_set_line_wrap(GTK_LABEL(transcript), TRUE);
            gtk_label_set_line_wrap_mode(GTK_LABEL(transcript), PANGO_WRAP_WORD_CHAR);
            gtk_label_set_max_width_chars(GTK_LABEL(transcript), 62);
            gtk_label_set_selectable(GTK_LABEL(transcript), TRUE);
            gtk_widget_set_hexpand(transcript, TRUE);
            pack(card, transcript);
            if (!e.text.empty() && (e.delivery.find("copy to insert") != std::string::npos ||
                    e.delivery.find("protected input") != std::string::npos)) {
                auto* warning = label(e.delivery, "secondary");
                gtk_label_set_line_wrap(GTK_LABEL(warning), TRUE);
                pack(card, warning);
            }
            auto* bottom = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
            auto* metadata = label(e.time + "  ·  " + duration(e.duration), "secondary");
            gtk_label_set_ellipsize(GTK_LABEL(metadata), PANGO_ELLIPSIZE_END);
            std::string details = e.destination + "\n" + e.delivery;
            if (e.inference_ms > 0) details += take_string(g_strdup_printf("\nTranscribed in %.0f ms", e.inference_ms));
            gtk_widget_set_tooltip_text(metadata, details.c_str());
            pack(bottom, metadata, true);
            auto* remove = icon_button("user-trash-symbolic", "Delete recording and transcription");
            gtk_actionable_set_action_target_value(GTK_ACTIONABLE(remove), g_variant_new_string(e.id.c_str()));
            gtk_actionable_set_action_name(GTK_ACTIONABLE(remove), "app.delete-recording");
            gtk_widget_set_sensitive(remove, e.id != active_id_);
            pack(bottom, remove);
            auto* play = icon_button(e.id == playing_id_ ? "media-playback-stop-symbolic" : "media-playback-start-symbolic", "Play recording");
            g_signal_connect(play, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) {
                auto* row = static_cast<Row*>(p); auto* a = row->app;
                if (a->mode_ == Mode::Recording) return;
                if (a->playing_id_ == row->id) { a->player_.stop(); a->playing_id_.clear(); }
                else if (auto* e = a->entry(row->id)) {
                    std::string error;
                    if (a->player_.play(a->history_.audio_path(*e), error)) a->playing_id_ = e->id;
                    else a->message(error);
                }
                a->render_history();
            }), row);
            pack(bottom, play);
            auto* copy = icon_button("edit-copy-symbolic", "Copy transcription");
            gtk_widget_set_sensitive(copy, !e.text.empty());
            g_signal_connect(copy, "clicked", G_CALLBACK(+[](GtkButton* button, gpointer p) {
                auto* row = static_cast<Row*>(p);
                if (auto* e = row->app->entry(row->id)) {
                    gtk_clipboard_set_text(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), e->text.c_str(), e->text.size());
                    gtk_button_set_image(button, gtk_image_new_from_icon_name("emblem-ok-symbolic", GTK_ICON_SIZE_BUTTON));
                    gtk_widget_set_tooltip_text(GTK_WIDGET(button), "Copied");
                }
            }), row);
            pack(bottom, copy); pack(card, bottom); add_history_row(card);
        }
        gtk_widget_show_all(history_box_);
    }
    void choose_file() {
        auto* chooser = gtk_file_chooser_dialog_new("Transcribe a recording", GTK_WINDOW(window_), GTK_FILE_CHOOSER_ACTION_OPEN,
            "Cancel", GTK_RESPONSE_CANCEL, "Transcribe", GTK_RESPONSE_ACCEPT, nullptr);
        auto* filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, "WAV recordings"); gtk_file_filter_add_pattern(filter, "*.wav");
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(chooser), filter);
        if (gtk_dialog_run(GTK_DIALOG(chooser)) == GTK_RESPONSE_ACCEPT) {
            auto path = take_string(gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(chooser)));
            gtk_widget_destroy(chooser); transcribe_file(path);
        } else gtk_widget_destroy(chooser);
    }
};

} // namespace desktop

int main(int argc, char** argv) {
    gst_init(nullptr, nullptr);
    g_set_application_name("Paraketto");
    auto* application = gtk_application_new("io.localoptima.Paraketto", G_APPLICATION_HANDLES_COMMAND_LINE);
    const GOptionEntry options[] = {
        {"toggle", 0, 0, G_OPTION_ARG_NONE, nullptr, "Start or stop dictation in the focused application", nullptr},
        {"cancel", 0, 0, G_OPTION_ARG_NONE, nullptr, "Cancel recording or pending insertion", nullptr},
        {"quit", 0, 0, G_OPTION_ARG_NONE, nullptr, "Quit the app and its server", nullptr},
        {"install-shortcut", 0, 0, G_OPTION_ARG_NONE, nullptr, "Enable the GNOME Ctrl+Alt+Space shortcut", nullptr},
        {"install-desktop", 0, 0, G_OPTION_ARG_NONE, nullptr, "Add a user application launcher and the dictation shortcut", nullptr},
        {"weights", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "FP16 weights (default: Ultra)", "FILE"},
        {"server-binary", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "Paraketto server executable", "FILE"},
        {"data-dir", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "History and log directory", "DIRECTORY"},
        {"audio-source", 0, 0, G_OPTION_ARG_STRING, nullptr, "GStreamer microphone source (default: pulsesrc)", "SOURCE"},
        {"transcribe", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "Transcribe a WAV into history", "FILE"},
        {nullptr, 0, 0, G_OPTION_ARG_NONE, nullptr, nullptr, nullptr}
    };
    g_application_add_main_option_entries(G_APPLICATION(application), options);
    std::unique_ptr<desktop::App> app;
    g_signal_connect(application, "command-line", G_CALLBACK(+[](GApplication* application, GApplicationCommandLine* command, gpointer data) -> gint {
        auto& app = *static_cast<std::unique_ptr<desktop::App>*>(data);
        auto* dict = g_application_command_line_get_options_dict(command);
        if (g_variant_dict_contains(dict, "quit")) { g_application_quit(application); return 0; }
        if (g_variant_dict_contains(dict, "install-shortcut") || g_variant_dict_contains(dict, "install-desktop")) {
            std::string error;
            if (!desktop::install_shortcut(desktop::executable_path(), "<Control><Alt>space", error)) {
                g_application_command_line_printerr(command, "%s\n", error.c_str()); return 1;
            }
            if (g_variant_dict_contains(dict, "install-desktop")) {
                try { desktop::install_launcher(); }
                catch (const std::exception& e) { g_application_command_line_printerr(command, "%s\n", e.what()); return 1; }
            }
            g_application_command_line_print(command, "Ctrl+Alt+Space toggles Paraketto dictation.\n");
            return 0;
        }
        if (g_variant_dict_contains(dict, "cancel")) { if (app) app->cancel(); return 0; }
        try {
            if (!app) app = std::make_unique<desktop::App>(GTK_APPLICATION(application), desktop::options_from(dict));
            if (g_variant_dict_contains(dict, "toggle")) app->toggle();
            else {
                app->show();
                std::string filename = desktop::option_string(dict, "transcribe");
                if (!filename.empty()) app->transcribe_file(filename);
            }
        } catch (const std::exception& error) {
            g_application_command_line_printerr(command, "%s\n", error.what()); return 1;
        }
        return 0;
    }), &app);
    guint sigint = g_unix_signal_add(SIGINT, [](gpointer p) -> gboolean { g_application_quit(G_APPLICATION(p)); return G_SOURCE_CONTINUE; }, application);
    guint sigterm = g_unix_signal_add(SIGTERM, [](gpointer p) -> gboolean { g_application_quit(G_APPLICATION(p)); return G_SOURCE_CONTINUE; }, application);
    int status = g_application_run(G_APPLICATION(application), argc, argv);
    g_source_remove(sigint); g_source_remove(sigterm);
    app.reset();
    g_object_unref(application);
    return status;
}
