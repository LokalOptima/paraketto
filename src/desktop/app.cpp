#include "audio.h"
#include "backend.h"
#include "dictionary_view.h"
#include "app_common.h"
#include "history.h"
#include "input.h"
#include "model_settings.h"
#include "transcript.h"
#include "session_view.h"
#include "replacement_highlights.h"
#include <map>
#include "wav.h"
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include <glib-unix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <set>
#include <signal.h>

namespace desktop {
using namespace paraketto::app;
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
.history-entry { padding: 16px 24px; }
.history-day { padding: 20px 24px 0; }
.transcript-separator { min-height: 1px; background-color: alpha(@theme_fg_color, 0.12); margin: 0 24px; }
.history-actions button { min-width: 24px; min-height: 24px; padding: 4px; }
.history-metadata { font-size: 0.92em; }
.server-row { padding: 12px 24px; }
.stats-row { padding: 10px 24px; }
.transcript { font-size: 1.08em; }
.reading-pane { background-color: @theme_base_color; }
.reading-toolbar { padding: 18px 24px 12px; }
.reading-content { padding: 8px 24px 16px; }
.reading-footer { padding: 8px 24px 16px; }
.transport button { min-width: 24px; min-height: 24px; padding: 8px; }
.replacement-toggle { min-width: 0; min-height: 0; border-radius: 999px; padding: 1px; background-image: none; background-color: alpha(@theme_fg_color, 0.18); box-shadow: none; transition: none; }
.replacement-toggle:checked { background-color: @theme_selected_bg_color; border-color: @theme_selected_bg_color; }
.recording-clock { font-family: monospace; }
.section-status { font-size: 0.85em; opacity: 0.6; }
.empty-transcript { font-size: 1.15em; opacity: 0.65; }
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
void set_notice(GtkWidget* widget, const std::string& text) {
    set_label(widget, text); gtk_widget_set_visible(widget, !text.empty());
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
void move_widget(GtkWidget* widget, GtkWidget* container) {
    g_object_ref(widget);
    if (auto* parent = gtk_widget_get_parent(widget)) gtk_container_remove(GTK_CONTAINER(parent), widget);
    gtk_container_add(GTK_CONTAINER(container), widget);
    g_object_unref(widget);
}
std::string duration(double seconds) {
    int s = std::max(0, static_cast<int>(seconds));
    return take_string(g_strdup_printf("%02d:%02d", s / 60, s % 60));
}
struct HistoryStamp { std::string day, heading, time; };
HistoryStamp session_stamp(const std::string& timestamp) {
    auto* utc = g_date_time_new_from_iso8601(timestamp.c_str(), nullptr);
    if (!utc) return {"unknown", "History", timestamp};
    auto* date = g_date_time_to_local(utc); g_date_time_unref(utc);
    auto* now = g_date_time_new_now_local(); auto* yesterday = g_date_time_add_days(now, -1);
    auto key = take_string(g_date_time_format(date, "%Y%m%d"));
    std::string title = key == take_string(g_date_time_format(now, "%Y%m%d")) ? "Today" :
        key == take_string(g_date_time_format(yesterday, "%Y%m%d")) ? "Yesterday" :
        take_string(g_date_time_format(date, "%e %B %Y"));
    auto time = take_string(g_date_time_format(date, "%H:%M"));
    g_date_time_unref(date); g_date_time_unref(now); g_date_time_unref(yesterday);
    return {key, title, time};
}
float waveform_level(float rms, float previous, double elapsed) {
    // Display-only scale: quiet below -50 dBFS, full height at -10 dBFS.
    // Keep headroom for loud speech without hiding quieter syllables.
    double db = 20.0 * std::log10(std::max(rms, 1e-6f));
    double target = std::clamp((db + 50.0) / 40.0, 0.0, 1.0);
    // Respond quickly to speech and let each bar fall gently between syllables.
    double time_constant = target > previous ? 0.030 : 0.160;
    return previous + (target - previous) * (1.0 - std::exp(-elapsed / time_constant));
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
    std::string service_url, data, audio_source = "pulsesrc";
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
    result.service_url = option("service-url");
    result.data = option("data-dir");
    if (result.data.empty()) result.data = std::string(g_get_user_data_dir()) + "/paraketto";
    auto source = option("audio-source");
    if (!source.empty()) result.audio_source = source;
    result.data = take_string(g_canonicalize_filename(result.data.c_str(), nullptr));
    return result;
}
}

class App {
    friend struct PaneTest;
public:
    App(GtkApplication* application, Options options) : application_(application), options_(std::move(options)) {
        auto* provider = gtk_css_provider_new();
        gtk_css_provider_load_from_data(provider, css, -1, nullptr);
        gtk_style_context_add_provider_for_screen(gdk_screen_get_default(), GTK_STYLE_PROVIDER(provider),
            GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
        g_object_unref(provider);
        shortcut_text_ = shortcut_label();
        backend_ = std::make_unique<Backend>(options_.service_url, options_.data);
        build_window();
        build_overlay();
        timer_ = g_timeout_add(40, [](gpointer p) -> gboolean { static_cast<App*>(p)->tick(); return G_SOURCE_CONTINUE; }, this);
        g_application_hold(G_APPLICATION(application_));
    }
    ~App() {
        g_source_remove(timer_);
        if(spelling_idle_)g_source_remove(spelling_idle_);
        close_spelling();
        backend_.reset();
        if (preferences_) gtk_widget_destroy(preferences_);
        dictionary_.reset();
        gtk_widget_destroy(overlay_);
        gtk_widget_destroy(window_);
    }
    void show() { gtk_widget_show_all(window_); update_pane(); gtk_window_present(GTK_WINDOW(window_)); }
    void toggle(bool = true) {
        if (mode_ == Mode::Recording || mode_ == Mode::Paused) { stop_recording(); return; }
        if (mode_ != Mode::Idle || starting_session_) return;
        starting_session_ = true;
        auto request = take_string(g_uuid_string_random());
        backend_->session_request("start", "POST", "/sessions", json_body({{"request_id", request},
            {"source", "microphone"}, {"audio_source", options_.audio_source}}));
        set_label(overlay_title_, "Starting recording…"); place_overlay(); gtk_widget_show_all(overlay_);
    }
    void cancel() {
        auto_copy_id_.clear();
        if (starting_session_) {
            cancel_after_start_ = true;
            gtk_widget_hide(overlay_);
            return;
        }
        if (!active_id_.empty() && mode_ != Mode::Idle) {
            backend_->session_request("cancel", "POST", "/sessions/" + active_id_ + "/cancel", "{}", active_id_);
        }
        gtk_widget_hide(overlay_);
    }
    void quit() {
        auto_copy_id_.clear();
        quitting_ = true;
        if (starting_session_) return; // Complete the start handshake, then save it.
        auto active = rows_.find(active_id_);
        if (!active_id_.empty() && (mode_ == Mode::Recording || mode_ == Mode::Paused || (active != rows_.end() && (active->second->data.capture == "recording" || active->second->data.capture == "paused")))) {
            backend_->session_request("quit-stop", "POST", "/sessions/" + active_id_ + "/stop", "{}", active_id_);
            set_label(overlay_title_, "Saving recording…");
        } else g_application_quit(G_APPLICATION(application_));
    }
    void transcribe_file(const std::string& path) {
        if (starting_session_) return;
        auto absolute = take_string(g_canonicalize_filename(path.c_str(), nullptr));
        backend_->session_request("import", "POST", "/sessions/import", json_body({{"path", absolute},
            {"request_id", take_string(g_uuid_string_random())}}));
    }
    void configure_shortcut() {
        std::string error;
        if (install_shortcut(executable_path(), "<Control><Alt>space", error)) {
            refresh_shortcut();
        } else message(error);
    }
    void refresh_shortcut() {
        shortcut_text_ = shortcut_label();
        set_label(shortcut_, shortcut_text_);
    }
private:
    enum class Mode { Idle, Recording, Paused, Transcribing };
    GtkApplication* application_;
    Options options_;
    Settings settings_;
    std::unique_ptr<Backend> backend_;
    GtkWidget *window_, *overlay_, *history_box_, *vram_, *record_button_;
    GtkWidget *stats_, *status_stack_, *shortcut_, *overlay_title_, *overlay_cancel_, *wave_, *server_actions_;
    GtkWidget* preferences_ = nullptr;
    std::unique_ptr<DictionaryView> dictionary_;
    struct SpellingPopover {
        GtkWidget *widget=nullptr,*anchor=nullptr,*entry=nullptr,*save=nullptr,*error=nullptr;
        PangoAttrList* anchor_attributes=nullptr;
        std::string session,selected,operation;
        int64_t segment=0,revision=0,generation=0,start=0,end=0;
        bool replacements_enabled=true;
        guint dismiss_idle=0;
        ~SpellingPopover(){if(dismiss_idle)g_source_remove(dismiss_idle);if(anchor_attributes)pango_attr_list_unref(anchor_attributes);if(anchor)g_object_unref(anchor);}
    };
    std::unique_ptr<SpellingPopover> spelling_;
    guint spelling_idle_=0;
    GtkWidget *session_host_ = nullptr, *pane_scroll_ = nullptr, *view_stack_ = nullptr;
    GtkWidget *pane_copy_ = nullptr, *replacements_toggle_ = nullptr, *reading_tools_ = nullptr;
    GtkWidget *transport_stack_ = nullptr, *transport_controls_ = nullptr, *pane_pause_ = nullptr, *recording_clock_ = nullptr;
    bool syncing_replacements_ = false, replacements_pending_ = false, capture_command_pending_ = false;
    GtkWidget *pane_title_ = nullptr, *pane_progress_ = nullptr, *pane_feedback_ = nullptr, *pane_empty_ = nullptr;
    GtkWidget *pane_jump_ = nullptr, *pane_first_ = nullptr, *pane_next_ = nullptr, *pane_current_ = nullptr, *pane_stop_ = nullptr;
    GtkAdjustment* pane_adjustment_ = nullptr;
    std::string selected_id_;
    bool follow_latest_ = true, adjusting_scroll_ = false, paging_older_ = false, pending_pane_updates_ = false;
    int64_t pane_page_start_ = 0;
    guint timer_ = 0;
    Mode mode_ = Mode::Idle;
    std::array<float, 48> bars_{};
    std::string active_id_, terminal_session_id_;
    std::string shortcut_text_;
    gint64 overlay_until_ = 0, waveform_time_ = 0, history_poll_ = 0;
    double total_vram_mb_ = -1;
    bool starting_session_ = false, cancel_after_start_ = false, quitting_ = false, history_loaded_ = false, list_pending_ = false;
    bool have_history_stats_ = false;
    int64_t history_words_ = 0, history_wpm_ = 0;
    int service_pid_ = 0;
    std::string auto_copy_id_, auto_copy_publication_, next_history_cursor_;
    int64_t auto_copy_epoch_ = -1;
    GtkWidget *empty_history_ = nullptr, *load_more_ = nullptr;
    struct SegmentRow {
        SessionSegment data;
        GtkWidget *edited = nullptr;
        GtkWidget *edited_section = nullptr, *status = nullptr;
        ReplacementHighlights rendered;
    };
    struct Row {
        App* app = nullptr;
        SessionView data;
        GtkWidget *list_row = nullptr, *card = nullptr, *preview = nullptr, *progress = nullptr, *metadata = nullptr;
        GtkWidget *details = nullptr, *edited = nullptr;
        GtkWidget *expand = nullptr, *copy = nullptr, *retry = nullptr, *remove = nullptr, *more = nullptr;
        bool expanded = false, details_loaded = false, detail_pending = false;
        bool has_more = false;
        bool has_later_sections = false;
        int64_t cursor = 0;
        std::map<int64_t, std::unique_ptr<SegmentRow>> segments;
    };
    std::map<std::string, std::unique_ptr<Row>> rows_;


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
        return backend_->state();
    }
    void server_action() {
        if (mode_ != Mode::Idle) return;
        auto state = server_state();
        if (state.ready || state.starting) backend_->unload();
        else backend_->load();
    }
    void save_preferences() {
        backend_->settings(settings_);
    }
    void open_server_log() {
        auto path = server_state().log_path;
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
            static_cast<App*>(p)->quit();
        }));
        add_action("delete-recording", G_CALLBACK(+[](GSimpleAction*, GVariant* value, gpointer p) {
            static_cast<App*>(p)->remove_entry(g_variant_get_string(value, nullptr));
        }), G_VARIANT_TYPE_STRING);
        auto* header = gtk_header_bar_new();
        gtk_header_bar_set_title(GTK_HEADER_BAR(header), "Paraketto");
        gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(header), TRUE);
        gtk_window_set_titlebar(GTK_WINDOW(window_), header);
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
        view_stack_ = gtk_stack_new();
        gtk_stack_set_homogeneous(GTK_STACK(view_stack_), TRUE);
        gtk_stack_set_transition_type(GTK_STACK(view_stack_), GTK_STACK_TRANSITION_TYPE_NONE);
        auto* navigation = gtk_stack_switcher_new();
        gtk_stack_switcher_set_stack(GTK_STACK_SWITCHER(navigation), GTK_STACK(view_stack_));
        gtk_header_bar_set_custom_title(GTK_HEADER_BAR(header), navigation);
        pack(root, view_stack_, true);
        auto* pane = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_style_context_add_class(gtk_widget_get_style_context(pane), "reading-pane");
        gtk_stack_add_titled(GTK_STACK(view_stack_), pane, "transcription", "Transcription");
        auto* pane_header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        gtk_style_context_add_class(gtk_widget_get_style_context(pane_header), "reading-toolbar");
        pane_current_ = icon_button("go-jump-symbolic", "Return to current recording");
        gtk_widget_set_no_show_all(pane_current_, TRUE);
        g_signal_connect_swapped(pane_current_, "clicked", G_CALLBACK(+[](App* a) {
            if (auto it = a->rows_.find(a->active_id_); it != a->rows_.end()) a->select_session(*it->second);
        }), this); pack(pane_header, pane_current_);
        pane_title_ = label("New transcription", "heading");
        gtk_label_set_ellipsize(GTK_LABEL(pane_title_), PANGO_ELLIPSIZE_END);
        pack(pane_header, pane_title_, true);
        reading_tools_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        pack(reading_tools_, label("Replacements", "secondary"));
        replacements_toggle_ = gtk_toggle_button_new();
        auto* replacement_knob = gtk_drawing_area_new();
        gtk_widget_set_size_request(replacement_knob, 46, 22);
        gtk_container_add(GTK_CONTAINER(replacements_toggle_), replacement_knob);
        // Keep the switch's pill/knob appearance, with ordinary button input.
        // The knob has only two positions and cannot track a drag or animation.
        g_signal_connect(replacement_knob, "draw", G_CALLBACK(+[](GtkWidget* widget, cairo_t* cr, gpointer p) -> gboolean {
            const double width = gtk_widget_get_allocated_width(widget);
            const double height = gtk_widget_get_allocated_height(widget);
            const double radius = (height - 2) / 2;
            const double x = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(p)) ? width - radius - 1 : radius + 1;
            cairo_set_source_rgb(cr, 1, 1, 1);
            cairo_arc(cr, x, height / 2, radius, 0, 2 * G_PI); cairo_fill(cr);
            return FALSE;
        }), replacements_toggle_);
        gtk_widget_set_valign(replacements_toggle_, GTK_ALIGN_CENTER);
        gtk_style_context_add_class(gtk_widget_get_style_context(replacements_toggle_), "replacement-toggle");
        gtk_widget_set_tooltip_text(replacements_toggle_, "Click to turn dictionary replacements on or off for this transcript and copied text");
        atk_object_set_name(gtk_widget_get_accessible(replacements_toggle_), "Replacements");
        g_signal_connect(replacements_toggle_, "toggled", G_CALLBACK(+[](GtkToggleButton* button, gpointer p) {
            gtk_widget_queue_draw(gtk_bin_get_child(GTK_BIN(button)));
            auto* a = static_cast<App*>(p); if (!a->syncing_replacements_) a->toggle_replacements();
        }), this); pack(reading_tools_, replacements_toggle_);
        pane_copy_ = icon_button("edit-copy-symbolic", "Copy transcript");
        atk_object_set_name(gtk_widget_get_accessible(pane_copy_), "Copy transcript");
        g_signal_connect_swapped(pane_copy_, "clicked", G_CALLBACK(+[](App* a) { a->copy_selected(false); }), this);
        pack(reading_tools_, pane_copy_); pack(pane_header, reading_tools_);
        pack(pane, pane_header);
        pane_progress_ = label("", "secondary");
        gtk_widget_set_no_show_all(pane_progress_, TRUE);
        gtk_label_set_line_wrap(GTK_LABEL(pane_progress_), TRUE);
        gtk_widget_set_margin_start(pane_progress_, 24); gtk_widget_set_margin_end(pane_progress_, 24);
        gtk_widget_set_size_request(pane_progress_, -1, 28); pack(pane, pane_progress_);
        pane_scroll_ = gtk_scrolled_window_new(nullptr, nullptr);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(pane_scroll_), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        pane_adjustment_ = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(pane_scroll_));
        g_signal_connect(pane_adjustment_, "value-changed", G_CALLBACK(+[](GtkAdjustment* adjustment, gpointer p) {
            auto* a = static_cast<App*>(p);
            if (a->adjusting_scroll_) return;
            if (gtk_adjustment_get_value(adjustment) + gtk_adjustment_get_page_size(adjustment) < gtk_adjustment_get_upper(adjustment) - 6) {
                a->follow_latest_ = false;
            }
        }), this);
        g_signal_connect(pane_adjustment_, "changed", G_CALLBACK(+[](GtkAdjustment*, gpointer p) {
            auto* a = static_cast<App*>(p); if (a->follow_latest_ && !a->pane_has_selection()) a->scroll_to_latest();
        }), this);
        session_host_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
        gtk_style_context_add_class(gtk_widget_get_style_context(session_host_), "reading-content");
        pane_empty_ = label("Your words will appear here.", "empty-transcript");
        gtk_widget_set_margin_top(pane_empty_, 48);
        gtk_label_set_line_wrap(GTK_LABEL(pane_empty_), TRUE); pack(session_host_, pane_empty_);
        gtk_container_add(GTK_CONTAINER(pane_scroll_), session_host_); pack(pane, pane_scroll_, true);
        auto* pane_bottom = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        pane_feedback_ = label("", "secondary"); gtk_label_set_line_wrap(GTK_LABEL(pane_feedback_), TRUE);
        gtk_widget_set_no_show_all(pane_feedback_, TRUE); pack(pane_bottom, pane_feedback_, true);
        pane_first_ = gtk_button_new_with_label("First sections"); gtk_widget_set_no_show_all(pane_first_, TRUE);
        g_signal_connect_swapped(pane_first_, "clicked", G_CALLBACK(+[](App* a) { a->page_sections(true); }), this); pack(pane_bottom, pane_first_);
        pane_next_ = gtk_button_new_with_label("Next sections");
        gtk_widget_set_no_show_all(pane_next_, TRUE);
        g_signal_connect_swapped(pane_next_, "clicked", G_CALLBACK(+[](App* a) { a->page_sections(false); }), this); pack(pane_bottom, pane_next_);
        pane_jump_ = gtk_button_new_with_label("Jump to latest"); gtk_widget_set_no_show_all(pane_jump_, TRUE);
        g_signal_connect_swapped(pane_jump_, "clicked", G_CALLBACK(+[](App* a) { a->jump_to_latest(); }), this); pack(pane_bottom, pane_jump_);
        auto* footer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_style_context_add_class(gtk_widget_get_style_context(footer), "reading-footer");
        gtk_widget_set_size_request(pane_bottom, -1, 36); pack(footer, pane_bottom);
        transport_stack_ = gtk_stack_new();
        gtk_stack_set_homogeneous(GTK_STACK(transport_stack_), TRUE);
        gtk_widget_set_halign(transport_stack_, GTK_ALIGN_CENTER);
        gtk_widget_set_size_request(transport_stack_, 240, 48);
        gtk_style_context_add_class(gtk_widget_get_style_context(transport_stack_), "transport");
        auto* start = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        gtk_widget_set_halign(start, GTK_ALIGN_CENTER);
        record_button_ = gtk_button_new_with_label("Record");
        gtk_button_set_always_show_image(GTK_BUTTON(record_button_), TRUE);
        reset_recording_button();
        g_signal_connect_swapped(record_button_, "clicked", G_CALLBACK(+[](App* a) { a->toggle(false); }), this);
        pack(start, record_button_);
        shortcut_ = label(shortcut_text_, "secondary"); pack(start, shortcut_);
        gtk_stack_add_named(GTK_STACK(transport_stack_), start, "idle");
        transport_controls_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        gtk_widget_set_halign(transport_controls_, GTK_ALIGN_CENTER);
        recording_clock_ = label("00:00", "recording-clock");
        gtk_label_set_width_chars(GTK_LABEL(recording_clock_), 8);
        pack(transport_controls_, recording_clock_);
        pane_pause_ = icon_button("media-playback-pause-symbolic", "Pause recording");
        pane_stop_ = icon_button("media-playback-stop-symbolic", "Finish recording and copy transcript");
        for (auto* button : {pane_pause_, pane_stop_}) gtk_widget_set_size_request(button, 44, 44);
        atk_object_set_name(gtk_widget_get_accessible(pane_pause_), "Pause recording");
        atk_object_set_name(gtk_widget_get_accessible(pane_stop_), "Stop recording");
        g_signal_connect_swapped(pane_pause_, "clicked", G_CALLBACK(+[](App* a) { a->pause_recording(); }), this);
        g_signal_connect_swapped(pane_stop_, "clicked", G_CALLBACK(+[](App* a) { a->stop_recording(); }), this);
        pack(transport_controls_, pane_pause_); pack(transport_controls_, pane_stop_);
        gtk_stack_add_named(GTK_STACK(transport_stack_), transport_controls_, "recording");
        auto* finishing = label("Finishing transcription…", "secondary");
        gtk_widget_set_halign(finishing, GTK_ALIGN_CENTER);
        gtk_stack_add_named(GTK_STACK(transport_stack_), finishing, "finishing");
        pack(footer, transport_stack_); pack(pane, footer);
        auto* history = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_stack_add_titled(GTK_STACK(view_stack_), history, "history", "History");
        auto* scroll = gtk_scrolled_window_new(nullptr, nullptr);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        gtk_style_context_add_class(gtk_widget_get_style_context(scroll), "transcriptions");
        pack(history, scroll, true);
        history_box_ = gtk_list_box_new();
        gtk_list_box_set_selection_mode(GTK_LIST_BOX(history_box_), GTK_SELECTION_NONE);
        g_signal_connect(history_box_, "row-activated", G_CALLBACK(+[](GtkListBox*, GtkListBoxRow* widget, gpointer p) {
            if (auto* row = static_cast<Row*>(g_object_get_data(G_OBJECT(widget), "session-row"))) static_cast<App*>(p)->select_session(*row);
        }), this);
        gtk_list_box_set_sort_func(GTK_LIST_BOX(history_box_), +[](GtkListBoxRow* a, GtkListBoxRow* b, gpointer) -> gint {
            auto* left = static_cast<Row*>(g_object_get_data(G_OBJECT(a), "session-row"));
            auto* right = static_cast<Row*>(g_object_get_data(G_OBJECT(b), "session-row"));
            if (!left || !right) return left ? -1 : right ? 1 : 0;
            return right->data.time.compare(left->data.time);
        }, nullptr, nullptr);
        gtk_style_context_add_class(gtk_widget_get_style_context(history_box_), "history-list");
        gtk_list_box_set_header_func(GTK_LIST_BOX(history_box_), [](GtkListBoxRow* row, GtkListBoxRow* before, gpointer) {
            auto* day = static_cast<const char*>(g_object_get_data(G_OBJECT(row), "day"));
            if (!day) { gtk_list_box_row_set_header(row, nullptr); return; }
            auto* previous = before ? static_cast<const char*>(g_object_get_data(G_OBJECT(before), "day")) : nullptr;
            GtkWidget* header;
            if (!previous || strcmp(day, previous)) {
                header = label(static_cast<const char*>(g_object_get_data(G_OBJECT(row), "day-heading")), "secondary");
                gtk_style_context_add_class(gtk_widget_get_style_context(header), "history-day");
            } else {
                header = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
                gtk_style_context_add_class(gtk_widget_get_style_context(header), "transcript-separator");
            }
            gtk_list_box_row_set_header(row, header);
            gtk_widget_show(header);
        }, nullptr, nullptr);
        gtk_container_add(GTK_CONTAINER(scroll), history_box_);
        pack(root, gtk_separator_new(GTK_ORIENTATION_HORIZONTAL));
        auto* server_panel = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        gtk_style_context_add_class(gtk_widget_get_style_context(server_panel), "server-panel");
        pack(root, server_panel);
        auto* server = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
        gtk_style_context_add_class(gtk_widget_get_style_context(server), "server-row");
        status_stack_ = gtk_stack_new();
        // Keep every label in a homogeneous stack: its size depends on all
        // possible states, never on the current state or a changing label.
        gtk_stack_set_homogeneous(GTK_STACK(status_stack_), TRUE);
        struct StateLabel { const char *name, *text, *style; };
        for (const auto& item : {
            StateLabel{"sleeping", "Sleeping", "state-sleeping"},
            StateLabel{"connecting", "Connecting…", nullptr},
            StateLabel{"unloading", "Unloading…", nullptr},
            StateLabel{"loading", "Loading…", nullptr},
            StateLabel{"load-error", "Load failed", "failure"},
            StateLabel{"offline", "Unavailable", "failure"},
            StateLabel{"busy", "Transcribing…", "state-ready"},
            StateLabel{"ready", "Ready", "state-ready"},
        }) {
            auto* state = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
            auto* dot = gtk_image_new_from_icon_name("media-record-symbolic", GTK_ICON_SIZE_MENU);
            gtk_image_set_pixel_size(GTK_IMAGE(dot), 10);
            pack(state, dot);
            auto* text = label(item.text, "server-state");
            gtk_label_set_single_line_mode(GTK_LABEL(text), TRUE);
            pack(state, text, true);
            if (item.style) for (auto* widget : {dot, text})
                gtk_style_context_add_class(gtk_widget_get_style_context(widget), item.style);
            gtk_stack_add_named(GTK_STACK(status_stack_), state, item.name);
        }
        pack(server, status_stack_);
        vram_ = label("", "vram");
        gtk_label_set_width_chars(GTK_LABEL(vram_), 19);
        gtk_label_set_max_width_chars(GTK_LABEL(vram_), 19);
        gtk_label_set_ellipsize(GTK_LABEL(vram_), PANGO_ELLIPSIZE_END);
        gtk_label_set_single_line_mode(GTK_LABEL(vram_), TRUE);
        gtk_label_set_xalign(GTK_LABEL(vram_), 0);
        pack(server, vram_, true);
        server_actions_ = gtk_stack_new();
        gtk_stack_set_homogeneous(GTK_STACK(server_actions_), TRUE);
        gtk_widget_set_valign(server_actions_, GTK_ALIGN_CENTER);
        for (const char* text : {"Load", "Cancel", "Unload", "Retry"}) {
            auto* button = gtk_button_new_with_label(text);
            gtk_actionable_set_action_name(GTK_ACTIONABLE(button), "app.model");
            gtk_stack_add_named(GTK_STACK(server_actions_), button, text);
        }
        pack(server, server_actions_);
        pack(server_panel, server);
        auto* stats_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
        gtk_style_context_add_class(gtk_widget_get_style_context(stats_row), "stats-row");
        stats_ = label("");
        gtk_style_context_add_class(gtk_widget_get_style_context(stats_), "secondary");
        gtk_label_set_ellipsize(GTK_LABEL(stats_), PANGO_ELLIPSIZE_END);
        gtk_widget_set_tooltip_text(stats_, "Saved transcriptions. WPM = total words ÷ total recording time in minutes.");
        pack(stats_row, stats_, true);
        pack(history, stats_row);
        dictionary_ = std::make_unique<DictionaryView>(GTK_WINDOW(window_),
            [this](std::string operation, std::string method, std::string path, std::string body) {
                backend_->session_request(std::move(operation), std::move(method), std::move(path), std::move(body));
            });
        gtk_stack_add_titled(GTK_STACK(view_stack_), dictionary_->widget(), "dictionary", "Dictionary");
        g_signal_connect(view_stack_, "notify::visible-child-name", G_CALLBACK(+[](GObject* stack, GParamSpec*, gpointer p) {
            auto* app = static_cast<App*>(p);
            if (g_strcmp0(gtk_stack_get_visible_child_name(GTK_STACK(stack)), "dictionary") == 0) app->dictionary_->activate();
        }), this);
        gtk_stack_set_visible_child_name(GTK_STACK(view_stack_), "transcription");
        refresh_history();
    }

    void show_preferences() {
        if (preferences_) { gtk_window_present(GTK_WINDOW(preferences_)); return; }
        settings_ = server_state().settings;
        preferences_ = gtk_dialog_new_with_buttons("Preferences", GTK_WINDOW(window_),
            static_cast<GtkDialogFlags>(GTK_DIALOG_DESTROY_WITH_PARENT | GTK_DIALOG_USE_HEADER_BAR), nullptr, nullptr);
        gtk_window_set_default_size(GTK_WINDOW(preferences_), 500, -1);
        gtk_window_set_resizable(GTK_WINDOW(preferences_), FALSE);
        g_signal_connect(preferences_, "response", G_CALLBACK(+[](GtkDialog* dialog, gint, gpointer) {
            gtk_widget_destroy(GTK_WIDGET(dialog));
        }), nullptr);
        g_signal_connect(preferences_, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer p) {
            auto* a = static_cast<App*>(p);
            a->preferences_ = nullptr;
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
        auto model_path = server_state().weights;
        auto* weights = label(model_path.empty() ? "Service unavailable" : take_string(g_path_get_basename(model_path.c_str())), "secondary");
        gtk_label_set_ellipsize(GTK_LABEL(weights), PANGO_ELLIPSIZE_MIDDLE);
        gtk_widget_set_tooltip_text(weights, model_path.c_str());
        pack(body, weights);
        pack(body, label("Service", "heading"));
        auto service_url = server_state().url;
        pack(body, label(service_url.empty() ? "Discovering local service…" : service_url, "secondary"));
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
    void reset_recording_button() {
        gtk_button_set_image(GTK_BUTTON(record_button_), gtk_image_new_from_icon_name("audio-input-microphone-symbolic", GTK_ICON_SIZE_BUTTON));
        gtk_widget_set_tooltip_text(record_button_, "Record and copy the saved transcript");
        atk_object_set_name(gtk_widget_get_accessible(record_button_), "Record");
        auto* style = gtk_widget_get_style_context(record_button_);
        gtk_style_context_remove_class(style, "destructive-action");
        gtk_style_context_add_class(style, "suggested-action");
    }
    void show_recording() {
        mode_ = Mode::Recording; overlay_until_ = 0;
        terminal_session_id_.clear();
        waveform_time_ = g_get_monotonic_time();
        gtk_style_context_remove_class(gtk_widget_get_style_context(overlay_title_), "failure");
        gtk_widget_set_tooltip_text(overlay_cancel_, "Cancel recording; keep captured audio");
        place_overlay(); gtk_widget_show_all(overlay_);
    }
    void stop_recording() {
        if (active_id_.empty()) return;
        auto_copy_id_ = active_id_; auto_copy_publication_.clear();
        auto row = rows_.find(active_id_);
        auto_copy_epoch_ = row == rows_.end() ? -1 : row->second->data.delivery_epoch;
        backend_->session_request("stop", "POST", "/sessions/" + active_id_ + "/stop", "{}", active_id_);
        mode_ = Mode::Transcribing;
        reset_recording_button(); set_label(overlay_title_, "Saving and finishing transcription…");
    }
    void tick() {
        auto state = server_state();
        auto now = g_get_monotonic_time();
        if (!state.connected || (service_pid_ && state.service_pid && state.service_pid != service_pid_)) {
            auto_copy_id_.clear(); auto_copy_publication_.clear();
        }
        if (state.service_pid) service_pid_ = state.service_pid;
        if (!preferences_ && state.connected) settings_ = state.settings;
        auto action_error = backend_->action_error();
        if (!action_error.empty()) message(action_error);
        bool transcribing = mode_ == Mode::Transcribing;
        bool failed = state.status == "offline" || state.status == "error";
        bool sleeping = state.status == "sleeping";
        gtk_stack_set_visible_child_name(GTK_STACK(status_stack_),
            failed ? (state.load_failed ? "load-error" : "offline") :
            transcribing && state.ready ? "busy" : state.status.c_str());
        std::string detail;
        if (sleeping) detail = "Loads when you start recording";
        else if (state.status == "offline") detail = "Start the Paraketto service to transcribe";
        else if (state.status == "connecting") detail = "Connecting to the transcription service";
        else if (state.status == "unloading") detail = "Releasing GPU memory";
        else if (state.starting) detail = "Loading the model; progress is not reported. Recording can start immediately.";
        else if (state.load_failed) detail = "Couldn’t load model. Retry, or view the server log for details";
        else if (!state.ready) detail = "Retry, or view the server log for details";
        else if (mode_ != Mode::Idle || state.busy) detail = "Model in use";
        else if (state.settings.keep_loaded) detail = "Automatic unloading disabled";
        else detail = "Unloads in " + duration(std::ceil(state.idle_remaining));
        std::string model = state.model.empty() ? state.weights : state.model;
        if (!state.device.empty()) model += "\n" + state.device;
        gtk_widget_set_tooltip_text(status_stack_, (detail + (model.empty() ? "" : "\n" + model)).c_str());
        if (state.total_vram_mb > 0) total_vram_mb_ = state.total_vram_mb;
        auto memory_field = [](double mb) {
            return mb >= 0 ? take_string(g_strdup_printf("%6.2f", mb / 1024)) : std::string(" --.--");
        };
        set_label(vram_, memory_field(state.server_vram_mb) + " / " + memory_field(total_vram_mb_) + " GiB");
        std::string memory = sleeping ? "The model is unloaded" : state.used_vram_mb >= 0 ? take_string(g_strdup_printf(
            "Model process memory. Entire GPU: %.2f / %.2f GiB used.", state.used_vram_mb / 1024, state.total_vram_mb / 1024)) : "Model VRAM usage is unavailable";
        gtk_widget_set_tooltip_text(vram_, memory.c_str());
        gtk_stack_set_visible_child_name(GTK_STACK(server_actions_),
            sleeping ? "Load" : state.starting ? "Cancel" : state.ready ? "Unload" : "Retry");
        auto* model_action = g_action_map_lookup_action(G_ACTION_MAP(application_), "model");
        g_simple_action_set_enabled(G_SIMPLE_ACTION(model_action), mode_ == Mode::Idle && state.connected && !state.busy && state.status != "unloading");
        gtk_widget_set_sensitive(record_button_, !starting_session_ && mode_ == Mode::Idle && state.connected);
        for (auto& event : backend_->session_events()) handle_session_event(event);
        update_pane();
        if (state.connected && state.sessions_available) {
            if (!history_loaded_ && !list_pending_ && now - history_poll_ > 1000000) refresh_history();
            if (!state.active_session_id.empty() && active_id_ != state.active_session_id && !starting_session_) {
                active_id_ = state.active_session_id;
                backend_->watch_session(active_id_);
                show_recording(); // A reconnected UI never arms automatic clipboard delivery.
            }
            if (now - history_poll_ > 5000000 && !list_pending_) { history_poll_ = now; refresh_history(); }
        }
        if (mode_ == Mode::Recording || mode_ == Mode::Paused) {
            auto levels = backend_->levels();
            float level = waveform_level(mode_ == Mode::Recording && levels.id == active_id_ ? levels.rms : 0, bars_.back(), (now - waveform_time_) / 1000000.0);
            waveform_time_ = now;
            std::move(bars_.begin() + 1, bars_.end(), bars_.begin()); bars_.back() = level;
            gtk_widget_queue_draw(wave_);
            set_label(overlay_title_, state.connected ? duration(levels.elapsed_seconds) + (mode_ == Mode::Paused ? "  ·  Paused" : "  ·  Recording") : "Service disconnected · reconnecting…");
        }
        if (overlay_until_ && mode_ == Mode::Idle && now > overlay_until_) {
            gtk_widget_hide(overlay_); overlay_until_ = 0;
        }

    }
    void refresh_history(const std::string& cursor = {}) {
        if (list_pending_) return;
        list_pending_ = true; history_poll_ = g_get_monotonic_time();
        backend_->session_request(cursor.empty() ? "history" : "more-history", "GET",
            "/sessions?limit=50&cursor=" + (cursor.empty() ? "0" : cursor));
    }
    void close_spelling() {
        if(spelling_)gtk_widget_destroy(spelling_->widget);
    }
    void save_spelling() {
        if(!spelling_ || !spelling_->operation.empty() || !gtk_widget_get_sensitive(spelling_->save))return;
        auto& s=*spelling_;
        auto* b=json_builder_new();json_builder_begin_object(b);
        json_int(b,"segment_id",s.segment);json_int(b,"revision_id",s.revision);json_int(b,"generation",s.generation);
        json_int(b,"start",s.start);json_int(b,"end",s.end);json_bool(b,"replacements_enabled",s.replacements_enabled);
        json_field(b,"selected",s.selected);json_field(b,"canonical",std::string(gtk_entry_get_text(GTK_ENTRY(s.entry))));
        json_builder_end_object(b);auto body=json_string(b);g_object_unref(b);
        s.operation="spelling:"+take_string(g_uuid_string_random());
        gtk_widget_set_sensitive(s.entry,FALSE);gtk_widget_set_sensitive(s.save,FALSE);gtk_widget_hide(s.error);
        backend_->session_request(s.operation,"POST","/sessions/"+s.session+"/spelling",body,s.session);
    }
    void show_spelling(GtkWidget* anchor) {
        auto row=rows_.find(selected_id_);if(row==rows_.end() || !gtk_widget_get_mapped(anchor))return;
        SegmentRow* segment=nullptr;
        for(const auto& [sequence,item]:row->second->segments)if(item->edited==anchor){segment=item.get();break;}
        int first=0,last=0;
        if(!segment || segment->data.state!="completed" || !segment->rendered.valid ||
           !gtk_label_get_selection_bounds(GTK_LABEL(anchor),&first,&last) || first==last)return;
        const auto& text=segment->rendered.text;
        if(text!=replacement_highlights(segment->data,row->second->data.replacements_enabled).text) {
            set_notice(pane_feedback_,"Transcript changed; clear the selection and select the text again.");return;
        }
        if(first>last)std::swap(first,last);
        const char* begin=g_utf8_offset_to_pointer(text.c_str(),first);
        const char* end=g_utf8_offset_to_pointer(text.c_str(),last);
        while(begin<end && g_unichar_isspace(g_utf8_get_char(begin)))begin=g_utf8_next_char(begin);
        while(begin<end && g_unichar_isspace(g_utf8_get_char(g_utf8_prev_char(end))))end=g_utf8_prev_char(end);
        if(begin==end || end-begin>4096)return;
        size_t start=begin-text.c_str(),finish=end-text.c_str();
        for(const auto& h:segment->rendered.highlights)
            if((start>h.start&&start<h.end)||(finish>h.start&&finish<h.end)) {
                set_notice(pane_feedback_,"Select the whole replacement to change its spelling.");return;
            }
        close_spelling();spelling_=std::make_unique<SpellingPopover>();auto& s=*spelling_;
        follow_latest_=false;
        s.session=selected_id_;s.segment=std::stoll(segment->data.id);s.revision=segment->data.revision;
        s.generation=row->second->data.generation;s.start=start;s.end=finish;s.selected=text.substr(start,finish-start);
        s.replacements_enabled=row->second->data.replacements_enabled;s.anchor=GTK_WIDGET(g_object_ref(anchor));
        auto* original_attributes=gtk_label_get_attributes(GTK_LABEL(anchor));
        s.anchor_attributes=original_attributes?pango_attr_list_ref(original_attributes):nullptr;
        auto* selection_attributes=original_attributes?pango_attr_list_copy(original_attributes):pango_attr_list_new();
        GdkRGBA background{},foreground{};auto* style=gtk_widget_get_style_context(anchor);
        if(!gtk_style_context_lookup_color(style,"theme_selected_bg_color",&background))gdk_rgba_parse(&background,"#3584e4");
        if(!gtk_style_context_lookup_color(style,"theme_selected_fg_color",&foreground))gdk_rgba_parse(&foreground,"white");
        for(auto* attribute:{pango_attr_background_new(background.red*65535,background.green*65535,background.blue*65535),
                            pango_attr_foreground_new(foreground.red*65535,foreground.green*65535,foreground.blue*65535)}) {
            attribute->start_index=start;attribute->end_index=finish;pango_attr_list_change(selection_attributes,attribute);
        }
        gtk_label_set_attributes(GTK_LABEL(anchor),selection_attributes);pango_attr_list_unref(selection_attributes);
        s.widget=gtk_popover_new(anchor);gtk_popover_set_position(GTK_POPOVER(s.widget),GTK_POS_TOP);
        // Anchor to the selected range on its first line, including RTL layouts.
        auto* layout=gtk_label_get_layout(GTK_LABEL(anchor));PangoRectangle position{};
        pango_layout_index_to_pos(layout,start,&position);
        int line=0,x=0;pango_layout_index_to_line_x(layout,start,FALSE,&line,&x);
        auto* first_line=pango_layout_get_line_readonly(layout,line);
        int* ranges=nullptr,count=0;
        pango_layout_line_get_x_ranges(first_line,start,std::min<size_t>(finish,first_line->start_index+first_line->length),&ranges,&count);
        int left=std::min(position.x,position.x+position.width),right=std::max(position.x,position.x+position.width);
        for(int i=0;i<count;++i){left=std::min(left,ranges[2*i]);right=std::max(right,ranges[2*i+1]);}g_free(ranges);
        int offset_x=0,offset_y=0;label_layout_offsets(anchor,&offset_x,&offset_y);
        GdkRectangle rect{offset_x+left/PANGO_SCALE,offset_y+position.y/PANGO_SCALE,
            std::max(1,(right-left)/PANGO_SCALE),std::max(1,position.height/PANGO_SCALE)};
        gtk_popover_set_pointing_to(GTK_POPOVER(s.widget),&rect);
        auto* box=gtk_box_new(GTK_ORIENTATION_VERTICAL,8);gtk_container_set_border_width(GTK_CONTAINER(box),12);
        gtk_container_add(GTK_CONTAINER(s.widget),box);
        pack(box,label("Desired spelling", "section-title"));
        s.entry=gtk_entry_new();gtk_widget_set_size_request(s.entry,280,-1);
        gtk_entry_set_max_length(GTK_ENTRY(s.entry),4096);gtk_entry_set_text(GTK_ENTRY(s.entry),s.selected.c_str());pack(box,s.entry);
        auto* hint=label("Save for this section and future transcriptions.","secondary");
        gtk_label_set_line_wrap(GTK_LABEL(hint),TRUE);gtk_label_set_max_width_chars(GTK_LABEL(hint),42);pack(box,hint);
        s.error=label("","secondary");gtk_label_set_line_wrap(GTK_LABEL(s.error),TRUE);
        gtk_label_set_max_width_chars(GTK_LABEL(s.error),42);gtk_widget_set_no_show_all(s.error,TRUE);pack(box,s.error);
        auto* buttons=gtk_box_new(GTK_ORIENTATION_HORIZONTAL,8);gtk_widget_set_halign(buttons,GTK_ALIGN_END);
        auto* cancel=gtk_button_new_with_label("Cancel");s.save=gtk_button_new_with_label("Save spelling");
        gtk_style_context_add_class(gtk_widget_get_style_context(s.save),"suggested-action");
        pack(buttons,cancel);pack(buttons,s.save);pack(box,buttons);gtk_widget_set_sensitive(s.save,FALSE);
        g_signal_connect_swapped(cancel,"clicked",G_CALLBACK(+[](App* a){a->close_spelling();}),this);
        g_signal_connect_swapped(s.save,"clicked",G_CALLBACK(+[](App* a){a->save_spelling();}),this);
        g_signal_connect_swapped(s.entry,"activate",G_CALLBACK(+[](App* a){a->save_spelling();}),this);
        g_signal_connect(s.entry,"changed",G_CALLBACK(+[](GtkEditable*,gpointer p){
            auto* a=static_cast<App*>(p);if(!a->spelling_)return;auto& s=*a->spelling_;
            const std::string value=gtk_entry_get_text(GTK_ENTRY(s.entry));
            gtk_widget_set_sensitive(s.save,s.operation.empty()&&!value.empty()&&value!=s.selected);
        }),this);
        g_signal_connect(s.widget,"closed",G_CALLBACK(+[](GtkPopover* popover,gpointer p){
            auto* a=static_cast<App*>(p);
            if(gtk_widget_in_destruction(GTK_WIDGET(popover)) || !a->spelling_ || a->spelling_->dismiss_idle)return;
            // GTK still uses the popover after emitting closed. Destroy it on
            // the next main-loop turn; replacing/closing it cancels this source.
            a->spelling_->dismiss_idle=g_idle_add(+[](gpointer p)->gboolean {
                auto* a=static_cast<App*>(p);
                if(a->spelling_){a->spelling_->dismiss_idle=0;a->close_spelling();}
                return G_SOURCE_REMOVE;
            },a);
        }),this);
        g_signal_connect(s.widget,"destroy",G_CALLBACK(+[](GtkWidget* widget,gpointer p){
            auto* a=static_cast<App*>(p);if(a->spelling_&&a->spelling_->widget==widget) {
                auto& s=*a->spelling_;
                if(!gtk_widget_in_destruction(s.anchor))gtk_label_set_attributes(GTK_LABEL(s.anchor),s.anchor_attributes);
                a->spelling_.reset();
            }
        }),this);
        gtk_widget_show_all(s.widget);gtk_popover_popup(GTK_POPOVER(s.widget));
        gtk_widget_grab_focus(s.entry);gtk_editable_select_region(GTK_EDITABLE(s.entry),0,-1);
    }
    void queue_spelling(GtkWidget* anchor) {
        if(spelling_idle_)g_source_remove(spelling_idle_);
        struct Pending {App* app;GtkWidget* anchor;};
        auto* pending=new Pending{this,GTK_WIDGET(g_object_ref(anchor))};
        spelling_idle_=g_idle_add_full(G_PRIORITY_DEFAULT_IDLE,+[](gpointer p)->gboolean {
            auto& pending=*static_cast<Pending*>(p);pending.app->spelling_idle_=0;
            pending.app->show_spelling(pending.anchor);return G_SOURCE_REMOVE;
        },pending,+[](gpointer p){auto* pending=static_cast<Pending*>(p);g_object_unref(pending->anchor);delete pending;});
    }
    static GtkWidget* transcript_label() {
        auto* widget = label("", "transcript");
        gtk_label_set_line_wrap(GTK_LABEL(widget), TRUE);
        gtk_label_set_line_wrap_mode(GTK_LABEL(widget), PANGO_WRAP_WORD_CHAR);
        gtk_label_set_max_width_chars(GTK_LABEL(widget), 62);
        gtk_label_set_selectable(GTK_LABEL(widget), TRUE);
        gtk_widget_set_hexpand(widget, TRUE);
        return widget;
    }
    static void label_layout_offsets(GtkWidget* widget,int* x,int* y) {
        // GtkLabel is windowless: its Pango offsets include the allocation in
        // the parent window. Popover/tooltip coordinates are widget-relative.
        gtk_label_get_layout_offsets(GTK_LABEL(widget),x,y);
        GtkAllocation allocation{};gtk_widget_get_allocation(widget,&allocation);
        *x-=allocation.x;*y-=allocation.y;
    }
    static void update_selected_label(GtkWidget* widget, const std::string& text) {
        if (text == gtk_label_get_text(GTK_LABEL(widget))) return;
        int start = 0, end = 0;
        bool selected = gtk_label_get_selection_bounds(GTK_LABEL(widget), &start, &end);
        if (selected && start != end) {
            g_object_set_data_full(G_OBJECT(widget), "pending-text", g_strdup(text.c_str()), g_free);
            return;
        }
        set_label(widget, text);
        g_object_set_data(G_OBJECT(widget), "pending-text", nullptr);
    }
    bool pane_has_selection() const {
        auto row = rows_.find(selected_id_);
        if (row == rows_.end()) return false;
        for (const auto& [sequence, section] : row->second->segments) for (auto* text : {section->edited}) {
            int start, end;
            if (gtk_label_get_selection_bounds(GTK_LABEL(text), &start, &end) && start != end) return true;
        }
        return false;
    }
    void scroll_to_latest() {
        if (!pane_adjustment_) return;
        adjusting_scroll_ = true;
        gtk_adjustment_set_value(pane_adjustment_, std::max(0.0, gtk_adjustment_get_upper(pane_adjustment_) - gtk_adjustment_get_page_size(pane_adjustment_)));
        adjusting_scroll_ = false;
    }
    void copy_selected(bool original) {
        if (selected_id_.empty()) return;
        backend_->session_request(original ? "copy-original" : "copy", "GET", "/sessions/" + selected_id_ +
            "/transcript?view=" + (original ? "original" : "edited"), "{}", selected_id_);
    }
    void select_session(Row& row) {
        if(selected_id_!=row.data.id)close_spelling();
        gtk_stack_set_visible_child_name(GTK_STACK(view_stack_), "transcription");
        if (selected_id_ == row.data.id) return;
        if (auto previous = rows_.find(selected_id_); previous != rows_.end()) {
            auto& old = *previous->second;
            move_widget(old.details, old.card); gtk_widget_hide(old.details); gtk_widget_set_no_show_all(old.details, TRUE); old.expanded = false;
            clear_segments(old); // Only the selected session owns section widgets.
        }
        selected_id_ = row.data.id; row.expanded = true;
        follow_latest_ = true; paging_older_ = false; pending_pane_updates_ = false;
        pane_page_start_ = 0;
        gtk_widget_hide(pane_jump_);
        move_widget(row.details, session_host_); gtk_widget_set_no_show_all(row.details, FALSE); gtk_widget_show_all(row.details);
        set_notice(pane_feedback_, "");
        load_details(row); update_pane();
    }
    void page_sections(bool first) {
        auto row = rows_.find(selected_id_); if (row == rows_.end() || row->second->detail_pending) return;
        auto& r = *row->second;
        pane_page_start_ = first ? 0 : pane_page_start_ + 50;
        clear_segments(r); r.has_later_sections = false;
        follow_latest_ = false; paging_older_ = true;
        adjusting_scroll_ = true; gtk_adjustment_set_value(pane_adjustment_, 0); adjusting_scroll_ = false;
        load_details(r);
    }
    void jump_to_latest() {
        auto row = rows_.find(selected_id_); if (row == rows_.end() || row->second->detail_pending) return;
        follow_latest_ = true; paging_older_ = false; pending_pane_updates_ = false; gtk_widget_hide(pane_jump_);
        clear_segments(*row->second); load_details(*row->second); scroll_to_latest();
    }
    void update_pane() {
        bool recording = mode_ == Mode::Recording || mode_ == Mode::Paused;
        gtk_stack_set_visible_child_name(GTK_STACK(transport_stack_),
            recording ? "recording" : mode_ == Mode::Transcribing || starting_session_ ? "finishing" : "idle");
        gtk_widget_set_visible(pane_current_, recording && selected_id_ != active_id_);
        gtk_widget_set_sensitive(pane_pause_, recording && !capture_command_pending_);
        gtk_widget_set_sensitive(pane_stop_, recording && !capture_command_pending_);
        bool paused = mode_ == Mode::Paused;
        const char* pause_action = paused ? "Resume recording" : "Pause recording";
        gtk_widget_set_tooltip_text(pane_pause_, pause_action);
        atk_object_set_name(gtk_widget_get_accessible(pane_pause_), pause_action);
        auto* image = gtk_button_get_image(GTK_BUTTON(pane_pause_));
        const char* icon = nullptr; gtk_image_get_icon_name(GTK_IMAGE(image), &icon, nullptr);
        const char* wanted = paused ? "media-playback-start-symbolic" : "media-playback-pause-symbolic";
        if (!icon || strcmp(icon, wanted)) gtk_image_set_from_icon_name(GTK_IMAGE(image), wanted, GTK_ICON_SIZE_BUTTON);
        auto levels = backend_->levels();
        set_label(recording_clock_, duration(levels.id == active_id_ ? levels.elapsed_seconds : 0));
        auto found = rows_.find(selected_id_);
        bool selected = found != rows_.end();
        // Retain the toolbar's geometry even in the empty state.
        gtk_widget_set_opacity(reading_tools_, selected ? 1.0 : 0.0);
        gtk_widget_set_sensitive(reading_tools_, selected);
        gtk_widget_set_sensitive(pane_next_, FALSE);
        if (!selected) {
            set_label(pane_title_, "New transcription"); set_notice(pane_progress_, "");
            set_label(pane_empty_, "Your words will appear here."); gtk_widget_show(pane_empty_);
            gtk_widget_hide(pane_first_); gtk_widget_hide(pane_next_); gtk_widget_hide(pane_jump_);
            return;
        }
        auto& row = *found->second;
        bool live = row.data.capture == "recording" || row.data.capture == "paused";
        bool is_paused = row.data.capture == "paused";
        set_label(pane_title_, live ? (is_paused ? "Paused" : "Recording") : "Transcription");
        syncing_replacements_ = true;
        if (!replacements_pending_) {
            // An external toggle can change the text while the selection keeps
            // its old rendering visible. Keep a way to reveal that deferred edit.
            if (!follow_latest_ && bool(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(replacements_toggle_))) != row.data.replacements_enabled)
                pending_pane_updates_ = true;
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(replacements_toggle_), row.data.replacements_enabled);
        }
        syncing_replacements_ = false;
        gtk_widget_set_sensitive(replacements_toggle_, !replacements_pending_);
        gtk_widget_set_sensitive(pane_copy_, !replacements_pending_ && (row.data.words > 0 || row.data.transcribed > 0));
        std::string notice = row.data.attention();
        if (live && row.data.id == active_id_) {
            auto service = server_state();
            if (!service.session_error.empty() && service.session_error != notice) {
                if (!notice.empty()) notice += " · ";
                notice += service.session_error;
            }
        }
        set_notice(pane_progress_, notice);
        gtk_widget_set_visible(pane_empty_, row.segments.empty());
        set_label(pane_empty_, live ? (is_paused ? "Resume when you’re ready." : "Audio is being recorded. Text arrives when each speech group finishes.") :
            row.detail_pending ? "Loading transcript…" : "No transcribed text.");
        gtk_widget_set_sensitive(pane_next_, paging_older_ && row.has_more && !row.detail_pending);
        gtk_widget_set_visible(pane_first_, row.segments.size() >= 50 || paging_older_ || row.has_more);
        gtk_widget_set_visible(pane_next_, paging_older_);
        if (pane_has_selection()) follow_latest_ = false;
        const bool away_from_end = gtk_adjustment_get_value(pane_adjustment_) + gtk_adjustment_get_page_size(pane_adjustment_) <
            gtk_adjustment_get_upper(pane_adjustment_) - 6;
        gtk_widget_set_visible(pane_jump_, paging_older_ || pending_pane_updates_ || away_from_end);
        for (auto& [sequence, section] : row.segments) {
            int start, end;
            if (gtk_label_get_selection_bounds(GTK_LABEL(section->edited), &start, &end) && start != end) continue;
            if (g_object_get_data(G_OBJECT(section->edited), "pending-render")) render_segment(*section, row.data.replacements_enabled);
        }
    }
    void toggle_replacements() {
        auto row = rows_.find(selected_id_);
        if (row == rows_.end() || replacements_pending_) return;
        for (auto& [sequence, section] : row->second->segments) gtk_label_select_region(GTK_LABEL(section->edited), 0, 0);
        replacements_pending_ = true;
        bool enabled = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(replacements_toggle_));
        backend_->session_request("replacements", "PATCH", "/sessions/" + selected_id_ + "/replacements",
            std::string("{\"enabled\":") + (enabled ? "true}" : "false}"), selected_id_);
        update_pane();
    }
    void pause_recording() {
        if (active_id_.empty() || capture_command_pending_ || (mode_ != Mode::Recording && mode_ != Mode::Paused)) return;
        capture_command_pending_ = true;
        const char* action = mode_ == Mode::Paused ? "resume" : "pause";
        backend_->session_request(action, "POST", "/sessions/" + active_id_ + "/" + action, "{}", active_id_);
        update_pane();
    }
    void load_details(Row& row) {
        if (row.detail_pending) return;
        row.detail_pending = true;
        backend_->session_request("details", "GET", "/sessions/" + row.data.id + "/updates?after=" +
            std::to_string(row.cursor) + "&generation=" + std::to_string(row.data.generation) + "&limit=50", "{}", row.data.id);
    }
    void expand(Row& row) {
        select_session(row);
    }
    Row& ensure_row(JsonObject* object) {
        auto id = json_text(object, "id");
        auto it = rows_.find(id);
        if (it == rows_.end()) {
            auto owned = std::make_unique<Row>(); owned->app = this; owned->data.update(object);
            auto* row = owned.get(); rows_[id] = std::move(owned);
            row->list_row = gtk_list_box_row_new();
            gtk_list_box_row_set_activatable(GTK_LIST_BOX_ROW(row->list_row), TRUE);
            gtk_list_box_row_set_selectable(GTK_LIST_BOX_ROW(row->list_row), FALSE);
            g_object_set_data(G_OBJECT(row->list_row), "session-row", row);
            row->card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
            gtk_style_context_add_class(gtk_widget_get_style_context(row->card), "history-entry");
            gtk_container_add(GTK_CONTAINER(row->list_row), row->card);
            row->preview = transcript_label(); pack(row->card, row->preview);
            gtk_label_set_lines(GTK_LABEL(row->preview), 4);
            gtk_label_set_ellipsize(GTK_LABEL(row->preview), PANGO_ELLIPSIZE_END);
            row->details = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
            gtk_widget_set_no_show_all(row->details, TRUE);
            row->edited = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
            pack(row->details, row->edited); pack(row->card, row->details);
            row->progress = label("", "secondary"); gtk_label_set_line_wrap(GTK_LABEL(row->progress), TRUE);
            gtk_widget_set_no_show_all(row->progress, TRUE); pack(row->card, row->progress);
            auto* bottom = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
            gtk_style_context_add_class(gtk_widget_get_style_context(bottom), "history-actions");
            row->metadata = label("", "secondary"); gtk_label_set_ellipsize(GTK_LABEL(row->metadata), PANGO_ELLIPSIZE_END); pack(bottom, row->metadata, true);
            row->expand = icon_button("go-next-symbolic", "Open transcript");
            g_signal_connect(row->expand, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) { auto& r = *static_cast<Row*>(p); r.app->expand(r); }), row);
            pack(bottom, row->expand);
            row->retry = icon_button("view-refresh-symbolic", "Retry failed transcription");
            gtk_widget_set_no_show_all(row->retry, TRUE);
            g_signal_connect(row->retry, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) {
                auto& r = *static_cast<Row*>(p); r.app->auto_copy_id_.clear();
                r.app->backend_->session_request("retry", "POST", "/sessions/" + r.data.id + "/retry", "{}", r.data.id);
            }), row); pack(bottom, row->retry);
            row->more = gtk_menu_button_new();
            gtk_style_context_add_class(gtk_widget_get_style_context(row->more), "flat");
            gtk_button_set_image(GTK_BUTTON(row->more), gtk_image_new_from_icon_name("view-more-symbolic", GTK_ICON_SIZE_BUTTON));
            gtk_widget_set_tooltip_text(row->more, "Recording options");
            auto* options = gtk_popover_new(row->more);
            auto* reprocess = gtk_button_new_with_label("Transcribe again from saved audio");
            gtk_style_context_add_class(gtk_widget_get_style_context(reprocess), "flat");
            gtk_widget_set_margin_top(reprocess, 4); gtk_widget_set_margin_bottom(reprocess, 4);
            gtk_container_add(GTK_CONTAINER(options), reprocess); gtk_widget_show(reprocess);
            gtk_menu_button_set_popover(GTK_MENU_BUTTON(row->more), options);
            g_signal_connect(reprocess, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) {
                auto& r = *static_cast<Row*>(p); r.app->auto_copy_id_.clear();
                gtk_popover_popdown(gtk_menu_button_get_popover(GTK_MENU_BUTTON(r.more)));
                r.app->backend_->session_request("reprocess", "POST", "/sessions/" + r.data.id + "/reprocess", "{\"audio\":true,\"silence_filter\":false}", r.data.id);
            }), row); pack(bottom, row->more);
            row->remove = icon_button("user-trash-symbolic", "Delete recording and transcription");
            gtk_actionable_set_action_target_value(GTK_ACTIONABLE(row->remove), g_variant_new_string(id.c_str()));
            gtk_actionable_set_action_name(GTK_ACTIONABLE(row->remove), "app.delete-recording"); pack(bottom, row->remove);
            row->copy = icon_button("edit-copy-symbolic", "Copy current transcript");
            g_signal_connect(row->copy, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) {
                auto& r = *static_cast<Row*>(p);
                r.app->backend_->session_request("copy", "GET", "/sessions/" + r.data.id + "/transcript?view=edited", "{}", r.data.id);
            }), row); pack(bottom, row->copy); pack(row->card, bottom);
            gtk_container_add(GTK_CONTAINER(history_box_), row->list_row); gtk_widget_show_all(row->list_row);
            gtk_widget_show_all(row->details); gtk_widget_hide(row->details);
            if (empty_history_) { gtk_widget_destroy(empty_history_); empty_history_ = nullptr; }
            it = rows_.find(id);
        }
        auto& row = *it->second;
        row.data.update(object);
        update_selected_label(row.preview, row.data.preview.empty() ? (row.data.capture == "recording" ? "Listening…" : "Open to inspect transcript") : row.data.preview);
        set_notice(row.progress, row.data.attention());
        auto stamp = session_stamp(row.data.time);
        g_object_set_data_full(G_OBJECT(row.list_row), "day", g_strdup(stamp.day.c_str()), g_free);
        g_object_set_data_full(G_OBJECT(row.list_row), "day-heading", g_strdup(stamp.heading.c_str()), g_free);
        set_label(row.metadata, stamp.time + (row.data.source == "import" ? " · Imported audio" : "") + " · " + duration(row.data.duration));
        gtk_widget_set_sensitive(row.remove, row.data.capture != "recording" && row.data.capture != "paused");
        gtk_widget_set_sensitive(row.more, row.data.capture != "recording" && row.data.capture != "paused");
        gtk_widget_set_visible(row.retry, row.data.transcription == "failed" || row.data.transcription == "incomplete");
        gtk_widget_set_sensitive(row.copy, row.data.transcription == "complete" || row.data.transcribed > 0 || row.data.words > 0);
        gtk_list_box_invalidate_sort(GTK_LIST_BOX(history_box_));
        gtk_list_box_invalidate_headers(GTK_LIST_BOX(history_box_));
        update_stats();
        return row;
    }
    void update_stats() {
        int64_t words = 0; double seconds = 0;
        for (const auto& [id, row] : rows_) if (row->data.transcription == "complete" && row->data.words > 0) {
            words += row->data.words; seconds += row->data.duration;
        }
        auto summary = take_string(g_strdup_printf("%'lld words · %'lld WPM", static_cast<long long>(have_history_stats_ ? history_words_ : words),
            static_cast<long long>(have_history_stats_ ? history_wpm_ : seconds > 0 ? std::llround(words * 60 / seconds) : 0)));
        set_label(stats_, summary);
        gtk_widget_set_tooltip_text(stats_, "Saved history. WPM uses full recording duration, including silence.");
    }
    void clear_segments(Row& row) {
        if(spelling_&&spelling_->session==row.data.id)close_spelling();
        for (auto* box : {row.edited}) {
            auto* children = gtk_container_get_children(GTK_CONTAINER(box));
            for (auto* c = children; c; c = c->next) gtk_widget_destroy(GTK_WIDGET(c->data));
            g_list_free(children);
        }
        row.segments.clear(); row.cursor = 0; row.details_loaded = false;
    }
    static std::string original_at_point(GtkWidget* widget, int x, int y) {
        auto* highlights = static_cast<std::vector<ReplacementHighlight>*>(g_object_get_data(G_OBJECT(widget), "replacement-highlights"));
        if (!highlights || highlights->empty()) return {};
        int left, top, index, trailing;
        label_layout_offsets(widget, &left, &top);
        if (!pango_layout_xy_to_index(gtk_label_get_layout(GTK_LABEL(widget)), (x-left)*PANGO_SCALE, (y-top)*PANGO_SCALE, &index, &trailing)) return {};
        for (const auto& highlight : *highlights)
            if (index >= 0 && size_t(index) >= highlight.start && size_t(index) < highlight.end) return highlight.original;
        return {};
    }
    void render_segment(SegmentRow& segment, bool enabled) {
        int start, end;
        if ((spelling_&&spelling_->anchor==segment.edited) ||
            (gtk_label_get_selection_bounds(GTK_LABEL(segment.edited), &start, &end) && start != end)) {
            g_object_set_data(G_OBJECT(segment.edited), "pending-render", GINT_TO_POINTER(1)); return;
        }
        const auto& data = segment.data;
        std::string gap = data.state == "failed" || data.state == "gap" ? "Transcription unavailable: " + data.error :
            data.state == "pending" || data.state == "running" ? "Transcribing…" : "";
        segment.rendered = gap.empty() ? replacement_highlights(data, enabled) : ReplacementHighlights{gap, {}, true};
        set_label(segment.edited, segment.rendered.text);
        auto* attrs = pango_attr_list_new();
        GdkRGBA base{}, accent{};
        auto* style = gtk_widget_get_style_context(segment.edited);
        if (!gtk_style_context_lookup_color(style, "theme_base_color", &base)) gdk_rgba_parse(&base, "#242424");
        gdk_rgba_parse(&accent, "#d6ad48");
        auto channel = [](double value) { return guint16(std::clamp(value, 0.0, 1.0) * 65535); };
        for (const auto& h : segment.rendered.highlights) {
            auto* background = pango_attr_background_new(channel(base.red*.78+accent.red*.22),
                channel(base.green*.78+accent.green*.22), channel(base.blue*.78+accent.blue*.22));
            background->start_index = h.start; background->end_index = h.end; pango_attr_list_insert(attrs, background);
            auto* underline = pango_attr_underline_new(PANGO_UNDERLINE_SINGLE);
            underline->start_index = h.start; underline->end_index = h.end; pango_attr_list_insert(attrs, underline);
            auto* color = pango_attr_underline_color_new(channel(accent.red), channel(accent.green), channel(accent.blue));
            color->start_index = h.start; color->end_index = h.end; pango_attr_list_insert(attrs, color);
        }
        gtk_label_set_attributes(GTK_LABEL(segment.edited), attrs); pango_attr_list_unref(attrs);
        g_object_set_data_full(G_OBJECT(segment.edited), "replacement-highlights",
            new std::vector<ReplacementHighlight>(segment.rendered.highlights), +[](gpointer p) { delete static_cast<std::vector<ReplacementHighlight>*>(p); });
        gtk_widget_set_has_tooltip(segment.edited, !segment.rendered.highlights.empty());
        g_object_set_data(G_OBJECT(segment.edited), "pending-render", nullptr);
    }
    void update_segment(Row& row, SessionSegment value) {
        if (row.data.id != selected_id_) return;
        auto found = row.segments.find(value.sequence);
        if (found == row.segments.end()) {
            auto segment = std::make_unique<SegmentRow>();
            segment->edited = transcript_label();
            gtk_label_set_selectable(GTK_LABEL(segment->edited), TRUE);
            g_signal_connect(segment->edited,"button-release-event",G_CALLBACK(+[](GtkWidget* widget,GdkEventButton* event,gpointer p)->gboolean {
                if(event->button==1)static_cast<App*>(p)->queue_spelling(widget);
                return FALSE;
            }),this);
            g_signal_connect(segment->edited,"key-release-event",G_CALLBACK(+[](GtkWidget* widget,GdkEventKey* event,gpointer p)->gboolean {
                if(event->keyval==GDK_KEY_Shift_L || event->keyval==GDK_KEY_Shift_R ||
                   ((event->state&GDK_CONTROL_MASK)&&(event->keyval==GDK_KEY_a||event->keyval==GDK_KEY_A)))
                    static_cast<App*>(p)->queue_spelling(widget);
                return FALSE;
            }),this);
            g_signal_connect(segment->edited,"populate-popup",G_CALLBACK(+[](GtkLabel* label,GtkMenu* menu,gpointer p){
                int start=0;int end=0;if(!gtk_label_get_selection_bounds(label,&start,&end)||start==end)return;
                auto* item=gtk_menu_item_new_with_label("Save spelling…");
                g_object_set_data_full(G_OBJECT(item),"transcript",g_object_ref(label),g_object_unref);
                g_signal_connect(item,"activate",G_CALLBACK(+[](GtkMenuItem* item,gpointer p){
                    static_cast<App*>(p)->queue_spelling(GTK_WIDGET(g_object_get_data(G_OBJECT(item),"transcript")));
                }),p);
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),item);gtk_widget_show(item);
            }),this);
            g_signal_connect(segment->edited, "query-tooltip", G_CALLBACK(+[](GtkWidget* widget, gint x, gint y, gboolean keyboard, GtkTooltip* tooltip, gpointer) -> gboolean {
                auto original = original_at_point(widget, x, y);
                if (keyboard || original.empty()) return FALSE;
                gtk_tooltip_set_text(tooltip, ("Original: " + original).c_str()); return TRUE;
            }), nullptr);
            segment->edited_section = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
            pack(segment->edited_section, segment->edited);
            segment->status = label("", "section-status");
            gtk_label_set_line_wrap(GTK_LABEL(segment->status), TRUE);
            pack(segment->edited_section, segment->status);
            pack(row.edited, segment->edited_section);
            found = row.segments.emplace(value.sequence, std::move(segment)).first;
            gtk_box_reorder_child(GTK_BOX(row.edited), found->second->edited_section, std::distance(row.segments.begin(), found));
        }
        auto& segment = *found->second;
        segment.data = std::move(value);
        render_segment(segment, row.data.replacements_enabled);
        set_label(segment.status, duration(segment.data.start / 16000.0) + "–" + duration(segment.data.end / 16000.0));
        gtk_widget_show_all(segment.edited_section);
        while (row.segments.size() > 50) {
            auto oldest = row.segments.begin(); gtk_widget_destroy(oldest->second->edited_section); row.segments.erase(oldest);
        }
    }
    void apply_updates(Row& row, JsonObject* root, bool explicit_page) {
        auto generation = json_number(root, "generation", row.data.generation);
        bool reset = json_flag(root, "reset");
        if (reset || (row.data.generation && generation != row.data.generation)) clear_segments(row);
        if (auto* metadata = json_object(root, "session")) row.data.update(metadata);
        row.data.generation = generation;
        if (row.data.id != selected_id_ || (!explicit_page && !follow_latest_)) {
            auto* segments = json_array(root, "segments");
            if (row.data.id == selected_id_ && (reset || (segments && json_array_get_length(segments)))) pending_pane_updates_ = true;
            maybe_deliver(row); return;
        }
        if (auto* segments = json_array(root, "segments")) for (guint i = 0; i < json_array_get_length(segments); ++i) {
            auto section = SessionSegment::parse(json_array_get_object_element(segments, i));
            if (paging_older_ && (section.sequence < pane_page_start_ || section.sequence >= pane_page_start_ + 50)) {
                if (section.sequence >= pane_page_start_ + 50) row.has_later_sections = true;
                continue;
            }
            update_segment(row, std::move(section));
        }
        row.cursor = reset ? 0 : json_number(root, "cursor", row.cursor); row.details_loaded = true;
        bool more_updates = json_flag(root, "has_more");
        row.has_more = paging_older_ ? row.has_later_sections : more_updates;
        set_notice(row.progress, row.data.attention());
        // Change cursors are revision order, not audio order. Older pages scan
        // bounded deltas and retain only their requested source sequence range,
        // so a recently edited early section cannot disappear from its page.
        if (more_updates || reset) load_details(row);
        maybe_deliver(row);
    }
    void maybe_deliver(Row& row) {
        if (row.data.id == auto_copy_id_ && row.data.delivery_epoch != auto_copy_epoch_) {
            auto_copy_id_.clear(); auto_copy_publication_.clear();
        }
        if (row.data.id == auto_copy_id_ && (row.data.transcription == "failed" || row.data.transcription == "incomplete" || row.data.capture == "interrupted" || row.data.capture == "cancelled")) {
            auto_copy_id_.clear(); auto_copy_publication_.clear();
        }
        if (row.data.id == auto_copy_id_ && row.data.publication_complete && auto_copy_publication_.empty()) {
            auto_copy_publication_ = row.data.publication_id;
            backend_->session_request("auto-copy", "GET", "/sessions/" + row.data.id + "/transcript?view=publication", "{}", row.data.id);
        }
        if (row.data.id != active_id_) return;
        if (row.data.capture == "recording" || row.data.capture == "paused") {
            if (mode_ == Mode::Idle) show_recording();
            if (mode_ != Mode::Transcribing) mode_ = row.data.capture == "paused" ? Mode::Paused : Mode::Recording;
            return;
        }
        reset_recording_button();
        if (row.data.transcription == "complete" || row.data.transcription == "failed" || row.data.transcription == "incomplete" ||
            row.data.capture == "cancelled" || row.data.capture == "interrupted") {
            bool announce = mode_ != Mode::Idle || terminal_session_id_ != row.data.id;
            mode_ = Mode::Idle;
            terminal_session_id_ = row.data.id;
            if (announce) {
                const auto notice = row.data.attention();
                if (!notice.empty()) overlay_error(notice);
                else { gtk_widget_hide(overlay_); overlay_until_ = 0; }
                if (!list_pending_) refresh_history();
            }
        } else mode_ = Mode::Transcribing;
    }
    void handle_session_event(const SessionEvent& event) {
        if (dictionary_->handle_event(event)) return;
        const bool spelling_event=event.operation.rfind("spelling:",0)==0;
        if (event.operation == "history" || event.operation == "more-history") list_pending_ = false;
        if (event.operation == "replacements") replacements_pending_ = false;
        if (event.operation == "pause" || event.operation == "resume") capture_command_pending_ = false;
        if (event.operation == "details") if (auto it = rows_.find(event.id); it != rows_.end()) it->second->detail_pending = false;
        if (!event.error.empty()) {
            if(spelling_event) {
                if(spelling_&&spelling_->operation==event.operation) {
                    spelling_->operation.clear();gtk_widget_set_sensitive(spelling_->entry,TRUE);
                    gtk_widget_set_sensitive(spelling_->save,TRUE);set_label(spelling_->error,event.error);
                    gtk_widget_show(spelling_->error);gtk_widget_grab_focus(spelling_->entry);
                } else if(selected_id_==event.id)set_notice(pane_feedback_,event.error);
                return;
            }
            if (event.operation == "start") { starting_session_ = false; mode_ = Mode::Idle; quitting_ = false; overlay_error(event.error); }
            else if (event.operation == "stop" || event.operation == "quit-stop") {
                auto_copy_id_.clear(); quitting_ = false;
                // A rejected Stop must leave the live controls available. The
                // next authoritative update can still report a completed Stop.
                if (auto it = rows_.find(active_id_); it != rows_.end()) {
                    if (it->second->data.capture == "paused") mode_ = Mode::Paused;
                    else if (it->second->data.capture == "recording") mode_ = Mode::Recording;
                }
                set_notice(pane_feedback_, event.error); update_pane(); overlay_error(event.error);
            }
            else if (event.operation != "history" && event.operation != "more-history") message(event.error);
            if (event.operation == "replacements") if (auto it = rows_.find(event.id); it != rows_.end()) {
                it->second->cursor = 0; load_details(*it->second);
            }
            return;
        }
        if (event.operation == "copy" || event.operation == "copy-original" || event.operation == "auto-copy") {
            bool automatic = event.operation == "auto-copy";
            if (automatic && (event.id != auto_copy_id_ || quitting_)) return;
            if (automatic) {
                auto row = rows_.find(event.id);
                if (row == rows_.end() || row->second->data.delivery_epoch != auto_copy_epoch_) return;
            }
            if (!event.body.empty()) {
                gtk_clipboard_set_text(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), event.body.c_str(), event.body.size());
                gtk_clipboard_set_can_store(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), nullptr, 0);
            }
            if (automatic) {
                auto_copy_id_.clear(); mode_ = Mode::Idle;
                if (event.body.empty()) overlay_error("No speech was transcribed. Check the microphone and try again.");
                else { gtk_widget_hide(overlay_); overlay_until_ = 0; }
            }
            if (selected_id_ == event.id) {
                auto row = rows_.find(event.id); bool partial = row != rows_.end() && row->second->data.transcription != "complete";
                set_notice(pane_feedback_, event.body.empty() ? "No speech was transcribed. Check the audio and try again." :
                    partial ? "Transcript is incomplete. Copy again when it finishes." : "");
            }
            return;
        }
        if (event.operation == "delete") {
            if (selected_id_ == event.id) {
                if (auto it = rows_.find(event.id); it != rows_.end()) move_widget(it->second->details, it->second->card);
                selected_id_.clear(); update_pane();
            }
            if (auto it = rows_.find(event.id); it != rows_.end()) { gtk_widget_destroy(it->second->list_row); rows_.erase(it); }
            if (active_id_ == event.id) { active_id_.clear(); backend_->watch_session(""); mode_ = Mode::Idle; }
            update_stats(); return;
        }
        if (event.operation == "quit-stop") { g_application_quit(G_APPLICATION(application_)); return; }
        try {
            Json parsed(event.body); auto* root = parsed.object();
            if (event.operation == "history" || event.operation == "more-history") {
                bool first_history = !history_loaded_;
                history_loaded_ = true;
                if (auto* stats = json_object(root, "stats")) {
                    have_history_stats_ = true; history_words_ = json_number(stats, "words"); history_wpm_ = json_number(stats, "words_per_minute");
                    update_stats();
                }
                if (auto* sessions = json_array(root, "sessions")) for (guint i = 0; i < json_array_get_length(sessions); ++i)
                    { auto& row = ensure_row(json_array_get_object_element(sessions, i)); if (row.expanded && row.data.id != active_id_ && follow_latest_ && !paging_older_) load_details(row); }
                auto next = json_text(root, "next_cursor");
                if (next.empty() && json_number(root, "next_cursor", -1) >= 0) next = std::to_string(json_number(root, "next_cursor"));
                bool more_history = json_flag(root, "has_more");
                if (!more_history) next_history_cursor_.clear();
                else if (first_history || event.operation == "more-history") next_history_cursor_ = next;
                if (!load_more_) {
                    load_more_ = gtk_button_new_with_label("Load older recordings");
                    g_signal_connect(load_more_, "clicked", G_CALLBACK(+[](GtkButton*, gpointer p) {
                        auto* a = static_cast<App*>(p); a->refresh_history(a->next_history_cursor_);
                    }), this);
                    gtk_container_add(GTK_CONTAINER(history_box_), load_more_); gtk_widget_show(load_more_);
                    gtk_widget_set_no_show_all(gtk_widget_get_parent(load_more_), TRUE);
                }
                gtk_widget_set_visible(gtk_widget_get_parent(load_more_), !next_history_cursor_.empty());
                if (rows_.empty() && !empty_history_) {
                    empty_history_ = gtk_list_box_row_new();
                    auto* text = label("Your transcriptions will appear here.", "secondary");
                    gtk_widget_set_margin_top(text, 48); gtk_widget_set_margin_bottom(text, 48);
                    gtk_label_set_xalign(GTK_LABEL(text), 0.5); gtk_container_add(GTK_CONTAINER(empty_history_), text);
                    gtk_container_add(GTK_CONTAINER(history_box_), empty_history_); gtk_widget_show_all(empty_history_);
                }
                return;
            }
            auto* metadata = json_object(root, "session");
            if (!metadata && !json_text(root, "id").empty()) metadata = root;
            Row* row = nullptr;
            if (metadata) row = &ensure_row(metadata);
            else if (auto it = rows_.find(event.id); it != rows_.end()) row = it->second.get();
            if (!row) return;
            if (selected_id_.empty()) select_session(*row);
            if (event.operation == "start") {
                starting_session_ = false; active_id_ = row->data.id; auto_copy_id_.clear(); backend_->watch_session(active_id_); show_recording();
                if (!row->expanded) expand(*row);
                if (quitting_) quit();
                else if (cancel_after_start_) { cancel_after_start_ = false; cancel(); }
            } else if (event.operation == "import") {
                if (active_id_.empty() || (mode_ != Mode::Recording && mode_ != Mode::Paused)) { active_id_ = row->data.id; backend_->watch_session(active_id_); }
                if (!row->expanded) expand(*row);
            } else if (event.operation == "details" || event.operation == "update") apply_updates(*row, root, event.operation == "details");
            else if (event.operation == "cancel") {
                if (row->data.id == active_id_) { active_id_.clear(); backend_->watch_session(""); mode_ = Mode::Idle; reset_recording_button(); }
            } else if (event.operation == "reprocess") { clear_segments(*row); load_details(*row); }
            else if (event.operation == "replacements") {
                // An explicit toggle refreshes this page, while background
                // updates still preserve selected text and reading position.
                row->cursor = 0; load_details(*row);
            } else if(spelling_event) {
                if(spelling_&&spelling_->operation==event.operation) {
                    gtk_label_select_region(GTK_LABEL(spelling_->anchor),0,0);close_spelling();
                }
                row->cursor=0;load_details(*row);
                if(selected_id_==event.id)set_notice(pane_feedback_, "");
            } else if (event.operation == "retry") load_details(*row);
            maybe_deliver(*row);
        } catch (const std::exception& error) { message(std::string("Invalid session response: ") + error.what()); }
    }
    void remove_entry(const std::string& id) {
        if (auto it = rows_.find(id); it != rows_.end() && it->second->data.capture != "recording" && it->second->data.capture != "paused") {
            if (auto_copy_id_ == id) auto_copy_id_.clear();
            backend_->session_request("delete", "DELETE", "/sessions/" + id, "{}", id);
        }
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
        {"toggle", 0, 0, G_OPTION_ARG_NONE, nullptr, "Start or stop recording and copy the saved transcript", nullptr},
        {"cancel", 0, 0, G_OPTION_ARG_NONE, nullptr, "Cancel recording or pending clipboard delivery", nullptr},
        {"quit", 0, 0, G_OPTION_ARG_NONE, nullptr, "Quit the app (service stays available)", nullptr},
        {"install-shortcut", 0, 0, G_OPTION_ARG_NONE, nullptr, "Enable the GNOME Ctrl+Alt+Space shortcut", nullptr},
        {"install-desktop", 0, 0, G_OPTION_ARG_NONE, nullptr, "Add a user application launcher and the dictation shortcut", nullptr},
        {"service-url", 0, 0, G_OPTION_ARG_STRING, nullptr, "Paraketto API (default: discover the local service)", "URL"},
        {"data-dir", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "History directory", "DIRECTORY"},
        {"audio-source", 0, 0, G_OPTION_ARG_STRING, nullptr, "GStreamer microphone source (default: pulsesrc)", "SOURCE"},
        {"transcribe", 0, 0, G_OPTION_ARG_FILENAME, nullptr, "Transcribe a WAV into history", "FILE"},
        {nullptr, 0, 0, G_OPTION_ARG_NONE, nullptr, nullptr, nullptr}
    };
    g_application_add_main_option_entries(G_APPLICATION(application), options);
    std::unique_ptr<desktop::App> app;
    g_signal_connect(application, "command-line", G_CALLBACK(+[](GApplication* application, GApplicationCommandLine* command, gpointer data) -> gint {
        auto& app = *static_cast<std::unique_ptr<desktop::App>*>(data);
        auto* dict = g_application_command_line_get_options_dict(command);
        if (g_variant_dict_contains(dict, "quit")) { if (app) app->quit(); else g_application_quit(application); return 0; }
        if (g_variant_dict_contains(dict, "install-shortcut") || g_variant_dict_contains(dict, "install-desktop")) {
            std::string error;
            if (!desktop::install_shortcut(desktop::executable_path(), "<Control><Alt>space", error)) {
                g_application_command_line_printerr(command, "%s\n", error.c_str()); return 1;
            }
            if (g_variant_dict_contains(dict, "install-desktop")) {
                try { desktop::install_launcher(); }
                catch (const std::exception& e) { g_application_command_line_printerr(command, "%s\n", e.what()); return 1; }
            }
            if (app) app->refresh_shortcut();
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
    auto signal_quit = +[](gpointer p) -> gboolean {
        auto& app = *static_cast<std::unique_ptr<desktop::App>*>(p);
        if (app) app->quit();
        return G_SOURCE_CONTINUE;
    };
    guint sigint = g_unix_signal_add(SIGINT, signal_quit, &app);
    guint sigterm = g_unix_signal_add(SIGTERM, signal_quit, &app);
    int status = g_application_run(G_APPLICATION(application), argc, argv);
    g_source_remove(sigint); g_source_remove(sigterm);
    app.reset();
    g_object_unref(application);
    return status;
}
