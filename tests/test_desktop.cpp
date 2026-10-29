// Integration checks run under a private X server and D-Bus session.
#include "desktop/audio.h"
#include "desktop/backend.h"
#include "service/worker.h"
#include "app_common.h"
#include "desktop/history.h"
#include "desktop/transcript.h"
#include "desktop/input.h"
#include "wav.h"
#include <gdk/gdkx.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <functional>
#include <glib/gstdio.h>
#include <signal.h>

using namespace desktop;
using namespace paraketto::app;

static void transcript_previews() {
    auto* label = gtk_label_new(nullptr);
    g_object_ref_sink(label);
    const std::string four = "First line\nSecond line\nThird line\nFourth line";
    auto exact = transcript_preview(label, four, 400);
    g_assert_false(exact.truncated);
    g_assert_cmpstr(exact.text.c_str(), ==, four.c_str());
    const std::string paragraphs = four + "\nFifth line\nSixth line";
    std::string unicode;
    for (int i = 0; i < 90; ++i) unicode += "Café 日本語 👩‍💻 ";
    for (const auto& source : {paragraphs, unicode}) {
        size_t narrower = 0;
        for (int width : {160, 400, 700}) {
            auto preview = transcript_preview(label, source, width);
            g_assert_true(preview.truncated);
            g_assert_true(g_utf8_validate(preview.text.c_str(), -1, nullptr));
            g_assert_true(g_str_has_suffix(preview.text.c_str(), "…"));
            auto prefix = preview.text.substr(0, preview.text.size() - strlen("…"));
            g_assert_true(source.compare(0, prefix.size(), prefix) == 0);
            g_assert_cmpuint(prefix.size(), >=, narrower);
            narrower = prefix.size();
            auto* layout = gtk_widget_create_pango_layout(label, preview.text.c_str());
            pango_layout_set_width(layout, width * PANGO_SCALE);
            pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
            g_assert_cmpint(pango_layout_get_line_count(layout), <=, 4);
            pango_layout_set_text(layout, source.c_str(), source.size());
            int count;
            auto* boundaries = pango_layout_get_log_attrs_readonly(layout, &count);
            auto boundary = g_utf8_strlen(prefix.c_str(), prefix.size());
            g_assert_true(boundaries[boundary].is_cursor_position);
            g_object_unref(layout);
        }
    }
    g_object_unref(label);
}

static bool until(const std::function<bool()>& condition, int milliseconds = 3000) {
    auto end = g_get_monotonic_time() + milliseconds * 1000LL;
    do {
        while (g_main_context_iteration(nullptr, FALSE)) {}
        if (condition()) return true;
        g_usleep(1000);
    } while (g_get_monotonic_time() < end);
    return false;
}

static void recorder_history() {
    Recorder recorder;
    std::string error;
    g_assert_true(recorder.start("audiotestsrc is-live=true wave=sine volume=0.3", error));
    g_assert_true(until([&] { return recorder.seconds() >= 0.25; }));
    g_assert_cmpfloat(recorder.rms(), >, 0.1);
    std::string wav = recorder.stop();
    auto decoded = paraketto::read_wav_from_memory(wav.data(), wav.size());
    g_assert_cmpint(decoded.sample_rate, ==, 16000);
    g_assert_cmpuint(decoded.samples.size(), >, 3000);

    auto directory = take_string(g_dir_make_tmp("paraketto-history-test-XXXXXX", nullptr));
    History history(directory);
    auto first = history.create(wav, "Text editor");
    first.text = "Caffè — árvíztűrő 日本語\nsecond line";
    first.delivery = "Inserted"; first.inference_ms = 42;
    history.save(first);
    auto second = history.create(wav, "Browser");
    auto loaded = history.load();
    g_assert_cmpuint(loaded.size(), ==, 2);
    g_assert_cmpstr(loaded[0].id.c_str(), ==, second.id.c_str());
    bool found = false;
    for (const auto& e : loaded) if (e.id == first.id) {
        g_assert_cmpstr(e.text.c_str(), ==, first.text.c_str()); found = true;
    }
    g_assert_true(found);
    g_assert_true(read_file(history.audio_path(first)) == wav);
    history.remove(first); history.remove(second);
    g_assert_true(history.load().empty());
    g_rmdir(directory.c_str());

    g_assert_false(recorder.start("not_a_real_audio_source", error));
    g_assert_false(error.empty());

    Entry short_recording, long_recording, failed_recording, unknown_duration;
    short_recording.text = "Don't split naïve words."; short_recording.duration = 2;
    long_recording.text = "Another four word message."; long_recording.duration = 6;
    failed_recording.text = "Ignore this failed result";
    failed_recording.error = "Failed"; failed_recording.duration = 40;
    auto stats = history_stats({short_recording, long_recording, failed_recording});
    g_assert_cmpuint(stats.words, ==, 8);
    g_assert_cmpint(stats.words_per_minute, ==, 60);
    unknown_duration.text = "Untimed words";
    stats = history_stats({short_recording, long_recording, unknown_duration});
    g_assert_cmpuint(stats.words, ==, 10);
    g_assert_cmpint(stats.words_per_minute, ==, 60);
    g_assert_cmpint(history_stats({}).words_per_minute, ==, 0);
}

static void backend_failure_and_shutdown() {
    using Backend = paraketto::service::Worker;
    auto directory = take_string(g_dir_make_tmp("paraketto-backend-test-XXXXXX", nullptr));
    {
        Backend backend("/does/not/exist", directory + "/missing-weights", directory + "/server.log");
        g_assert_true(until([&] { return !backend.state().starting; }));
        g_assert_false(backend.state().ready);
        g_assert_false(backend.state().message.empty());
    }
    write_file(directory + "/weights", "fixture");
    std::string pid_path = directory + "/pid";
    auto quoted = take_string(g_shell_quote(pid_path.c_str()));
    write_file(directory + "/server", "#!/bin/sh\nprintf '%s' \"$$\" > " + quoted + "\nexec sleep 30\n");
    g_chmod((directory + "/server").c_str(), 0700);
    int pid = 0;
    gint64 stopped = 0;
    {
        Backend backend(directory + "/server", directory + "/weights", directory + "/server.log");
        g_assert_true(until([&] { return g_file_test(pid_path.c_str(), G_FILE_TEST_IS_REGULAR); }));
        pid = std::stoi(read_file(pid_path));
        stopped = g_get_monotonic_time();
    }
    g_assert_cmpint(g_get_monotonic_time() - stopped, <, 2000000);
    g_assert_cmpint(kill(pid, 0), ==, -1);
    for (const char* name : {"weights", "server", "server.log", "pid"}) g_unlink((directory + "/" + name).c_str());
    g_rmdir(directory.c_str());
}

static void clipboard_delivery() {
    auto* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    auto* field = gtk_text_view_new();
    auto* buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(field));
    auto* clipboard = gtk_clipboard_get(GDK_SELECTION_CLIPBOARD);
    gtk_container_add(GTK_CONTAINER(window), field);
    gtk_widget_show_all(window);
    gtk_widget_grab_focus(field);
    auto* display = gdk_x11_display_get_xdisplay(gdk_display_get_default());
    Window xid = gdk_x11_window_get_xid(gtk_widget_get_window(window));
    auto set_active = [&](Window target) {
        XSetInputFocus(display, target, RevertToParent, CurrentTime);
        XChangeProperty(display, DefaultRootWindow(display), XInternAtom(display, "_NET_ACTIVE_WINDOW", False),
            XA_WINDOW, 32, PropModeReplace, reinterpret_cast<unsigned char*>(&target), 1);
        XSync(display, False);
    };
    auto contents = [&] {
        GtkTextIter start, end;
        gtk_text_buffer_get_bounds(buffer, &start, &end);
        return take_string(gtk_text_buffer_get_text(buffer, &start, &end, TRUE));
    };
    set_active(xid);
    g_assert_true(until([&] { return gtk_widget_has_focus(field); }));
    // A normal GTK editor uses its built-in Ctrl+V. The terminal fixture accepts
    // only Ctrl+Shift+V, so sending the wrong shortcut cannot pass this test.
    bool terminal = false;
    g_signal_connect(field, "key-press-event", G_CALLBACK(+[](GtkWidget* field, GdkEventKey* event, gpointer data) -> gboolean {
        if (!*static_cast<bool*>(data)) return FALSE;
        auto modifiers = event->state & gtk_accelerator_get_default_mod_mask();
        if (gdk_keyval_to_lower(event->keyval) != GDK_KEY_v) return FALSE;
        if (modifiers == (GDK_CONTROL_MASK | GDK_SHIFT_MASK))
            gtk_text_buffer_paste_clipboard(gtk_text_view_get_buffer(GTK_TEXT_VIEW(field)),
                gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), nullptr, TRUE);
        return TRUE;
    }), &terminal);
    std::string long_text;
    for (int i = 0; i < 500; ++i) long_text += "Caffè — árvíztűrő 日本語 👩‍💻\nAnother paragraph.\n";
    TextInput input;
    for (const char* window_class : {"paraketto-test-editor", "com.mitchellh.ghostty"}) {
        terminal = g_str_has_suffix(window_class, "ghostty");
        XClassHint hint{const_cast<char*>(window_class), const_cast<char*>(window_class)};
        XSetClassHint(display, xid, &hint);
        XSync(display, False);
        for (const auto& text : {std::string("Hallo, hallo."), std::string(378, 'a'), long_text}) {
            gtk_text_buffer_set_text(buffer, "", 0);
            input.begin(true);
            g_assert_cmpuint(input.target_window(), ==, xid);
            auto delivered = input.deliver(text);
            g_assert_true(g_str_has_prefix(delivered.c_str(), "Paste sent to "));
            g_assert_true(until([&] { return contents() == text; }));
            auto copied = take_string(gtk_clipboard_wait_for_text(clipboard));
            g_assert_cmpstr(copied.c_str(), ==, text.c_str());
        }
    }

    gtk_text_buffer_set_text(buffer, "", 0);
    gtk_clipboard_set_text(clipboard, "keep this clipboard", -1);
    input.begin(false);
    auto saved = input.deliver("history only");
    g_assert_cmpstr(saved.c_str(), ==, "Saved to history");
    input.begin(true);
    input.end();
    saved = input.deliver("cancelled");
    g_assert_cmpstr(saved.c_str(), ==, "Saved to history");

    // Switching windows invalidates delivery even if the user switches back.
    auto* other = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_widget_show_all(other);
    Window other_xid = gdk_x11_window_get_xid(gtk_widget_get_window(other));
    set_active(xid);
    input.begin(true);
    set_active(other_xid);
    input.poll();
    set_active(xid);
    input.poll();
    g_assert_true(input.changed());
    auto rejected = input.deliver("must not follow focus");
    g_assert_true(g_str_has_prefix(rejected.c_str(), "Focus changed"));
    g_assert_false(until([&] { return !contents().empty(); }, 150));
    auto copied = take_string(gtk_clipboard_wait_for_text(clipboard));
    g_assert_cmpstr(copied.c_str(), ==, "keep this clipboard");
    gtk_widget_destroy(other);
    gtk_widget_destroy(window);
}

int main(int argc, char** argv) {
    gtk_init(&argc, &argv);
    gst_init(nullptr, nullptr);
    g_test_init(&argc, &argv, nullptr);
    g_test_add_func("/desktop/recording-history", recorder_history);
    g_test_add_func("/desktop/transcript-previews", transcript_previews);
    g_test_add_func("/desktop/clipboard-delivery", clipboard_delivery);
    g_test_add_func("/desktop/backend-failure-shutdown", backend_failure_and_shutdown);
    return g_test_run();
}
