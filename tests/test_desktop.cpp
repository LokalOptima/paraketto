// Integration checks run under a private X server and private IBus session.
#include "desktop/audio.h"
#include "desktop/backend.h"
#include "service/worker.h"
#include "app_common.h"
#include "desktop/history.h"
#include "desktop/input.h"
#include "wav.h"
#include <gdk/gdkx.h>
#include <X11/Xatom.h>
#include <functional>
#include <glib/gstdio.h>
#include <signal.h>

using namespace desktop;
using namespace paraketto::app;

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

static void ibus_delivery() {
    auto* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    auto* field = gtk_entry_new();
    gtk_container_add(GTK_CONTAINER(window), field);
    gtk_widget_show_all(window);
    auto* display = gdk_x11_display_get_xdisplay(gdk_display_get_default());
    Window xid = gdk_x11_window_get_xid(gtk_widget_get_window(window));
    XSetInputFocus(display, xid, RevertToParent, CurrentTime);
    auto set_active = [&](Window target) {
        XChangeProperty(display, DefaultRootWindow(display), XInternAtom(display, "_NET_ACTIVE_WINDOW", False),
            XA_WINDOW, 32, PropModeReplace, reinterpret_cast<unsigned char*>(&target), 1);
        XSync(display, False);
    };
    set_active(xid);
    ibus_init();
    auto* bus = ibus_bus_new();
    g_assert_true(ibus_bus_is_connected(bus));
    auto* context = ibus_bus_create_input_context(bus, "gtk3-im:paraketto-test");
    g_assert_nonnull(context);
    ibus_input_context_set_capabilities(context, IBUS_CAP_FOCUS | IBUS_CAP_PREEDIT_TEXT);
    std::string committed;
    g_signal_connect(context, "commit-text", G_CALLBACK(+[](IBusInputContext*, IBusText* text, gpointer p) {
        *static_cast<std::string*>(p) += ibus_text_get_text(text);
    }), &committed);
    {
        TextInput input;
        gtk_clipboard_set_text(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD), "keep this clipboard", -1);
        ibus_input_context_focus_in(context);
        input.begin(true);
        g_assert_true(until([&] { return input.native_target() && input.settled(); }));
        auto delivered = input.deliver("Caffè — 日本語");
        g_test_message("Delivery: %s", delivered.c_str());
        g_assert_true(until([&] { return !committed.empty(); }));
        g_assert_cmpstr(committed.c_str(), ==, "Caffè — 日本語");
        auto clipboard = take_string(gtk_clipboard_wait_for_text(gtk_clipboard_get(GDK_SELECTION_CLIPBOARD)));
        g_assert_cmpstr(clipboard.c_str(), ==, "keep this clipboard");

        g_assert_true(until([&] {
            auto* engine = ibus_bus_get_global_engine(bus);
            bool restored = engine && g_strcmp0(ibus_engine_desc_get_name(engine), "paraketto-dictation");
            if (engine) g_object_unref(engine);
            return restored;
        }));
        committed.clear();
        // A shortcut's keyboard grab temporarily removes the real IBus field.
        // Delivery must wait for that same field to return, then work normally.
        input.begin(true);
        g_assert_true(until([&] { return input.native_target() && input.settled(); }));
        ibus_input_context_focus_out(context);
        g_assert_true(until([&] { return input.changed(); }));
        g_assert_false(input.settled());
        ibus_input_context_focus_in(context);
        g_assert_true(until([&] { return !input.changed() && input.settled(); }));
        input.deliver("After shortcut");
        g_assert_true(until([&] { return committed == "After shortcut"; }));
        committed.clear();

        input.begin(true);
        g_assert_true(until([&] { return input.native_target() && input.settled(); }));
        ibus_input_context_focus_out(context);
        g_assert_true(until([&] { return input.changed(); }));
        auto rejected = input.deliver("must not be inserted");
        g_assert_true(rejected.find("Focus changed") != std::string::npos);
        g_assert_false(until([&] { return !committed.empty(); }, 150));

        // Moving to a different real field remains invalid even if the user
        // returns to the original field before recognition completes.
        auto* other = ibus_bus_create_input_context(bus, "gtk3-im:other-field");
        ibus_input_context_set_capabilities(other, IBUS_CAP_FOCUS | IBUS_CAP_PREEDIT_TEXT);
        ibus_input_context_focus_in(context);
        input.begin(true);
        g_assert_true(until([&] { return input.native_target() && input.settled(); }));
        ibus_input_context_focus_in(other);
        g_assert_true(until([&] { return input.changed() && input.settled(); }));
        ibus_input_context_focus_in(context);
        g_assert_true(until([&] { return input.settled(); }));
        g_assert_true(input.changed());
        rejected = input.deliver("must not follow the focus");
        g_assert_true(rejected.find("Focus changed") != std::string::npos);
        g_assert_false(until([&] { return !committed.empty(); }, 150));
        g_object_unref(other);

        // A rapid cancellation must restore the user's engine even if activation
        // was still waiting for the factory's asynchronous response.
        ibus_input_context_focus_in(context);
        input.begin(true); input.end();
        g_assert_true(until([&] { return input.settled(); }));
    }
    ibus_input_context_focus_out(context);
    g_object_unref(context); g_object_unref(bus);
    gtk_widget_destroy(window);
}

int main(int argc, char** argv) {
    gtk_init(&argc, &argv);
    gst_init(nullptr, nullptr);
    g_test_init(&argc, &argv, nullptr);
    g_test_add_func("/desktop/recording-history", recorder_history);
    g_test_add_func("/desktop/ibus-delivery", ibus_delivery);
    g_test_add_func("/desktop/backend-failure-shutdown", backend_failure_and_shutdown);
    return g_test_run();
}
