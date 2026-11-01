// Real GTK pane interaction, driven against a private CPU-only service.
#include "cpp-httplib/httplib.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#define main paraketto_desktop_application_main
#include "../src/desktop/app.cpp"
#undef main
#pragma GCC diagnostic pop
#include "service/session_store.h"
#include <filesystem>
#include <fstream>
#include <functional>

static bool wait_ui(const std::function<bool()>& check, int ms = 8000) {
    auto until = g_get_monotonic_time() + ms * 1000LL;
    while (g_get_monotonic_time() < until) {
        while (g_main_context_iteration(nullptr, FALSE)) {}
        if (check()) return true;
        g_usleep(10000);
    }
    return false;
}
static void seed(const std::string& data) {
    using namespace paraketto::service;
    SessionStore store(data);
    store.set_dictionary("{\"entries\":[{\"canonical\":\"Paraketto\",\"aliases\":[\"paraketo\"]}]}");
    auto session = store.create_session("pane-fixture", "import", "{\"silence_filter\":false}");
    auto directory = std::filesystem::path(data) / "sessions" / session.id / "audio";
    std::filesystem::create_directories(directory);
    for (int block = 0; block < 2; ++block) {
        auto path = directory / (block ? "000001.pcm" : "000000.pcm");
        { std::ofstream file(path, std::ios::binary); std::string second(32000, '\0'); for (int i = 0; i < 30; ++i) file.write(second.data(), second.size()); }
        store.put_audio_block({session.id,path.string(),block,block*30*16000,30*16000,true});
    }
    for (int i = 0; i < 60; ++i) {
        store.plan_segment(session.id,session.generation,i*16000,(i+1)*16000,{{i*16000,(i+1)*16000}});
        auto job = store.next_asr(); g_assert_true(job.has_value());
        store.commit_asr(*job,"Section " + std::to_string(i + 1) + ". Café notes: paraketo keeps the original wording; paraketo makes dictionary changes reversible.","fixture",1);
    }
    store.seal_capture(session.id); store.publish(session.id);
}
namespace desktop {
struct PaneTest {
    static void spin(int milliseconds = 120) {
        auto until = g_get_monotonic_time() + milliseconds * 1000LL;
        while (g_get_monotonic_time() < until) {
            while (g_main_context_iteration(nullptr, FALSE)) {}
            g_usleep(10000);
        }
    }
    static void failed_stop_restores_controls(App& app) {
        const auto previous = app.mode_;
        g_assert_true(previous == App::Mode::Recording || previous == App::Mode::Paused);
        app.auto_copy_id_ = app.active_id_;
        app.mode_ = App::Mode::Transcribing; app.update_pane();
        g_assert_cmpstr(gtk_stack_get_visible_child_name(GTK_STACK(app.transport_stack_)), ==, "finishing");
        app.handle_session_event({"stop", app.active_id_, {}, "Fixture rejected Stop", 503});
        g_assert_true(gtk_widget_get_visible(app.pane_feedback_));
        g_assert_cmpstr(gtk_label_get_text(GTK_LABEL(app.pane_feedback_)), ==, "Fixture rejected Stop");
        g_assert_true(app.mode_ == previous); g_assert_true(app.auto_copy_id_.empty());
        g_assert_cmpstr(gtk_stack_get_visible_child_name(GTK_STACK(app.transport_stack_)), ==, "recording");
        g_assert_true(gtk_widget_get_sensitive(app.pane_pause_)); g_assert_true(gtk_widget_get_sensitive(app.pane_stop_));
        // An ordinary metadata refresh must keep the recovered controls usable.
        app.maybe_deliver(*app.rows_.at(app.active_id_));
        g_assert_true(app.mode_ == previous);
    }
    static void screenshot(App& app, const char* name) {
        const char* directory = g_getenv("PARAKETTO_VISUAL_DIR");
        if (!directory || !*directory) return;
        std::filesystem::create_directories(directory);
        spin();
        auto* window = gtk_widget_get_window(app.window_);
        int root_x=0,root_y=0;gdk_window_get_origin(window,&root_x,&root_y);
        auto* pixels = gdk_pixbuf_get_from_window(gdk_get_default_root_window(), root_x, root_y,
            gdk_window_get_width(window), gdk_window_get_height(window));
        g_assert_nonnull(pixels);
        auto path = std::filesystem::path(directory) / (std::string(name) + ".png");
        GError* error = nullptr;
        g_assert_true(gdk_pixbuf_save(pixels, path.c_str(), "png", &error, nullptr));
        g_assert_no_error(error); g_object_unref(pixels);
    }
    static std::array<int,4> geometry(App& app, GtkWidget* widget) {
        int x = 0, y = 0;
        g_assert_true(gtk_widget_translate_coordinates(widget, app.window_, 0, 0, &x, &y));
        return {x,y,gtk_widget_get_allocated_width(widget),gtk_widget_get_allocated_height(widget)};
    }
    static std::string transcript(httplib::Client& client, const std::string& id) {
        auto response = client.Get("/sessions/" + id + "/transcript?view=edited");
        g_assert_true(response && response->status == 200); return response->body;
    }
    static Json metadata(httplib::Client& client, const std::string& id) {
        auto response = client.Get("/sessions/" + id);
        g_assert_true(response && response->status == 200); return Json(response->body);
    }
    static std::string hover(GtkWidget* label, size_t byte) {
        auto* layout = gtk_label_get_layout(GTK_LABEL(label));
        PangoRectangle box{}; pango_layout_index_to_pos(layout, byte, &box);
        int x = 0, y = 0; gtk_label_get_layout_offsets(GTK_LABEL(label), &x, &y);
        GtkAllocation allocation{};gtk_widget_get_allocation(label,&allocation);
        x-=allocation.x;y-=allocation.y;
        x += (box.x + box.width / 2) / PANGO_SCALE;
        y += (box.y + box.height / 2) / PANGO_SCALE;
        g_assert_cmpint(y,>=,0);g_assert_cmpint(y,<,allocation.height);
        return App::original_at_point(label, x, y);
    }
    static void spelling_popover(App& app,httplib::Client& client,App::Row& row) {
        app.select_session(row);gtk_button_clicked(GTK_BUTTON(app.pane_jump_));
        g_assert_true(wait_ui([&]{return row.segments.count(59)&&!row.detail_pending;}));
        auto& section=*row.segments.at(59);
        const auto raw=section.data.raw;
        auto publication=client.Get("/sessions/"+row.data.id+"/publication");
        g_assert_true(publication&&publication->status==200);const auto saved_publication=publication->body;
        auto select=[&](const std::string& word,bool keyboard=false,const char* trimmed=nullptr) {
            const auto text=std::string(gtk_label_get_text(GTK_LABEL(section.edited)));
            const auto byte=text.find(word);g_assert_cmpuint(byte,!=,std::string::npos);
            int start=g_utf8_pointer_to_offset(text.c_str(),text.c_str()+byte);
            gtk_label_select_region(GTK_LABEL(section.edited),start,start+g_utf8_strlen(word.data(),word.size()));
            auto* event=gdk_event_new(keyboard?GDK_KEY_RELEASE:GDK_BUTTON_RELEASE);
            auto* seat=gdk_display_get_default_seat(gdk_display_get_default());
            gboolean handled=FALSE;
            if(keyboard) {
                event->key.keyval=GDK_KEY_Shift_L;
                event->key.window=GDK_WINDOW(g_object_ref(gtk_widget_get_window(section.edited)));
                gdk_event_set_device(event,gdk_seat_get_keyboard(seat));
                g_signal_emit_by_name(section.edited,"key-release-event",&event->key,&handled);
            } else {
                event->button.button=1;
                event->button.window=GDK_WINDOW(g_object_ref(gtk_widget_get_window(section.edited)));
                gdk_event_set_device(event,gdk_seat_get_pointer(seat));
                g_signal_emit_by_name(section.edited,"button-release-event",&event->button,&handled);
            }
            gdk_event_free(event);
            bool visible=wait_ui([&]{return app.spelling_&&gtk_widget_get_mapped(app.spelling_->widget);});
            if(!visible && app.spelling_) {
                auto* popover=app.spelling_->widget;auto* parent=gtk_widget_get_parent(popover);
                GdkRectangle target{};gtk_popover_get_pointing_to(GTK_POPOVER(popover),&target);
                g_printerr("popover visible=%d child-visible=%d realized=%d mapped=%d parent=%s parent-mapped=%d anchor-mapped=%d target=%d,%d,%d,%d\n",
                    gtk_widget_get_visible(popover),gtk_widget_get_child_visible(popover),gtk_widget_get_realized(popover),gtk_widget_get_mapped(popover),
                    parent?G_OBJECT_TYPE_NAME(parent):"none",parent?gtk_widget_get_mapped(parent):0,gtk_widget_get_mapped(section.edited),target.x,target.y,target.width,target.height);
            }
            g_assert_true(visible);
            g_assert_cmpint(gtk_popover_get_position(GTK_POPOVER(app.spelling_->widget)),==,GTK_POS_TOP);
            GdkRectangle rect{};g_assert_true(gtk_popover_get_pointing_to(GTK_POPOVER(app.spelling_->widget),&rect));
            g_assert_cmpint(rect.width,>,0);g_assert_cmpint(rect.height,>,0);
            g_assert_cmpint(rect.y,>=,0);g_assert_cmpint(rect.y,<,gtk_widget_get_allocated_height(section.edited));
            g_assert_cmpstr(gtk_entry_get_text(GTK_ENTRY(app.spelling_->entry)),==,trimmed ? trimmed : word.c_str());
        };
        // Exercise GTK's dismissal signal, as used by Escape/outside-click,
        // before reopening through the keyboard-selection handler.
        select("wording");
        gtk_popover_popdown(GTK_POPOVER(app.spelling_->widget));
        g_assert_true(wait_ui([&]{return !app.spelling_;}));
        select("wording",true);
        gtk_entry_set_text(GTK_ENTRY(app.spelling_->entry),"phrasing");
        screenshot(app,"spelling-popover");
        gtk_button_clicked(GTK_BUTTON(app.spelling_->save));
        g_assert_true(wait_ui([&]{return !app.spelling_&&!row.detail_pending&&section.data.edited.find("phrasing")!=std::string::npos;}));
        g_assert_cmpstr(section.data.raw.c_str(),==,raw.c_str());
        auto dictionary=client.Get("/dictionary");g_assert_true(dictionary&&dictionary->status==200);
        g_assert_nonnull(strstr(dictionary->body.c_str(),"wording"));g_assert_nonnull(strstr(dictionary->body.c_str(),"phrasing"));
        publication=client.Get("/sessions/"+row.data.id+"/publication");
        g_assert_true(publication&&publication->body==saved_publication);
        g_assert_false(gtk_widget_get_visible(app.pane_feedback_));
        // Dragging through the surrounding spaces must learn just the phrase,
        // with its internal space and the original ASR bytes preserved.
        select(" the original ",false,"the original");
        g_assert_cmpstr(app.spelling_->selected.c_str(),==,"the original");
        gtk_entry_set_text(GTK_ENTRY(app.spelling_->entry),"the source");
        gtk_button_clicked(GTK_BUTTON(app.spelling_->save));
        g_assert_true(wait_ui([&]{return !app.spelling_&&!row.detail_pending&&section.data.edited.find("the source")!=std::string::npos;}));
        dictionary=client.Get("/dictionary");g_assert_true(dictionary&&dictionary->status==200);
        Json learned(dictionary->body);auto* entries=json_array(learned.object(),"entries");bool found=false;
        for(guint i=0;i<json_array_get_length(entries);++i) {
            auto* entry=json_array_get_object_element(entries,i);
            if(json_text(entry,"canonical")!="the source")continue;
            auto* aliases=json_array(entry,"aliases");g_assert_cmpuint(json_array_get_length(aliases),==,1);
            g_assert_cmpstr(json_array_get_string_element(aliases,0),==,"the original");found=true;
        }
        g_assert_true(found);g_assert_cmpstr(section.data.raw.c_str(),==,raw.c_str());
        g_assert_true(client.Get("/sessions/"+row.data.id+"/publication")->body==saved_publication);
        g_assert_false(gtk_widget_get_visible(app.pane_feedback_));
        // A concurrent toggle invalidates the captured display, and errors stay
        // in the editor instead of closing it or learning an unintended alias.
        select("notes");gtk_entry_set_text(GTK_ENTRY(app.spelling_->entry),"Notes");
        auto changed=client.Patch("/sessions/"+row.data.id+"/replacements","{\"enabled\":false}","application/json");
        g_assert_true(changed&&changed->status==200);
        gtk_button_clicked(GTK_BUTTON(app.spelling_->save));
        g_assert_true(wait_ui([&]{return app.spelling_&&app.spelling_->operation.empty()&&gtk_widget_get_visible(app.spelling_->error);}));
        g_assert_nonnull(strstr(gtk_label_get_text(GTK_LABEL(app.spelling_->error)),"Transcript changed"));
        g_assert_true(client.Get("/dictionary")->body==dictionary->body);
        app.close_spelling();g_assert_null(app.spelling_.get());
        changed=client.Patch("/sessions/"+row.data.id+"/replacements","{\"enabled\":true}","application/json");
        g_assert_true(changed&&changed->status==200);
        row.cursor=0;app.load_details(row);g_assert_true(wait_ui([&]{return !row.detail_pending;}));
        screenshot(app,"spelling-saved");
    }
    static void dictionary_tab(App& app, httplib::Client& client) {
        auto& view = *app.dictionary_;
        gtk_stack_set_visible_child_name(GTK_STACK(app.view_stack_), "dictionary");
        g_assert_true(wait_ui([&] { return view.revision_ > 0 && view.pending_.empty(); }));
        g_assert_true(gtk_widget_get_mapped(view.widget()));
        auto children = [](GtkWidget* box) {
            std::vector<GtkWidget*> result; auto* list = gtk_container_get_children(GTK_CONTAINER(box));
            for (auto* child = list; child; child = child->next) result.push_back(GTK_WIDGET(child->data));
            g_list_free(list); return result;
        };
        auto search = [&](const char* text, size_t count) {
            gtk_entry_set_text(GTK_ENTRY(view.search_), text);
            g_assert_true(wait_ui([&] { return !view.search_idle_ && view.matches_.size() == count; }));
        };
        auto saved = [&] {
            g_assert_true(wait_ui([&] { return view.pending_.empty() && !view.editor_; }));
            g_assert_false(gtk_widget_get_visible(view.message_));
        };
        gtk_button_clicked(GTK_BUTTON(view.add_)); g_assert_nonnull(view.editor_);
        gtk_entry_set_text(GTK_ENTRY(view.canonical_), "CaféBoard");
        auto aliases = children(view.aliases_);
        gtk_entry_set_text(GTK_ENTRY(g_object_get_data(G_OBJECT(aliases.front()), "alias")), "board; room");
        view.alias_field("cafe board");
        gtk_dialog_response(GTK_DIALOG(view.editor_), GTK_RESPONSE_ACCEPT); saved();
        search("CAFÉBOARD", 1); search("BOARD; ROOM", 1);
        auto rows = children(view.rows_); g_assert_cmpuint(rows.size(), ==, 1);
        auto row = children(rows.front());
        gtk_button_clicked(GTK_BUTTON(row[1])); g_assert_nonnull(view.editor_);
        g_assert_cmpstr(gtk_entry_get_text(GTK_ENTRY(view.canonical_)), ==, "CaféBoard");
        aliases = children(view.aliases_); g_assert_cmpuint(aliases.size(), ==, 2);
        g_assert_cmpstr(gtk_entry_get_text(GTK_ENTRY(g_object_get_data(G_OBJECT(aliases[0]), "alias"))), ==, "board; room");
        gtk_entry_set_text(GTK_ENTRY(view.canonical_), "CaféDesk");
        gtk_dialog_response(GTK_DIALOG(view.editor_), GTK_RESPONSE_ACCEPT); saved();
        search("CAFÉDESK", 1);
        screenshot(app, "dictionary");
        // A concurrent addition must fence a removal based on old list indices.
        rows = children(view.rows_); row = children(rows.front()); gtk_button_clicked(GTK_BUTTON(row[2]));
        auto concurrent = client.Post("/dictionary/entries", "{\"canonical\":\"Concurrent\",\"aliases\":[\"concurrent alias\"]}", "application/json");
        g_assert_true(concurrent && concurrent->status == 200);
        gtk_dialog_response(GTK_DIALOG(view.editor_), GTK_RESPONSE_ACCEPT);
        g_assert_true(wait_ui([&] { return view.pending_.empty() && view.editor_conflict_; }));
        g_assert_true(gtk_widget_get_mapped(view.editor_error_));
        g_assert_nonnull(strstr(client.Get("/dictionary")->body.c_str(), "CaféDesk"));
        gtk_dialog_response(GTK_DIALOG(view.editor_), GTK_RESPONSE_CANCEL);
        gtk_button_clicked(GTK_BUTTON(view.reload_));
        g_assert_true(wait_ui([&] { return view.pending_.empty() && view.matches_.size() == 1; }));
        rows = children(view.rows_); row = children(rows.front()); gtk_button_clicked(GTK_BUTTON(row[2]));
        gtk_dialog_response(GTK_DIALOG(view.editor_), GTK_RESPONSE_ACCEPT); saved();
        g_assert_cmpuint(view.matches_.size(), ==, 0);
        g_assert_true(gtk_widget_get_visible(view.empty_));
        g_assert_nonnull(strstr(client.Get("/dictionary")->body.c_str(), "Concurrent"));
        // Only the current page creates widgets; search still covers every entry.
        auto* b = json_builder_new(); json_builder_begin_object(b); json_builder_set_member_name(b, "entries"); json_builder_begin_array(b);
        for (int i = 0; i < 125; ++i) {
            json_builder_begin_object(b); json_field(b, "canonical", "Project" + std::to_string(i));
            json_builder_set_member_name(b, "aliases"); json_builder_begin_array(b);
            json_builder_add_string_value(b, ("project number " + std::to_string(i)).c_str());
            json_builder_end_array(b); json_builder_end_object(b);
        }
        json_builder_end_array(b); json_builder_end_object(b); const auto body = json_string(b); g_object_unref(b);
        g_assert_cmpint(client.Put("/dictionary", body, "application/json")->status, ==, 200);
        gtk_entry_set_text(GTK_ENTRY(view.search_), ""); gtk_button_clicked(GTK_BUTTON(view.reload_));
        g_assert_true(wait_ui([&] { return view.pending_.empty() && !view.search_idle_ && view.entries_.size() == 125; }));
        g_assert_cmpuint(children(view.rows_).size(), ==, 50);
        gtk_button_clicked(GTK_BUTTON(view.next_)); g_assert_cmpuint(view.page_, ==, 1);
        g_assert_cmpuint(children(view.rows_).size(), ==, 50);
        search("PROJECT NUMBER 124", 1); g_assert_cmpuint(view.page_, ==, 0);
        g_assert_cmpstr(view.entries_[view.matches_.front()].canonical.c_str(), ==, "Project124");
        // Loading failures remain inside the tab and disable writes until retry.
        view.pending_ = "dictionary:test-failure"; view.loading_ = true;
        app.handle_session_event({view.pending_, {}, {}, "Fixture unavailable", 503});
        g_assert_false(gtk_widget_get_sensitive(view.add_));
        g_assert_nonnull(strstr(gtk_label_get_text(GTK_LABEL(view.message_)), "Fixture unavailable"));
        gtk_button_clicked(GTK_BUTTON(view.reload_));
        g_assert_true(wait_ui([&] { return view.pending_.empty() && gtk_widget_get_sensitive(view.add_); }));
        gtk_stack_set_visible_child_name(GTK_STACK(app.view_stack_), "history");
        gtk_stack_set_visible_child_name(GTK_STACK(app.view_stack_), "dictionary");
        g_assert_true(wait_ui([&] { return view.pending_.empty(); }));
        g_assert_cmpuint(view.matches_.size(), ==, 1);
    }
    static void replacement_clicks(App& app, App::Row& row) {
        auto* toggle = app.replacements_toggle_;
        g_assert_true(GTK_IS_TOGGLE_BUTTON(toggle));
        g_assert_false(GTK_IS_SWITCH(toggle));
        g_assert_true(GTK_IS_DRAWING_AREA(gtk_bin_get_child(GTK_BIN(toggle))));
        const auto initial_geometry = geometry(app, toggle);
        g_assert_cmpint(initial_geometry[2], <=, 60);
        g_assert_cmpint(initial_geometry[3], <=, 32);
        auto* pointer = xdo_new(nullptr); g_assert_nonnull(pointer);
        auto move = [&](double fraction) {
            int origin_x = 0, origin_y = 0;
            gdk_window_get_origin(gtk_widget_get_window(app.window_), &origin_x, &origin_y);
            const auto box = geometry(app, toggle);
            g_assert_cmpint(xdo_move_mouse(pointer, origin_x + box[0] + int(box[2] * fraction),
                origin_y + box[1] + box[3] / 2, 0), ==, 0);
            spin(30);
        };
        auto settled = [&](bool enabled) {
            const bool ready = wait_ui([&] { return !app.replacements_pending_ &&
                row.data.replacements_enabled == enabled && !row.detail_pending && row.segments.count(59) &&
                row.segments.at(59)->rendered.highlights.size() == (enabled ? 2 : 0); });
            if (!ready) {
                g_printerr("toggle expected=%d actual=%d server=%d pending=%d details=%d highlights=%zu mapped=%d sensitive=%d\n",
                    enabled, gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)), row.data.replacements_enabled,
                    app.replacements_pending_, row.detail_pending, row.segments.count(59) ? row.segments.at(59)->rendered.highlights.size() : 999,
                    gtk_widget_get_mapped(toggle), gtk_widget_is_sensitive(toggle));
                screenshot(app, "replacement-toggle-failure");
            }
            g_assert_true(ready);
            g_assert_true(bool(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle))) == enabled);
            g_assert_true(geometry(app, toggle) == initial_geometry);
        };
        // Actual pointer events exercise both ends and the middle of the pill;
        // setting the widget's active property would miss interaction bugs.
        for (const auto& test : {std::make_pair(0.1, false), std::make_pair(0.9, true), std::make_pair(0.5, false)}) {
            move(test.first); g_assert_cmpint(xdo_click_window(pointer, CURRENTWINDOW, 1), ==, 0); settled(test.second);
        }
        screenshot(app, "replacement-toggle-off");
        auto space = [&] {
            g_assert_cmpint(xdo_focus_window(pointer, gdk_x11_window_get_xid(gtk_widget_get_window(app.window_))), ==, 0);
            gtk_widget_grab_focus(toggle);
            g_assert_true(wait_ui([&] { return gtk_widget_has_focus(toggle); }));
            g_assert_cmpint(xdo_send_keysequence_window_down(pointer, CURRENTWINDOW, "space", 12000), ==, 0);
            spin(60); // Let GTK process the press before sending its release.
            g_assert_cmpint(xdo_send_keysequence_window_up(pointer, CURRENTWINDOW, "space", 12000), ==, 0);
        };
        space(); settled(true);
        // A drag cannot place the control between states, or submit a change
        // before release. Releasing within the button is one ordinary click.
        move(0.1); g_assert_cmpint(xdo_mouse_down(pointer, CURRENTWINDOW, 1), ==, 0); move(0.5);
        g_assert_false(app.replacements_pending_); g_assert_true(row.data.replacements_enabled);
        g_assert_cmpint(xdo_mouse_up(pointer, CURRENTWINDOW, 1), ==, 0); settled(false);
        space(); settled(true);
        screenshot(app, "replacement-toggle-on");
        xdo_free(pointer);
    }
    static void inspect_controls(App& app) {
        std::function<void(GtkWidget*)> inspect = [&](GtkWidget* widget) {
            // GtkStackSwitcher uses radio buttons, a GtkCheckButton subclass.
            g_assert_false(GTK_IS_CHECK_BUTTON(widget) && !GTK_IS_RADIO_BUTTON(widget));
            const char* text = GTK_IS_LABEL(widget) ? gtk_label_get_text(GTK_LABEL(widget)) :
                GTK_IS_BUTTON(widget) ? gtk_button_get_label(GTK_BUTTON(widget)) : nullptr;
            if (text) {
                auto lower = take_string(g_utf8_strdown(text, -1));
                g_assert_null(strstr(lower.c_str(), "cleanup"));
                g_assert_null(strstr(lower.c_str(), "clean up"));
                g_assert_null(strstr(lower.c_str(), "s1"));
            }
            if (GTK_IS_BUTTON(widget)) {
                auto* image = gtk_button_get_image(GTK_BUTTON(widget));
                if (image && gtk_image_get_storage_type(GTK_IMAGE(image)) == GTK_IMAGE_ICON_NAME) {
                    const char* icon = nullptr; gtk_image_get_icon_name(GTK_IMAGE(image), &icon, nullptr);
                    g_assert_cmpstr(icon, !=, "audio-x-generic-symbolic");
                    if (icon && (std::string(icon) == "media-playback-start-symbolic" || std::string(icon) == "media-playback-pause-symbolic"))
                        g_assert_true(widget == app.pane_pause_); // Resume/Pause capture only; no playback controls.
                }
            }
            if (GTK_IS_CONTAINER(widget)) {
                auto* children = gtk_container_get_children(GTK_CONTAINER(widget));
                for (auto* child = children; child; child = child->next) inspect(GTK_WIDGET(child->data));
                g_list_free(children);
            }
        };
        inspect(app.window_);
        app.show_preferences(); inspect(app.preferences_); gtk_widget_destroy(app.preferences_);
        auto* image = gtk_button_get_image(GTK_BUTTON(app.pane_copy_));
        g_assert_true(GTK_IS_IMAGE(image));
        const char* icon = nullptr; gtk_image_get_icon_name(GTK_IMAGE(image), &icon, nullptr);
        g_assert_cmpstr(icon, ==, "edit-copy-symbolic");
    }
    static void run(App& app, const std::string& url) {
        httplib::Client client(url);
        g_assert_true(wait_ui([&] { return !app.rows_.empty(); }));
        for(const auto& [id,entry]:app.rows_) {
            g_assert_true(entry->data.attention().empty());
            g_assert_false(gtk_widget_get_visible(entry->progress));
        }
        g_assert_cmpstr(gtk_stack_get_visible_child_name(GTK_STACK(app.view_stack_)), ==, "transcription");
        g_assert_false(gtk_widget_get_mapped(app.transport_controls_));
        g_assert_true(gtk_widget_get_mapped(app.record_button_));
        screenshot(app, "idle");
        auto& row = *app.rows_.begin()->second;
        gtk_stack_set_visible_child_name(GTK_STACK(app.view_stack_), "history"); spin();
        g_assert_true(gtk_widget_get_mapped(app.history_box_));
        g_assert_nonnull(app.load_more_);
        g_assert_false(gtk_widget_get_visible(gtk_widget_get_parent(app.load_more_)));
        g_assert_false(gtk_widget_get_mapped(app.transport_controls_));
        g_signal_emit_by_name(app.history_box_, "row-activated", row.list_row);
        g_assert_true(wait_ui([&] { return row.segments.size() == 50 && row.segments.count(59) && !row.detail_pending; }));
        g_assert_cmpstr(gtk_stack_get_visible_child_name(GTK_STACK(app.view_stack_)), ==, "transcription");
        g_assert_false(gtk_widget_get_mapped(app.history_box_));
        g_assert_false(gtk_widget_get_visible(app.pane_progress_));
        g_assert_cmpuint(row.segments.begin()->first, ==, 10);
        g_assert_true(gtk_adjustment_get_value(app.pane_adjustment_) > 0);
        auto& last = *row.segments.at(59);
        g_assert_nonnull(strstr(gtk_label_get_text(GTK_LABEL(last.edited)), "Paraketto"));
        g_assert_nonnull(strstr(last.data.raw.c_str(), "paraketo"));
        g_assert_true(last.rendered.valid); g_assert_cmpuint(last.rendered.highlights.size(), ==, 2);
        g_assert_nonnull(gtk_label_get_attributes(GTK_LABEL(last.edited)));
        for (const auto& span : last.rendered.highlights) g_assert_true(hover(last.edited, span.start) == "paraketo");
        g_assert_true(hover(last.edited, 0).empty());
        g_assert_true(App::original_at_point(last.edited, -10, -10).empty());
        g_assert_true(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(app.replacements_toggle_)));
        replacement_clicks(app, row);
        screenshot(app, "transcription");
        auto* clipboard = gtk_clipboard_get(GDK_SELECTION_CLIPBOARD);
        gtk_clipboard_set_text(clipboard, "Before manual copy", -1);
        gtk_button_clicked(GTK_BUTTON(app.pane_copy_));
        g_assert_true(wait_ui([&] { return take_string(gtk_clipboard_wait_for_text(clipboard)).find("Paraketto") != std::string::npos; }));
        g_assert_false(gtk_widget_get_visible(app.pane_feedback_));

        // A different client toggles all replacements while the user is reading.
        // The saved text changes immediately, but the selected presentation stays.
        gtk_label_select_region(GTK_LABEL(last.edited), 0, 9);
        g_assert_true(wait_ui([&] { return !app.follow_latest_; }));
        const std::string selected_text = gtk_label_get_text(GTK_LABEL(last.edited));
        double position = gtk_adjustment_get_value(app.pane_adjustment_);
        auto state = metadata(client, row.data.id);
        auto changed = client.Patch("/sessions/" + row.data.id + "/replacements",
            "{\"enabled\":false,\"expected_revision\":" + std::to_string(static_cast<int64_t>(state.number("revision"))) + "}", "application/json");
        g_assert_true(changed && changed->status == 200);
        g_assert_true(wait_ui([&] { return !row.data.replacements_enabled || app.pending_pane_updates_; }, 7000));
        spin();
        int start = -1, end = -1;
        g_assert_true(gtk_label_get_selection_bounds(GTK_LABEL(last.edited), &start, &end));
        g_assert_cmpint(start, ==, 0); g_assert_cmpint(end, ==, 9);
        g_assert_cmpstr(gtk_label_get_text(GTK_LABEL(last.edited)), ==, selected_text.c_str());
        g_assert_cmpfloat(gtk_adjustment_get_value(app.pane_adjustment_), ==, position);
        g_assert_cmpuint(last.rendered.highlights.size(), ==, 2);
        for (const auto& span : last.rendered.highlights) g_assert_true(hover(last.edited, span.start) == "paraketo");
        g_assert_true(gtk_widget_get_visible(app.pane_jump_));
        g_assert_true(transcript(client,row.data.id).find("Paraketto") == std::string::npos);
        gtk_label_select_region(GTK_LABEL(last.edited), 0, 0);
        gtk_button_clicked(GTK_BUTTON(app.pane_jump_));
        g_assert_true(wait_ui([&] { return row.segments.count(59) && !row.detail_pending &&
            row.segments.at(59)->rendered.highlights.empty() &&
            std::string(gtk_label_get_text(GTK_LABEL(row.segments.at(59)->edited))).find("paraketo") != std::string::npos; }));
        gtk_button_clicked(GTK_BUTTON(app.pane_copy_));
        g_assert_true(wait_ui([&] { return take_string(gtk_clipboard_wait_for_text(clipboard)).find("paraketo") != std::string::npos; }));
        screenshot(app, "replacements-off");

        // The one pill applies to the complete session, including unloaded pages.
        gtk_button_clicked(GTK_BUTTON(app.replacements_toggle_));
        g_assert_true(wait_ui([&] { return row.data.replacements_enabled && row.segments.count(59) &&
            row.segments.at(59)->rendered.highlights.size() == 2; }));
        g_assert_true(transcript(client,row.data.id).find("paraketo") == std::string::npos);
        gtk_button_clicked(GTK_BUTTON(app.pane_first_));
        g_assert_true(wait_ui([&] { return !row.detail_pending && row.segments.size() == 50 && row.segments.count(0); }));
        g_assert_false(app.follow_latest_); g_assert_true(row.has_more);
        g_assert_cmpuint(row.segments.at(0)->rendered.highlights.size(), ==, 2);
        g_assert_cmpfloat(gtk_adjustment_get_value(app.pane_adjustment_), ==, 0);
        gtk_button_clicked(GTK_BUTTON(app.pane_next_));
        g_assert_true(wait_ui([&] { return !row.detail_pending && row.segments.count(59); }));
        g_assert_cmpuint(row.segments.size(), <=, 50);

        gtk_widget_hide(app.window_); app.toggle();
        g_assert_true(wait_ui([&] { return app.mode_ == App::Mode::Recording && app.selected_id_ == app.active_id_; }));
        g_assert_false(gtk_widget_get_visible(app.window_));
        g_assert_true(row.segments.empty());
        app.show(); spin();
        g_assert_false(gtk_widget_get_visible(gtk_widget_get_parent(app.load_more_)));
        g_assert_true(gtk_widget_get_mapped(app.transport_controls_));
        g_assert_false(gtk_widget_get_mapped(app.record_button_));
        const auto slot = geometry(app, app.transport_stack_);
        const auto pause_button = geometry(app, app.pane_pause_), stop_button = geometry(app, app.pane_stop_);
        g_assert_cmpint(pause_button[2], >=, 44); g_assert_cmpint(pause_button[3], >=, 44);
        g_assert_cmpint(pause_button[2], ==, stop_button[2]); g_assert_cmpint(pause_button[3], ==, stop_button[3]);
        g_assert_true(wait_ui([&] { return metadata(client, app.active_id_).number("durable_samples") >= 16000; }));
        screenshot(app, "recording");
        gtk_button_clicked(GTK_BUTTON(app.pane_pause_));
        g_assert_true(wait_ui([&] { return app.mode_ == App::Mode::Paused && !app.capture_command_pending_; }));
        g_assert_true(metadata(client, app.active_id_).text("capture_state") == "paused");
        auto paused_samples = metadata(client, app.active_id_).number("durable_samples");
        spin(350);
        g_assert_cmpfloat(metadata(client, app.active_id_).number("durable_samples"), ==, paused_samples);
        g_assert_true(geometry(app, app.pane_pause_) == pause_button);
        g_assert_true(geometry(app, app.pane_stop_) == stop_button);
        g_assert_true(geometry(app, app.transport_stack_) == slot);
        screenshot(app, "paused");
        failed_stop_restores_controls(app);

        // A global choice made before ASR finishes also governs future text.
        gtk_button_clicked(GTK_BUTTON(app.replacements_toggle_));
        g_assert_true(wait_ui([&] { return !app.rows_.at(app.active_id_)->data.replacements_enabled && !app.replacements_pending_; }));
        gtk_stack_set_visible_child_name(GTK_STACK(app.view_stack_), "history"); spin();
        g_assert_true(gtk_widget_get_mapped(app.history_box_));
        g_assert_false(gtk_widget_get_mapped(app.transport_controls_));
        screenshot(app, "history");
        g_signal_emit_by_name(app.history_box_, "row-activated", row.list_row); spin();
        g_assert_cmpstr(gtk_stack_get_visible_child_name(GTK_STACK(app.view_stack_)), ==, "transcription");
        g_assert_true(gtk_widget_get_sensitive(app.pane_current_));
        gtk_button_clicked(GTK_BUTTON(app.pane_current_)); spin();
        g_assert_cmpstr(app.selected_id_.c_str(), ==, app.active_id_.c_str());
        g_assert_true(geometry(app, app.pane_pause_) == pause_button);
        gtk_button_clicked(GTK_BUTTON(app.pane_pause_));
        g_assert_true(wait_ui([&] { return app.mode_ == App::Mode::Recording && !app.capture_command_pending_; }));
        g_assert_true(wait_ui([&] { return metadata(client, app.active_id_).number("durable_samples") > paused_samples + 4000; }));
        g_assert_true(geometry(app, app.pane_pause_) == pause_button);
        g_assert_true(geometry(app, app.pane_stop_) == stop_button);
        failed_stop_restores_controls(app);
        const auto recorded_id = app.active_id_;
        gtk_button_clicked(GTK_BUTTON(app.pane_stop_));
        g_assert_true(wait_ui([&] { return app.mode_ == App::Mode::Idle && app.rows_.at(recorded_id)->data.publication_complete; }, 12000));
        spin();
        g_assert_false(gtk_widget_get_mapped(app.transport_controls_));
        g_assert_true(gtk_widget_get_mapped(app.record_button_));
        g_assert_true(geometry(app, app.transport_stack_) == slot);
        auto recorded_text = transcript(client, recorded_id);
        g_assert_true(recorded_text.find("A live paraketo recording.") != std::string::npos);
        g_assert_true(recorded_text.find("Paraketto") == std::string::npos);
        inspect_controls(app);
        spelling_popover(app,client,row);
        dictionary_tab(app,client);
        puts("PASS: main transcription/history navigation, global replacements and copy, inline hover, selection/scroll and bounded paging");
        puts("PASS: replacement pill responds to real pointer clicks, drag release and Space with only on/off states and fixed geometry");
        puts("PASS: capture Pause/Resume/Stop, future replacements, fixed transport geometry and no playback controls");
        puts("PASS: failed Stop restores recording and paused controls; real Resume and Stop still complete");
        puts("PASS: selected text opens an anchored spelling editor, saves an alias, preserves originals/publications and rejects a stale selection");
        puts("PASS: Dictionary tab add/search/change/remove, Unicode aliases, revision conflicts, bounded paging and error recovery");
    }
};
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--seed") { seed(argv[2]); return 0; }
    g_assert_cmpint(argc, ==, 3);
    gtk_init(&argc, &argv); gst_init(nullptr, nullptr);
    auto* application = gtk_application_new("io.localoptima.ParakettoPaneTest", G_APPLICATION_NON_UNIQUE);
    g_assert_true(g_application_register(G_APPLICATION(application), nullptr, nullptr));
    {
        desktop::Options options; options.service_url = argv[1]; options.data = argv[2];
        options.audio_source = "audiotestsrc is-live=true wave=sine";
        desktop::App app(application, options); app.show(); desktop::PaneTest::run(app, argv[1]);
    }
    g_object_unref(application);
}
