#include "dictionary_view.h"
#include "app_common.h"
#include <algorithm>

namespace desktop {
using namespace paraketto::app;
namespace {
void pack(GtkWidget* box, GtkWidget* child, bool expand = false) {
    gtk_box_pack_start(GTK_BOX(box), child, expand, expand, 0);
}
GtkWidget* label(const std::string& text) {
    auto* widget = gtk_label_new(text.c_str());
    gtk_label_set_xalign(GTK_LABEL(widget), 0);
    return widget;
}
std::string fold(const std::string& text) { return take_string(g_utf8_casefold(text.c_str(), text.size())); }
void clear(GtkWidget* box) {
    auto* children = gtk_container_get_children(GTK_CONTAINER(box));
    for (auto* child = children; child; child = child->next) gtk_widget_destroy(GTK_WIDGET(child->data));
    g_list_free(children);
}
}

DictionaryView::DictionaryView(GtkWindow* parent, Send send) : parent_(parent), send_(std::move(send)) {
    root_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    g_object_ref_sink(root_);
    gtk_container_set_border_width(GTK_CONTAINER(root_), 24);
    auto* toolbar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    search_ = gtk_search_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(search_), "Search spellings or aliases");
    gtk_entry_set_max_length(GTK_ENTRY(search_), 256);
    pack(toolbar, search_, true);
    add_ = gtk_button_new_with_label("Add entry");
    gtk_style_context_add_class(gtk_widget_get_style_context(add_), "suggested-action");
    pack(toolbar, add_);
    reload_ = gtk_button_new_from_icon_name("view-refresh-symbolic", GTK_ICON_SIZE_BUTTON);
    gtk_widget_set_tooltip_text(reload_, "Reload dictionary"); pack(toolbar, reload_); pack(root_, toolbar);
    auto* hint = label("Your preferred spellings and the words to replace. Changes apply to new recordings.");
    gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
    gtk_style_context_add_class(gtk_widget_get_style_context(hint), "secondary"); pack(root_, hint);
    message_ = label(""); gtk_label_set_line_wrap(GTK_LABEL(message_), TRUE);
    gtk_widget_set_no_show_all(message_, TRUE); pack(root_, message_);
    auto* scroll = gtk_scrolled_window_new(nullptr, nullptr);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    auto* content = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_container_add(GTK_CONTAINER(scroll), content); pack(root_, scroll, true);
    rows_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12); pack(content, rows_);
    empty_ = label("No entries yet. Add a spelling to get started.");
    gtk_label_set_line_wrap(GTK_LABEL(empty_), TRUE); gtk_widget_set_no_show_all(empty_, TRUE); pack(content, empty_);
    auto* pages = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    count_ = label(""); pack(pages, count_, true);
    previous_ = gtk_button_new_with_label("Previous"); next_ = gtk_button_new_with_label("Next");
    for (auto* button : {previous_, next_}) { gtk_widget_set_no_show_all(button, TRUE); pack(pages, button); }
    pack(root_, pages);
    g_signal_connect(search_, "changed", G_CALLBACK(+[](GtkEditable*, gpointer p) {
        auto* view = static_cast<DictionaryView*>(p);
        if (view->search_idle_) g_source_remove(view->search_idle_);
        view->search_idle_ = g_timeout_add(120, +[](gpointer p) -> gboolean {
            auto* view = static_cast<DictionaryView*>(p); view->search_idle_ = 0; view->filter(); return G_SOURCE_REMOVE;
        }, view);
    }), this);
    g_signal_connect_swapped(add_, "clicked", G_CALLBACK(+[](DictionaryView* v) { v->edit(Action::Add); }), this);
    g_signal_connect_swapped(reload_, "clicked", G_CALLBACK(+[](DictionaryView* v) { v->load(); }), this);
    g_signal_connect_swapped(previous_, "clicked", G_CALLBACK(+[](DictionaryView* v) { if (v->page_) { --v->page_; v->render(); } }), this);
    g_signal_connect_swapped(next_, "clicked", G_CALLBACK(+[](DictionaryView* v) {
        if ((v->page_ + 1) * page_size < v->matches_.size()) { ++v->page_; v->render(); }
    }), this);
    controls();
}
DictionaryView::~DictionaryView() {
    if (editor_) gtk_widget_destroy(editor_);
    if (search_idle_) g_source_remove(search_idle_);
    gtk_widget_destroy(root_); g_object_unref(root_);
}
void DictionaryView::activate() { if (pending_.empty() && !editor_) load(); }
void DictionaryView::request(std::string method, std::string path, std::string body) {
    pending_ = "dictionary:" + take_string(g_uuid_string_random());
    controls(); send_(pending_, std::move(method), std::move(path), std::move(body));
}
void DictionaryView::load() {
    if (!pending_.empty() || editor_) return;
    loading_ = true; feedback(""); request("GET", "/dictionary");
}
void DictionaryView::feedback(const std::string& text, bool error) {
    gtk_label_set_text(GTK_LABEL(message_), text.c_str()); gtk_widget_set_visible(message_, !text.empty());
    auto* style = gtk_widget_get_style_context(message_);
    if (error) gtk_style_context_add_class(style, "failure"); else gtk_style_context_remove_class(style, "failure");
}
void DictionaryView::controls() {
    const bool ready = revision_ > 0 && pending_.empty();
    gtk_widget_set_sensitive(add_, ready); gtk_widget_set_sensitive(rows_, ready);
    gtk_widget_set_sensitive(reload_, pending_.empty() && !editor_);
    gtk_widget_set_sensitive(previous_, ready && page_ > 0);
    gtk_widget_set_sensitive(next_, ready && (page_ + 1) * page_size < matches_.size());
    if (editor_) {
        gtk_widget_set_sensitive(editor_fields_, pending_.empty());
        gtk_dialog_set_response_sensitive(GTK_DIALOG(editor_), GTK_RESPONSE_ACCEPT,
            pending_.empty() && !editor_conflict_ && (action_ == Action::Remove || *gtk_entry_get_text(GTK_ENTRY(canonical_))));
    }
}
void DictionaryView::receive(const std::string& json) {
    Json parsed(json);
    auto* node = json_object_get_member(parsed.object(), "entries");
    const auto revision = int64_t(parsed.number("revision"));
    if (revision < 1 || !node || !JSON_NODE_HOLDS_ARRAY(node)) throw std::runtime_error("Invalid dictionary response");
    std::vector<Entry> entries;
    auto* array = json_node_get_array(node);
    for (guint i = 0; i < json_array_get_length(array); ++i) {
        auto* item = json_array_get_element(array, i);
        if (!JSON_NODE_HOLDS_OBJECT(item)) throw std::runtime_error("Invalid dictionary entry");
        auto* object = json_node_get_object(item);
        auto* canonical = json_object_get_member(object, "canonical");
        if (!canonical || json_node_get_value_type(canonical) != G_TYPE_STRING) throw std::runtime_error("Invalid dictionary spelling");
        Entry entry; entry.canonical = json_node_get_string(canonical); entry.search = fold(entry.canonical);
        auto* aliases = json_object_get_member(object, "aliases");
        if (!aliases || !JSON_NODE_HOLDS_ARRAY(aliases)) throw std::runtime_error("Invalid dictionary aliases");
        auto* values = json_node_get_array(aliases);
        for (guint j = 0; j < json_array_get_length(values); ++j) {
            auto* value = json_array_get_element(values, j);
            if (json_node_get_value_type(value) != G_TYPE_STRING) throw std::runtime_error("Invalid dictionary alias");
            entry.aliases.emplace_back(json_node_get_string(value)); entry.search += '\n' + fold(entry.aliases.back());
        }
        entries.push_back(std::move(entry));
    }
    entries_ = std::move(entries); revision_ = revision; filter();
}
bool DictionaryView::handle_event(const SessionEvent& event) {
    if (event.operation.rfind("dictionary:", 0) != 0) return false;
    if (event.operation != pending_) return true;
    pending_.clear(); const bool load = loading_; loading_ = false;
    std::string error = event.error;
    if (error.empty()) {
        try { receive(event.body); }
        catch (const std::exception& e) { error = e.what(); }
    }
    if (!error.empty()) {
        if (load) revision_ = 0; // A failed refresh must never enable edits against stale data.
        if (editor_) {
            editor_conflict_ = event.status == 409;
            if (editor_conflict_) error += " Cancel, reload the dictionary and try again.";
            gtk_label_set_text(GTK_LABEL(editor_error_), error.c_str()); gtk_widget_show(editor_error_);
        } else feedback(std::string(load ? "Could not load dictionary: " : "Could not save dictionary: ") + error, true);
    } else {
        if (editor_) gtk_widget_destroy(editor_);
        feedback("");
    }
    controls(); return true;
}
void DictionaryView::filter() {
    if (search_idle_) { g_source_remove(search_idle_); search_idle_ = 0; }
    const auto query = fold(gtk_entry_get_text(GTK_ENTRY(search_)));
    matches_.clear();
    for (size_t i = 0; i < entries_.size(); ++i) if (entries_[i].search.find(query) != std::string::npos) matches_.push_back(i);
    page_ = 0; render();
}
void DictionaryView::render() {
    clear(rows_);
    const auto begin = page_ * page_size, end = std::min(begin + page_size, matches_.size());
    for (auto pos = begin; pos < end; ++pos) {
        const auto index = matches_[pos]; const auto& entry = entries_[index];
        auto* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        auto* text = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
        auto* spelling = label(entry.canonical); gtk_style_context_add_class(gtk_widget_get_style_context(spelling), "heading");
        std::string aliases;
        for (const auto& alias : entry.aliases) { if (!aliases.empty()) aliases += " · "; aliases += alias; }
        auto* alternate = label(aliases.empty() ? "No aliases" : aliases);
        gtk_style_context_add_class(gtk_widget_get_style_context(alternate), "secondary");
        for (auto* value : {spelling, alternate}) {
            gtk_label_set_ellipsize(GTK_LABEL(value), PANGO_ELLIPSIZE_END);
            gtk_label_set_max_width_chars(GTK_LABEL(value), 44);
            gtk_widget_set_tooltip_text(value, gtk_label_get_text(GTK_LABEL(value))); pack(text, value);
        }
        pack(row, text, true);
        for (const auto* name : {"Change", "Remove"}) {
            auto* button = gtk_button_new_with_label(name); gtk_widget_set_valign(button, GTK_ALIGN_CENTER);
            g_object_set_data(G_OBJECT(button), "entry-index", GSIZE_TO_POINTER(index));
            g_signal_connect(button, "clicked", G_CALLBACK(+[](GtkButton* button, gpointer p) {
                auto* view = static_cast<DictionaryView*>(p);
                view->edit(std::string(gtk_button_get_label(button)) == "Change" ? Action::Change : Action::Remove,
                    GPOINTER_TO_SIZE(g_object_get_data(G_OBJECT(button), "entry-index")));
            }), this);
            pack(row, button);
        }
        pack(rows_, row);
    }
    gtk_widget_show_all(rows_);
    gtk_label_set_text(GTK_LABEL(empty_), entries_.empty() ? "No entries yet. Add a spelling to get started." : "No matching entries.");
    gtk_widget_set_visible(empty_, revision_ > 0 && matches_.empty());
    const auto count = matches_.empty() ? "0 entries" : std::to_string(begin + 1) + "–" + std::to_string(end) + " of " + std::to_string(matches_.size()) + " entries";
    gtk_label_set_text(GTK_LABEL(count_), count.c_str());
    for (auto* button : {previous_, next_}) gtk_widget_set_visible(button, matches_.size() > page_size);
    controls();
}
void DictionaryView::alias_field(const std::string& value) {
    auto* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    auto* entry = gtk_entry_new(); gtk_entry_set_text(GTK_ENTRY(entry), value.c_str());
    gtk_entry_set_placeholder_text(GTK_ENTRY(entry), "Transcribed spelling");
    g_object_set_data(G_OBJECT(row), "alias", entry); pack(row, entry, true);
    auto* remove = gtk_button_new_from_icon_name("list-remove-symbolic", GTK_ICON_SIZE_BUTTON);
    gtk_widget_set_tooltip_text(remove, "Remove alias");
    g_signal_connect_swapped(remove, "clicked", G_CALLBACK(gtk_widget_destroy), row); pack(row, remove);
    pack(aliases_, row); gtk_widget_show_all(row);
}
void DictionaryView::edit(Action action, size_t index) {
    if (!pending_.empty() || revision_ < 1 || editor_ || (action != Action::Add && index >= entries_.size())) return;
    action_ = action; editing_ = index; editor_conflict_ = false;
    editor_ = gtk_dialog_new_with_buttons(action == Action::Add ? "Add entry" : action == Action::Change ? "Change entry" : "Remove entry",
        parent_, static_cast<GtkDialogFlags>(GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT | GTK_DIALOG_USE_HEADER_BAR),
        "Cancel", GTK_RESPONSE_CANCEL, action == Action::Remove ? "Remove" : "Save", GTK_RESPONSE_ACCEPT, nullptr);
    gtk_window_set_default_size(GTK_WINDOW(editor_), 460, action == Action::Remove ? -1 : 360);
    auto* save = gtk_dialog_get_widget_for_response(GTK_DIALOG(editor_), GTK_RESPONSE_ACCEPT);
    gtk_style_context_add_class(gtk_widget_get_style_context(save), action == Action::Remove ? "destructive-action" : "suggested-action");
    auto* content = gtk_dialog_get_content_area(GTK_DIALOG(editor_)); gtk_container_set_border_width(GTK_CONTAINER(content), 16);
    gtk_box_set_spacing(GTK_BOX(content), 12);
    editor_fields_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8); pack(content, editor_fields_, true);
    if (action == Action::Remove) {
        auto* question = label("Remove “" + entries_[index].canonical + "” and its aliases?");
        gtk_label_set_line_wrap(GTK_LABEL(question), TRUE); pack(editor_fields_, question);
    } else {
        pack(editor_fields_, label("Desired spelling")); canonical_ = gtk_entry_new();
        if (action == Action::Change) gtk_entry_set_text(GTK_ENTRY(canonical_), entries_[index].canonical.c_str());
        gtk_entry_set_activates_default(GTK_ENTRY(canonical_), TRUE); pack(editor_fields_, canonical_);
        pack(editor_fields_, label("Replace these transcribed spellings"));
        auto* scroll = gtk_scrolled_window_new(nullptr, nullptr); gtk_widget_set_size_request(scroll, -1, 120);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        aliases_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8); gtk_container_add(GTK_CONTAINER(scroll), aliases_); pack(editor_fields_, scroll, true);
        if (action == Action::Change) for (const auto& alias : entries_[index].aliases) alias_field(alias);
        else alias_field();
        auto* add = gtk_button_new_with_label("Add alias"); gtk_widget_set_halign(add, GTK_ALIGN_START); pack(editor_fields_, add);
        g_signal_connect_swapped(add, "clicked", G_CALLBACK(+[](DictionaryView* v) { v->alias_field(); }), this);
        g_signal_connect_swapped(canonical_, "changed", G_CALLBACK(+[](DictionaryView* v) { v->controls(); }), this);
    }
    editor_error_ = label(""); gtk_label_set_line_wrap(GTK_LABEL(editor_error_), TRUE);
    gtk_label_set_max_width_chars(GTK_LABEL(editor_error_), 52); gtk_widget_set_no_show_all(editor_error_, TRUE);
    gtk_style_context_add_class(gtk_widget_get_style_context(editor_error_), "failure"); pack(content, editor_error_);
    g_signal_connect(editor_, "response", G_CALLBACK(+[](GtkDialog* dialog, gint response, gpointer p) {
        if (response == GTK_RESPONSE_ACCEPT) static_cast<DictionaryView*>(p)->save(); else gtk_widget_destroy(GTK_WIDGET(dialog));
    }), this);
    g_signal_connect(editor_, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer p) {
        auto* v = static_cast<DictionaryView*>(p); v->editor_ = v->editor_fields_ = v->canonical_ = v->aliases_ = v->editor_error_ = nullptr; v->controls();
    }), this);
    gtk_dialog_set_default_response(GTK_DIALOG(editor_), GTK_RESPONSE_ACCEPT);
    gtk_widget_show_all(editor_); controls(); if (canonical_) gtk_widget_grab_focus(canonical_);
}
void DictionaryView::save() {
    if (!editor_ || !pending_.empty() || editor_conflict_) return;
    auto* builder = json_builder_new(); json_builder_begin_object(builder);
    if (action_ != Action::Add) json_int(builder, "expected_revision", revision_);
    if (action_ != Action::Remove) {
        json_field(builder, "canonical", std::string(gtk_entry_get_text(GTK_ENTRY(canonical_))));
        json_builder_set_member_name(builder, "aliases"); json_builder_begin_array(builder);
        auto* rows = gtk_container_get_children(GTK_CONTAINER(aliases_));
        for (auto* row = rows; row; row = row->next) {
            auto* value = gtk_entry_get_text(GTK_ENTRY(g_object_get_data(G_OBJECT(row->data), "alias")));
            if (*value) json_builder_add_string_value(builder, value);
        }
        g_list_free(rows); json_builder_end_array(builder);
    }
    json_builder_end_object(builder); const auto body = json_string(builder); g_object_unref(builder);
    gtk_widget_hide(editor_error_);
    request(action_ == Action::Add ? "POST" : action_ == Action::Change ? "PUT" : "DELETE",
        action_ == Action::Add ? "/dictionary/entries" : "/dictionary/entries/" + std::to_string(editing_), body);
}
} // namespace desktop
