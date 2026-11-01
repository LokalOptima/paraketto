#pragma once
#include "backend.h"
#include <gtk/gtk.h>
#include <functional>
#include <string>
#include <vector>

namespace desktop {
class DictionaryView {
public:
    using Send = std::function<void(std::string, std::string, std::string, std::string)>;
    DictionaryView(GtkWindow* parent, Send send);
    ~DictionaryView();
    GtkWidget* widget() const { return root_; }
    void activate();
    bool handle_event(const SessionEvent& event);
private:
    friend struct PaneTest;
    struct Entry { std::string canonical, search; std::vector<std::string> aliases; };
    enum class Action { Add, Change, Remove };
    static constexpr size_t page_size = 50;
    GtkWindow* parent_;
    Send send_;
    GtkWidget *root_, *search_, *add_, *reload_, *rows_, *empty_, *message_, *count_, *previous_, *next_;
    GtkWidget *editor_ = nullptr, *editor_fields_ = nullptr, *canonical_ = nullptr, *aliases_ = nullptr, *editor_error_ = nullptr;
    std::vector<Entry> entries_;
    std::vector<size_t> matches_;
    size_t page_ = 0, editing_ = 0;
    int64_t revision_ = 0;
    guint search_idle_ = 0;
    std::string pending_;
    bool loading_ = false, editor_conflict_ = false;
    Action action_ = Action::Add;
    void request(std::string method, std::string path, std::string body = "{}");
    void load();
    void receive(const std::string& json);
    void filter();
    void render();
    void controls();
    void feedback(const std::string& text, bool error = false);
    void edit(Action action, size_t index = 0);
    void alias_field(const std::string& value = {});
    void save();
};
} // namespace desktop
