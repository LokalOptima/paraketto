#pragma once

#include <gtk/gtk.h>
#include <ibus.h>
extern "C" {
#include <xdo.h>
}
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace desktop {

// All methods and IBus callbacks run on the GTK main context.
class TextInput {
public:
    TextInput();
    ~TextInput();
    void begin(bool insert);
    void poll();
    void end();
    std::string deliver(const std::string& text);
    const std::string& destination() const { return destination_; }
    bool changed() const {
        return invalid_ || (armed_ && !context_.empty() &&
            (!focused_ || current_context_ != context_));
    }
    Window target_window() const { return target_; }
    bool native_target() const { return !context_.empty(); }
    bool settled() const;

    void focus_in(IBusEngine*, const char* path, const char* client);
    void focus_legacy(IBusEngine*);
    void focus_out(IBusEngine*);
private:
    IBusBus* bus_ = nullptr;
    IBusFactory* factory_ = nullptr;
    std::vector<IBusEngine*> engines_;
    unsigned next_engine_id_ = 0;
    IBusEngine* focused_ = nullptr;
    xdo_t* xdo_ = nullptr;
    std::string current_context_, context_, previous_engine_, destination_, window_class_;
    Window target_ = 0;
    bool armed_ = false, insert_ = false, invalid_ = false;
    bool legacy_focus_ = false;
    unsigned revision_ = 0, target_revision_ = 0;
    struct Lifetime { TextInput* owner; };
    std::shared_ptr<Lifetime> lifetime_;
    unsigned activating_ = 0;
    gint64 started_ = 0;
    void register_engine();
    void select_engine(const std::string& name, std::function<void(bool)> done);
    Window active_window();
    static IBusEngine* create_engine(IBusFactory*, const gchar*, gpointer);
};

bool install_shortcut(const std::string& executable, const std::string& accelerator, std::string& error);
std::string shortcut_label();

} // namespace desktop
