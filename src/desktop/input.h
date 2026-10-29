#pragma once

#include <gtk/gtk.h>
extern "C" {
#include <xdo.h>
}
#include <string>

namespace desktop {

// All methods run on the GTK main context. X11 tracks the destination window.
class TextInput {
public:
    TextInput();
    ~TextInput();
    void begin(bool insert);
    void poll();
    void end();
    std::string deliver(const std::string& text);
    const std::string& destination() const { return destination_; }
    bool changed() const { return invalid_; }
    Window target_window() const { return target_; }
private:
    xdo_t* xdo_ = nullptr;
    std::string destination_, window_class_;
    Window target_ = 0;
    bool armed_ = false, insert_ = false, invalid_ = false;
    Window active_window();
};

bool install_shortcut(const std::string& executable, const std::string& accelerator, std::string& error);
std::string shortcut_label();

} // namespace desktop
