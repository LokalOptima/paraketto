#pragma once

#include <gtk/gtk.h>
#include <algorithm>
#include <string>
#include <utility>

namespace desktop {

struct TranscriptPreview { std::string text; bool truncated = false; };

// GtkLabel's line limit applies per paragraph. Measure the full layout so the
// preview also stays bounded when a transcript contains explicit line breaks.
inline TranscriptPreview transcript_preview(GtkWidget* widget, const std::string& text, int width, int lines = 4) {
    auto* layout = gtk_widget_create_pango_layout(widget, text.c_str());
    pango_layout_set_width(layout, std::max(1, width) * PANGO_SCALE);
    pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
    if (pango_layout_get_line_count(layout) <= lines) {
        g_object_unref(layout);
        return {text, false};
    }
    auto* last = pango_layout_get_line_readonly(layout, lines - 1);
    int end = last->start_index + last->length;
    int character = g_utf8_pointer_to_offset(text.c_str(), text.c_str() + end);
    int count = 0;
    const auto* boundaries = pango_layout_get_log_attrs_readonly(layout, &count);
    auto previous = [&] {
        do {
            const char* start = g_utf8_find_prev_char(text.c_str(), text.c_str() + end);
            end = start ? start - text.c_str() : 0;
            --character;
        } while (end && !boundaries[character].is_cursor_position);
    };
    // Trim at a grapheme boundary, preserving combining marks and joined emoji.
    while (end && (!boundaries[character].is_cursor_position ||
        g_unichar_isspace(g_utf8_get_char(g_utf8_find_prev_char(text.c_str(), text.c_str() + end))))) previous();
    auto* measure = pango_layout_copy(layout);
    std::string preview;
    do {
        preview = text.substr(0, end) + "…";
        pango_layout_set_text(measure, preview.c_str(), preview.size());
        if (pango_layout_get_line_count(measure) <= lines || !end) break;
        previous();
    } while (true);
    g_object_unref(measure);
    g_object_unref(layout);
    return {std::move(preview), true};
}

} // namespace desktop
