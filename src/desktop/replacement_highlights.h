#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace desktop {

struct ReplacementHighlight {
    // Half-open UTF-8 byte offsets in the displayed transcript, suitable for
    // Pango attributes and the byte index returned by pango_layout_xy_to_index.
    size_t start = 0, end = 0;
    std::string original;
};

struct ReplacementHighlights {
    std::string text;
    std::vector<ReplacementHighlight> highlights;
    bool valid = true;
};

namespace replacement_detail {
inline bool utf8(const std::string& text) {
    for (size_t i = 0; i < text.size();) {
        auto first = static_cast<unsigned char>(text[i]);
        if (first < 0x80) { ++i; continue; }
        size_t length = first >= 0xc2 && first <= 0xdf ? 2 :
            first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        if (!length || length > text.size() - i) return false;
        for (size_t j = 1; j < length; ++j) {
            auto byte = static_cast<unsigned char>(text[i + j]);
            if (byte < 0x80 || byte > 0xbf) return false;
        }
        auto second = static_cast<unsigned char>(text[i + 1]);
        if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second > 0x9f) ||
            (first == 0xf0 && second < 0x90) || (first == 0xf4 && second > 0x8f)) return false;
        i += length;
    }
    return true;
}
inline bool boundary(const std::string& text, size_t offset) {
    return offset <= text.size() && (offset == text.size() ||
        (static_cast<unsigned char>(text[offset]) & 0xc0) != 0x80);
}
} // namespace replacement_detail

// Accepts SessionSegment without introducing GTK/GLib dependencies here. Edits
// refer to immutable raw bytes; highlights refer to rendered bytes. The saved
// edited text remains authoritative if an incomplete or invalid response cannot
// be mapped exactly. Disabled edits never create highlights.
template<class Segment>
ReplacementHighlights replacement_highlights(const Segment& segment, bool replacements_enabled = true) {
    auto fallback = [&] { return ReplacementHighlights{segment.edited, {}, false}; };
    if (!replacement_detail::utf8(segment.raw)) return fallback();
    if (!replacements_enabled) return {segment.raw, {}, true};

    using Edit = typename std::decay_t<decltype(segment.edits)>::value_type;
    std::vector<const Edit*> active;
    for (const auto& edit : segment.edits) {
        if (!edit.enabled || edit.origin != "dictionary") continue;
        if (edit.start < 0 || edit.end <= edit.start ||
            static_cast<size_t>(edit.end) > segment.raw.size()) return fallback();
        auto start = static_cast<size_t>(edit.start), end = static_cast<size_t>(edit.end);
        if (!replacement_detail::boundary(segment.raw, start) || !replacement_detail::boundary(segment.raw, end) ||
            segment.raw.compare(start, end - start, edit.before) != 0 || !replacement_detail::utf8(edit.after)) return fallback();
        active.push_back(&edit);
    }
    std::sort(active.begin(), active.end(), [](const auto* a, const auto* b) { return a->start < b->start; });
    size_t source = 0;
    for (const auto* edit : active) {
        if (static_cast<size_t>(edit->start) < source) return fallback();
        source = static_cast<size_t>(edit->end);
    }

    ReplacementHighlights result;
    source = 0;
    for (const auto* edit : active) {
        result.text.append(segment.raw, source, static_cast<size_t>(edit->start) - source);
        auto start = result.text.size();
        result.text += edit->after;
        if (!edit->after.empty() && edit->before != edit->after)
            result.highlights.push_back({start, result.text.size(), edit->before});
        source = static_cast<size_t>(edit->end);
    }
    result.text.append(segment.raw, source, std::string::npos);
    if (result.text != segment.edited) return fallback();
    return result;
}

} // namespace desktop
