#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <utility>

namespace paraketto::service {
struct TextEdit {
    size_t start = 0, end = 0;
    std::string before, after, origin;
    bool enabled = true;
};
std::string text_hash(const std::string& text);
void validate_edits(const std::string& source, const std::vector<TextEdit>& edits);
std::string render_edits(const std::string& source, const std::vector<TextEdit>& edits);
// Translate a displayed UTF-8 byte selection back to immutable ASR bytes.
// A boundary inside replacement text is ambiguous and must be reselected.
std::pair<size_t,size_t> selection_source_span(const std::string& source,
    const std::vector<TextEdit>& edits, size_t start, size_t end, const std::string& selected);
} // namespace paraketto::service
