#include "desktop/replacement_highlights.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {
struct Edit {
    int64_t start = -1, end = -1;
    std::string before, after;
    bool enabled = true;
    std::string origin = "dictionary";
};
struct Segment {
    std::string raw, edited;
    std::vector<Edit> edits;
};
void check(bool condition, const char* explanation) {
    if (!condition) { std::cerr << "FAIL: " << explanation << '\n'; std::exit(1); }
}
void rejected(const Segment& segment) {
    auto result = desktop::replacement_highlights(segment);
    check(!result.valid && result.highlights.empty() && result.text == segment.edited,
          "invalid mapping retains saved text without misleading highlights");
}
}

int main() {
    using desktop::replacement_highlights;
    auto plain = replacement_highlights(Segment{"Unchanged 日本語.", "Unchanged 日本語.", {}});
    check(plain.valid && plain.highlights.empty() && plain.text == "Unchanged 日本語.", "unchanged Unicode text");
    auto empty = replacement_highlights(Segment{});
    check(empty.valid && empty.text.empty() && empty.highlights.empty(), "empty transcript");

    Segment repeated{"paraketo met paraketo.", "paraketo met Paraketto.", {{13, 21, "paraketo", "Paraketto"}}};
    auto result = replacement_highlights(repeated);
    check(result.valid && result.highlights.size() == 1 && result.highlights[0].start == 13 &&
          result.highlights[0].end == 22 && result.highlights[0].original == "paraketo", "exact repeated alias occurrence");

    // Input edits can arrive out of order. A shorter first replacement changes
    // the second highlight's displayed byte offset; source offsets stay fixed.
    Segment unicode{"Café paraketo 日本語 paraketo.", "Café P 日本語 Paraketto.",
        {{25, 33, "paraketo", "Paraketto"}, {6, 14, "paraketo", "P"}}};
    result = replacement_highlights(unicode);
    check(result.valid && result.text == unicode.edited && result.highlights.size() == 2, "multiple Unicode replacements");
    check(result.highlights[0].start == 6 && result.highlights[0].end == 7 &&
          result.highlights[1].start == 18 && result.highlights[1].end == 27, "rendered offsets are UTF-8 bytes");

    Segment expanding{"a and b", "日本語 and Caffè", {{0, 1, "a", "日本語"}, {6, 7, "b", "Caffè"}}};
    result = replacement_highlights(expanding);
    check(result.valid && result.highlights[0].end == 9 && result.highlights[1].start == 14 &&
          result.highlights[1].end == 20, "multibyte expansion shifts later highlights");

    Segment disabled{"paraketo met paraketo.", "paraketo met Paraketto.",
        {{0, 8, "paraketo", "Paraketto", false}, {13, 21, "paraketo", "Paraketto"}}};
    result = replacement_highlights(disabled);
    check(result.valid && result.highlights.size() == 1 && result.highlights[0].start == 13, "disabled individual decision");
    result = replacement_highlights(disabled, false);
    check(result.valid && result.text == disabled.raw && result.highlights.empty(), "global replacements disabled");

    Segment archived{"raw", "raw", {{0, 3, "raw", "rewritten", true, "legacy"}}};
    result = replacement_highlights(archived);
    check(result.valid && result.text == "raw" && result.highlights.empty(), "archived non-dictionary edit ignored");
    Segment deletion{"one two", " two", {{0, 3, "one", ""}}};
    result = replacement_highlights(deletion);
    check(result.valid && result.text == " two" && result.highlights.empty(), "empty replacement has no phantom hover range");
    auto unchanged_edit = replacement_highlights(Segment{"same", "same", {{0, 4, "same", "same"}}});
    check(unchanged_edit.valid && unchanged_edit.highlights.empty(), "no-op replacement is not highlighted");

    rejected({"raw", "Saved authoritative text", {{-1, 3, "raw", "changed"}}});
    rejected({"raw", "Saved authoritative text", {{0, 9, "raw", "changed"}}});
    rejected({"raw", "Saved authoritative text", {{1, 1, "", "changed"}}});
    rejected({"raw", "Saved authoritative text", {{0, 3, "mismatch", "changed"}}});
    rejected({"abcd", "Saved authoritative text", {{0, 3, "abc", "A"}, {2, 4, "cd", "B"}}});
    rejected({"Café", "Saved authoritative text", {{4, 5, std::string("\xa9"), "x"}}});
    rejected({"Café", "Saved authoritative text", {{3, 4, std::string("\xc3"), "x"}}});
    rejected({"raw", "Saved authoritative text", {{0, 3, "raw", "different generated text"}}});
    rejected({"raw", "Saved authoritative text", {}});
    for (const auto& invalid : {std::string("\xc0\xaf"), std::string("\xe0\x80\xaf"), std::string("\xed\xa0\x80"),
            std::string("\xf4\x90\x80\x80"), std::string("\xf0\x9f"), std::string("\x80")}) {
        rejected({invalid, "Saved authoritative text", {}});
        rejected({"raw", "Saved authoritative text", {{0, 3, "raw", invalid}}});
    }
    std::cout << "PASS: replacement highlighting, Unicode byte offsets, exact occurrences, disabled decisions and safe fallback\n";
}
