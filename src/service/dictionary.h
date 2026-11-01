#pragma once
#include "edits.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace paraketto::service {
struct DictionaryEntry { std::string canonical; std::vector<std::string> aliases; };
struct DictionaryText {
    std::string text;
    std::vector<TextEdit> edits;
};
std::vector<DictionaryEntry> parse_dictionary(const std::string& json);
std::string serialize_dictionary(const std::vector<DictionaryEntry>& entries);
// Merge one explicit spelling/alias rule. Conflicting aliases are rejected;
// existing unrelated entries and aliases are retained.
void add_dictionary_entry(std::vector<DictionaryEntry>& entries, const DictionaryEntry& entry);

// Immutable Aho-Corasick matcher. Build once per dictionary revision and reuse
// for every segment. Per-request scratch belongs to apply(), so readers can
// safely share a prepared matcher.
class DictionaryMatcher {
public:
    explicit DictionaryMatcher(const std::vector<DictionaryEntry>& entries);
    DictionaryText apply(const std::string& source) const;
    size_t storage_bytes() const;
private:
    static constexpr uint32_t none = UINT32_MAX;
    struct Rule { std::string alias, canonical; };
    struct State { uint32_t first=0, count=0, failure=0, output=none, rule=none; };
    struct Edge { uint32_t next; unsigned char byte; };
    std::vector<Rule> rules_;
    std::vector<State> states_;
    std::vector<Edge> edges_;
    std::array<uint32_t,256> root_{};
    uint32_t edge(uint32_t state, unsigned char byte) const;
    uint32_t advance(uint32_t state, unsigned char byte) const;
};
// Convenience for one-off callers; streaming transcription retains a matcher.
DictionaryText apply_dictionary(const std::string& source, const std::vector<DictionaryEntry>& entries);
} // namespace paraketto::service
