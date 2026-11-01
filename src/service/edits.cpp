#include "edits.h"
#include "app_common.h"
#include <algorithm>
#include <stdexcept>

namespace paraketto::service {
namespace {
bool boundary(const std::string& s, size_t n) { return n <= s.size() && (n == s.size() || (static_cast<unsigned char>(s[n]) & 0xc0) != 0x80); }
}
std::string text_hash(const std::string& text) {
    return app::take_string(g_compute_checksum_for_data(G_CHECKSUM_SHA256, reinterpret_cast<const guchar*>(text.data()), text.size()));
}
void validate_edits(const std::string& source, const std::vector<TextEdit>& edits) {
    if (!g_utf8_validate(source.data(), source.size(), nullptr)) throw std::invalid_argument("Invalid UTF-8 source");
    std::vector<const TextEdit*> active;
    for (auto& e : edits) {
        if (e.start >= e.end || !boundary(source,e.start) || !boundary(source,e.end) ||
            source.substr(e.start,e.end-e.start) != e.before || !g_utf8_validate(e.after.data(),e.after.size(),nullptr))
            throw std::invalid_argument("Edit does not match its immutable source");
        if (e.enabled) active.push_back(&e);
    }
    std::sort(active.begin(),active.end(),[](auto a,auto b){return a->start < b->start;});
    size_t end = 0;
    for (auto e : active) { if (e->start < end) throw std::invalid_argument("Enabled edits overlap"); end=e->end; }
}
std::string render_edits(const std::string& source, const std::vector<TextEdit>& edits) {
    validate_edits(source,edits);
    std::vector<TextEdit> active;
    for (auto& e : edits) if(e.enabled) active.push_back(e);
    std::sort(active.begin(),active.end(),[](auto& a,auto& b){return a.start < b.start;});
    std::string result; size_t pos=0;
    for(auto& e:active) {result.append(source,pos,e.start-pos);result+=e.after;pos=e.end;}
    result.append(source,pos,std::string::npos); return result;
}
std::pair<size_t,size_t> selection_source_span(const std::string& source,
        const std::vector<TextEdit>& edits,size_t start,size_t end,const std::string& selected) {
    const auto rendered=render_edits(source,edits);
    if(start>=end || !boundary(rendered,start) || !boundary(rendered,end) ||
       rendered.substr(start,end-start)!=selected)
        throw std::invalid_argument("Selection no longer matches the transcript");
    std::vector<const TextEdit*> active;
    for(const auto& edit:edits)if(edit.enabled)active.push_back(&edit);
    std::sort(active.begin(),active.end(),[](auto a,auto b){return a->start<b->start;});
    auto map=[&](size_t position) {
        size_t raw=0,display=0;
        for(const auto* edit:active) {
            size_t begin=display+edit->start-raw;
            if(position<=begin)return raw+position-display;
            size_t after=begin+edit->after.size();
            if(position<after)throw std::invalid_argument("Select the whole replacement to change its spelling");
            raw=edit->end;display=after;
        }
        return raw+position-display;
    };
    return {map(start),map(end)};
}
} // namespace paraketto::service
