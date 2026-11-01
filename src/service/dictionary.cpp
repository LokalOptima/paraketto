#include "dictionary.h"
#include "app_common.h"
#include <algorithm>
#include <map>
#include <limits>
#include <stdexcept>

namespace paraketto::service {
namespace {
std::string fold(const std::string& s) { return app::take_string(g_utf8_casefold(s.data(),s.size())); }
bool word(gunichar c) { return g_unichar_isalnum(c)||g_unichar_ismark(c)||c=='_'||c=='\''||c==0x2019; }
void valid_term(const std::string& s) {
    if(s.empty() || s.size()>4096 || !g_utf8_validate(s.data(),s.size(),nullptr) || s.find('\0')!=std::string::npos)
        throw std::invalid_argument("Dictionary terms must be nonempty UTF-8 strings up to 4096 bytes");
    if(g_unichar_isspace(g_utf8_get_char(s.c_str())) || g_unichar_isspace(g_utf8_get_char(g_utf8_prev_char(s.c_str()+s.size()))))
        throw std::invalid_argument("Dictionary terms cannot have leading or trailing whitespace");
}
struct Rule { std::string alias,canonical; };
std::vector<Rule> rules(const std::vector<DictionaryEntry>& entries) {
    if(entries.size()>10000)throw std::invalid_argument("Too many dictionary entries");
    std::map<std::string,std::string> unique;
    size_t bytes=0;
    for(auto& entry:entries) {
        valid_term(entry.canonical);
        if(entry.aliases.size()>1000)throw std::invalid_argument("Too many aliases");
        auto aliases=entry.aliases;aliases.push_back(entry.canonical);
        for(auto& alias:aliases) {
            bytes+=alias.size();
            if(bytes>4*1024*1024)throw std::invalid_argument("Dictionary terms exceed 4 MiB");
            valid_term(alias);auto key=fold(alias);auto it=unique.find(key);
            if(it!=unique.end() && it->second!=entry.canonical)
                throw std::invalid_argument("Alias '"+alias+"' already maps to '"+it->second+"'; edit that dictionary entry first");
            unique[key]=entry.canonical;
        }
    }
    std::vector<Rule> out;for(auto& [a,c]:unique)out.push_back({a,c});
    return out;
}
}
std::vector<DictionaryEntry> parse_dictionary(const std::string& json) {
    app::Json document(json);auto* node=json_object_get_member(document.object(),"entries");
    if(!node || !JSON_NODE_HOLDS_ARRAY(node)) throw std::invalid_argument("Dictionary requires an entries array");
    auto* array=json_node_get_array(node); if(json_array_get_length(array)>10000) throw std::invalid_argument("Too many dictionary entries");
    std::vector<DictionaryEntry> result;
    for(guint i=0;i<json_array_get_length(array);++i) {
        auto* entry_node=json_array_get_element(array,i);
        if(!JSON_NODE_HOLDS_OBJECT(entry_node)) throw std::invalid_argument("Invalid dictionary entry");
        auto* object=json_node_get_object(entry_node);auto* canonical=json_object_get_member(object,"canonical");
        if(!canonical || json_node_get_value_type(canonical)!=G_TYPE_STRING) throw std::invalid_argument("Missing canonical spelling");
        DictionaryEntry entry;entry.canonical=json_node_get_string(canonical);
        auto* aliases=json_object_get_member(object,"aliases");
        if(aliases) {
            if(!JSON_NODE_HOLDS_ARRAY(aliases)) throw std::invalid_argument("Aliases must be an array");
            auto* list=json_node_get_array(aliases);
            if(json_array_get_length(list)>1000) throw std::invalid_argument("Too many aliases");
            for(guint j=0;j<json_array_get_length(list);++j) {
                auto* value=json_array_get_element(list,j);
                if(json_node_get_value_type(value)!=G_TYPE_STRING) throw std::invalid_argument("Alias must be a string");
                entry.aliases.emplace_back(json_node_get_string(value));
            }
        }
        result.push_back(std::move(entry));
    }
    rules(result);return result;
}
std::string serialize_dictionary(const std::vector<DictionaryEntry>& entries) {
    rules(entries);auto* b=json_builder_new();json_builder_begin_object(b);
    app::json_field(b,"matching",std::string("Unicode case folding, complete words/phrases, longest match, no cascading"));
    json_builder_set_member_name(b,"entries");json_builder_begin_array(b);
    for(auto& e:entries) {json_builder_begin_object(b);app::json_field(b,"canonical",e.canonical);json_builder_set_member_name(b,"aliases");json_builder_begin_array(b);for(auto& a:e.aliases)json_builder_add_string_value(b,a.c_str());json_builder_end_array(b);json_builder_end_object(b);}
    json_builder_end_array(b);json_builder_end_object(b);auto out=app::json_string(b);g_object_unref(b);return out;
}
void add_dictionary_entry(std::vector<DictionaryEntry>& entries,const DictionaryEntry& entry) {
    valid_term(entry.canonical);
    auto found=std::find_if(entries.begin(),entries.end(),[&](const auto& e){return e.canonical==entry.canonical;});
    if(found==entries.end()){entries.push_back(entry);found=std::prev(entries.end());}
    else for(const auto& alias:entry.aliases) {
        auto key=fold(alias);
        if(std::none_of(found->aliases.begin(),found->aliases.end(),[&](const auto& old){return fold(old)==key;}))
            found->aliases.push_back(alias);
    }
    rules(entries); // Conflicting mappings never silently replace another term.
}

DictionaryMatcher::DictionaryMatcher(const std::vector<DictionaryEntry>& entries) {
    for(auto& rule:rules(entries))rules_.push_back({std::move(rule.alias),std::move(rule.canonical)});
    struct BuildState {std::vector<Edge> edges;uint32_t rule=none;};
    std::vector<BuildState> build(1);
    for(uint32_t i=0;i<rules_.size();++i) {
        uint32_t state=0;
        for(unsigned char byte:rules_[i].alias) {
            uint32_t next=none;
            for(const auto& e:build[state].edges)if(e.byte==byte){next=e.next;break;}
            if(next==none) {
                if(build.size()>=none)throw std::length_error("Dictionary trie is too large");
                next=static_cast<uint32_t>(build.size());
                build[state].edges.push_back({next,byte});build.emplace_back();
            }
            state=next;
        }
        build[state].rule=i;
    }
    states_.resize(build.size());edges_.reserve(build.size()-1);
    for(size_t i=0;i<build.size();++i) {
        auto& children=build[i].edges;
        std::sort(children.begin(),children.end(),[](const auto& a,const auto& b){return a.byte<b.byte;});
        states_[i].first=edges_.size();states_[i].count=children.size();states_[i].rule=build[i].rule;
        edges_.insert(edges_.end(),children.begin(),children.end());
    }
    std::vector<uint32_t> queue;queue.reserve(states_.size());
    for(const auto& child:build[0].edges){root_[child.byte]=child.next;queue.push_back(child.next);}
    for(size_t i=0;i<queue.size();++i) {
        const auto parent=queue[i];
        for(const auto& child:build[parent].edges) {
            auto& state=states_[child.next];
            state.failure=advance(states_[parent].failure,child.byte);
            const auto& fallback=states_[state.failure];
            state.output=fallback.rule!=none?state.failure:fallback.output;
            queue.push_back(child.next);
        }
    }
}
uint32_t DictionaryMatcher::edge(uint32_t state,unsigned char byte) const {
    const auto& s=states_[state];
    auto begin=edges_.begin()+s.first,end=begin+s.count;
    auto found=std::lower_bound(begin,end,byte,[](const auto& e,unsigned char c){return e.byte<c;});
    return found!=end && found->byte==byte?found->next:none;
}
uint32_t DictionaryMatcher::advance(uint32_t state,unsigned char byte) const {
    while(state) {auto next=edge(state,byte);if(next!=none)return next;state=states_[state].failure;}
    return root_[byte];
}
size_t DictionaryMatcher::storage_bytes() const {
    size_t bytes=sizeof(*this)+states_.capacity()*sizeof(State)+edges_.capacity()*sizeof(Edge)+rules_.capacity()*sizeof(Rule);
    for(const auto& r:rules_)bytes+=r.alias.capacity()+r.canonical.capacity();
    return bytes;
}
DictionaryText DictionaryMatcher::apply(const std::string& source) const {
    if(source.size()>=none || !g_utf8_validate(source.data(),source.size(),nullptr))
        throw std::invalid_argument("Invalid or oversized UTF-8 source");
    if(rules_.empty() || source.empty())return {source,{}};
    const bool ascii=std::all_of(source.begin(),source.end(),[](unsigned char c){return c<128;});
    auto lower=[](unsigned char c){return char(c>='A'&&c<='Z'?c+('a'-'A'):c);};
    auto ascii_word=[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='\'';};
    std::string folded;folded.reserve(source.size());
    std::vector<uint32_t> offsets;
    if(ascii) {folded=source;for(auto& c:folded)c=lower(c);}
    else {
        // Dense byte mapping preserves complete Unicode characters and folding
        // expansions (ß -> ss), without tree lookups or per-ASCII allocations.
        offsets.reserve(source.size()+1);
        for(size_t i=0;i<source.size();) {
            size_t begin=folded.size(),next=i+1;
            if(static_cast<unsigned char>(source[i])<128)folded+=lower(source[i]);
            else {next=g_utf8_next_char(source.data()+i)-source.data();folded+=fold(source.substr(i,next-i));}
            offsets.resize(folded.size(),none);offsets[begin]=i;i=next;
        }
        offsets.push_back(source.size());
    }
    struct Match {uint32_t rule=none,end=0;};
    std::vector<Match> best(source.size());
    uint32_t state=0;
    for(size_t i=0;i<folded.size();++i) {
        state=advance(state,static_cast<unsigned char>(folded[i]));
        for(uint32_t output=states_[state].rule!=none?state:states_[state].output;output!=none;output=states_[output].output) {
            uint32_t rule=states_[output].rule;
            size_t begin=i+1-rules_[rule].alias.size(),end=i+1;
            if(!ascii){begin=offsets[begin];end=offsets[end];if(begin==none||end==none)continue;}
            bool left=begin==0 || (ascii?!ascii_word(source[begin-1]):!word(g_utf8_get_char(g_utf8_prev_char(source.data()+begin))));
            bool right=end==source.size() || (ascii?!ascii_word(source[end]):!word(g_utf8_get_char(source.data()+end)));
            if(left&&right&&end>best[begin].end)best[begin]={rule,static_cast<uint32_t>(end)};
        }
    }
    DictionaryText out;out.text.reserve(source.size());size_t copied=0;
    for(size_t pos=0;pos<source.size();) {
        auto match=best[pos];
        if(match.rule==none){++pos;continue;}
        const auto& replacement=rules_[match.rule].canonical;
        out.text.append(source,copied,pos-copied);out.text+=replacement;
        auto before=source.substr(pos,match.end-pos);
        if(before!=replacement)out.edits.push_back({pos,match.end,std::move(before),replacement,"dictionary",true});
        pos=copied=match.end;
    }
    out.text.append(source,copied,std::string::npos);return out;
}
DictionaryText apply_dictionary(const std::string& source,const std::vector<DictionaryEntry>& entries) {
    return DictionaryMatcher(entries).apply(source);
}
} // namespace paraketto::service
