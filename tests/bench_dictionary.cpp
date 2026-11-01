// CPU-only matcher benchmark. The prepared path models a cache hit during ASR;
// DICTIONARY_PREPARED=0 can also compile against the pre-cache dictionary.cpp.
#include "service/dictionary.h"
#include "app_common.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef DICTIONARY_PREPARED
#define DICTIONARY_PREPARED 1
#endif
using namespace paraketto::service;
using Clock=std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
int main() {
    for(int count:{10,100,1000,10000}) {
        std::vector<DictionaryEntry> entries{{"Paraketto",{"paraketo"}},
            {"New York",{"new york city"}},{"Straße",{"strasse"}}};
        for(int i=3;i<count;++i)entries.push_back({"Project"+std::to_string(i),{"project number "+std::to_string(i)}});
        const auto json=serialize_dictionary(entries);
        auto start=Clock::now();
#if DICTIONARY_PREPARED
        DictionaryMatcher prepared(parse_dictionary(json));
#else
        auto parsed=parse_dictionary(json);
#endif
        const auto prepare_ms=milliseconds(start);
        for(int words:{100,500}) {
            std::vector<std::string> vocabulary;
            std::istringstream phrase("i want to send the paraketo notes to project number 7 tomorrow and meet in new york city after checking the Straße address with José. please keep the original recording and the uncertain parts of this thought because i have not decided on a date yet.");
            for(std::string word;phrase>>word;)vocabulary.push_back(word);
            std::string text;
            for(int i=0;i<words;++i){if(i)text+=' ';text+=vocabulary[i%vocabulary.size()];}
            std::vector<double> samples;DictionaryText output;
            for(int repeat=0;repeat<14;++repeat) {
                start=Clock::now();
#if DICTIONARY_PREPARED
                output=prepared.apply(text);
#else
                output=apply_dictionary(text,parse_dictionary(json));
#endif
                const auto elapsed=milliseconds(start);if(repeat>=3)samples.push_back(elapsed);
            }
            std::sort(samples.begin(),samples.end());
            auto hash=paraketto::app::take_string(g_compute_checksum_for_data(G_CHECKSUM_SHA256,
                reinterpret_cast<const guchar*>(output.text.data()),output.text.size()));
            std::cout<<std::fixed<<std::setprecision(6)<<"{\"entries\":"<<count<<",\"words\":"<<words
                <<",\"prepare_ms\":"<<prepare_ms<<",\"median_ms\":"<<samples[samples.size()/2]
                <<",\"edits\":"<<output.edits.size()<<",\"output_sha256\":\""<<hash<<"\"}\n";
        }
    }
}
