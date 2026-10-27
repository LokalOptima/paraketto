#pragma once
#include <string>
#include <vector>

namespace desktop {

struct Entry {
    std::string id, time, text, destination, delivery, error;
    double duration = 0, inference_ms = 0;
};

struct HistoryStats {
    size_t words = 0;
    long long words_per_minute = 0;
};
HistoryStats history_stats(const std::vector<Entry>& entries);

class History {
public:
    explicit History(std::string directory);
    std::vector<Entry> load() const;
    Entry create(const std::string& wav, const std::string& destination);
    void save(const Entry& entry) const;
    void remove(const Entry& entry) const;
    std::string audio_path(const Entry& entry) const;
    const std::string& directory() const { return directory_; }
private:
    std::string directory_;
};

} // namespace desktop
