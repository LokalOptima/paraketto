#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace paraketto::service {

struct StoreError : std::runtime_error {
    int status;
    StoreError(int status, const std::string& message) : std::runtime_error(message), status(status) {}
};
struct SourceSpan { int64_t start = 0, end = 0; };
struct AudioBlock {
    std::string session_id, path;
    int64_t sequence = 0, start_sample = 0, durable_frames = 0;
    bool finalized = false;
};
struct SessionRecord {
    std::string id, source, capture_state, error, settings_json;
    int64_t generation = 1, durable_samples = 0, planned_samples = 0, final_samples = -1;
    int64_t dictionary_revision = 0, created_us = 0;
    bool replacements_enabled = true;
};
struct AsrJob {
    int64_t id = 0, generation = 0, sequence = 0, attempt = 0;
    int64_t coverage_start = 0, coverage_end = 0;
    std::string session_id, config_json;
    std::vector<SourceSpan> spans;
};
struct Publication {
    std::string id, session_id, text;
    int64_t generation = 0;
    bool complete = false;
};

// One instance per service data directory, after the service's process-lifetime lock.
// Calls serialize internally. Never call from the GStreamer callback. Files must
// already be synchronized before put_audio_block publishes a durable position.
class SessionStore {
public:
    explicit SessionStore(const std::string& data_directory);
    ~SessionStore();
    SessionStore(const SessionStore&) = delete;
    SessionStore& operator=(const SessionStore&) = delete;
    const std::string& data_directory() const;

    SessionRecord create_session(const std::string& idempotency_key,
                                const std::string& source = "microphone",
                                const std::string& settings_json = "{}");
    std::optional<SessionRecord> get_session(const std::string& id);
    std::optional<SessionRecord> active_session();
    std::vector<SessionRecord> recoverable_sessions();
    std::vector<SessionRecord> processing_sessions();
    void put_audio_block(const AudioBlock& block);
    void record_capture_gap(const std::string& id, int64_t sample_position,
                            int64_t missing_samples, const std::string& reason);
    std::vector<AudioBlock> audio_blocks(const std::string& id, int64_t start = 0, int64_t end = INT64_MAX);
    // Covers [start,end) exactly once, including any skipped silence between spans.
    // Empty spans represent skipped silence; gap=true records missing source audio.
    int64_t plan_segment(const std::string& id, int64_t generation,
                         int64_t start, int64_t end, const std::vector<SourceSpan>& spans,
                         const std::string& asr_config_json = "{}", bool gap = false);
    // Pausing retains an open endpoint and the same capture identity.
    void set_capture_paused(const std::string& id, bool paused);
    void seal_capture(const std::string& id, const std::string& state = "stopped",
                      const std::string& error = "");
    void cancel_session(const std::string& id);
    void delete_session(const std::string& id);
    std::vector<std::string> pending_deletions();
    // Only original legacy files within this store's own history directory.
    // Empty for ordinary WAV imports; rejects symlinks and outside paths.
    std::vector<std::string> deletion_paths(const std::string& id);
    void finish_deletion(const std::string& id);

    std::optional<AsrJob> next_asr(); // atomically claims a pending job
    bool commit_asr(const AsrJob& job, const std::string& raw_text,
                    const std::string& model_revision, double inference_ms = 0);
    bool fail_asr(const AsrJob& job, const std::string& reason, bool requeue = false);
    void retry(const std::string& id, const std::string& stage = "all");
    int64_t reprocess(const std::string& id, bool audio,
                      const std::string& settings_json = "{}", int64_t dictionary_revision = 0);
    std::optional<Publication> publish(const std::string& id);
    std::optional<Publication> publication(const std::string& id);
    std::string transcript(const std::string& id, const std::string& view = "edited");

    std::string dictionary_json();
    int64_t set_dictionary(const std::string& json); // {"entries":[{"canonical":...,"aliases":[...]}]}
    int64_t add_dictionary_entry(const std::string& json); // atomic merge: {"canonical":...,"aliases":[...]}
    // A list index is meaningful only together with its required expected_revision.
    int64_t update_dictionary_entry(size_t index, const std::string& json, bool remove = false);
    // Save a displayed selection as an alias, update that ASR section, and use
    // the new rule for subsequent segments. Capture and saved publications stay intact.
    void correct_spelling(const std::string& id, const std::string& json);
    // An explicit target is idempotent and normally last-writer-wins. Clients
    // may supply the session revision for an optimistic concurrency check.
    void set_replacements_enabled(const std::string& id, bool enabled, int64_t expected_revision = -1);
    // Metadata stays bounded. Text/edits are paged via updates or explicit exports.
    std::string session_json(const std::string& id);
    std::string sessions_json(int limit = 50, int offset = 0);
    std::string updates_json(const std::string& id, int64_t after = 0, int limit = 50,
                             int64_t expected_generation = 0);
    std::string publication_json(const std::string& id); // explicit full snapshot retrieval
    std::string counts_json();
    void import_legacy_history(const std::string& directory);
    void checkpoint();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace paraketto::service
