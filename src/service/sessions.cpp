#include "sessions.h"
#include "app_common.h"
#include "cpp-httplib/httplib.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace paraketto::service {
using namespace paraketto::app;
namespace fs = std::filesystem;
namespace {
bool active_capture(const std::string& state) { return state == "recording" || state == "paused"; }
void object_field(JsonBuilder* b, const char* key, const std::string& text) {
    Json json(text);
    auto* node = json_node_init_object(json_node_alloc(), json.object());
    json_builder_set_member_name(b,key); json_builder_add_value(b,node);
}
std::string dump(JsonBuilder* b) { auto s = json_string(b); g_object_unref(b); return s; }
int64_t parameter(const httplib::Request& r, const char* name, int64_t fallback, int64_t maximum = INT64_MAX) {
    if (!r.has_param(name)) return fallback;
    auto text = r.get_param_value(name);
    size_t used = 0; int64_t result;
    try { result = std::stoll(text,&used); } catch (...) { throw StoreError(400,"Invalid query parameter"); }
    if (used != text.size() || result < 0 || result > maximum) throw StoreError(400,"Invalid query parameter");
    return result;
}
std::string pipeline_file(const std::string& path) {
    gchar* escaped = g_strescape(path.c_str(),nullptr);
    std::string out = "filesrc location=\"" + std::string(escaped) + "\" ! wavparse";
    g_free(escaped); return out;
}
std::string source_json(bool filtering) {
    return std::string("{\"silence_filter\":") + (filtering ? "true" : "false") +
        ",\"policy\":\"silero-v5-conservative-v2\",\"vad_sha256\":\"0776449541b2b34cea5479bda89f6028e304ba0728fd4a1f053e17169f74a3eb\","
        "\"frame_samples\":512,\"onset\":0.15,\"continuation\":0.10,\"vad_target_rms\":0.01,\"vad_max_gain\":16,"
        "\"prepad_samples\":4800,\"postpad_samples\":8000,\"long_pause_samples\":32000,"
        "\"preferred_group_samples\":480000,\"preferred_pause_samples\":11200,\"maximum_group_samples\":1440000}";
}
}

Sessions::Sessions(SessionOptions options, Transcribe transcribe, SpeechActivity activity, AsrIdentity identity)
    : options_(std::move(options)), store_(options_.data_directory), transcribe_(std::move(transcribe)),
      activity_(std::move(activity)), asr_identity_(std::move(identity)) {
    // Recovery never opens the microphone. A valid active-file tail is adopted
    // before sealing the interrupted endpoint and rebuilding unplanned work.
    for (auto session : store_.recoverable_sessions()) {
        try {
            recover_audio(store_,session);
            if (active_capture(session.capture_state)) store_.seal_capture(session.id,"interrupted","Service stopped during recording");
        } catch (const std::exception& e) { store_.seal_capture(session.id,"interrupted",e.what()); }
    }
    store_.import_legacy_history(options_.data_directory + "/history");
    thread_ = std::thread([this] { run(); });
}
Sessions::~Sessions() { stop(); }
void Sessions::stop() {
    if (stopping_.exchange(true)) return;
    pause_ready_.notify_all();
    std::string active; { std::lock_guard<std::mutex> lock(mutex_); active = capture_id_; }
    if (!active.empty()) { try { finish_capture(active,false,true); } catch (...) {} }
    if (thread_.joinable()) thread_.join();
    store_.checkpoint();
}
std::string Sessions::start(const std::string& body, bool file) {
    Json request(body.empty() ? "{}" : body);
    auto key = request.text("request_id");
    if (key.empty()) key = request.text("idempotency_key");
    if (!g_uuid_string_is_valid(key.c_str())) throw StoreError(400,"A UUID request_id is required");
    std::lock_guard<std::mutex> control(capture_control_);
    if (stopping_) throw StoreError(503,"Service is stopping");
    { std::lock_guard<std::mutex> lock(mutex_);
        if (capture_ && capture_key_ != key) throw StoreError(409,"A capture session is already active");
    }
    std::string source = request.text("audio_source", "autoaudiosrc");
    if (source.size() > 4096) throw StoreError(400,"Audio source is too long");
    if (file) {
        auto path = request.text("path");
        if (!fs::path(path).is_absolute() || !fs::is_regular_file(path)) throw StoreError(400,"Import requires an absolute WAV file path");
        std::ifstream input(path,std::ios::binary); char h[12]{}; input.read(h,12);
        if (input.gcount()!=12 || std::string(h,4)!="RIFF" || std::string(h+8,4)!="WAVE") throw StoreError(400,"Import requires a WAV recording");
        source = pipeline_file(path);
    }
    bool filtering = request.boolean("silence_filter",true);
    auto session = store_.create_session(key,file ? "import" : "microphone",source_json(filtering));
    { std::lock_guard<std::mutex> lock(mutex_);
        if (session.capture_state != "recording" || capture_id_ == session.id) return store_.session_json(session.id);
        if (capture_) throw StoreError(409,"A capture session is already active");
    }
    auto id = session.id;
    auto capture = std::make_shared<AudioCapture>(options_.data_directory,id,source,!file,
        [this](const AudioBlock& b) { store_.put_audio_block(b); },
        [this,id](int64_t position,int64_t missing,const std::string& reason) { store_.record_capture_gap(id,position,missing ? missing : -1,reason); });
    try { capture->start(); }
    catch (const std::exception& e) { store_.seal_capture(id,"interrupted",e.what()); throw StoreError(503,e.what()); }
    { std::lock_guard<std::mutex> lock(mutex_); capture_ = capture; capture_id_ = id; capture_key_ = key; pause_planned_ = -1; }
    std::clog << "stage=capture event=start session=" << id << " source=" << (file ? "import" : "microphone") << '\n';
    return store_.session_json(id);
}
std::string Sessions::pause_capture(const std::string& id, bool paused) {
    std::lock_guard<std::mutex> control(capture_control_);
    if (stopping_) throw StoreError(503,"Service is stopping");
    auto session = store_.get_session(id);
    if (!session) throw StoreError(404,"Session not found");
    if (!active_capture(session->capture_state)) throw StoreError(409,"Recording is no longer active");
    std::shared_ptr<AudioCapture> capture;
    { std::lock_guard<std::mutex> lock(mutex_); if (capture_id_ == id) capture = capture_; }
    if (!capture) throw StoreError(409,"Recording is no longer active");
    if (!paused && session->capture_state == "recording" && !capture->state().paused) return store_.session_json(id);
    auto await_boundary = [&](int64_t target) {
        // Flush the detector through the durable boundary before allowing a
        // rapid Resume. ASR jobs remain asynchronous and no publication is made.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        std::unique_lock<std::mutex> lock(mutex_);
        while (pause_planned_ < target) {
            if (stopping_) throw StoreError(503,"Service is stopping");
            auto state = capture->state();
            if (state.finished || !state.error.empty())
                throw StoreError(503,state.error.empty() ? "Recording ended while pausing" : state.error);
            if (std::chrono::steady_clock::now() >= deadline)
                throw StoreError(503,"Recording paused, but its transcription boundary could not be saved");
            pause_ready_.wait_for(lock,std::chrono::milliseconds(50));
        }
    };
    try {
        if (paused) {
            capture->pause();
            const auto target = capture->state().durable;
            { std::lock_guard<std::mutex> lock(mutex_); speech_active_ = false; if (session->capture_state != "paused") pause_planned_ = -1; }
            store_.set_capture_paused(id,true);
            await_boundary(target);
        } else {
            if (capture->state().paused || session->capture_state == "paused") {
                // An earlier Pause may have stopped capture successfully while
                // its metadata or planning write failed. Retrying either action
                // must preserve that boundary before acquisition can restart.
                store_.set_capture_paused(id,true);
                await_boundary(capture->state().durable);
            }
            // Do not restart acquisition if persisting the state change fails.
            store_.set_capture_paused(id,false);
            capture->resume();
            std::lock_guard<std::mutex> lock(mutex_); pause_planned_ = -1;
        }
    } catch (const StoreError&) { throw; }
    catch (const std::exception& e) { throw StoreError(503,e.what()); }
    std::clog << "stage=capture event=" << (paused ? "pause" : "resume") << " session=" << id << '\n';
    return store_.session_json(id);
}
void Sessions::finish_capture(const std::string& id, bool cancel, bool interrupted) {
    std::lock_guard<std::mutex> control(capture_control_);
    std::shared_ptr<AudioCapture> capture;
    { std::lock_guard<std::mutex> lock(mutex_); if (capture_id_ == id) capture = capture_; }
    auto session = store_.get_session(id);
    if (!session) throw StoreError(404,"Session not found");
    if (capture) {
        capture->stop(); auto state = capture->state();
        if (!state.error.empty()) interrupted = true;
        store_.seal_capture(id,cancel ? "cancelled" : interrupted ? "interrupted" : "stopped",state.error);
        std::clog << "stage=capture event=stop session=" << id << " durable_samples=" << state.durable
                  << " interrupted=" << interrupted << " cancelled=" << cancel << '\n';
        std::lock_guard<std::mutex> lock(mutex_); capture_.reset(); capture_id_.clear(); capture_key_.clear(); speech_active_ = false;
    } else if (active_capture(session->capture_state)) store_.seal_capture(id,cancel ? "cancelled" : interrupted ? "interrupted" : "stopped");
    if (cancel) store_.cancel_session(id);
}
std::string Sessions::levels(const std::string& id) {
    std::shared_ptr<AudioCapture> capture; bool speech;
    { std::lock_guard<std::mutex> lock(mutex_); if (capture_id_ == id) capture = capture_; speech = speech_active_; }
    auto session = store_.get_session(id); if (!session) throw StoreError(404,"Session not found");
    CaptureState s; s.captured = s.written = s.durable = session->durable_samples;
    if (capture) s = capture->state();
    auto* b = json_builder_new(); json_builder_begin_object(b);
    json_field(b,"rms",s.rms); json_field(b,"elapsed_seconds",s.captured / double(audio_rate));
    json_int(b,"captured_samples",s.captured); json_int(b,"written_samples",s.written); json_int(b,"durable_samples",s.durable);
    json_bool(b,"speech_active",speech && capture && !s.paused); json_bool(b,"paused",s.paused); json_field(b,"error",s.error);
    json_builder_end_object(b); return dump(b);
}
std::string Sessions::status() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* b = json_builder_new(); json_builder_begin_object(b);
    json_field(b,"active_session_id",capture_id_);
    json_field(b,"vad_status",vad_status_); json_field(b,"session_error",engine_error_);
    object_field(b,"queues",store_.counts_json());
    json_builder_end_object(b); return dump(b);
}
void Sessions::plan(const SessionRecord& session) {
    auto found = planners_.find(session.id);
    if (found == planners_.end() || found->second.generation != session.generation ||
        (found->second.reset_on_resume && session.capture_state == "recording")) {
        bool filter = Json(session.settings_json).boolean("silence_filter",true);
        auto id = session.id; auto generation = session.generation;
        Planner planner{generation,std::make_unique<Segmenter>(session.planned_samples,options_.vad_weights,filter,
            [this,id,generation](int64_t start,int64_t end,const std::vector<SourceSpan>& spans) {
                auto revision = asr_identity_ ? asr_identity_() : options_.asr_revision;
                Json provenance(revision.empty() ? options_.asr_revision : revision);
                auto planner = planners_.find(id);
                bool filtering = planner != planners_.end() && planner->second.detector->filtering();
                Json policy(source_json(filtering));
                json_object_set_object_member(provenance.object(),"segmentation",json_object_ref(policy.object()));
                auto* node = json_node_init_object(json_node_alloc(),provenance.object());
                auto configuration = take_string(json_to_string(node,FALSE)); json_node_free(node);
                store_.plan_segment(id,generation,start,end,spans,configuration);
            })};
        if (session.planned_samples > 0) {
            auto from = std::max<int64_t>(0,session.planned_samples - 32*512);
            auto samples = read_audio(store_,id,from,session.planned_samples);
            planner.detector->warmup(samples.data(),samples.size());
        }
        found = planners_.insert_or_assign(session.id,std::move(planner)).first;
    }
    auto& detector = *found->second.detector;
    auto position = detector.position();
    auto available = session.durable_samples - position;
    bool paused = session.capture_state == "paused";
    bool flush = !active_capture(session.capture_state) || paused;
    if (available > 0) {
        int64_t count = std::min<int64_t>(available,512*32);
        bool final = flush && count == available;
        if (!final) count = count / 512 * 512;
        if (count > 0) {
            auto samples = read_audio(store_,session.id,position,position+count);
            detector.process(samples.data(),samples.size(),final);
        }
    } else if (flush) detector.flush();
    if (paused && detector.position() == session.durable_samples) found->second.reset_on_resume = true;
    { std::lock_guard<std::mutex> lock(mutex_);
        if (capture_id_ == session.id) {
            vad_status_ = detector.status(); speech_active_ = !paused && detector.speech_active();
            if (paused && detector.position() == session.durable_samples) {
                pause_planned_ = session.durable_samples; pause_ready_.notify_all();
            }
        }
    }
}
void Sessions::run() {
    size_t planner_turn = 0;
    while (!stopping_) {
        try {
            std::shared_ptr<AudioCapture> capture; std::string capture_id;
            { std::lock_guard<std::mutex> lock(mutex_);
                capture = capture_; capture_id = capture_id_;
            }
            if (capture && capture->state().finished) finish_capture(capture_id,false,!capture->state().error.empty());
            auto sessions = store_.processing_sessions();
            // Round-robin bounded planning keeps queued imports from delaying
            // an active microphone's transcription progress.
            if (!sessions.empty()) {
                const auto& session = sessions[planner_turn++ % sessions.size()];
                if (session.durable_samples > session.planned_samples || active_capture(session.capture_state)) plan(session);
                if (!active_capture(session.capture_state)) store_.publish(session.id);
            }
            for (auto it=planners_.begin(); it!=planners_.end();) {
                auto s = store_.get_session(it->first);
                if (!s || (!active_capture(s->capture_state) && s->planned_samples == s->durable_samples)) it=planners_.erase(it); else ++it;
            }
            std::string deletion_error;
            for (const auto& id : store_.pending_deletions()) {
                try {
                    if (!g_uuid_string_is_valid(id.c_str())) throw std::runtime_error("Invalid deleted session identity");
                    for (const auto& path : store_.deletion_paths(id)) {
                        fs::remove(path); sync_directory(fs::path(path).parent_path().string());
                    }
                    auto directory = fs::path(options_.data_directory)/"sessions"/id;
                    fs::remove_all(directory);
                    if (fs::exists(directory.parent_path())) sync_directory(directory.parent_path().string());
                    store_.finish_deletion(id);
                } catch (const std::exception& e) {
                    deletion_error = "Could not finish deleting session " + id + ": " + e.what();
                }
            }
            if (asr_result_ && asr_result_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto future = std::move(*asr_result_); asr_result_.reset();
                auto result = future.get();
                if (result.status == 200) {
                    Json j(result.body);
                    auto revision = result.model_revision.empty() ? options_.asr_revision : result.model_revision;
                    Json expected(asr_job_->config_json), actual(revision);
                    bool matches = true;
                    for (const auto* key : {"weights_sha256", "runtime_sha256", "precision"}) {
                        auto required = expected.text(key);
                        if (!required.empty() && required != actual.text(key)) matches = false;
                    }
                    if (matches) store_.commit_asr(*asr_job_,j.text("text"),revision,j.number("inference_ms",j.number("inference_time_s")*1000));
                    else store_.fail_asr(*asr_job_,"ASR model or runtime changed; use Reprocess to transcribe with the current assets");
                } else store_.fail_asr(*asr_job_,Json(result.body).text("error","Transcription failed"),result.status == 429);
                std::clog << "stage=asr event=complete session=" << asr_job_->session_id << " job=" << asr_job_->id << " status=" << result.status << '\n';
                asr_result_.reset(); asr_job_.reset();
            }
            if (!asr_job_) asr_job_ = store_.next_asr();
            if (asr_job_ && !asr_result_) {
                std::string wav; int64_t frames = 0;
                for (auto span : asr_job_->spans) frames += span.end-span.start;
                if (frames > 90*audio_rate) throw std::runtime_error("ASR job exceeds input bound");
                wav = wav_header(frames); wav.reserve(44 + frames*2);
                for (auto span : asr_job_->spans) { auto pcm = read_audio(store_,asr_job_->session_id,span.start,span.end); wav.append(reinterpret_cast<const char*>(pcm.data()),pcm.size()*2); }
                asr_result_.emplace(transcribe_(std::move(wav)));
                std::clog << "stage=asr event=start session=" << asr_job_->session_id << " job=" << asr_job_->id << " samples=" << frames << '\n';
            }
            bool speaking;
            { std::lock_guard<std::mutex> lock(mutex_); engine_error_ = deletion_error; speaking = speech_active_; }
            if (activity_) activity_(speaking);
        } catch (const std::exception& e) {
            { std::lock_guard<std::mutex> lock(mutex_); engine_error_ = e.what(); }
            if (asr_job_ && !asr_result_) {
                try { store_.fail_asr(*asr_job_,e.what()); asr_job_.reset(); }
                catch (...) { /* Retain the claimed job until storage is writable again. */ }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (activity_) activity_(false);
    if (asr_job_) { try { store_.fail_asr(*asr_job_,"Service stopped",true); } catch (...) {} }
}

void Sessions::routes(httplib::Server& http) {
    auto wrap = [](auto handler) {
        return [handler](const httplib::Request& req, httplib::Response& res) {
            try { res.set_content(handler(req),"application/json"); }
            catch (const StoreError& e) { res.status=e.status; res.set_content(json_error(e.what()),"application/json"); }
        };
    };
    http.Post("/sessions",wrap([this](const auto& r) { return start(r.body,false); }));
    http.Post("/sessions/import",wrap([this](const auto& r) { return start(r.body,true); }));
    http.Get("/sessions",wrap([this](const auto& r) { return store_.sessions_json(parameter(r,"limit",50,100),parameter(r,"cursor",0,INT32_MAX)); }));
    http.Get(R"(/sessions/([a-zA-Z0-9-]+))",wrap([this](const auto& r) { return store_.session_json(r.matches[1]); }));
    http.Get(R"(/sessions/([a-zA-Z0-9-]+)/updates)",wrap([this](const auto& r) { return store_.updates_json(r.matches[1],parameter(r,"after",0),parameter(r,"limit",50,100),parameter(r,"generation",0)); }));
    http.Get(R"(/sessions/([a-zA-Z0-9-]+)/levels)",wrap([this](const auto& r) { return levels(r.matches[1]); }));
    http.Get(R"(/sessions/([a-zA-Z0-9-]+)/publication)",wrap([this](const auto& r) { return store_.publication_json(r.matches[1]); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/pause)",wrap([this](const auto& r) { return pause_capture(r.matches[1],true); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/resume)",wrap([this](const auto& r) { return pause_capture(r.matches[1],false); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/stop)",wrap([this](const auto& r) { std::string id=r.matches[1]; finish_capture(id); return store_.session_json(id); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/cancel)",wrap([this](const auto& r) { std::string id=r.matches[1]; finish_capture(id,true); return store_.session_json(id); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/retry)",wrap([this](const auto& r) { Json j(r.body.empty()?"{}":r.body); store_.retry(r.matches[1],j.text("stage","all")); return store_.session_json(r.matches[1]); }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/reprocess)",wrap([this](const auto& r) { Json j(r.body.empty()?"{}":r.body); store_.reprocess(r.matches[1],j.boolean("audio",true),source_json(j.boolean("silence_filter",false)),int64_t(j.number("dictionary_revision"))); return store_.session_json(r.matches[1]); }));
    http.Patch(R"(/sessions/([a-zA-Z0-9-]+)/replacements)",wrap([this](const auto& r) {
        Json j(r.body);
        if (!json_object_has_member(j.object(),"enabled") || json_node_get_value_type(json_object_get_member(j.object(),"enabled")) != G_TYPE_BOOLEAN)
            throw StoreError(400,"enabled must be a boolean");
        auto revision = j.number("expected_revision",-1);
        if (json_object_has_member(j.object(),"expected_revision") &&
            (!std::isfinite(revision) || revision < 0 || std::floor(revision) != revision || revision >= double(INT64_MAX)))
            throw StoreError(400,"expected_revision must be a nonnegative integer");
        store_.set_replacements_enabled(r.matches[1],j.boolean("enabled"),int64_t(revision)); return store_.session_json(r.matches[1]);
    }));
    http.Delete(R"(/sessions/([a-zA-Z0-9-]+))",wrap([this](const auto& r) {
        std::string id=r.matches[1]; finish_capture(id,true); store_.delete_session(id); return std::string("{}");
    }));
    http.Get("/dictionary",wrap([this](const auto&) { return store_.dictionary_json(); }));
    http.Put("/dictionary",wrap([this](const auto& r) { store_.set_dictionary(r.body); return store_.dictionary_json(); }));
    http.Post("/dictionary/entries",wrap([this](const auto& r) { store_.add_dictionary_entry(r.body); return store_.dictionary_json(); }));
    http.Put(R"(/dictionary/entries/([0-9]{1,5}))",wrap([this](const auto& r) {
        store_.update_dictionary_entry(std::stoul(r.matches[1]),r.body);return store_.dictionary_json();
    }));
    http.Delete(R"(/dictionary/entries/([0-9]{1,5}))",wrap([this](const auto& r) {
        store_.update_dictionary_entry(std::stoul(r.matches[1]),r.body,true);return store_.dictionary_json();
    }));
    http.Post(R"(/sessions/([a-zA-Z0-9-]+)/spelling)",wrap([this](const auto& r) {
        store_.correct_spelling(r.matches[1],r.body);return store_.session_json(r.matches[1]);
    }));
    http.Get(R"(/sessions/([a-zA-Z0-9-]+)/transcript)",[this](const auto& r,auto& res) {
        try { auto view=r.has_param("view")?r.get_param_value("view"):"edited"; res.set_content(store_.transcript(r.matches[1],view),"text/plain; charset=utf-8"); }
        catch (const StoreError& e) { res.status=e.status; res.set_content(json_error(e.what()),"application/json"); }
    });
    serve_audio(http);
}
void Sessions::serve_audio(httplib::Server& http) {
    http.Get(R"(/sessions/([a-zA-Z0-9-]+)/audio)",[this](const auto& r,auto& res) {
        try {
            std::string id=r.matches[1]; auto session=store_.get_session(id);
            if (!session) throw StoreError(404,"Session not found");
            int64_t start=parameter(r,"start",0,session->durable_samples), end=parameter(r,"end",session->durable_samples,session->durable_samples);
            if (end<start) throw StoreError(400,"Invalid audio range");
            auto header=wav_header(end-start); size_t length=44+(end-start)*2;
            res.set_content_provider(length,"audio/wav",[this,id,start,header](size_t offset,size_t requested,httplib::DataSink& sink) {
                try {
                    if (offset<44) { size_t n=std::min<size_t>(requested,44-offset); if (!sink.write(header.data()+offset,n)) return false; offset+=n; requested-=n; }
                    size_t n=std::min<size_t>(requested,65536);
                    if (n) {
                        int64_t first=start+(offset-44)/2; size_t skip=(offset-44)%2;
                        auto samples=read_audio(store_,id,first,first+(n+skip+1)/2);
                        return sink.write(reinterpret_cast<const char*>(samples.data())+skip,n);
                    }
                    return true;
                } catch (...) { return false; }
            });
        } catch (const StoreError& e) { res.status=e.status; res.set_content(json_error(e.what()),"application/json"); }
    });
}
}
