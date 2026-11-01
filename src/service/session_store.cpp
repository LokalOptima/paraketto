#include "session_store.h"
#include "dictionary.h"
#include "edits.h"
#include "app_common.h"
#include "sqlite/sqlite3.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#include <glib/gstdio.h>
#include <sys/stat.h>
#include <sys/random.h>
#include <cerrno>
#include <unistd.h>

namespace paraketto::service {
using namespace app;
namespace {
class Statement {
    sqlite3* db_; sqlite3_stmt* stmt_ = nullptr;
public:
    Statement(sqlite3* db,const char* sql):db_(db) {if(sqlite3_prepare_v2(db,sql,-1,&stmt_,nullptr)!=SQLITE_OK)throw std::runtime_error(sqlite3_errmsg(db));}
    ~Statement(){sqlite3_finalize(stmt_);}
    Statement& bind(int n,const std::string& value){if(sqlite3_bind_text(stmt_,n,value.data(),value.size(),SQLITE_TRANSIENT)!=SQLITE_OK)throw std::runtime_error(sqlite3_errmsg(db_));return *this;}
    Statement& bind(int n,const char* value){return bind(n,std::string(value));}
    Statement& bind(int n,int64_t value){sqlite3_bind_int64(stmt_,n,value);return *this;}
    Statement& bind(int n,int value){return bind(n,int64_t(value));}
    Statement& bind(int n,long long value){return bind(n,int64_t(value));}
    Statement& real(int n,double value){sqlite3_bind_double(stmt_,n,value);return *this;}
    bool row(){int rc=sqlite3_step(stmt_);if(rc==SQLITE_ROW)return true;if(rc==SQLITE_DONE)return false;throw std::runtime_error(sqlite3_errmsg(db_));}
    void done(){if(row())throw std::runtime_error("Unexpected SQLite row");}
    int64_t number(int c)const{return sqlite3_column_int64(stmt_,c);}
    double real(int c)const{return sqlite3_column_double(stmt_,c);}
    std::string text(int c)const {auto* s=sqlite3_column_text(stmt_,c);return s?std::string(reinterpret_cast<const char*>(s),sqlite3_column_bytes(stmt_,c)):std::string();}
};
void sql(sqlite3* db,const char* text){char* error=nullptr;if(sqlite3_exec(db,text,nullptr,nullptr,&error)!=SQLITE_OK){std::string message=error?error:sqlite3_errmsg(db);sqlite3_free(error);throw std::runtime_error(message);}}
struct Transaction {sqlite3* db;bool committed=false;explicit Transaction(sqlite3* db):db(db){sql(db,"BEGIN IMMEDIATE");}~Transaction(){if(!committed)sqlite3_exec(db,"ROLLBACK",nullptr,nullptr,nullptr);}void commit(){sql(db,"COMMIT");committed=true;}};
std::string uuid(){
    unsigned char bytes[16];size_t done=0;
    while(done<sizeof bytes){ssize_t count=getrandom(bytes+done,sizeof bytes-done,0);if(count<0&&errno==EINTR)continue;if(count<=0)throw std::runtime_error("Could not generate session identity");done+=count;}
    bytes[6]=(bytes[6]&0x0f)|0x40;bytes[8]=(bytes[8]&0x3f)|0x80;
    char result[37];g_snprintf(result,sizeof result,"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",bytes[0],bytes[1],bytes[2],bytes[3],bytes[4],bytes[5],bytes[6],bytes[7],bytes[8],bytes[9],bytes[10],bytes[11],bytes[12],bytes[13],bytes[14],bytes[15]);return result;
}
std::string spans_json(const std::vector<SourceSpan>& spans){auto*b=json_builder_new();json_builder_begin_object(b);json_builder_set_member_name(b,"spans");json_builder_begin_array(b);for(auto&s:spans){json_builder_begin_object(b);json_int(b,"start",s.start);json_int(b,"end",s.end);json_builder_end_object(b);}json_builder_end_array(b);json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;}
std::vector<SourceSpan> parse_spans(const std::string& value){Json j(value);std::vector<SourceSpan> out;auto*a=json_object_get_array_member(j.object(),"spans");for(guint i=0;i<json_array_get_length(a);++i){auto*o=json_array_get_object_element(a,i);out.push_back({json_object_get_int_member(o,"start"),json_object_get_int_member(o,"end")});}return out;}
int64_t word_count(const std::string& value){int64_t n=0;bool previous=false;for(auto*p=value.c_str();*p;p=g_utf8_next_char(p)){gunichar c=g_utf8_get_char(p);bool w=g_unichar_isalnum(c)||g_unichar_ismark(c)||((c=='\''||c==0x2019)&&previous);if(w&&!previous)++n;previous=w;}return n;}
std::string clip(const std::string& value,size_t limit){const char*p=value.data();size_t n=0;while(p<value.data()+value.size()&&n++<limit)p=g_utf8_next_char(p);return value.substr(0,p-value.data());}
int64_t legacy_time(const std::filesystem::path& path){
    auto stamp=path.stem().string();int year=0,month=0,day=0,hour=0,minute=0,second=0;
    if(stamp.size()>=15 && sscanf(stamp.c_str(),"%4d%2d%2dT%2d%2d%2d",&year,&month,&day,&hour,&minute,&second)==6){
        auto* time=g_date_time_new_local(year,month,day,hour,minute,second);
        if(time){auto result=g_date_time_to_unix(time)*G_USEC_PER_SEC;g_date_time_unref(time);return result;}
    }
    struct stat metadata{};if(g_stat(path.c_str(),&metadata)==0)return int64_t(metadata.st_mtime)*G_USEC_PER_SEC;
    return g_get_real_time();
}
std::string timestamp(int64_t micros){auto*t=g_date_time_new_from_unix_local(micros/1000000);auto s=take_string(g_date_time_format_iso8601(t));g_date_time_unref(t);return s;}

}

struct SessionStore::Impl {
    sqlite3* db=nullptr;std::string directory;std::recursive_mutex mutex;
    struct PreparedDictionary {
        std::vector<DictionaryEntry> entries;
        DictionaryMatcher matcher;
        explicit PreparedDictionary(std::vector<DictionaryEntry> value):entries(std::move(value)),matcher(entries){}
        size_t bytes()const {
            size_t size=matcher.storage_bytes()+entries.capacity()*sizeof(DictionaryEntry);
            for(const auto& e:entries){size+=e.canonical.capacity()+e.aliases.capacity()*sizeof(std::string);for(const auto& a:e.aliases)size+=a.capacity();}
            return size;
        }
    };
    struct CachedDictionary {int64_t id;std::shared_ptr<const PreparedDictionary> prepared;};
    std::list<CachedDictionary> dictionaries;
    void cache_dictionary(int64_t id,std::shared_ptr<const PreparedDictionary> prepared) {
        dictionaries.remove_if([&](const auto& d){return d.id==id;});
        dictionaries.push_front({id,std::move(prepared)});
        size_t bytes=0;for(const auto& d:dictionaries)bytes+=d.prepared->bytes();
        // Retain at most four revisions / 64 MiB, or one unusually large active
        // dictionary. Shared readers keep an evicted revision alive only as needed.
        while(dictionaries.size()>1 && (dictionaries.size()>4 || bytes>64*1024*1024)) {
            bytes-=dictionaries.back().prepared->bytes();dictionaries.pop_back();
        }
    }
    bool has_retired_runs=false;
    explicit Impl(std::string path):directory(std::move(path)) {
        if(g_mkdir_with_parents(directory.c_str(),0700))throw std::runtime_error("Could not create session data directory");
        std::string file=directory+"/sessions.sqlite3";bool existed=g_file_test(file.c_str(),G_FILE_TEST_EXISTS);
        if(sqlite3_open_v2(file.c_str(),&db,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_FULLMUTEX,nullptr)!=SQLITE_OK){std::string error=sqlite3_errmsg(db);sqlite3_close(db);db=nullptr;throw std::runtime_error(error);}
        try {
            g_chmod(file.c_str(),0600);sqlite3_busy_timeout(db,3000);
            sql(db,"PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA wal_autocheckpoint=1000;");
            {Statement s(db,"PRAGMA journal_mode");if(!s.row()||s.text(0)!="wal")throw std::runtime_error("SQLite WAL unavailable");}
            {Statement s(db,"PRAGMA synchronous");if(!s.row()||s.number(0)!=2)throw std::runtime_error("SQLite FULL durability unavailable");}
            {Statement s(db,"PRAGMA foreign_keys");if(!s.row()||s.number(0)!=1)throw std::runtime_error("SQLite foreign keys unavailable");}
            int64_t version=0;{Statement s(db,"PRAGMA user_version");s.row();version=s.number(0);}
            if(version>4)throw std::runtime_error("Session database belongs to a newer application");
            if(version<1){if(existed)backup(directory+"/sessions.pre-v1.sqlite3");create_schema();version=4;}
            if(version<2){backup(directory+"/sessions.pre-v2.sqlite3");Transaction migration(db);sql(db,"ALTER TABLE sessions ADD COLUMN delivery_epoch INTEGER NOT NULL DEFAULT 0; PRAGMA user_version=2;");migration.commit();version=2;}
            if(version<3){backup(directory+"/sessions.pre-v3.sqlite3");retire_model_edits();version=3;}
            if(version<4){backup(directory+"/sessions.pre-v4.sqlite3");migrate_replacements();}
            {Statement archived(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='retired_cleanup_runs'");has_retired_runs=archived.row();}
            // The service's directory lock excludes another controller. Running
            // work can safely be retried; attempt fences reject old completions.
            Transaction tx(db);
            sql(db,"UPDATE sessions SET delivery_epoch=delivery_epoch+1,revision=revision+1 WHERE deleted=0; UPDATE sessions SET capture_state='interrupted',final_samples=durable_samples,error='Service stopped during recording' WHERE capture_state IN ('recording','paused') AND deleted=0; UPDATE generations SET final_samples=(SELECT final_samples FROM sessions WHERE sessions.id=generations.session_id) WHERE generation=(SELECT generation FROM sessions WHERE sessions.id=generations.session_id) AND final_samples<0 AND (SELECT capture_state FROM sessions WHERE sessions.id=generations.session_id)='interrupted'; UPDATE segments SET state='pending' WHERE state='running';");tx.commit();
        }catch(...){sqlite3_close(db);db=nullptr;throw;}
    }
    ~Impl(){if(db){sqlite3_wal_checkpoint_v2(db,nullptr,SQLITE_CHECKPOINT_PASSIVE,nullptr,nullptr);sqlite3_close(db);}}
    void backup(const std::string& path){sqlite3* target=nullptr;if(sqlite3_open(path.c_str(),&target)!=SQLITE_OK){sqlite3_close(target);throw std::runtime_error("Cannot create database migration backup");}auto* copy=sqlite3_backup_init(target,"main",db,"main");if(!copy){sqlite3_close(target);throw std::runtime_error("Cannot initialize database backup");}int rc=sqlite3_backup_step(copy,-1);int finish=sqlite3_backup_finish(copy);sqlite3_close(target);g_chmod(path.c_str(),0600);if(rc!=SQLITE_DONE||finish!=SQLITE_OK)throw std::runtime_error("Database backup failed");}
    void create_schema(){Transaction tx(db);sql(db,R"SQL(
CREATE TABLE dictionaries(id INTEGER PRIMARY KEY, json TEXT NOT NULL, hash TEXT NOT NULL UNIQUE, created_us INTEGER NOT NULL);
INSERT INTO dictionaries VALUES(1,'{"entries":[]}','empty',0);
CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
INSERT INTO settings VALUES('dictionary','1');
CREATE TABLE sessions(id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, source TEXT NOT NULL, created_us INTEGER NOT NULL,
 capture_state TEXT NOT NULL,settings TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 1,dict_id INTEGER NOT NULL REFERENCES dictionaries(id),
 durable_samples INTEGER NOT NULL DEFAULT 0,planned_samples INTEGER NOT NULL DEFAULT 0,final_samples INTEGER NOT NULL DEFAULT -1,
 error TEXT NOT NULL DEFAULT '',deleted INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL DEFAULT 1,delivery_epoch INTEGER NOT NULL DEFAULT 0,
 replacements_enabled INTEGER NOT NULL DEFAULT 1 CHECK(replacements_enabled IN (0,1)));
CREATE TABLE generations(session_id TEXT REFERENCES sessions(id),generation INTEGER,settings TEXT NOT NULL,dict_id INTEGER REFERENCES dictionaries(id),
 planned_samples INTEGER NOT NULL DEFAULT 0,final_samples INTEGER NOT NULL DEFAULT -1,
 replacements_enabled INTEGER NOT NULL DEFAULT 1 CHECK(replacements_enabled IN (0,1)),PRIMARY KEY(session_id,generation));
CREATE TABLE audio_blocks(session_id TEXT REFERENCES sessions(id),sequence INTEGER,path TEXT NOT NULL,start_sample INTEGER NOT NULL,
 durable_frames INTEGER NOT NULL,finalized INTEGER NOT NULL,PRIMARY KEY(session_id,sequence));
CREATE TABLE capture_gaps(id INTEGER PRIMARY KEY,session_id TEXT REFERENCES sessions(id),position INTEGER NOT NULL,missing_samples INTEGER NOT NULL,reason TEXT NOT NULL,
 UNIQUE(session_id,position,reason));
CREATE TABLE segments(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,generation INTEGER NOT NULL,sequence INTEGER NOT NULL,
 coverage_start INTEGER NOT NULL,coverage_end INTEGER NOT NULL,spans TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',attempt INTEGER NOT NULL DEFAULT 0,
 config TEXT NOT NULL,revision_id INTEGER,edited_text TEXT NOT NULL DEFAULT '',words INTEGER NOT NULL DEFAULT 0,error TEXT NOT NULL DEFAULT '',cursor INTEGER NOT NULL DEFAULT 0,
 FOREIGN KEY(session_id,generation) REFERENCES generations(session_id,generation),UNIQUE(session_id,generation,sequence),UNIQUE(session_id,generation,coverage_start));
CREATE TABLE dispositions(session_id TEXT NOT NULL,generation INTEGER NOT NULL,start_sample INTEGER NOT NULL,end_sample INTEGER NOT NULL,kind TEXT NOT NULL,
 FOREIGN KEY(session_id,generation) REFERENCES generations(session_id,generation),PRIMARY KEY(session_id,generation,start_sample));
CREATE TABLE transcript_revisions(id INTEGER PRIMARY KEY,segment_id INTEGER REFERENCES segments(id),text TEXT NOT NULL,hash TEXT NOT NULL,
 model TEXT NOT NULL,inference_ms REAL NOT NULL,provenance TEXT NOT NULL DEFAULT 'asr',created_us INTEGER NOT NULL);
CREATE TABLE edits(id INTEGER PRIMARY KEY,revision_id INTEGER REFERENCES transcript_revisions(id),start_byte INTEGER NOT NULL,end_byte INTEGER NOT NULL,
 before_text TEXT NOT NULL,after_text TEXT NOT NULL,origin TEXT NOT NULL,origin_id INTEGER NOT NULL,enabled INTEGER NOT NULL,decision_revision INTEGER NOT NULL DEFAULT 1,user_decided INTEGER NOT NULL DEFAULT 0,
 UNIQUE(revision_id,start_byte,end_byte,before_text,after_text,origin));
CREATE TABLE publications(id TEXT PRIMARY KEY,session_id TEXT REFERENCES sessions(id),generation INTEGER NOT NULL,text TEXT NOT NULL,
 source_refs TEXT NOT NULL,complete INTEGER NOT NULL,created_us INTEGER NOT NULL,UNIQUE(session_id,generation));
CREATE TABLE legacy_imports(path TEXT PRIMARY KEY,session_id TEXT REFERENCES sessions(id));
CREATE INDEX pending_asr ON segments(state,session_id,generation,sequence);
CREATE INDEX changed_segments ON segments(session_id,generation,cursor);
CREATE INDEX edit_revision ON edits(revision_id,start_byte);
PRAGMA user_version=4;
)SQL");tx.commit();}
    void retire_model_edits(){
        Transaction migration(db);
        // Old model outputs remain as audit evidence only. The renamed table
        // has no scheduler, and every former job is explicitly retired. Preserve
        // its previous outcome separately without changing saved publications.
        sql(db,"DROP INDEX pending_cleanup; ALTER TABLE cleanup_runs RENAME TO retired_cleanup_runs; "
               "ALTER TABLE retired_cleanup_runs ADD COLUMN retired_state TEXT NOT NULL DEFAULT ''; "
               "UPDATE retired_cleanup_runs SET retired_state=state,state='retired'; "
               "UPDATE edits SET enabled=0,decision_revision=decision_revision+1 WHERE origin!='dictionary' AND enabled!=0; "
               "ALTER TABLE publications DROP COLUMN cleanup_complete;");
        int64_t last=0;
        for(;;){
            int64_t segment=0;std::string session;
            {Statement affected(db,"SELECT a.id,a.session_id FROM segments a WHERE a.id>? AND EXISTS(SELECT 1 FROM edits e WHERE e.revision_id=a.revision_id AND e.origin!='dictionary') ORDER BY a.id LIMIT 1");
             affected.bind(1,last);if(!affected.row())break;segment=affected.number(0);session=affected.text(1);}
            render_segment(segment);touch(session,segment);last=segment;
        }
        sql(db,"PRAGMA user_version=3;");migration.commit();
    }
    void migrate_replacements(){
        Transaction migration(db);
        sql(db,"ALTER TABLE sessions ADD COLUMN replacements_enabled INTEGER NOT NULL DEFAULT 1 CHECK(replacements_enabled IN (0,1)); "
               "ALTER TABLE generations ADD COLUMN replacements_enabled INTEGER NOT NULL DEFAULT 1 CHECK(replacements_enabled IN (0,1)); "
               // A single switch cannot represent mixed old decisions. Keep
               // prior opt-outs by turning that whole generation off, rather
               // than silently enabling a replacement the user had rejected.
               "UPDATE generations SET replacements_enabled=0 WHERE EXISTS(SELECT 1 FROM segments a JOIN edits e ON e.revision_id=a.revision_id "
               "WHERE a.session_id=generations.session_id AND a.generation=generations.generation AND e.origin='dictionary' AND e.enabled=0); "
               "UPDATE sessions SET replacements_enabled=coalesce((SELECT g.replacements_enabled FROM generations g WHERE g.session_id=sessions.id AND g.generation=sessions.generation),1);");
        Statement generations(db,"SELECT session_id,generation FROM generations WHERE replacements_enabled=0 ORDER BY session_id,generation");
        while(generations.row())update_replacement_edits(generations.text(0),generations.number(1),false,false);
        sql(db,"PRAGMA user_version=4;");migration.commit();
    }
    void update_replacement_edits(const std::string& id,int64_t generation,bool enabled,bool user_decided){
        int64_t last=0;
        for(;;){
            int64_t segment=0,revision=0;
            {Statement changed(db,"SELECT a.id,a.revision_id FROM segments a WHERE a.session_id=? AND a.generation=? AND a.id>? "
                                  "AND EXISTS(SELECT 1 FROM edits e WHERE e.revision_id=a.revision_id AND e.origin='dictionary' AND e.enabled!=?) ORDER BY a.id LIMIT 1");
             changed.bind(1,id).bind(2,generation).bind(3,last).bind(4,int(enabled));if(!changed.row())break;segment=changed.number(0);revision=changed.number(1);}
            Statement edits(db,"UPDATE edits SET enabled=?,decision_revision=decision_revision+1,user_decided=CASE WHEN ? THEN 1 ELSE user_decided END "
                               "WHERE revision_id=? AND origin='dictionary' AND enabled!=?");
            edits.bind(1,int(enabled)).bind(2,int(user_decided)).bind(3,revision).bind(4,int(enabled)).done();
            render_segment(segment);touch(id,segment);last=segment;
        }
    }
    SessionRecord get(const std::string& id,bool deleted=false){Statement s(db,"SELECT id,source,capture_state,error,settings,generation,durable_samples,planned_samples,final_samples,dict_id,created_us,deleted,replacements_enabled FROM sessions WHERE id=?");s.bind(1,id);if(!s.row()||(!deleted&&s.number(11)))throw StoreError(404,"Session not found");return {s.text(0),s.text(1),s.text(2),s.text(3),s.text(4),s.number(5),s.number(6),s.number(7),s.number(8),s.number(9),s.number(10),s.number(12)!=0};}
    int64_t touch(const std::string& id,int64_t segment=0){Statement s(db,"UPDATE sessions SET revision=revision+1 WHERE id=? RETURNING revision");s.bind(1,id);s.row();int64_t rev=s.number(0);if(segment){Statement t(db,"UPDATE segments SET cursor=? WHERE id=?");t.bind(1,rev).bind(2,segment).done();}return rev;}
    std::shared_ptr<const PreparedDictionary> dictionary(int64_t id) {
        for(auto it=dictionaries.begin();it!=dictionaries.end();++it)if(it->id==id) {
            auto result=it->prepared;dictionaries.splice(dictionaries.begin(),dictionaries,it);return result;
        }
        Statement s(db,"SELECT json FROM dictionaries WHERE id=?");s.bind(1,id);
        if(!s.row())throw StoreError(400,"Unknown dictionary revision");
        auto result=std::make_shared<PreparedDictionary>(parse_dictionary(s.text(0)));
        cache_dictionary(id,result);return result;
    }
    int64_t current_dictionary() {Statement s(db,"SELECT value FROM settings WHERE key='dictionary'");s.row();return std::stoll(s.text(0));}
    int64_t insert_dictionary(const PreparedDictionary& prepared) {
        const auto json=serialize_dictionary(prepared.entries),hash=text_hash(json);
        Statement insert(db,"INSERT OR IGNORE INTO dictionaries(json,hash,created_us) VALUES(?,?,?)");
        insert.bind(1,json).bind(2,hash).bind(3,g_get_real_time()).done();
        Statement select(db,"SELECT id FROM dictionaries WHERE hash=?");select.bind(1,hash);select.row();return select.number(0);
    }
    void select_dictionary(int64_t id) {Statement s(db,"UPDATE settings SET value=? WHERE key='dictionary'");s.bind(1,std::to_string(id)).done();}
    std::vector<TextEdit> edits(int64_t revision){Statement s(db,"SELECT start_byte,end_byte,before_text,after_text,origin,enabled FROM edits WHERE revision_id=? AND origin='dictionary' ORDER BY start_byte,id");s.bind(1,revision);std::vector<TextEdit> out;while(s.row())out.push_back({size_t(s.number(0)),size_t(s.number(1)),s.text(2),s.text(3),s.text(4),s.number(5)!=0});return out;}
    void insert_edit(int64_t revision,const TextEdit& edit,int64_t origin_id){Statement s(db,"INSERT OR IGNORE INTO edits(revision_id,start_byte,end_byte,before_text,after_text,origin,origin_id,enabled) VALUES(?,?,?,?,?,?,?,?)");s.bind(1,revision).bind(2,int64_t(edit.start)).bind(3,int64_t(edit.end)).bind(4,edit.before).bind(5,edit.after).bind(6,edit.origin).bind(7,origin_id).bind(8,int(edit.enabled)).done();}
    void render_segment(int64_t segment){Statement s(db,"SELECT r.id,r.text FROM segments s JOIN transcript_revisions r ON r.id=s.revision_id WHERE s.id=?");s.bind(1,segment);if(!s.row())return;auto rendered=render_edits(s.text(1),edits(s.number(0)));Statement u(db,"UPDATE segments SET edited_text=?,words=? WHERE id=?");u.bind(1,rendered).bind(2,word_count(rendered)).bind(3,segment).done();}
    void apply_dictionary_edits(const SessionRecord& session,int64_t revision,const std::string& raw,const DictionaryMatcher& matcher){
        auto mapped=matcher.apply(raw);
        for(auto& edit:mapped.edits){edit.enabled=session.replacements_enabled;insert_edit(revision,edit,session.dictionary_revision);}
    }
    void apply_dictionary_edits(const SessionRecord& session,int64_t revision,const std::string& raw){
        auto prepared=dictionary(session.dictionary_revision);apply_dictionary_edits(session,revision,raw,prepared->matcher);
    }
    std::string transcript(const SessionRecord& session,bool original){
        // Export is a rendering. Source text remains immutable in its revision;
        // missing sections must not disappear between successful neighbors.
        std::vector<std::pair<int64_t,std::string>> gaps;
        {Statement g(db,"SELECT position,reason FROM capture_gaps WHERE session_id=? UNION ALL SELECT start_sample,'Missing source audio' FROM dispositions WHERE session_id=? AND generation=? AND kind='gap' ORDER BY 1");g.bind(1,session.id).bind(2,session.id).bind(3,session.generation);while(g.row())gaps.emplace_back(g.number(0),g.text(1));}
        Statement s(db,"SELECT a.coverage_start,a.coverage_end,a.state,a.error,r.text,a.edited_text FROM segments a LEFT JOIN transcript_revisions r ON r.id=a.revision_id WHERE a.session_id=? AND a.generation=? ORDER BY a.sequence");s.bind(1,session.id).bind(2,session.generation);
        std::string out;size_t gap=0;
        auto append=[&](const std::string& text){if(text.empty())return;if(!out.empty())out+=' ';out+=text;};
        auto gap_marker=[&](const std::pair<int64_t,std::string>& value){append("[Audio gap near "+std::to_string(value.first/16000.0)+" seconds: "+value.second+"]");};
        while(s.row()){
            // There is no word alignment. A source dropout is identified at the
            // containing section, rather than invented at a particular word.
            while(gap<gaps.size()&&gaps[gap].first<s.number(1))gap_marker(gaps[gap++]);
            if(s.text(2)!="completed")append(s.text(2)=="pending"||s.text(2)=="running"?"[Transcription pending]":"[Transcription unavailable"+(s.text(3).empty()?std::string():": "+s.text(3))+"]");
            append(s.text(original?4:5));
        }
        while(gap<gaps.size())gap_marker(gaps[gap++]);
        return out;
    }
    struct Progress {int64_t pending=0,failed=0,transcribed=0,words=0;bool complete=false,coverage_complete=false,source_gap=false;};
    Progress progress(const SessionRecord& session){
        Progress p;Statement s(db,"SELECT state,count(*),sum(words) FROM segments WHERE session_id=? AND generation=? GROUP BY state");s.bind(1,session.id).bind(2,session.generation);while(s.row()){if(s.text(0)=="failed")p.failed+=s.number(1);else if(s.text(0)!="completed"&&s.text(0)!="cancelled")p.pending+=s.number(1);p.words+=s.number(2);}
        p.transcribed=session.planned_samples;
        Statement f(db,"SELECT min(coverage_start) FROM segments WHERE session_id=? AND generation=? AND state!='completed'");f.bind(1,session.id).bind(2,session.generation);if(f.row()&&!f.text(0).empty())p.transcribed=std::min(p.transcribed,f.number(0));
        Statement g(db,"SELECT min(position) FROM capture_gaps WHERE session_id=?");g.bind(1,session.id);bool gap=false;if(g.row()&&!g.text(0).empty()){gap=true;p.transcribed=std::min(p.transcribed,g.number(0));}
        Statement d(db,"SELECT min(start_sample) FROM dispositions WHERE session_id=? AND generation=? AND kind='gap'");d.bind(1,session.id).bind(2,session.generation);if(d.row()&&!d.text(0).empty()){gap=true;p.transcribed=std::min(p.transcribed,d.number(0));}
        p.source_gap=gap;p.coverage_complete=session.capture_state!="recording"&&session.capture_state!="paused"&&session.final_samples>=0&&session.planned_samples>=session.final_samples&&!p.pending;
        p.complete=p.coverage_complete&&!p.failed&&!gap&&session.capture_state!="cancelled";
        return p;
    }
    void session_object(JsonBuilder* b,const SessionRecord& s){
        auto p=progress(s);json_builder_begin_object(b);json_field(b,"id",s.id);json_field(b,"source",s.source);json_field(b,"capture_state",s.capture_state);json_bool(b,"replacements_enabled",s.replacements_enabled);
        json_field(b,"transcription_state",std::string(p.failed?"failed":p.complete?"completed":s.capture_state=="cancelled"?"cancelled":p.coverage_complete?"incomplete":"pending"));
        json_int(b,"generation",s.generation);json_int(b,"dictionary_revision",s.dictionary_revision);json_int(b,"durable_samples",s.durable_samples);json_int(b,"planned_samples",s.planned_samples);json_int(b,"final_samples",s.final_samples);
        json_field(b,"duration_seconds",double(s.durable_samples)/16000);json_field(b,"transcribed_through",double(p.transcribed)/16000);json_int(b,"word_count",p.words);
        json_field(b,"created_at",timestamp(s.created_us));json_field(b,"error",s.error.empty()&&p.source_gap?std::string("Recording contains audio gaps"):s.error);json_bool(b,"complete",p.complete);
        Statement revision(db,"SELECT revision,delivery_epoch FROM sessions WHERE id=?");revision.bind(1,s.id);revision.row();json_int(b,"revision",revision.number(0));json_int(b,"delivery_epoch",revision.number(1));
        Statement preview(db,"SELECT edited_text FROM segments WHERE session_id=? AND generation=? AND state='completed' ORDER BY sequence LIMIT 1");preview.bind(1,s.id).bind(2,s.generation);json_field(b,"preview",preview.row()?clip(preview.text(0),600):std::string());
        json_builder_set_member_name(b,"gaps");json_builder_begin_array(b);Statement gaps(db,"SELECT position,missing_samples,reason FROM capture_gaps WHERE session_id=? ORDER BY position LIMIT 100");gaps.bind(1,s.id);while(gaps.row()){json_builder_begin_object(b);json_int(b,"sample_position",gaps.number(0));json_int(b,"missing_samples",gaps.number(1));json_field(b,"reason",gaps.text(2));json_builder_end_object(b);}json_builder_end_array(b);
        Statement pub(db,"SELECT id,complete FROM publications WHERE session_id=? AND generation=?");pub.bind(1,s.id).bind(2,s.generation);json_builder_set_member_name(b,"publication");if(pub.row()){json_builder_begin_object(b);json_field(b,"id",pub.text(0));json_bool(b,"complete",pub.number(1));json_builder_end_object(b);}else json_builder_add_null_value(b);
        json_builder_end_object(b);
    }
};

SessionStore::SessionStore(const std::string& path):impl_(std::make_unique<Impl>(path)){}
SessionStore::~SessionStore()=default;
const std::string& SessionStore::data_directory() const{return impl_->directory;}
SessionRecord SessionStore::create_session(const std::string& key,const std::string& source,const std::string& settings){
    if(key.empty()||key.size()>256)throw StoreError(400,"A bounded idempotency key is required");
    if(source!="microphone"&&source!="import"&&source!="legacy")throw StoreError(400,"Unsupported session source");
    Json check(settings);
    std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);
    Statement exists(impl_->db,"SELECT id,source,deleted FROM sessions WHERE request_key=?");exists.bind(1,key);if(exists.row()){if(exists.number(2))throw StoreError(410,"This request refers to a deleted session");if(exists.text(1)!=source)throw StoreError(409,"Idempotency key used for a different source");auto session=impl_->get(exists.text(0));tx.commit();return session;}
    if(source=="microphone"){Statement active(impl_->db,"SELECT id FROM sessions WHERE capture_state IN ('recording','paused') AND source='microphone' AND deleted=0");if(active.row())throw StoreError(409,"A microphone recording is already active");}
    Statement dict(impl_->db,"SELECT value FROM settings WHERE key='dictionary'");dict.row();auto dictionary=std::stoll(dict.text(0));auto id=uuid();
    Statement insert(impl_->db,"INSERT INTO sessions(id,request_key,source,created_us,capture_state,settings,dict_id) VALUES(?,?,?,?,?,?,?)");insert.bind(1,id).bind(2,key).bind(3,source).bind(4,g_get_real_time()).bind(5,"recording").bind(6,settings).bind(7,dictionary).done();
    Statement generation(impl_->db,"INSERT INTO generations(session_id,generation,settings,dict_id) VALUES(?,1,?,?)");generation.bind(1,id).bind(2,settings).bind(3,dictionary).done();tx.commit();return impl_->get(id);
}
std::optional<SessionRecord> SessionStore::get_session(const std::string& id){std::lock_guard lock(impl_->mutex);try{return impl_->get(id);}catch(const StoreError& e){if(e.status==404)return {};throw;}}
std::optional<SessionRecord> SessionStore::active_session(){std::lock_guard lock(impl_->mutex);Statement s(impl_->db,"SELECT id FROM sessions WHERE capture_state IN ('recording','paused') AND source='microphone' AND deleted=0 ORDER BY created_us LIMIT 1");if(s.row())return impl_->get(s.text(0));return {};}
std::vector<SessionRecord> SessionStore::recoverable_sessions(){std::lock_guard lock(impl_->mutex);Statement s(impl_->db,"SELECT id FROM sessions WHERE deleted=0 AND capture_state!='cancelled' AND source!='legacy' ORDER BY created_us");std::vector<SessionRecord> out;while(s.row())out.push_back(impl_->get(s.text(0)));return out;}
std::vector<SessionRecord> SessionStore::processing_sessions(){
    std::lock_guard lock(impl_->mutex);
    Statement s(impl_->db,"SELECT s.id FROM sessions s WHERE s.deleted=0 AND s.capture_state!='cancelled' AND s.source!='legacy' AND (s.capture_state IN ('recording','paused') OR s.planned_samples<s.durable_samples OR (s.final_samples>=0 AND NOT EXISTS(SELECT 1 FROM publications p WHERE p.session_id=s.id AND p.generation=s.generation) AND NOT EXISTS(SELECT 1 FROM segments a WHERE a.session_id=s.id AND a.generation=s.generation AND a.state!='completed') AND NOT EXISTS(SELECT 1 FROM capture_gaps g WHERE g.session_id=s.id) AND NOT EXISTS(SELECT 1 FROM dispositions d WHERE d.session_id=s.id AND d.generation=s.generation AND d.kind='gap'))) ORDER BY s.created_us");
    std::vector<SessionRecord> out;while(s.row())out.push_back(impl_->get(s.text(0)));return out;
}
void SessionStore::put_audio_block(const AudioBlock& b){
    if(b.sequence<0||b.start_sample<0||b.durable_frames<0||b.path.empty())throw StoreError(400,"Invalid audio block metadata");
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(b.session_id);Transaction tx(impl_->db);
    Statement old(impl_->db,"SELECT start_sample,durable_frames,finalized FROM audio_blocks WHERE session_id=? AND sequence=?");old.bind(1,b.session_id).bind(2,b.sequence);
    if(old.row()&&(old.number(0)!=b.start_sample||old.number(1)>b.durable_frames||(old.number(2)&&!b.finalized)))throw StoreError(409,"Audio block cannot lose durable frames or change its start");
    Statement s(impl_->db,"INSERT INTO audio_blocks VALUES(?,?,?,?,?,?) ON CONFLICT(session_id,sequence) DO UPDATE SET path=excluded.path,durable_frames=excluded.durable_frames,finalized=excluded.finalized");s.bind(1,b.session_id).bind(2,b.sequence).bind(3,b.path).bind(4,b.start_sample).bind(5,b.durable_frames).bind(6,int(b.finalized)).done();
    int64_t durable=b.start_sample+b.durable_frames;
    Statement u(impl_->db,"UPDATE sessions SET durable_samples=max(durable_samples,?),final_samples=CASE WHEN capture_state NOT IN ('recording','paused') AND final_samples>=0 THEN max(final_samples,?) ELSE final_samples END WHERE id=?");u.bind(1,durable).bind(2,durable).bind(3,b.session_id).done();
    Statement g(impl_->db,"UPDATE generations SET final_samples=(SELECT final_samples FROM sessions WHERE id=?) WHERE session_id=? AND generation=?");g.bind(1,b.session_id).bind(2,b.session_id).bind(3,session.generation).done();impl_->touch(b.session_id);tx.commit();
}
void SessionStore::record_capture_gap(const std::string& id,int64_t position,int64_t missing,const std::string& reason){
    if(position<0||missing < -1||reason.empty())throw StoreError(400,"Invalid capture gap");
    std::lock_guard lock(impl_->mutex);impl_->get(id);Transaction tx(impl_->db);
    Statement s(impl_->db,"INSERT OR IGNORE INTO capture_gaps(session_id,position,missing_samples,reason) VALUES(?,?,?,?)");s.bind(1,id).bind(2,position).bind(3,missing).bind(4,reason).done();impl_->touch(id);tx.commit();
}
std::vector<AudioBlock> SessionStore::audio_blocks(const std::string& id,int64_t start,int64_t end){std::lock_guard lock(impl_->mutex);impl_->get(id);Statement s(impl_->db,"SELECT sequence,path,start_sample,durable_frames,finalized FROM audio_blocks WHERE session_id=? AND start_sample<? AND start_sample+durable_frames>? ORDER BY sequence");s.bind(1,id).bind(2,end).bind(3,start);std::vector<AudioBlock> out;while(s.row())out.push_back({id,s.text(1),s.number(0),s.number(2),s.number(3),s.number(4)!=0});return out;}
int64_t SessionStore::plan_segment(const std::string& id,int64_t generation,int64_t start,int64_t end,const std::vector<SourceSpan>& spans,const std::string& config,bool gap){
    Json validate(config);std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);if(session.generation!=generation)throw StoreError(409,"Superseded processing generation");
    if(start<0||end<=start||end>session.durable_samples)throw StoreError(400,"Planning must cover a nonempty durable audio range");
    int64_t previous=start;for(auto& span:spans){if(span.start<previous||span.end<=span.start||span.end>end)throw StoreError(400,"Invalid ordered source spans");previous=span.end;}
    if(gap&&!spans.empty())throw StoreError(400,"A gap cannot contain speech spans");
    Transaction tx(impl_->db);
    if(start<session.planned_samples){Statement prior(impl_->db,"SELECT id,coverage_end,spans FROM segments WHERE session_id=? AND generation=? AND coverage_start=?");prior.bind(1,id).bind(2,generation).bind(3,start);if(prior.row()&&prior.number(1)==end&&prior.text(2)==spans_json(spans)){auto existing=prior.number(0);tx.commit();return existing;}Statement d(impl_->db,"SELECT end_sample,kind FROM dispositions WHERE session_id=? AND generation=? AND start_sample=?");d.bind(1,id).bind(2,generation).bind(3,start);if(spans.empty()&&d.row()&&d.number(0)==end&&d.text(1)==(gap?"gap":"silence")){tx.commit();return 0;}throw StoreError(409,"Planning range was already committed differently");}
    if(start!=session.planned_samples)throw StoreError(409,"Planning cannot skip an unclassified audio range");
    int64_t segment=0;
    if(!spans.empty()){
        Statement seq(impl_->db,"SELECT coalesce(max(sequence)+1,0) FROM segments WHERE session_id=? AND generation=?");seq.bind(1,id).bind(2,generation);seq.row();
        Statement s(impl_->db,"INSERT INTO segments(session_id,generation,sequence,coverage_start,coverage_end,spans,config) VALUES(?,?,?,?,?,?,?)");s.bind(1,id).bind(2,generation).bind(3,seq.number(0)).bind(4,start).bind(5,end).bind(6,spans_json(spans)).bind(7,config).done();segment=sqlite3_last_insert_rowid(impl_->db);
    }
    Statement d(impl_->db,"INSERT INTO dispositions VALUES(?,?,?,?,?)");d.bind(1,id).bind(2,generation).bind(3,start).bind(4,end).bind(5,gap?"gap":spans.empty()?"silence":"speech").done();
    Statement s(impl_->db,"UPDATE sessions SET planned_samples=? WHERE id=?");s.bind(1,end).bind(2,id).done();Statement g(impl_->db,"UPDATE generations SET planned_samples=? WHERE session_id=? AND generation=?");g.bind(1,end).bind(2,id).bind(3,generation).done();impl_->touch(id,segment);tx.commit();return segment;
}
void SessionStore::set_capture_paused(const std::string& id,bool paused){
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);
    if(session.capture_state!="recording"&&session.capture_state!="paused")throw StoreError(409,"Session capture is no longer active");
    const std::string state=paused?"paused":"recording";if(session.capture_state==state)return;
    Transaction tx(impl_->db);Statement update(impl_->db,"UPDATE sessions SET capture_state=? WHERE id=?");update.bind(1,state).bind(2,id).done();impl_->touch(id);tx.commit();
}
void SessionStore::seal_capture(const std::string& id,const std::string& state,const std::string& error){if(state!="stopped"&&state!="interrupted"&&state!="cancelled")throw StoreError(400,"Invalid terminal capture state");std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);Transaction tx(impl_->db);Statement s(impl_->db,"UPDATE sessions SET capture_state=?,final_samples=durable_samples,error=?,delivery_epoch=delivery_epoch+CASE WHEN ?='interrupted' THEN 1 ELSE 0 END WHERE id=?");s.bind(1,state).bind(2,error).bind(3,state).bind(4,id).done();Statement g(impl_->db,"UPDATE generations SET final_samples=? WHERE session_id=? AND generation=?");g.bind(1,session.durable_samples).bind(2,id).bind(3,session.generation).done();impl_->touch(id);tx.commit();}
void SessionStore::cancel_session(const std::string& id){std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);Transaction tx(impl_->db);Statement s(impl_->db,"UPDATE sessions SET capture_state='cancelled',final_samples=durable_samples,delivery_epoch=delivery_epoch+1 WHERE id=?");s.bind(1,id).done();Statement a(impl_->db,"UPDATE segments SET state='cancelled' WHERE session_id=? AND generation=? AND state IN ('pending','running','failed')");a.bind(1,id).bind(2,session.generation).done();std::vector<int64_t> ids;{Statement changed(impl_->db,"SELECT id FROM segments WHERE session_id=? AND generation=?");changed.bind(1,id).bind(2,session.generation);while(changed.row())ids.push_back(changed.number(0));}for(auto segment:ids)impl_->touch(id,segment);
    Statement generation(impl_->db,"UPDATE generations SET final_samples=? WHERE session_id=? AND generation=?");generation.bind(1,session.durable_samples).bind(2,id).bind(3,session.generation).done();impl_->touch(id);tx.commit();}
void SessionStore::delete_session(const std::string& id){std::lock_guard lock(impl_->mutex);auto s=impl_->get(id,true);Statement mark(impl_->db,"UPDATE sessions SET deleted=1,capture_state='cancelled',revision=revision+1 WHERE id=? AND deleted=0");mark.bind(1,id).done();}
std::vector<std::string> SessionStore::pending_deletions(){std::lock_guard lock(impl_->mutex);Statement s(impl_->db,"SELECT id FROM sessions WHERE deleted=1");std::vector<std::string> out;while(s.row())out.push_back(s.text(0));return out;}
std::vector<std::string> SessionStore::deletion_paths(const std::string& id){
    namespace fs=std::filesystem;std::lock_guard lock(impl_->mutex);
    Statement session(impl_->db,"SELECT source,deleted FROM sessions WHERE id=?");session.bind(1,id);
    if(!session.row()||session.number(1)!=1)throw StoreError(409,"Session is not pending deletion");
    if(session.text(0)!="legacy")return {};
    auto history=fs::absolute(fs::path(impl_->directory)/"history").lexically_normal();
    if(fs::is_symlink(fs::symlink_status(history)))throw StoreError(409,"Refusing to delete through a symlinked history directory");
    auto canonical_history=fs::weakly_canonical(history);
    auto canonical_data=fs::weakly_canonical(impl_->directory);
    if(canonical_history.parent_path()!=canonical_data)throw StoreError(409,"History is outside the owned data directory");
    std::vector<std::string> result;Statement files(impl_->db,"SELECT path FROM legacy_imports WHERE session_id=?");files.bind(1,id);
    while(files.row()){
        auto metadata=fs::path(files.text(0)).lexically_normal();auto wav=metadata;wav.replace_extension(".wav");
        for(auto& path:{metadata,wav}){
            if(path.parent_path()!=history)throw StoreError(409,"Legacy recording path is outside owned history");
            auto status=fs::symlink_status(path);
            if(fs::is_symlink(status))throw StoreError(409,"Refusing to delete a symlinked legacy recording");
            if(!fs::exists(status))continue;
            if(!fs::is_regular_file(status)||fs::weakly_canonical(path).parent_path()!=canonical_history)throw StoreError(409,"Legacy recording path is not an owned regular file");
            result.push_back(path.string());
        }
    }
    return result;
}
void SessionStore::finish_deletion(const std::string& id){
    std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);
    Statement check(impl_->db,"SELECT deleted FROM sessions WHERE id=?");check.bind(1,id);if(!check.row()||check.number(0)!=1){tx.commit();return;}
    if(impl_->has_retired_runs){Statement erase(impl_->db,"DELETE FROM retired_cleanup_runs WHERE session_id=?");erase.bind(1,id).done();}
    for(const char* query:{
        "DELETE FROM edits WHERE revision_id IN (SELECT r.id FROM transcript_revisions r JOIN segments a ON a.id=r.segment_id WHERE a.session_id=?)",
        "DELETE FROM publications WHERE session_id=?",
        "DELETE FROM transcript_revisions WHERE segment_id IN (SELECT id FROM segments WHERE session_id=?)",
        "DELETE FROM segments WHERE session_id=?","DELETE FROM dispositions WHERE session_id=?",
        "DELETE FROM capture_gaps WHERE session_id=?","DELETE FROM audio_blocks WHERE session_id=?","DELETE FROM generations WHERE session_id=?"}){
        Statement erase(impl_->db,query);erase.bind(1,id).done();
    }
    // Retain only the tombstone/idempotency identity so late completions and
    // retained legacy source metadata cannot resurrect explicitly deleted data.
    Statement s(impl_->db,"UPDATE sessions SET deleted=2,settings='{}',durable_samples=0,planned_samples=0,final_samples=0,error='' WHERE id=?");s.bind(1,id).done();tx.commit();
}
std::optional<AsrJob> SessionStore::next_asr(){
    std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);
    Statement s(impl_->db,"SELECT a.id,a.session_id,a.generation,a.sequence,a.attempt,a.coverage_start,a.coverage_end,a.spans,a.config FROM segments a JOIN sessions s ON s.id=a.session_id AND s.generation=a.generation WHERE a.state='pending' AND s.deleted=0 AND s.capture_state!='cancelled' ORDER BY (s.capture_state IN ('recording','paused')) DESC,s.created_us,a.sequence LIMIT 1");if(!s.row()){tx.commit();return {};}
    AsrJob job;job.id=s.number(0);job.session_id=s.text(1);job.generation=s.number(2);job.sequence=s.number(3);job.attempt=s.number(4)+1;job.coverage_start=s.number(5);job.coverage_end=s.number(6);job.spans=parse_spans(s.text(7));job.config_json=s.text(8);
    Statement u(impl_->db,"UPDATE segments SET state='running',attempt=?,error='' WHERE id=?");u.bind(1,job.attempt).bind(2,job.id).done();impl_->touch(job.session_id,job.id);tx.commit();return job;
}
bool SessionStore::commit_asr(const AsrJob& job,const std::string& text,const std::string& model,double milliseconds){
    if(text.size()>1024*1024||!g_utf8_validate(text.data(),text.size(),nullptr)||!std::isfinite(milliseconds))throw StoreError(400,"Invalid ASR result");
    std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);
    Statement valid(impl_->db,"SELECT a.state,a.attempt,a.revision_id FROM segments a JOIN sessions s ON s.id=a.session_id WHERE a.id=? AND a.session_id=? AND a.generation=? AND s.generation=a.generation AND s.deleted=0 AND s.capture_state!='cancelled'");valid.bind(1,job.id).bind(2,job.session_id).bind(3,job.generation);if(!valid.row()||valid.number(1)!=job.attempt){tx.commit();return false;}
    if(valid.text(0)=="completed"){Statement old(impl_->db,"SELECT hash FROM transcript_revisions WHERE id=?");old.bind(1,valid.number(2));bool same=old.row()&&old.text(0)==text_hash(text);tx.commit();return same;}
    if(valid.text(0)!="running"){tx.commit();return false;}
    auto session=impl_->get(job.session_id);
    Statement raw(impl_->db,"INSERT INTO transcript_revisions(segment_id,text,hash,model,inference_ms,created_us) VALUES(?,?,?,?,?,?)");raw.bind(1,job.id).bind(2,text).bind(3,text_hash(text)).bind(4,model).real(5,milliseconds).bind(6,g_get_real_time()).done();auto revision=sqlite3_last_insert_rowid(impl_->db);
    Statement update(impl_->db,"UPDATE segments SET state='completed',revision_id=?,error='' WHERE id=?");update.bind(1,revision).bind(2,job.id).done();
    impl_->apply_dictionary_edits(session,revision,text);impl_->render_segment(job.id);impl_->touch(job.session_id,job.id);tx.commit();return true;
}
bool SessionStore::fail_asr(const AsrJob& job,const std::string& reason,bool requeue){std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);Statement s(impl_->db,"UPDATE segments SET state=?,error=? WHERE id=? AND session_id=? AND generation=? AND attempt=? AND state='running' AND EXISTS(SELECT 1 FROM sessions WHERE id=? AND generation=? AND deleted=0 AND capture_state!='cancelled')");s.bind(1,requeue?"pending":"failed").bind(2,reason).bind(3,job.id).bind(4,job.session_id).bind(5,job.generation).bind(6,job.attempt).bind(7,job.session_id).bind(8,job.generation).done();bool changed=sqlite3_changes(impl_->db);if(changed){impl_->touch(job.session_id,job.id);if(!requeue){Statement delivery(impl_->db,"UPDATE sessions SET delivery_epoch=delivery_epoch+1 WHERE id=?");delivery.bind(1,job.session_id).done();}}tx.commit();return changed;}
void SessionStore::retry(const std::string& id,const std::string& stage){
    if(stage!="all"&&stage!="asr")throw StoreError(400,"Unknown retry stage");
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);Transaction tx(impl_->db);
    if(session.capture_state=="cancelled"){Statement c(impl_->db,"UPDATE sessions SET capture_state='stopped',final_samples=durable_samples WHERE id=?");c.bind(1,id).done();}
    {Statement s(impl_->db,"UPDATE segments SET state='pending',error='' WHERE session_id=? AND generation=? AND state IN ('failed','cancelled')");s.bind(1,id).bind(2,session.generation).done();}
    Statement delivery(impl_->db,"UPDATE sessions SET delivery_epoch=delivery_epoch+1 WHERE id=?");delivery.bind(1,id).done();std::vector<int64_t> changed_ids;{Statement changed(impl_->db,"SELECT id FROM segments WHERE session_id=? AND generation=?");changed.bind(1,id).bind(2,session.generation);while(changed.row())changed_ids.push_back(changed.number(0));}for(auto segment:changed_ids)impl_->touch(id,segment);impl_->touch(id);tx.commit();
}
int64_t SessionStore::reprocess(const std::string& id,bool audio,const std::string& settings,int64_t dictionary_revision){
    Json checked(settings);std::lock_guard lock(impl_->mutex);auto previous=impl_->get(id);if(previous.capture_state=="recording"||previous.capture_state=="paused")throw StoreError(409,"Stop recording before reprocessing");
    if(!dictionary_revision){Statement d(impl_->db,"SELECT value FROM settings WHERE key='dictionary'");d.row();dictionary_revision=std::stoll(d.text(0));}impl_->dictionary(dictionary_revision);
    Transaction tx(impl_->db);int64_t generation=previous.generation+1;
    Statement g(impl_->db,"INSERT INTO generations(session_id,generation,settings,dict_id,planned_samples,final_samples,replacements_enabled) VALUES(?,?,?,?,?,?,?)");g.bind(1,id).bind(2,generation).bind(3,settings).bind(4,dictionary_revision).bind(5,audio?0:previous.planned_samples).bind(6,previous.durable_samples).bind(7,int(previous.replacements_enabled)).done();
    Statement s(impl_->db,"UPDATE sessions SET generation=?,settings=?,dict_id=?,planned_samples=?,final_samples=durable_samples,capture_state='stopped',error='',delivery_epoch=delivery_epoch+1 WHERE id=?");s.bind(1,generation).bind(2,settings).bind(3,dictionary_revision).bind(4,audio?0:previous.planned_samples).bind(5,id).done();
    auto session=impl_->get(id);
    if(!audio){
        Statement d(impl_->db,"INSERT INTO dispositions SELECT session_id,?,start_sample,end_sample,kind FROM dispositions WHERE session_id=? AND generation=?");d.bind(1,generation).bind(2,id).bind(3,previous.generation).done();
        Statement old(impl_->db,"SELECT a.sequence,a.coverage_start,a.coverage_end,a.spans,a.config,r.text,r.model,r.inference_ms,r.provenance,r.id FROM segments a LEFT JOIN transcript_revisions r ON r.id=a.revision_id WHERE a.session_id=? AND a.generation=? ORDER BY a.sequence");old.bind(1,id).bind(2,previous.generation);
        while(old.row()){
            bool has_text=old.number(9)>0;
            Statement insert(impl_->db,"INSERT INTO segments(session_id,generation,sequence,coverage_start,coverage_end,spans,config,state) VALUES(?,?,?,?,?,?,?,?)");insert.bind(1,id).bind(2,generation).bind(3,old.number(0)).bind(4,old.number(1)).bind(5,old.number(2)).bind(6,old.text(3)).bind(7,old.text(4)).bind(8,has_text?"completed":"pending").done();int64_t segment=sqlite3_last_insert_rowid(impl_->db);
            if(has_text){Statement raw(impl_->db,"INSERT INTO transcript_revisions(segment_id,text,hash,model,inference_ms,provenance,created_us) VALUES(?,?,?,?,?,?,?)");raw.bind(1,segment).bind(2,old.text(5)).bind(3,text_hash(old.text(5))).bind(4,old.text(6)).real(5,old.real(7)).bind(6,old.text(8)).bind(7,g_get_real_time()).done();int64_t revision=sqlite3_last_insert_rowid(impl_->db);
                Statement r(impl_->db,"UPDATE segments SET revision_id=? WHERE id=?");r.bind(1,revision).bind(2,segment).done();impl_->apply_dictionary_edits(session,revision,old.text(5));impl_->render_segment(segment);}
            impl_->touch(id,segment);
        }
    }
    impl_->touch(id);tx.commit();return generation;
}
std::optional<Publication> SessionStore::publication(const std::string& id){std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);Statement s(impl_->db,"SELECT id,text,complete,generation FROM publications WHERE session_id=? AND generation=?");s.bind(1,id).bind(2,session.generation);if(!s.row())return {};return Publication{s.text(0),id,s.text(1),s.number(3),s.number(2)!=0};}
std::optional<Publication> SessionStore::publish(const std::string& id){
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);if(auto old=publication(id))return old;auto progress=impl_->progress(session);if(!progress.complete)return {};
    Transaction tx(impl_->db);Publication out{uuid(),id,impl_->transcript(session,false),session.generation,true};
    auto*b=json_builder_new();json_builder_begin_object(b);json_builder_set_member_name(b,"segments");json_builder_begin_array(b);
    Statement refs(impl_->db,"SELECT id,revision_id FROM segments WHERE session_id=? AND generation=? ORDER BY sequence");refs.bind(1,id).bind(2,session.generation);while(refs.row()){json_builder_begin_object(b);json_int(b,"segment_id",refs.number(0));json_int(b,"revision_id",refs.number(1));json_builder_set_member_name(b,"decisions");json_builder_begin_array(b);Statement edits(impl_->db,"SELECT id,decision_revision,enabled FROM edits WHERE revision_id=? AND origin='dictionary' ORDER BY id");edits.bind(1,refs.number(1));while(edits.row()){json_builder_begin_object(b);json_int(b,"edit_id",edits.number(0));json_int(b,"decision_revision",edits.number(1));json_bool(b,"enabled",edits.number(2));json_builder_end_object(b);}json_builder_end_array(b);json_builder_end_object(b);}json_builder_end_array(b);json_builder_end_object(b);auto source_refs=json_string(b);g_object_unref(b);
    Statement s(impl_->db,"INSERT INTO publications(id,session_id,generation,text,source_refs,complete,created_us) VALUES(?,?,?,?,?,?,?)");s.bind(1,out.id).bind(2,id).bind(3,session.generation).bind(4,out.text).bind(5,source_refs).bind(6,1).bind(7,g_get_real_time()).done();impl_->touch(id);tx.commit();return out;
}
std::string SessionStore::transcript(const std::string& id,const std::string& view){std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);if(view=="publication"){auto pub=publication(id);if(!pub)throw StoreError(409,"No complete publication is available");return pub->text;}if(view!="edited"&&view!="cleaned"&&view!="original"&&view!="raw")throw StoreError(400,"Unknown transcript view");return impl_->transcript(session,view=="original"||view=="raw");}
std::string SessionStore::dictionary_json(){std::lock_guard lock(impl_->mutex);Statement s(impl_->db,"SELECT d.id,d.json,d.hash FROM dictionaries d JOIN settings s ON s.key='dictionary' AND CAST(s.value AS INTEGER)=d.id");s.row();auto*b=json_builder_new();json_builder_begin_object(b);json_int(b,"revision",s.number(0));json_field(b,"hash",s.text(2));Json doc(s.text(1));json_builder_set_member_name(b,"entries");json_builder_add_value(b,json_node_copy(json_object_get_member(doc.object(),"entries")));json_field(b,"matching",std::string("Unicode case folding; longest complete phrase; non-cascading"));json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;}
int64_t SessionStore::set_dictionary(const std::string& json) {
    Json request(json);int64_t expected=0;
    if(auto* value=json_object_get_member(request.object(),"expected_revision")) {
        if(json_node_get_value_type(value)!=G_TYPE_INT64 || json_node_get_int(value)<1)
            throw StoreError(400,"Invalid dictionary revision");
        expected=json_node_get_int(value);
    }
    auto prepared=std::make_shared<Impl::PreparedDictionary>(parse_dictionary(json));
    std::lock_guard lock(impl_->mutex);Transaction tx(impl_->db);
    if(expected && expected!=impl_->current_dictionary())throw StoreError(409,"Dictionary changed; reopen it before saving");
    auto id=impl_->insert_dictionary(*prepared);impl_->select_dictionary(id);tx.commit();
    impl_->cache_dictionary(id,prepared);return id;
}
int64_t SessionStore::add_dictionary_entry(const std::string& json) {
    Json checked(json); // The endpoint accepts exactly one JSON object.
    auto additions=parse_dictionary("{\"entries\":["+json+"]}");
    std::lock_guard lock(impl_->mutex);
    auto entries=impl_->dictionary(impl_->current_dictionary())->entries;
    paraketto::service::add_dictionary_entry(entries,additions.front());
    auto prepared=std::make_shared<Impl::PreparedDictionary>(std::move(entries));
    Transaction tx(impl_->db);auto id=impl_->insert_dictionary(*prepared);
    impl_->select_dictionary(id);tx.commit();impl_->cache_dictionary(id,prepared);return id;
}
int64_t SessionStore::update_dictionary_entry(size_t index,const std::string& json,bool remove) {
    Json request(json);
    auto* expected=json_object_get_member(request.object(),"expected_revision");
    if(!expected || json_node_get_value_type(expected)!=G_TYPE_INT64 || json_node_get_int(expected)<1)
        throw StoreError(400,"An expected dictionary revision is required");
    auto replacement=remove?std::vector<DictionaryEntry>{}:parse_dictionary("{\"entries\":["+json+"]}");
    std::lock_guard lock(impl_->mutex);
    auto current=impl_->current_dictionary();
    if(current!=json_node_get_int(expected))throw StoreError(409,"Dictionary changed since this entry was loaded.");
    auto entries=impl_->dictionary(current)->entries;
    if(index>=entries.size())throw StoreError(404,"Dictionary entry not found");
    if(remove)entries.erase(entries.begin()+index);else entries[index]=std::move(replacement.front());
    auto prepared=std::make_shared<Impl::PreparedDictionary>(std::move(entries));
    Transaction tx(impl_->db);auto id=impl_->insert_dictionary(*prepared);
    impl_->select_dictionary(id);tx.commit();impl_->cache_dictionary(id,prepared);return id;
}
void SessionStore::correct_spelling(const std::string& id,const std::string& json) {
    Json request(json);
    auto integer=[&](const char* key,int64_t minimum) {
        auto* value=json_object_get_member(request.object(),key);
        if(!value || json_node_get_value_type(value)!=G_TYPE_INT64 || json_node_get_int(value)<minimum)
            throw StoreError(400,std::string("Invalid ")+key);
        return json_node_get_int(value);
    };
    const auto segment=integer("segment_id",1),expected=integer("revision_id",1),generation=integer("generation",1);
    const auto start=integer("start",0),end=integer("end",1);
    auto* enabled=json_object_get_member(request.object(),"replacements_enabled");
    if(!enabled || json_node_get_value_type(enabled)!=G_TYPE_BOOLEAN)
        throw StoreError(400,"replacements_enabled must be a boolean");
    const auto selected=request.text("selected"),canonical=request.text("canonical");
    // Validate before touching persisted settings, including empty/oversized terms.
    DictionaryMatcher checked({{canonical,{selected}}});
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);
    if(session.generation!=generation || session.replacements_enabled!=bool(json_node_get_boolean(enabled)))
        throw StoreError(409,"Transcript changed; select the text again");
    Statement original(impl_->db,"SELECT a.revision_id,r.text,r.model,r.inference_ms,r.provenance FROM segments a JOIN transcript_revisions r ON r.id=a.revision_id WHERE a.id=? AND a.session_id=? AND a.generation=? AND a.state='completed'");
    original.bind(1,segment).bind(2,id).bind(3,generation);
    if(!original.row() || original.number(0)!=expected)
        throw StoreError(409,"Transcript changed; select the text again");
    const auto raw=original.text(1);
    const auto span=selection_source_span(raw,impl_->edits(expected),size_t(start),size_t(end),selected);
    DictionaryEntry addition{canonical,{raw.substr(span.first,span.second-span.first)}};
    const auto global_id=impl_->current_dictionary();
    auto global_entries=impl_->dictionary(global_id)->entries;
    paraketto::service::add_dictionary_entry(global_entries,addition);
    auto global=std::make_shared<Impl::PreparedDictionary>(std::move(global_entries));
    std::shared_ptr<Impl::PreparedDictionary> local=global;
    if(session.dictionary_revision!=global_id) {
        auto entries=impl_->dictionary(session.dictionary_revision)->entries;
        paraketto::service::add_dictionary_entry(entries,addition);
        local=std::make_shared<Impl::PreparedDictionary>(std::move(entries));
    }
    auto mapped=local->matcher.apply(raw);
    if(std::none_of(mapped.edits.begin(),mapped.edits.end(),[&](const auto& e) {
        return e.start==span.first && e.end==span.second && e.after==canonical;
    }))throw StoreError(400,"Select complete words or phrases that can use this spelling");
    Transaction tx(impl_->db);
    const auto global_revision=impl_->insert_dictionary(*global);
    session.dictionary_revision=local==global?global_revision:impl_->insert_dictionary(*local);
    impl_->select_dictionary(global_revision);
    Statement update_session(impl_->db,"UPDATE sessions SET dict_id=? WHERE id=?");
    update_session.bind(1,session.dictionary_revision).bind(2,id).done();
    Statement update_generation(impl_->db,"UPDATE generations SET dict_id=? WHERE session_id=? AND generation=?");
    update_generation.bind(1,session.dictionary_revision).bind(2,id).bind(3,generation).done();
    // Keep the previous revision and its edits for saved publication references.
    // The raw ASR bytes/model provenance are unchanged; only dictionary edits differ.
    Statement revision(impl_->db,"INSERT INTO transcript_revisions(segment_id,text,hash,model,inference_ms,provenance,created_us) VALUES(?,?,?,?,?,?,?)");
    revision.bind(1,segment).bind(2,raw).bind(3,text_hash(raw)).bind(4,original.text(2)).real(5,original.real(3)).bind(6,original.text(4)).bind(7,g_get_real_time()).done();
    auto new_revision=sqlite3_last_insert_rowid(impl_->db);
    for(auto& edit:mapped.edits) {
        edit.enabled=session.replacements_enabled;impl_->insert_edit(new_revision,edit,session.dictionary_revision);
    }
    Statement update_segment(impl_->db,"UPDATE segments SET revision_id=? WHERE id=?");
    update_segment.bind(1,new_revision).bind(2,segment).done();
    impl_->render_segment(segment);impl_->touch(id,segment);tx.commit();
    // Cache only committed IDs: a rolled-back row ID may be reused for other rules.
    impl_->cache_dictionary(global_revision,global);
    if(local!=global)impl_->cache_dictionary(session.dictionary_revision,local);
}

void SessionStore::set_replacements_enabled(const std::string& id,bool enabled,int64_t expected){
    if(expected < -1)throw StoreError(400,"Invalid session revision");
    std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);Transaction tx(impl_->db);
    // Repeating an already applied target is safe after a lost HTTP response,
    // even if capture/progress has advanced the shared session revision.
    if(session.replacements_enabled==enabled){tx.commit();return;}
    if(expected>=0){Statement revision(impl_->db,"SELECT revision FROM sessions WHERE id=?");revision.bind(1,id);revision.row();if(revision.number(0)!=expected)throw StoreError(409,"Session changed; refresh its replacements state");}
    Statement update(impl_->db,"UPDATE sessions SET replacements_enabled=? WHERE id=?");update.bind(1,int(enabled)).bind(2,id).done();
    Statement generation(impl_->db,"UPDATE generations SET replacements_enabled=? WHERE session_id=? AND generation=?");generation.bind(1,int(enabled)).bind(2,id).bind(3,session.generation).done();
    impl_->update_replacement_edits(id,session.generation,enabled,true);impl_->touch(id);tx.commit();
}
std::string SessionStore::session_json(const std::string& id){std::lock_guard lock(impl_->mutex);auto*b=json_builder_new();try{impl_->session_object(b,impl_->get(id));auto out=json_string(b);g_object_unref(b);return out;}catch(...){g_object_unref(b);throw;}}
std::string SessionStore::sessions_json(int limit,int offset){
    limit=std::clamp(limit,1,200);offset=std::max(offset,0);std::lock_guard lock(impl_->mutex);auto*b=json_builder_new();json_builder_begin_object(b);json_builder_set_member_name(b,"sessions");json_builder_begin_array(b);
    Statement s(impl_->db,"SELECT id FROM sessions WHERE deleted=0 ORDER BY created_us DESC,id LIMIT ? OFFSET ?");s.bind(1,limit+1).bind(2,offset);int n=0;bool more=false;while(s.row()){if(n++==limit){more=true;break;}impl_->session_object(b,impl_->get(s.text(0)));}json_builder_end_array(b);json_bool(b,"has_more",more);json_int(b,"next_offset",offset+std::min(n,limit));json_int(b,"next_cursor",offset+std::min(n,limit));
    Statement totals(impl_->db,"SELECT coalesce(sum(a.words),0),coalesce(sum(CASE WHEN a.words>0 AND s.error='' THEN s.durable_samples ELSE 0 END),0) FROM sessions s JOIN (SELECT session_id,generation,sum(words) words FROM segments WHERE state='completed' GROUP BY session_id,generation) a ON a.session_id=s.id AND a.generation=s.generation WHERE s.deleted=0 AND s.error='' AND s.capture_state NOT IN ('recording','paused') AND s.final_samples>=0 AND s.planned_samples>=s.final_samples AND NOT EXISTS(SELECT 1 FROM capture_gaps g WHERE g.session_id=s.id) AND NOT EXISTS(SELECT 1 FROM segments f WHERE f.session_id=s.id AND f.generation=s.generation AND f.state!='completed')");totals.row();auto words=totals.number(0);double minutes=double(totals.number(1))/16000/60;
    json_builder_set_member_name(b,"stats");json_builder_begin_object(b);json_int(b,"words",words);json_int(b,"words_per_minute",minutes>0?std::llround(words/minutes):0);json_builder_end_object(b);json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;
}
std::string SessionStore::updates_json(const std::string& id,int64_t after,int limit,int64_t expected_generation){
    limit=std::clamp(limit,1,100);std::lock_guard lock(impl_->mutex);auto session=impl_->get(id);bool reset=expected_generation&&expected_generation!=session.generation;if(reset)after=0;
    auto*b=json_builder_new();json_builder_begin_object(b);json_builder_set_member_name(b,"session");impl_->session_object(b,session);json_int(b,"generation",session.generation);json_bool(b,"reset",reset);json_builder_set_member_name(b,"segments");json_builder_begin_array(b);
    Statement segments(impl_->db,"SELECT a.id,a.sequence,a.state,a.error,a.revision_id,r.text,a.edited_text,a.spans,a.cursor,r.provenance FROM segments a LEFT JOIN transcript_revisions r ON r.id=a.revision_id WHERE a.session_id=? AND a.generation=? AND a.cursor>? ORDER BY a.cursor,a.id LIMIT ?");segments.bind(1,id).bind(2,session.generation).bind(3,after).bind(4,limit+1);int n=0;bool more=false;int64_t cursor=after;
    while(segments.row()){
        if(n++==limit){more=true;break;}auto segment=segments.number(0);cursor=segments.number(8);json_builder_begin_object(b);json_int(b,"id",segment);json_int(b,"sequence",segments.number(1));json_field(b,"state",segments.text(2));json_field(b,"error",segments.text(3));json_int(b,"revision_id",segments.number(4));json_field(b,"raw_text",segments.text(5));json_field(b,"edited_text",segments.text(6));json_field(b,"provenance",segments.text(9));
        Json source(segments.text(7));json_builder_set_member_name(b,"source_spans");json_builder_add_value(b,json_node_copy(json_object_get_member(source.object(),"spans")));
        Statement edits(impl_->db,"SELECT id,start_byte,end_byte,before_text,after_text,origin,enabled,decision_revision FROM edits WHERE revision_id=? AND origin='dictionary' ORDER BY start_byte,id");edits.bind(1,segments.number(4));json_builder_set_member_name(b,"edits");json_builder_begin_array(b);while(edits.row()){json_builder_begin_object(b);json_int(b,"id",edits.number(0));json_int(b,"start",edits.number(1));json_int(b,"end",edits.number(2));json_field(b,"before",edits.text(3));json_field(b,"after",edits.text(4));json_field(b,"origin",edits.text(5));json_bool(b,"enabled",edits.number(6));json_int(b,"decision_revision",edits.number(7));json_builder_end_object(b);}json_builder_end_array(b);json_builder_end_object(b);
    }
    if(!more){Statement r(impl_->db,"SELECT revision FROM sessions WHERE id=?");r.bind(1,id);r.row();cursor=r.number(0);}json_builder_end_array(b);json_int(b,"cursor",cursor);json_int(b,"next_cursor",cursor);json_bool(b,"has_more",more);json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;
}
std::string SessionStore::publication_json(const std::string& id){auto result=publication(id);if(!result)throw StoreError(409,"No publication available");auto*b=json_builder_new();json_builder_begin_object(b);json_field(b,"id",result->id);json_field(b,"session_id",id);json_int(b,"generation",result->generation);json_field(b,"text",result->text);json_bool(b,"complete",result->complete);json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;}
std::string SessionStore::counts_json(){
    std::lock_guard lock(impl_->mutex);Statement count(impl_->db,"SELECT count(*) FROM segments a JOIN sessions s ON s.id=a.session_id AND s.generation=a.generation WHERE s.deleted=0 AND a.state IN ('pending','running')");count.row();
    auto*b=json_builder_new();json_builder_begin_object(b);json_int(b,"asr",count.number(0));json_builder_end_object(b);auto out=json_string(b);g_object_unref(b);return out;
}
void SessionStore::checkpoint(){std::lock_guard lock(impl_->mutex);int rc=sqlite3_wal_checkpoint_v2(impl_->db,nullptr,SQLITE_CHECKPOINT_PASSIVE,nullptr,nullptr);if(rc!=SQLITE_OK&&rc!=SQLITE_BUSY)throw std::runtime_error(sqlite3_errmsg(impl_->db));}
void SessionStore::import_legacy_history(const std::string& directory){
    namespace fs=std::filesystem;if(!fs::is_directory(directory))return;
    for(auto& file:fs::directory_iterator(directory)){
        if(file.path().extension()!=".json"||!file.is_regular_file())continue;
        auto absolute=fs::absolute(file.path()).lexically_normal().string();
        std::lock_guard lock(impl_->mutex);Statement exists(impl_->db,"SELECT session_id FROM legacy_imports WHERE path=?");exists.bind(1,absolute);if(exists.row())continue;
        try {
            if(file.file_size()>16*1024*1024)continue;
            Json old(read_file(absolute));std::string text=old.text("text");if(!g_utf8_validate(text.data(),text.size(),nullptr))continue;
            auto wav=file.path();wav.replace_extension(".wav");bool has_audio=fs::is_regular_file(wav);double seconds=old.number("duration");if(!std::isfinite(seconds)||seconds<0)seconds=0;int64_t samples=std::llround(seconds*16000);
            Transaction tx(impl_->db);std::string id=uuid();Statement insert(impl_->db,"INSERT INTO sessions(id,request_key,source,created_us,capture_state,settings,dict_id,durable_samples,planned_samples,final_samples,error) VALUES(?,?, 'legacy',?,'stopped','{}',1,?,?,?,?)");insert.bind(1,id).bind(2,"legacy:"+text_hash(absolute)).bind(3,legacy_time(file.path())).bind(4,samples).bind(5,samples).bind(6,samples).bind(7,old.text("error")).done();
            Statement gen(impl_->db,"INSERT INTO generations(session_id,generation,settings,dict_id,planned_samples,final_samples) VALUES(?,1,'{}',1,?,?)");gen.bind(1,id).bind(2,samples).bind(3,samples).done();
            if(has_audio){Statement audio(impl_->db,"INSERT INTO audio_blocks VALUES(?,0,?,0,?,1)");audio.bind(1,id).bind(2,fs::absolute(wav).string()).bind(3,samples).done();}
            Statement segment(impl_->db,"INSERT INTO segments(session_id,generation,sequence,coverage_start,coverage_end,spans,config,state,cursor) VALUES(?,1,0,0,?,?,'{}',?,1)");segment.bind(1,id).bind(2,samples).bind(3,spans_json(samples?std::vector<SourceSpan>{{0,samples}}:std::vector<SourceSpan>{})).bind(4,old.text("error").empty()?"completed":"failed").done();int64_t segment_id=sqlite3_last_insert_rowid(impl_->db);
            Statement revision(impl_->db,"INSERT INTO transcript_revisions(segment_id,text,hash,model,inference_ms,provenance,created_us) VALUES(?,?,?,'legacy',?,'legacy',?)");revision.bind(1,segment_id).bind(2,text).bind(3,text_hash(text)).real(4,old.number("inference_ms")).bind(5,g_get_real_time()).done();int64_t revision_id=sqlite3_last_insert_rowid(impl_->db);
            Statement update(impl_->db,"UPDATE segments SET revision_id=?,edited_text=?,words=? WHERE id=?");update.bind(1,revision_id).bind(2,text).bind(3,word_count(text)).bind(4,segment_id).done();Statement imported(impl_->db,"INSERT INTO legacy_imports VALUES(?,?)");imported.bind(1,absolute).bind(2,id).done();tx.commit();
        }catch(const std::invalid_argument& e){g_warning("Cannot import legacy history metadata: %s",e.what());}
    }
}
} // namespace paraketto::service
