#include "service/session_store.h"
#include "service/dictionary.h"
#include "service/edits.h"
#include "app_common.h"
#include "sqlite/sqlite3.h"
#include <cassert>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <set>
#include <map>
#include <random>
#include <sys/wait.h>
#include <unistd.h>

using namespace paraketto::service;
using paraketto::app::Json;
namespace fs=std::filesystem;
template<class F> void rejects(F operation){bool rejected=false;try{operation();}catch(const std::exception&){rejected=true;}assert(rejected);}
static void durable(SessionStore& store,const SessionRecord& session,int64_t samples=16000){store.put_audio_block({session.id,"sessions/"+session.id+"/audio/000000.pcm.part",0,0,samples,false});}
static AsrJob plan(SessionStore& store,const SessionRecord& session,int64_t start=0,int64_t end=16000){store.plan_segment(session.id,session.generation,start,end,{{start,end}});auto job=store.next_asr();assert(job&&job->session_id==session.id);return *job;}
static void migration_tests(const fs::path& temp);
static void replacements_pause_tests(const fs::path& temp);
static void dictionary_matcher_tests();
static void spelling_tests(const fs::path& temp);
int main(){
    auto temp=fs::temp_directory_path()/("paraketto-store-test-"+std::to_string(getpid()));fs::remove_all(temp);fs::create_directories(temp);
    dictionary_matcher_tests();
    spelling_tests(temp/"spelling");
    auto entries=parse_dictionary(R"({"entries":[{"canonical":"5070Ti","aliases":["57 TTI"]},{"canonical":"Straße","aliases":["STRASSE"]},{"canonical":"New York","aliases":["new york city","ny"]}]})");
    auto mapped=apply_dictionary("hello 57 tti, STRASSE and new york city; tiny",entries);
    assert(mapped.text=="hello 5070Ti, Straße and New York; tiny");assert(render_edits("hello 57 tti, STRASSE and new york city; tiny",mapped.edits)==mapped.text);
    rejects([]{parse_dictionary(R"({"entries":[{"canonical":"A","aliases":["thing"]},{"canonical":"B","aliases":["THING"]}]})");});
    auto no_cascade=parse_dictionary(R"({"entries":[{"canonical":"foo","aliases":["bar"]},{"canonical":"baz","aliases":["foo foo"]}]})");assert(apply_dictionary("bar bar",no_cascade).text=="foo foo");
    // Source offsets are UTF-8 byte ranges; decisions render only enabled,
    // nonoverlapping replacements and can always recover the exact source.
    assert(render_edits("árvíz",{{0,2,"á","Á","dictionary",true}})=="Árvíz");
    assert(render_edits("árvíz",{{0,2,"á","Á","dictionary",false}})=="árvíz");
    rejects([]{validate_edits("é",{{1,2,"x","y","dictionary",true}});});
    rejects([]{validate_edits("hello",{{0,5,"other","Hello","dictionary",true}});});
    rejects([]{validate_edits("hello",{{0,5,"hello","Hello","dictionary",true},{0,1,"h","H","dictionary",true}});});
    std::string id;
    {
        SessionStore store(temp.string());store.set_dictionary(R"({"entries":[{"canonical":"5070Ti","aliases":["57 TTI"]}]})");
        auto s=store.create_session("start-one");id=s.id;assert(Json(store.session_json(id)).number("delivery_epoch")==0);assert(store.create_session("start-one").id==id);rejects([&]{store.create_session("second-mic");});
        durable(store,s,32000);rejects([&]{store.plan_segment(id,1,0,32001,{{0,32001}});});
        auto job=plan(store,s,0,16000);assert(store.commit_asr(job,"hello 57 TTI","model-sha256",5));assert(store.commit_asr(job,"hello 57 TTI","model-sha256",5));
        assert(store.transcript(id,"original")=="hello 57 TTI");assert(store.transcript(id)=="hello 5070Ti");
        store.plan_segment(id,1,16000,32000,{});store.seal_capture(id);
        auto pub=store.publish(id);assert(pub&&pub->text=="hello 5070Ti"&&pub->complete);
        store.retry(id);assert(Json(store.session_json(id)).number("delivery_epoch")==1);
        Json updates(store.updates_json(id));auto* segments=json_object_get_array_member(updates.object(),"segments");assert(json_array_get_length(segments)==1);auto* segment=json_array_get_object_element(segments,0);auto* edits=json_object_get_array_member(segment,"edits");assert(json_array_get_length(edits)==1);
        auto* replacement=json_array_get_object_element(edits,0);assert(std::string(json_object_get_string_member(replacement,"origin"))=="dictionary");
        store.set_replacements_enabled(id,false);assert(store.transcript(id)=="hello 57 TTI");rejects([&]{store.set_replacements_enabled(id,true,1);});
        assert(store.commit_asr(job,"hello 57 TTI","model-sha256",5));assert(store.transcript(id)=="hello 57 TTI");assert(store.publication(id)->text=="hello 5070Ti");
        store.retry(id);assert(store.transcript(id)=="hello 57 TTI");rejects([&]{store.retry(id,"cleanup");});
        store.set_dictionary(R"({"entries":[{"canonical":"RTX 5070 Ti","aliases":["57 TTI"]}]})");
        auto dictionary_generation=store.reprocess(id,false);assert(dictionary_generation==2);assert(!store.next_asr());assert(store.transcript(id,"original")=="hello 57 TTI");assert(store.transcript(id)=="hello 57 TTI");assert(!store.get_session(id)->replacements_enabled);store.set_replacements_enabled(id,true);assert(store.transcript(id)=="hello RTX 5070 Ti");assert(store.publish(id)->text=="hello RTX 5070 Ti");
        auto generation=store.reprocess(id,true,R"({"vad":false})");assert(generation==3);assert(!store.commit_asr(job,"stale","model"));assert(!store.publication(id));
        auto current=*store.get_session(id);auto re=plan(store,current,0,32000);assert(store.commit_asr(re,"um","model"));assert(store.transcript(id)=="um");
        assert(store.session_json(id).find("cleanup")==std::string::npos);assert(store.updates_json(id).find("cleanup")==std::string::npos);assert(store.counts_json().find("cleanup")==std::string::npos);
        store.delete_session(id);assert(!store.get_session(id));assert(!store.commit_asr(re,"resurrection","model"));rejects([&]{store.create_session("start-one");});assert(store.pending_deletions().size()==1);store.finish_deletion(id);
    }
    // A committed durable prefix with no planned ASR job survives a process
    // crash. The restarted controller receives it for segmentation replay.
    pid_t child=fork();assert(child>=0);
    if(child==0){SessionStore store(temp.string());auto s=store.create_session("crash-unplanned");durable(store,s,720000);_exit(0);}int status=0;waitpid(child,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);
    {
        SessionStore store(temp.string());auto sessions=store.recoverable_sessions();assert(sessions.size()==1);auto s=sessions.front();assert(s.capture_state=="interrupted"&&s.durable_samples==720000&&s.planned_samples==0);assert(Json(store.session_json(s.id)).number("delivery_epoch")>0);assert(!store.publish(s.id));
        auto job=plan(store,s,0,720000);store.record_capture_gap(s.id,1000,-1,"Device discontinuity");assert(store.commit_asr(job,"speech retained","model"));assert(!store.publish(s.id));
        Json metadata(store.session_json(s.id));assert(metadata.text("transcription_state")=="incomplete");assert(store.transcript(s.id).find("[Audio gap")!=std::string::npos);assert(store.transcript(s.id).find("speech retained")!=std::string::npos);assert(metadata.number("transcribed_through")==1000.0/16000);
    }
    // Running jobs are reset on startup and old attempts are fenced out.
    std::string restart;AsrJob stale;
    {
        SessionStore store(temp.string());auto s=store.create_session("restart-running");restart=s.id;durable(store,s);stale=plan(store,s);store.seal_capture(s.id);
    }
    {
        SessionStore store(temp.string());auto fresh=store.next_asr();assert(fresh&&fresh->id==stale.id&&fresh->attempt>stale.attempt);assert(!store.commit_asr(stale,"old","m"));assert(store.commit_asr(*fresh,"new","m"));
        // Concurrent readers and durable writes share one serialized connection.
        std::thread reader([&]{for(int i=0;i<30;++i)store.sessions_json(10);});for(int i=0;i<30;++i)store.session_json(restart);reader.join();
        fs::create_directories(temp/"history");std::ofstream(temp/"history"/"old.json")<<R"({"text":"Previously corrected text","duration":3,"inference_ms":20})";
        std::ofstream(temp/"history"/"old.wav")<<"legacy audio";store.import_legacy_history((temp/"history").string());store.import_legacy_history((temp/"history").string());Json history(store.sessions_json());auto*a=json_object_get_array_member(history.object(),"sessions");int legacy=0;std::string legacy_id;for(guint i=0;i<json_array_get_length(a);++i)if(std::string(json_object_get_string_member(json_array_get_object_element(a,i),"source"))=="legacy"){++legacy;legacy_id=json_object_get_string_member(json_array_get_object_element(a,i),"id");}assert(legacy==1);
        assert(fs::exists(temp/"history"/"old.wav")&&fs::exists(temp/"history"/"old.json"));
        store.delete_session(legacy_id);auto owned=store.deletion_paths(legacy_id);assert(owned.size()==2);for(auto& path:owned)fs::remove(path);store.finish_deletion(legacy_id);
        assert(!fs::exists(temp/"history"/"old.wav")&&!fs::exists(temp/"history"/"old.json"));
        auto external=store.create_session("external-import","import");store.delete_session(external.id);assert(store.deletion_paths(external.id).empty());store.finish_deletion(external.id);
        store.checkpoint();
    }
    {
        SessionStore store((temp/"paged").string());auto s=store.create_session("paging");durable(store,s,48000);
        for(int i=0;i<3;++i){auto job=plan(store,s,i*16000,(i+1)*16000);assert(store.fail_asr(job,"temporary failure"));}
        store.seal_capture(s.id);store.retry(s.id,"asr");
        int64_t cursor=0;std::set<int64_t> seen;
        for(int i=0;i<3;++i){Json page(store.updates_json(s.id,cursor,1));auto*a=json_object_get_array_member(page.object(),"segments");assert(json_array_get_length(a)==1);seen.insert(json_object_get_int_member(json_array_get_object_element(a,0),"id"));cursor=page.number("cursor");}
        assert(seen.size()==3);store.cancel_session(s.id);Json cancelled(store.updates_json(s.id,cursor,10));auto*a=json_object_get_array_member(cancelled.object(),"segments");assert(json_array_get_length(a)==3);
        for(guint i=0;i<json_array_get_length(a);++i)assert(std::string(json_object_get_string_member(json_array_get_object_element(a,i),"state"))=="cancelled");
    }
    migration_tests(temp);
    replacements_pause_tests(temp);
    fs::remove_all(temp);std::cout<<"Session store, recovery, dictionaries, edits and migration passed\n";
}

namespace {
struct TestDatabase {
    sqlite3* handle=nullptr;
    explicit TestDatabase(const fs::path& path){assert(sqlite3_open(path.c_str(),&handle)==SQLITE_OK);}
    ~TestDatabase(){assert(sqlite3_close(handle)==SQLITE_OK);}
    void execute(const char* statement){
        char* error=nullptr;int result=sqlite3_exec(handle,statement,nullptr,nullptr,&error);
        if(result!=SQLITE_OK){std::cerr<<(error?error:"SQLite fixture failure")<<'\n';sqlite3_free(error);}
        assert(result==SQLITE_OK);
    }
    std::string text(const char* statement){
        sqlite3_stmt* query=nullptr;assert(sqlite3_prepare_v2(handle,statement,-1,&query,nullptr)==SQLITE_OK);assert(sqlite3_step(query)==SQLITE_ROW);
        auto* value=sqlite3_column_text(query,0);std::string result=value?reinterpret_cast<const char*>(value):"";sqlite3_finalize(query);return result;
    }
};
// This is the previous on-disk v2 layout, not the current schema with its
// version number lowered. Include live, failed and completed old model runs,
// both kinds of edit decision, a snapshot, legacy history, and interrupted ASR.
void make_v2_fixture(const fs::path& directory){
    fs::create_directories(directory/"sessions"/"saved"/"audio");
    fs::create_directories(directory/"history");
    std::ofstream(directory/"sessions"/"saved"/"audio"/"000000.pcm")<<"saved recording bytes";
    std::ofstream(directory/"history"/"old.json")<<R"({"text":"Legacy corrected text","duration":1})";
    std::ofstream(directory/"history"/"old.wav")<<"saved legacy audio";
    TestDatabase db(directory/"sessions.sqlite3");
    db.execute(R"SQL(
PRAGMA foreign_keys=ON;
CREATE TABLE dictionaries(id INTEGER PRIMARY KEY,json TEXT NOT NULL,hash TEXT NOT NULL UNIQUE,created_us INTEGER NOT NULL);
CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE sessions(id TEXT PRIMARY KEY,request_key TEXT UNIQUE NOT NULL,source TEXT NOT NULL,created_us INTEGER NOT NULL,
 capture_state TEXT NOT NULL,settings TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 1,dict_id INTEGER NOT NULL REFERENCES dictionaries(id),
 durable_samples INTEGER NOT NULL DEFAULT 0,planned_samples INTEGER NOT NULL DEFAULT 0,final_samples INTEGER NOT NULL DEFAULT -1,
 error TEXT NOT NULL DEFAULT '',deleted INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL DEFAULT 1,delivery_epoch INTEGER NOT NULL DEFAULT 0);
CREATE TABLE generations(session_id TEXT REFERENCES sessions(id),generation INTEGER,settings TEXT NOT NULL,dict_id INTEGER REFERENCES dictionaries(id),
 planned_samples INTEGER NOT NULL DEFAULT 0,final_samples INTEGER NOT NULL DEFAULT -1,PRIMARY KEY(session_id,generation));
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
CREATE TABLE cleanup_runs(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,generation INTEGER NOT NULL,segment_id INTEGER REFERENCES segments(id),revision_id INTEGER REFERENCES transcript_revisions(id),
 dict_id INTEGER REFERENCES dictionaries(id),source_start INTEGER NOT NULL,source_end INTEGER NOT NULL,input TEXT NOT NULL,source_hash TEXT NOT NULL,
 config TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',attempt INTEGER NOT NULL DEFAULT 0,candidate TEXT NOT NULL DEFAULT '',metadata TEXT NOT NULL DEFAULT '{}',error TEXT NOT NULL DEFAULT '',
 UNIQUE(revision_id,dict_id,source_start,source_end));
CREATE TABLE edits(id INTEGER PRIMARY KEY,revision_id INTEGER REFERENCES transcript_revisions(id),start_byte INTEGER NOT NULL,end_byte INTEGER NOT NULL,
 before_text TEXT NOT NULL,after_text TEXT NOT NULL,origin TEXT NOT NULL,origin_id INTEGER NOT NULL,enabled INTEGER NOT NULL,decision_revision INTEGER NOT NULL DEFAULT 1,user_decided INTEGER NOT NULL DEFAULT 0,
 UNIQUE(revision_id,start_byte,end_byte,before_text,after_text,origin));
CREATE TABLE publications(id TEXT PRIMARY KEY,session_id TEXT REFERENCES sessions(id),generation INTEGER NOT NULL,text TEXT NOT NULL,
 source_refs TEXT NOT NULL,complete INTEGER NOT NULL,cleanup_complete INTEGER NOT NULL,created_us INTEGER NOT NULL,UNIQUE(session_id,generation));
CREATE TABLE legacy_imports(path TEXT PRIMARY KEY,session_id TEXT REFERENCES sessions(id));
CREATE INDEX pending_asr ON segments(state,session_id,generation,sequence);
CREATE INDEX pending_cleanup ON cleanup_runs(state,session_id,generation,id);
CREATE INDEX changed_segments ON segments(session_id,generation,cursor);
CREATE INDEX edit_revision ON edits(revision_id,start_byte);
INSERT INTO dictionaries VALUES(1,'{"entries":[]}','empty',0),
 (2,'{"entries":[{"canonical":"5070Ti","aliases":["57 TTI"]},{"canonical":"New York","aliases":["ny"]}]}','dictionary-fixture',1);
INSERT INTO settings VALUES('dictionary','2');
INSERT INTO sessions(id,request_key,source,created_us,capture_state,settings,dict_id,durable_samples,planned_samples,final_samples,revision) VALUES
 ('saved','saved-request','microphone',1,'stopped','{}',2,16000,16000,16000,50),
 ('legacy','legacy-request','legacy',2,'stopped','{}',1,16000,16000,16000,20),
 ('interrupted','interrupted-request','microphone',3,'recording','{}',2,32000,16000,-1,30);
INSERT INTO generations VALUES('saved',1,'{}',2,16000,16000),('legacy',1,'{}',1,16000,16000),('interrupted',1,'{}',2,16000,-1);
INSERT INTO audio_blocks VALUES('saved',0,'sessions/saved/audio/000000.pcm',0,16000,1),('legacy',0,'history/old.wav',0,16000,1),('interrupted',0,'sessions/interrupted/audio/000000.pcm.part',0,32000,0);
INSERT INTO segments(id,session_id,generation,sequence,coverage_start,coverage_end,spans,state,attempt,config,revision_id,edited_text,words,cursor) VALUES
 (1,'saved',1,0,0,16000,'{"spans":[{"start":0,"end":16000}]}','completed',1,'{"vad":true}',1,'Hello 5070Ti and ny',4,40),
 (2,'legacy',1,0,0,16000,'{"spans":[{"start":0,"end":16000}]}','completed',1,'{}',2,'Legacy corrected text',3,20),
 (3,'interrupted',1,0,0,16000,'{"spans":[{"start":0,"end":16000}]}','running',2,'{}',NULL,'',0,30);
INSERT INTO dispositions VALUES('saved',1,0,16000,'speech'),('legacy',1,0,16000,'speech'),('interrupted',1,0,16000,'speech');
INSERT INTO transcript_revisions VALUES(1,1,'hello 57 TTI and ny','raw-hash','{"weights_sha256":"saved-model"}',5,'asr',10),
 (2,2,'Legacy corrected text','legacy-hash','legacy/import',0,'legacy',11);
INSERT INTO cleanup_runs(id,session_id,generation,segment_id,revision_id,dict_id,source_start,source_end,input,source_hash,config,state,attempt,candidate,metadata,error) VALUES
 (100,'saved',1,1,1,2,0,5,'hello','raw-hash','{"prompt":"pinned"}','completed',1,'Hello','{"runtime":"original"}',''),
 (101,'saved',1,1,1,2,6,12,'5070Ti','raw-hash','{"prompt":"pinned"}','running',2,'','{}',''),
 (102,'saved',1,1,1,2,13,16,'and','raw-hash','{"prompt":"pinned"}','pending',0,'','{}',''),
 (103,'saved',1,1,1,2,17,19,'ny','raw-hash','{"prompt":"pinned"}','failed',1,'unsupported rewrite','{"stop":"limit"}','validation failed'),
 (104,'saved',1,1,1,2,0,19,'hello 5070Ti and ny','raw-hash','{"prompt":"pinned"}','cancelled',1,'','{}','cancelled');
INSERT INTO edits VALUES(10,1,6,12,'57 TTI','5070Ti','dictionary',2,1,1,0),
 (11,1,17,19,'ny','New York','dictionary',2,0,4,1),
 (20,1,0,5,'hello','Hello','s1-mini',100,1,1,0),
 (21,1,13,16,'and','plus','other-model',103,0,3,1);
INSERT INTO publications VALUES('saved-publication','saved',1,'Hello 5070Ti and ny','{"segments":[{"segment_id":1,"revision_id":1,"decisions":[{"edit_id":20,"decision_revision":1,"enabled":true}]}]}',1,0,100);
PRAGMA user_version=2;
)SQL");
    auto* insert=sqlite3_mprintf("INSERT INTO legacy_imports VALUES(%Q,'legacy')",(directory/"history"/"old.json").c_str());db.execute(insert);sqlite3_free(insert);
    auto hash=text_hash("hello 57 TTI and ny");auto* hashes=sqlite3_mprintf("UPDATE transcript_revisions SET hash=%Q WHERE id=1; UPDATE cleanup_runs SET source_hash=%Q",hash.c_str(),hash.c_str());db.execute(hashes);sqlite3_free(hashes);
}
}

static void migration_tests(const fs::path& temp){
    auto directory=temp/"migration-v2";make_v2_fixture(directory);
    std::string source_refs;
    {TestDatabase db(directory/"sessions.sqlite3");source_refs=db.text("SELECT source_refs FROM publications WHERE id='saved-publication'");}
    {
        SessionStore store(directory.string());assert(fs::exists(directory/"sessions.pre-v3.sqlite3"));assert(fs::exists(directory/"sessions.pre-v4.sqlite3"));assert(!store.get_session("saved")->replacements_enabled);
        assert(store.transcript("saved","original")=="hello 57 TTI and ny");assert(store.transcript("saved")=="hello 57 TTI and ny");
        assert(store.publication("saved")->id=="saved-publication");assert(store.publication("saved")->text=="Hello 5070Ti and ny");
        assert(store.publish("saved")->text=="Hello 5070Ti and ny");assert(store.publication_json("saved").find("cleanup")==std::string::npos);
        assert(store.get_session("saved")->durable_samples==16000);assert(store.audio_blocks("saved").front().path=="sessions/saved/audio/000000.pcm");
        assert(fs::file_size(directory/"sessions"/"saved"/"audio"/"000000.pcm")==21);
        assert(store.transcript("legacy","original")=="Legacy corrected text");assert(fs::exists(directory/"history"/"old.wav"));assert(fs::exists(directory/"history"/"old.json"));
        store.import_legacy_history((directory/"history").string());Json history(store.sessions_json());assert(json_array_get_length(json_object_get_array_member(history.object(),"sessions"))==3);
        Json changes(store.updates_json("saved",50));auto* segments=json_object_get_array_member(changes.object(),"segments");assert(json_array_get_length(segments)==1);
        auto* segment=json_array_get_object_element(segments,0);assert(std::string(json_object_get_string_member(segment,"edited_text"))=="hello 57 TTI and ny");
        auto* edits=json_object_get_array_member(segment,"edits");assert(json_array_get_length(edits)==2);
        for(guint i=0;i<json_array_get_length(edits);++i)assert(std::string(json_object_get_string_member(json_array_get_object_element(edits,i),"origin"))=="dictionary");
        rejects([&]{store.retry("saved","cleanup");});
        store.retry("saved","all");store.retry("saved","asr");assert(store.transcript("saved")=="hello 57 TTI and ny");
        store.set_replacements_enabled("saved",true);assert(store.transcript("saved")=="hello 5070Ti and New York");
        store.set_replacements_enabled("saved",false);assert(store.transcript("saved")=="hello 57 TTI and ny");
        auto interrupted=store.get_session("interrupted");assert(interrupted&&interrupted->capture_state=="interrupted"&&interrupted->durable_samples==32000&&interrupted->planned_samples==16000);
        auto recovered=store.next_asr();assert(recovered&&recovered->session_id=="interrupted"&&recovered->attempt==3);assert(store.commit_asr(*recovered,"recovered raw text","recovered-model"));
        auto tail=plan(store,*interrupted,16000,32000);assert(store.commit_asr(tail,"and tail","recovered-model"));assert(store.publish("interrupted"));
        TestDatabase db(directory/"sessions.sqlite3");assert(db.text("PRAGMA user_version")=="4");
        assert(db.text("SELECT count(*) FROM sqlite_schema WHERE name IN ('cleanup_runs','pending_cleanup')")=="0");
        assert(db.text("SELECT count(*) FROM retired_cleanup_runs WHERE state='retired'")=="5");
        assert(db.text("SELECT group_concat(retired_state,',') FROM (SELECT retired_state FROM retired_cleanup_runs ORDER BY id)")=="completed,running,pending,failed,cancelled");
        assert(db.text("SELECT candidate||metadata FROM retired_cleanup_runs WHERE id=100")=="Hello{\"runtime\":\"original\"}");
        assert(db.text("SELECT count(*) FROM edits WHERE origin!='dictionary' AND enabled!=0")=="0");
        assert(db.text("SELECT decision_revision FROM edits WHERE id=20")=="2");assert(db.text("SELECT decision_revision FROM edits WHERE id=21")=="3");
        assert(db.text("SELECT source_refs FROM publications WHERE id='saved-publication'")==source_refs);
        assert(db.text("SELECT model FROM transcript_revisions WHERE id=1")=="{\"weights_sha256\":\"saved-model\"}");
        assert(db.text("SELECT text FROM transcript_revisions WHERE id=1")=="hello 57 TTI and ny");
    }
    {
        TestDatabase backup(directory/"sessions.pre-v3.sqlite3");assert(backup.text("PRAGMA user_version")=="2");
        assert(backup.text("SELECT count(*) FROM cleanup_runs WHERE state IN ('running','pending')")=="2");
        assert(backup.text("SELECT enabled FROM edits WHERE id=20")=="1");assert(backup.text("SELECT source_refs FROM publications WHERE id='saved-publication'")==source_refs);
    }
    {
        SessionStore store(directory.string());assert(store.transcript("saved")=="hello 57 TTI and ny");assert(!store.next_asr());
        TestDatabase db(directory/"sessions.sqlite3");assert(db.text("SELECT decision_revision FROM edits WHERE id=20")=="2");
        store.set_dictionary(R"({"entries":[{"canonical":"RTX 5070 Ti","aliases":["57 TTI"]}]})");
        assert(store.reprocess("saved",false)==2);assert(!store.next_asr());assert(store.transcript("saved","original")=="hello 57 TTI and ny");assert(store.transcript("saved")=="hello 57 TTI and ny");store.set_replacements_enabled("saved",true);assert(store.transcript("saved")=="hello RTX 5070 Ti and ny");
        assert(db.text("SELECT text FROM publications WHERE id='saved-publication'")=="Hello 5070Ti and ny");assert(store.publish("saved")->text=="hello RTX 5070 Ti and ny");
        store.delete_session("saved");store.finish_deletion("saved");assert(db.text("SELECT count(*) FROM retired_cleanup_runs WHERE session_id='saved'")=="0");
        assert(db.text("SELECT count(*) FROM edits WHERE revision_id=1")=="0");assert(!store.get_session("saved"));
    }
    // A v1 database still follows both migrations and gets a backup before each.
    auto first=temp/"migration-v1";make_v2_fixture(first);
    {TestDatabase db(first/"sessions.sqlite3");db.execute("ALTER TABLE sessions DROP COLUMN delivery_epoch; PRAGMA user_version=1;");}
    {SessionStore store(first.string());assert(fs::exists(first/"sessions.pre-v2.sqlite3"));assert(fs::exists(first/"sessions.pre-v3.sqlite3"));assert(store.transcript("saved")=="hello 57 TTI and ny");}
}

static void replacements_pause_tests(const fs::path& temp){
    auto revision=[](SessionStore& store,const std::string& id){return int64_t(Json(store.session_json(id)).number("revision"));};
    auto directory=temp/"replacements";
    std::string identity;
    {
        SessionStore store(directory.string());store.set_dictionary(R"({"entries":[{"canonical":"Purple","aliases":["red blue"]}]})");
        auto s=store.create_session("replacements");identity=s.id;assert(s.replacements_enabled);
        assert(Json(store.session_json(s.id)).boolean("replacements_enabled"));auto initial=revision(store,s.id);
        store.set_replacements_enabled(s.id,false,initial);auto disabled=revision(store,s.id);assert(disabled>initial);
        store.set_replacements_enabled(s.id,false,initial);assert(revision(store,s.id)==disabled); // lost-response retry
        rejects([&]{store.set_replacements_enabled(s.id,true,initial);});rejects([&]{store.set_replacements_enabled(s.id,true,-2);});
        durable(store,s,64000);
        auto first=plan(store,s,0,16000);assert(store.commit_asr(first,"red blue","model"));assert(store.transcript(s.id)=="red blue");
        auto second=plan(store,s,16000,32000);store.set_replacements_enabled(s.id,true);assert(store.transcript(s.id).find("Purple")==0);
        assert(store.commit_asr(second,"red blue","model"));assert(store.transcript(s.id)=="Purple Purple");
        auto third=plan(store,s,32000,48000);store.set_replacements_enabled(s.id,false);assert(store.commit_asr(third,"red blue","model"));
        auto fourth=plan(store,s,48000,64000);assert(store.commit_asr(fourth,"red blue","model"));
        const std::string original="red blue red blue red blue red blue",replaced="Purple Purple Purple Purple";
        assert(store.transcript(s.id)==original);assert(store.transcript(s.id,"original")==original);
        TestDatabase db(directory/"sessions.sqlite3");
        assert(db.text("SELECT count(*) FROM edits WHERE enabled!=0")=="0");
        assert(db.text("SELECT count(*) FROM edits")=="4"); // disabled edits remain available for hover/re-enable
        // A failure halfway through a many-segment change rolls back both the
        // switch and every previously rebuilt segment/cursor in the transaction.
        auto before_failure=revision(store,s.id);
        db.execute("CREATE TRIGGER reject_replacement BEFORE UPDATE OF enabled ON edits WHEN NEW.enabled=1 AND EXISTS(SELECT 1 FROM segments a WHERE a.revision_id=OLD.revision_id AND a.sequence=1) BEGIN SELECT RAISE(ABORT,'fixture write failure'); END;");
        rejects([&]{store.set_replacements_enabled(s.id,true);});
        assert(!store.get_session(s.id)->replacements_enabled);assert(store.transcript(s.id)==original);assert(revision(store,s.id)==before_failure);
        assert(db.text("SELECT count(*) FROM edits WHERE enabled!=0")=="0");db.execute("DROP TRIGGER reject_replacement;");
        auto cursor=revision(store,s.id);store.set_replacements_enabled(s.id,true,cursor);
        assert(store.transcript(s.id)==replaced);assert(Json(store.session_json(s.id)).number("word_count")==4);
        std::set<int64_t> changed;
        for(int i=0;i<4;++i){Json page(store.updates_json(s.id,cursor,1));auto* a=json_object_get_array_member(page.object(),"segments");assert(json_array_get_length(a)==1);auto* item=json_array_get_object_element(a,0);changed.insert(json_object_get_int_member(item,"id"));assert(std::string(json_object_get_string_member(item,"edited_text"))=="Purple");cursor=page.number("cursor");auto* metadata=json_object_get_object_member(page.object(),"session");assert(json_object_get_boolean_member(metadata,"replacements_enabled"));}
        assert(changed.size()==4);
        store.seal_capture(s.id);assert(store.publish(s.id)->text==replaced);
        auto publication_refs=db.text("SELECT source_refs FROM publications LIMIT 1");
        std::thread reader([&]{for(int i=0;i<50;++i){auto text=store.transcript(s.id);assert(text==original||text==replaced);}});
        for(int i=0;i<10;++i)store.set_replacements_enabled(s.id,i%2!=0);
        reader.join();
        store.set_replacements_enabled(s.id,false);assert(store.transcript(s.id)==original);assert(Json(store.session_json(s.id)).number("word_count")==8);
        assert(store.publication(s.id)->text==replaced);assert(db.text("SELECT source_refs FROM publications LIMIT 1")==publication_refs);
        assert(store.reprocess(s.id,false)==2);assert(!store.get_session(s.id)->replacements_enabled);assert(!store.next_asr());assert(store.transcript(s.id)==original);
        assert(db.text("SELECT replacements_enabled FROM generations WHERE generation=2")=="0");
        assert(store.reprocess(s.id,true)==3);assert(!store.get_session(s.id)->replacements_enabled);
        auto current=*store.get_session(s.id);auto re=plan(store,current,0,64000);assert(store.commit_asr(re,"red blue","model"));assert(store.transcript(s.id)=="red blue");
        store.set_replacements_enabled(s.id,true);assert(store.transcript(s.id)=="Purple");
        assert(db.text("SELECT group_concat(replacements_enabled,',') FROM (SELECT replacements_enabled FROM generations ORDER BY generation)")=="0,0,1");
        store.set_replacements_enabled(s.id,false);assert(Json(store.session_json(s.id)).number("delivery_epoch")==2);
    }
    {
        SessionStore store(directory.string());assert(!store.get_session(identity)->replacements_enabled);assert(store.transcript(identity)=="red blue");
        store.delete_session(identity);rejects([&]{store.set_replacements_enabled(identity,true);});rejects([&]{store.set_capture_paused(identity,true);});store.finish_deletion(identity);
    }
    {
        SessionStore store((temp/"pause").string());
        auto old=store.create_session("old-import","import");durable(store,old);store.plan_segment(old.id,1,0,16000,{{0,16000}});store.seal_capture(old.id);
        auto active=store.create_session("paused-capture");durable(store,active);store.plan_segment(active.id,1,0,16000,{{0,16000}});
        auto before=revision(store,active.id);store.set_capture_paused(active.id,true);auto paused=store.get_session(active.id);assert(paused&&paused->capture_state=="paused"&&paused->final_samples==-1);assert(revision(store,active.id)>before);
        before=revision(store,active.id);store.set_capture_paused(active.id,true);assert(revision(store,active.id)==before);
        assert(store.active_session()->id==active.id);assert(store.create_session("paused-capture").capture_state=="paused");rejects([&]{store.create_session("another-microphone");});
        rejects([&]{store.reprocess(active.id,false);});rejects([&]{store.reprocess(active.id,true);});
        auto job=store.next_asr();assert(job&&job->session_id==active.id);assert(store.commit_asr(*job,"paused text","model"));
        assert(!store.publish(active.id));assert(!Json(store.session_json(active.id)).boolean("complete"));assert(Json(store.session_json(active.id)).text("transcription_state")=="pending");
        auto processing=store.processing_sessions();assert(std::any_of(processing.begin(),processing.end(),[&](auto& s){return s.id==active.id;}));
        assert(Json(store.session_json(active.id)).number("delivery_epoch")==0);store.set_capture_paused(active.id,false);assert(store.get_session(active.id)->capture_state=="recording");
        durable(store,active,32000);assert(store.get_session(active.id)->final_samples==-1);store.plan_segment(active.id,1,16000,32000,{});
        store.set_capture_paused(active.id,true);assert(!store.publish(active.id));store.seal_capture(active.id);assert(store.publish(active.id)->text=="paused text");
        assert(Json(store.session_json(active.id)).number("delivery_epoch")==0);rejects([&]{store.set_capture_paused(active.id,true);});rejects([&]{store.set_capture_paused(active.id,false);});
        auto old_job=store.next_asr();assert(old_job&&old_job->session_id==old.id);assert(store.commit_asr(*old_job,"import","model"));
        auto cancelled=store.create_session("cancel-paused");durable(store,cancelled);store.set_capture_paused(cancelled.id,true);store.cancel_session(cancelled.id);assert(store.get_session(cancelled.id)->capture_state=="cancelled");assert(!store.publish(cancelled.id));rejects([&]{store.set_capture_paused(cancelled.id,false);});
    }
    auto crash=temp/"paused-crash";pid_t child=fork();assert(child>=0);
    if(child==0){SessionStore store(crash.string());auto s=store.create_session("paused-before-crash");durable(store,s,32000);store.set_replacements_enabled(s.id,false);store.set_capture_paused(s.id,true);_exit(0);}
    int status=0;waitpid(child,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);
    {
        SessionStore store(crash.string());assert(!store.active_session());auto sessions=store.recoverable_sessions();assert(sessions.size()==1);auto s=sessions.front();assert(s.capture_state=="interrupted"&&s.durable_samples==32000&&s.planned_samples==0&&s.final_samples==32000);assert(!s.replacements_enabled);
        assert(Json(store.session_json(s.id)).number("delivery_epoch")>0);assert(!store.publish(s.id));auto job=plan(store,s,0,32000);assert(store.commit_asr(job,"saved before pause","model"));assert(store.publish(s.id));
    }
    // Open the real v3 snapshot directly: its mixed decisions become all-off,
    // while its immutable publication and default-on empty legacy generation survive.
    auto direct=temp/"migration-v3";fs::create_directories(direct);fs::copy_file(temp/"migration-v2"/"sessions.pre-v4.sqlite3",direct/"sessions.sqlite3");
    {
        SessionStore store(direct.string());assert(fs::exists(direct/"sessions.pre-v4.sqlite3"));assert(!store.get_session("saved")->replacements_enabled);assert(store.get_session("legacy")->replacements_enabled);
        assert(store.transcript("saved")=="hello 57 TTI and ny");assert(store.publication("saved")->text=="Hello 5070Ti and ny");
        TestDatabase backup(direct/"sessions.pre-v4.sqlite3");assert(backup.text("PRAGMA user_version")=="3");assert(backup.text("SELECT enabled FROM edits WHERE id=10")=="1");assert(backup.text("SELECT enabled FROM edits WHERE id=11")=="0");
        TestDatabase db(direct/"sessions.sqlite3");assert(db.text("PRAGMA user_version")=="4");assert(db.text("SELECT count(*) FROM edits WHERE origin='dictionary' AND enabled!=0")=="0");
        store.set_replacements_enabled("saved",true);assert(store.transcript("saved")=="hello 5070Ti and New York");assert(db.text("SELECT replacements_enabled FROM generations WHERE session_id='saved'")=="1");
    }
}

static DictionaryText brute_dictionary(const std::string& source,const std::vector<DictionaryEntry>& entries) {
    auto fold=[](const std::string& text){return paraketto::app::take_string(g_utf8_casefold(text.data(),text.size()));};
    auto word=[](gunichar c){return g_unichar_isalnum(c)||g_unichar_ismark(c)||c=='_'||c=='\''||c==0x2019;};
    std::map<std::string,std::string> rules;size_t longest=0;
    for(const auto& entry:entries) {
        auto aliases=entry.aliases;aliases.push_back(entry.canonical);
        for(const auto& alias:aliases){auto key=fold(alias);longest=std::max(longest,key.size());rules[key]=entry.canonical;}
    }
    DictionaryText result;
    for(size_t pos=0;pos<source.size();) {
        size_t next=g_utf8_next_char(source.data()+pos)-source.data(),best=pos;std::string replacement;
        if(!pos||!word(g_utf8_get_char(g_utf8_prev_char(source.data()+pos))))
            for(size_t end=next;end<=source.size()&&end-pos<=longest*4;) {
                if(end==source.size()||!word(g_utf8_get_char(source.data()+end))) {
                    auto found=rules.find(fold(source.substr(pos,end-pos)));
                    if(found!=rules.end()){best=end;replacement=found->second;}
                }
                if(end==source.size())break;
                end=g_utf8_next_char(source.data()+end)-source.data();
            }
        if(best>pos) {
            result.text+=replacement;auto before=source.substr(pos,best-pos);
            if(before!=replacement)result.edits.push_back({pos,best,before,replacement,"dictionary",true});
            pos=best;
        } else {result.text.append(source,pos,next-pos);pos=next;}
    }
    return result;
}
static void dictionary_matcher_tests() {
    const std::vector<DictionaryEntry> entries{{"Pair",{"alpha beta"}},{"B",{"beta"}},
        {"Suffix",{"beta gamma"}},{"Greek",{"gamma"}},{"Café",{"café"}},
        {"Road",{"strasse"}},{"Double",{"ss"}},{"Single",{"s"}},
        {"Name",{"o’neil"}},{"Contraction",{"don't"}},{"Accent",{"á"}}};
    DictionaryMatcher matcher(entries);
    auto compare=[&](const std::string& text) {
        auto expected=brute_dictionary(text,entries),actual=matcher.apply(text);
        assert(actual.text==expected.text);assert(actual.edits.size()==expected.edits.size());
        assert(render_edits(text,actual.edits)==actual.text);
        for(size_t i=0;i<actual.edits.size();++i) {
            const auto& a=actual.edits[i];const auto& b=expected.edits[i];
            assert(a.start==b.start&&a.end==b.end&&a.before==b.before&&a.after==b.after);
        }
    };
    for(const auto* text:{"alpha beta gamma", "ALPHA BETA, beta gamma!", "ß s ss Straße STRASSE",
        "Café café O’NEIL don't don'ts á", "🙂alpha beta🙂", "beta_gamma 12beta beta12", "", "plain untouched text"})compare(text);
    assert(DictionaryMatcher({{"Single",{"s"}}}).apply("ß").text=="ß");
    assert(DictionaryMatcher({{"foo",{"bar"}},{"baz",{"foo foo"}}}).apply("bar bar").text=="foo foo");
    std::mt19937 random(5070);
    const std::vector<std::string> tokens{"alpha","beta","gamma","ALPHA","Café","café","ß","ss","s","O’NEIL","don't","á","unchanged","项目","🙂"};
    const std::vector<std::string> separators{" ",", ","\n","_","’"," "};
    for(int sample=0;sample<750;++sample) {
        std::string text;const int count=1+random()%18;
        for(int i=0;i<count;++i){if(i)text+=separators[random()%separators.size()];text+=tokens[random()%tokens.size()];}
        compare(text);
    }
    rejects([&]{matcher.apply(std::string("\xff",1));});
    auto mapped=DictionaryMatcher({{"Paraketto",{"paraketo"}}}).apply("José likes paraketo today");
    const auto begin=mapped.text.find("today");
    auto span=selection_source_span("José likes paraketo today",mapped.edits,begin,begin+5,"today");
    assert(std::string("José likes paraketo today").substr(span.first,span.second-span.first)=="today");
    const auto replaced=mapped.text.find("Paraketto");
    span=selection_source_span("José likes paraketo today",mapped.edits,replaced,replaced+9,"Paraketto");
    assert(std::string("José likes paraketo today").substr(span.first,span.second-span.first)=="paraketo");
    rejects([&]{selection_source_span("José likes paraketo today",mapped.edits,replaced+1,replaced+4,"ara");});
    rejects([&]{selection_source_span("José likes paraketo today",mapped.edits,4,5,"x");});
}
static std::string spelling_request(SessionStore& store,const std::string& id,const std::string& selected,const std::string& canonical) {
    using namespace paraketto::app;
    Json updates(store.updates_json(id));auto* list=json_object_get_array_member(updates.object(),"segments");
    JsonObject* segment=nullptr;size_t start=std::string::npos;
    for(guint i=0;i<json_array_get_length(list);++i) {
        auto* candidate=json_array_get_object_element(list,i);std::string text=json_object_get_string_member(candidate,"edited_text");
        auto position=text.find(selected);if(position!=std::string::npos){segment=candidate;start=position;break;}
    }
    assert(segment);auto session=*store.get_session(id);
    auto* b=json_builder_new();json_builder_begin_object(b);
    json_int(b,"segment_id",json_object_get_int_member(segment,"id"));json_int(b,"revision_id",json_object_get_int_member(segment,"revision_id"));
    json_int(b,"generation",session.generation);json_int(b,"start",start);json_int(b,"end",start+selected.size());
    json_bool(b,"replacements_enabled",session.replacements_enabled);json_field(b,"selected",selected);json_field(b,"canonical",canonical);
    json_builder_end_object(b);auto body=json_string(b);g_object_unref(b);return body;
}
static void spelling_tests(const fs::path& directory) {
    SessionStore store(directory.string());
    store.set_dictionary(R"({"entries":[{"canonical":"Paraketto","aliases":["paraketo"]}]})");
    auto s=store.create_session("spelling-live","import");durable(store,s,48000);
    const auto first=plan(store,s,0,16000);const std::string raw="José uses paraketo and project sevn.";
    assert(store.commit_asr(first,raw,"fixture"));
    auto second=plan(store,s,16000,32000); // Claim before the spelling changes.
    const auto request=spelling_request(store,s.id,"project sevn","ProjectSeven");
    store.correct_spelling(s.id,request);
    assert(store.transcript(s.id)=="José uses Paraketto and ProjectSeven. [Transcription pending]");
    assert(store.transcript(s.id,"original")==raw+" [Transcription pending]");
    assert(store.get_session(s.id)->capture_state=="recording"&&store.get_session(s.id)->generation==s.generation);
    assert(store.commit_asr(first,raw,"fixture"));
    const auto saved=store.dictionary_json();
    rejects([&]{store.correct_spelling(s.id,request);});assert(store.dictionary_json()==saved);
    rejects([&]{store.correct_spelling(s.id,spelling_request(store,s.id,"rojectSeven","Wrong"));});
    rejects([&]{store.correct_spelling(s.id,spelling_request(store,s.id,"ProjectSeven","Different"));});
    assert(store.dictionary_json()==saved);
    // A segment already in flight uses the selected session's updated rules.
    assert(store.commit_asr(second,"project sevn again","fixture"));
    assert(store.transcript(s.id).find("ProjectSeven again")!=std::string::npos);
    store.plan_segment(s.id,s.generation,32000,48000,{});store.seal_capture(s.id);
    const auto publication=*store.publish(s.id);
    store.correct_spelling(s.id,spelling_request(store,s.id,"uses","Uses"));
    assert(store.publication(s.id)->text==publication.text);
    assert(store.transcript(s.id).find("José Uses")!=std::string::npos);
    assert(store.transcript(s.id,"original").find(raw)!=std::string::npos);
    // Incremental API merges preserve other clients' entries; a stale full edit is rejected.
    auto revision=int64_t(Json(store.dictionary_json()).number("revision"));
    std::thread a([&]{store.add_dictionary_entry(R"({"canonical":"Alpha","aliases":["alfa"]})");});
    std::thread b([&]{store.add_dictionary_entry(R"({"canonical":"Beta","aliases":["betta"]})");});a.join();b.join();
    const auto merged=store.dictionary_json();assert(merged.find("Alpha")!=std::string::npos&&merged.find("Beta")!=std::string::npos);
    rejects([&]{store.set_dictionary("{\"entries\":[],\"expected_revision\":"+std::to_string(revision)+"}");});
    assert(store.dictionary_json()==merged);
    // Evict cached revisions, then process a pinned old session after another write.
    auto pinned=store.create_session("pinned","import");durable(store,pinned);
    for(int i=0;i<8;++i)store.set_dictionary("{\"entries\":[{\"canonical\":\"Other"+std::to_string(i)+"\",\"aliases\":[\"alfa\"]}]}");
    const auto pinned_job=plan(store,pinned);assert(store.commit_asr(pinned_job,"alfa betta project sevn","fixture"));
    assert(store.transcript(pinned.id)=="Alpha Beta ProjectSeven");
    auto future=store.create_session("future","import");durable(store,future);
    assert(store.commit_asr(plan(store,future),"alfa","fixture"));assert(store.transcript(future.id)=="Other7");
}
