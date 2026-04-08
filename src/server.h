// server.h — HTTP server for Parakeet backends (header-only, templated)
//
// PipelineT must expose:
//   std::string transcribe(const float* samples, int num_samples)
//   double last_mel_ms, last_enc_ms, last_dec_ms
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "cpp-httplib/httplib.h"
#include "wav.h"

using namespace paraketto;

#ifdef WITH_CORRECTOR
#include "corrector.h"
#endif

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

static thread_local std::string t_log_detail;

static inline void log_request(const httplib::Request& req, const httplib::Response& res) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char ts[20];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    fprintf(stderr, "%s  %s %s  %d\n", ts, req.method.c_str(), req.path.c_str(), res.status);

    if (!t_log_detail.empty()) {
        fprintf(stderr, "         %s\n", t_log_detail.c_str());
        t_log_detail.clear();
    }
}

#ifdef WITH_CORRECTOR
template<typename PipelineT>
static void run_server(PipelineT& pipeline, Corrector* corrector, const std::string& host, int port) {
#else
template<typename PipelineT>
static void run_server(PipelineT& pipeline, const std::string& host, int port) {
#endif
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/shutdown", [&svr](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"shutting down\"}", "application/json");
        svr.stop();
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>paraketto</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #09090b;
         color: #e0e0e0; display: flex; justify-content: center; padding: 3rem 1.5rem;
         min-height: 100vh; }
  .container { width: 100%; max-width: 580px; }
  header { text-align: center; margin-bottom: 2rem; }
  header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
  header p { font-size: 0.82rem; color: #52525b; margin-top: 2px; }
  .rec-area { display: flex; flex-direction: column; align-items: center; gap: 12px; }
  #recbtn { width: 72px; height: 72px; border-radius: 50%; border: 2px solid #27272a;
            background: #18181b; cursor: pointer; display: flex; align-items: center;
            justify-content: center; transition: all 0.2s; padding: 0; }
  #recbtn:hover { border-color: #3f3f46; background: #1c1c1f; transform: scale(1.05); }
  #recbtn.recording { border-color: #dc2626; background: #1c1017;
                      animation: pulse 1.5s ease-in-out infinite; }
  #recbtn .mic { width: 28px; height: 28px; fill: #a1a1aa; color: #a1a1aa; transition: fill 0.2s, color 0.2s; }
  #recbtn:hover .mic { fill: #d4d4d8; color: #d4d4d8; }
  #recbtn.recording .mic { fill: #f87171; }
  #recbtn .stop { width: 22px; height: 22px; fill: #f87171; display: none; }
  #recbtn.recording .mic { display: none; }
  #recbtn.recording .stop { display: block; }
  @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.3); }
                     50% { box-shadow: 0 0 0 12px rgba(220,38,38,0); } }
  .rec-hint { font-size: 0.78rem; color: #3f3f46; }
  #waveform { width: 100%; height: 56px; display: none; margin-top: 12px; border-radius: 8px; }
  #waveform.visible { display: block; }
  .rec-time { font-size: 0.82rem; color: #71717a; font-variant-numeric: tabular-nums;
              display: none; margin-top: 4px; }
  .rec-time.visible { display: block; }
  .divider { display: flex; align-items: center; gap: 12px; margin: 20px 0;
             color: #27272a; font-size: 0.75rem; }
  .divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: #1c1c1f; }
  .drop-zone { border: 1px dashed #27272a; border-radius: 10px; padding: 14px;
               text-align: center; cursor: pointer; transition: all 0.2s; }
  .drop-zone:hover, .drop-zone.dragover { border-color: #3f3f46; background: #0f0f11; }
  .drop-zone p { color: #3f3f46; font-size: 0.82rem; }
  .result { margin-top: 24px; display: none; }
  .result.visible { display: block; }
  .transcript { background: #0f0f11; border: 1px solid #1c1c1f; border-radius: 12px;
                padding: 20px; font-size: 1.05rem; line-height: 1.7;
                color: #f4f4f5; min-height: 60px; white-space: pre-wrap; }
  .meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
  .meta span { font-size: 0.75rem; color: #3f3f46; font-variant-numeric: tabular-nums; }
  .meta .val { color: #71717a; }
  .spinner { display: none; margin: 20px auto; width: 20px; height: 20px;
             border: 2px solid #27272a; border-top-color: #71717a; border-radius: 50%;
             animation: spin 0.6s linear infinite; }
  .spinner.visible { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  input[type="file"] { display: none; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>parakettő</h1>
    <p>speech-to-text</p>
  </header>

  <div class="rec-area">
    <button id="recbtn" onclick="toggleRecord()">
      <svg class="mic" viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="12" y1="19" x2="12" y2="22" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="8" y1="22" x2="16" y2="22" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      <svg class="stop" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
    </button>
    <div class="rec-hint">click to record</div>
    <canvas id="waveform"></canvas>
    <div class="rec-time" id="rectime">0:00</div>
  </div>

  <div class="divider">or</div>

  <div class="drop-zone" id="dropzone">
    <p>drop or click to upload an audio file</p>
  </div>
  <input type="file" id="fileinput" accept="audio/*">

  <div class="spinner" id="spinner"></div>

  <div class="result" id="result">
    <div class="transcript" id="transcript"></div>
    <div class="meta" id="meta"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);

// --- File upload / drop ---
$('dropzone').addEventListener('click', () => $('fileinput').click());
$('fileinput').addEventListener('change', e => { if (e.target.files[0]) sendFile(e.target.files[0]); });
$('dropzone').addEventListener('dragover', e => { e.preventDefault(); $('dropzone').classList.add('dragover'); });
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop', e => {
  e.preventDefault();
  $('dropzone').classList.remove('dragover');
  if (e.dataTransfer.files[0]) sendFile(e.dataTransfer.files[0]);
});

// --- WAV encoding helper ---
function pcmToWav(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8,'WAVE');
  w(12,'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36,'data'); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++) {
    const x = Math.max(-1, Math.min(1, c[i]));
    v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7FFF, true); o += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// --- Microphone: AudioWorklet capture + waveform visualization, transcribe on stop ---
let micStream = null, micCtx = null, analyser = null, recording = false, recPcm = [];
let waveRaf = null, recStart = 0;
async function initMic() {
  if (micStream) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    micCtx = new AudioContext({ sampleRate: 16000 });
    await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class R extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0]?.[0];
          if (ch?.length) this.port.postMessage(new Float32Array(ch));
          return true;
        }
      }
      registerProcessor('rec', R);
    `], { type: 'application/javascript' })));
    const source = micCtx.createMediaStreamSource(micStream);
    const node = new AudioWorkletNode(micCtx, 'rec');
    node.port.onmessage = e => { if (recording) recPcm.push(e.data); };
    source.connect(node);
    // Analyser for waveform visualization
    analyser = micCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    return true;
  } catch(e) {
    alert('Microphone access denied');
    return false;
  }
}
let rmsBuf = null, rmsHead = 0, rmsLen = 0, lastRmsPush = 0;
function drawWaveform() {
  if (!recording) return;
  const canvas = $('waveform'), ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const barW = 3, gap = 2, step = barW + gap;
  const maxBars = Math.floor(w / step);
  // Resize ring buffer if canvas width changed
  if (!rmsBuf || rmsBuf.length !== maxBars) {
    rmsBuf = new Float32Array(maxBars);
    rmsHead = 0; rmsLen = 0;
  }
  // Sample RMS at 20Hz
  const now = Date.now();
  if (now - lastRmsPush >= 50) {
    const buf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
    rmsBuf[rmsHead] = Math.sqrt(sum / buf.length);
    rmsHead = (rmsHead + 1) % maxBars;
    if (rmsLen < maxBars) rmsLen++;
    lastRmsPush = now;
  }
  ctx.clearRect(0, 0, w, h);
  const offset = maxBars - rmsLen;
  for (let i = 0; i < rmsLen; i++) {
    const idx = (rmsHead - rmsLen + i + maxBars) % maxBars;
    const barH = Math.max(2, Math.min(h - 4, rmsBuf[idx] * h * 4));
    const x = (offset + i) * step;
    const y = (h - barH) / 2;
    ctx.fillStyle = '#f87171';
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, 1.5);
    ctx.fill();
  }
  const elapsed = (now - recStart) / 1000;
  const m = Math.floor(elapsed / 60), s = Math.floor(elapsed % 60);
  $('rectime').textContent = m + ':' + String(s).padStart(2, '0');
  waveRaf = requestAnimationFrame(drawWaveform);
}
async function toggleRecord() {
  if (recording) { stopRecord(); return; }
  if (!await initMic()) return;
  recPcm = [];
  rmsLen = 0; rmsHead = 0; lastRmsPush = 0;
  recording = true;
  recStart = Date.now();
  $('recbtn').classList.add('recording');
  $('waveform').classList.add('visible');
  $('rectime').classList.add('visible');
  $('transcript').textContent = '';
  $('meta').innerHTML = '';
  $('result').classList.remove('visible');
  waveRaf = requestAnimationFrame(drawWaveform);
}
function stopRecord() {
  recording = false;
  if (waveRaf) { cancelAnimationFrame(waveRaf); waveRaf = null; }
  $('recbtn').classList.remove('recording');
  $('waveform').classList.remove('visible');
  $('rectime').classList.remove('visible');
  if (recPcm.length) sendFile(pcmToWav(recPcm), 'recording.wav');
}
// --- Transcribe (file upload or final recording) ---
async function sendFile(file, name) {
  $('spinner').classList.add('visible');
  const fd = new FormData();
  fd.append('file', file, name || file.name);
  try {
    const r = await fetch('/transcribe', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) { $('transcript').textContent = 'Error: ' + (j.error || r.status); }
    else {
      $('transcript').textContent = j.text;
      const dur = j.audio_duration_s.toFixed(1);
      const inf = (j.inference_time_s * 1000).toFixed(0);
      const rtfx = (j.audio_duration_s / j.inference_time_s).toFixed(0);
      $('meta').innerHTML =
        `<span>${dur}s audio</span>` +
        `<span><span class="val">${inf}ms</span> inference</span>` +
        `<span><span class="val">${rtfx}x</span> RTFx</span>`;
    }
    $('result').classList.add('visible');
  } catch(e) {
    $('transcript').textContent = 'Connection error';
    $('result').classList.add('visible');
  }
  $('spinner').classList.remove('visible');
}
</script>
</body>
</html>)html", "text/html");
    });

    svr.Post("/transcribe", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'file' field\"}", "application/json");
            return;
        }
        const auto& file = req.get_file_value("file");
        auto wav = read_wav_from_memory(file.content.data(), file.content.size());
        if (wav.samples.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid WAV (need 16kHz mono, int16/float32)\"}", "application/json");
            return;
        }

        double audio_dur = (double)wav.samples.size() / wav.sample_rate;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text;
        {
            std::lock_guard<std::mutex> lock(mtx);
            text = pipeline.transcribe(wav.samples.data(), wav.samples.size());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::string corrected_text;
        double correct_ms = 0;
#ifdef WITH_CORRECTOR
        if (corrector) {
            corrected_text = corrector->correct(text);
            correct_ms = corrector->last_correct_ms;
        }
#endif

        // Stash detail for the logger callback (same thread)
        const std::string& display_text = corrected_text.empty() ? text : corrected_text;
        std::string preview = display_text.substr(0, 80);
        if (display_text.size() > 80) preview += "...";
        char detail[256];
        snprintf(detail, sizeof(detail), "audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                 audio_dur, elapsed * 1000, audio_dur / elapsed, preview.c_str());
        t_log_detail = detail;

        std::string body = "{\"text\":\"" + json_escape(text) + "\"";
        if (!corrected_text.empty())
            body += ",\"corrected_text\":\"" + json_escape(corrected_text) + "\"";
        body += ",\"audio_duration_s\":" + std::to_string(audio_dur) +
            ",\"inference_time_s\":" + std::to_string(elapsed) +
            ",\"mel_ms\":" + std::to_string(pipeline.last_mel_ms) +
            ",\"enc_ms\":" + std::to_string(pipeline.last_enc_ms) +
            ",\"dec_ms\":" + std::to_string(pipeline.last_dec_ms);
        if (correct_ms > 0)
            body += ",\"correct_ms\":" + std::to_string(correct_ms);
        body += "}";
        res.set_content(body, "application/json");
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}
