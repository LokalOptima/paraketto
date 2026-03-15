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

#ifdef WITH_CORRECTOR
#include "corrector.h"
#endif

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
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

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>paraketto</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0;
         display: flex; justify-content: center; padding: 2rem; min-height: 100vh; }
  .container { width: 100%; max-width: 640px; }
  h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.3rem; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #666; margin-bottom: 1.5rem; }
  .drop-zone { border: 2px dashed #333; border-radius: 12px; padding: 2.5rem 1.5rem;
               text-align: center; cursor: pointer; transition: all 0.2s; }
  .drop-zone:hover, .drop-zone.dragover { border-color: #2563eb; background: #0d1117; }
  .drop-zone p { color: #888; font-size: 0.9rem; }
  .drop-zone .icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .controls { display: flex; gap: 10px; margin-top: 16px; justify-content: center; }
  button { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; border-radius: 8px;
           padding: 10px 20px; font-size: 0.9rem; font-weight: 500; cursor: pointer;
           transition: all 0.15s; display: flex; align-items: center; gap: 6px; }
  button:hover { background: #222; border-color: #555; }
  button.recording { background: #7f1d1d; border-color: #dc2626; color: #fca5a5; }
  button:disabled { opacity: 0.4; cursor: default; }
  .result { margin-top: 20px; display: none; }
  .result.visible { display: block; }
  .transcript { background: #111; border: 1px solid #222; border-radius: 10px;
                padding: 16px 20px; font-size: 1.05rem; line-height: 1.6;
                color: #f0f0f0; min-height: 60px; white-space: pre-wrap; }
  .meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
  .meta span { font-size: 0.8rem; color: #555; font-variant-numeric: tabular-nums; }
  .meta .val { color: #888; }
  .spinner { display: none; margin: 20px auto; width: 24px; height: 24px;
             border: 2px solid #333; border-top-color: #2563eb; border-radius: 50%;
             animation: spin 0.6s linear infinite; }
  .spinner.visible { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  input[type="file"] { display: none; }
  .wave { height: 40px; margin-top: 12px; display: none; }
  .wave.visible { display: block; }
  .wave canvas { width: 100%; height: 100%; border-radius: 6px; }
</style>
</head>
<body>
<div class="container">
  <h1>paraketto</h1>
  <p class="subtitle">speech-to-text</p>

  <div class="drop-zone" id="dropzone">
    <div class="icon">&#127908;</div>
    <p>Drop a WAV file here, or click to upload</p>
  </div>
  <input type="file" id="fileinput" accept="audio/*">

  <div class="controls">
    <button id="recbtn" onclick="toggleRecord()">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="8" cy="5" r="3.5"/>
        <path d="M3 5a5 5 0 0010 0M5 10v1a3 3 0 006 0v-1M8 13v2M6 15h4" fill="none"
              stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      Record
    </button>
  </div>

  <div class="wave" id="wave"><canvas id="wavecanvas"></canvas></div>
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

// --- Microphone recording ---
let mediaRec = null, audioChunks = [];
async function toggleRecord() {
  if (mediaRec && mediaRec.state === 'recording') {
    mediaRec.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    audioChunks = [];
    mediaRec = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                                            ? 'audio/webm;codecs=opus' : 'audio/webm' });
    mediaRec.ondataavailable = e => audioChunks.push(e.data);
    mediaRec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      $('recbtn').classList.remove('recording');
      $('recbtn').innerHTML = micSvg + ' Record';
      stopWave();
      const blob = new Blob(audioChunks, { type: mediaRec.mimeType });
      sendFile(blob, 'recording.webm');
    };
    mediaRec.start();
    $('recbtn').classList.add('recording');
    $('recbtn').innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="3" width="10" height="10" rx="2"/></svg> Stop';
    startWave(stream);
  } catch(e) {
    alert('Microphone access denied');
  }
}
const micSvg = $('recbtn').innerHTML.split('Record')[0];

// --- Waveform visualizer ---
let waveAnim = null, analyser = null, waveCtx = null;
function startWave(stream) {
  const ac = new AudioContext();
  analyser = ac.createAnalyser();
  analyser.fftSize = 256;
  ac.createMediaStreamSource(stream).connect(analyser);
  const canvas = $('wavecanvas');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 80;
  waveCtx = canvas.getContext('2d');
  $('wave').classList.add('visible');
  drawWave();
}
function drawWave() {
  if (!analyser) return;
  waveAnim = requestAnimationFrame(drawWave);
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  const ctx = waveCtx, w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const step = w / data.length;
  for (let i = 0; i < data.length; i++) {
    const y = (data[i] / 255) * h;
    i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
  }
  ctx.stroke();
}
function stopWave() {
  cancelAnimationFrame(waveAnim);
  $('wave').classList.remove('visible');
  analyser = null;
}

// --- Transcribe ---
async function sendFile(file, name) {
  $('spinner').classList.add('visible');
  $('result').classList.remove('visible');
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
      const mel = j.mel_ms.toFixed(1), enc = j.enc_ms.toFixed(1), dec = j.dec_ms.toFixed(1);
      $('meta').innerHTML =
        `<span>${dur}s audio</span>` +
        `<span>inference <span class="val">${inf}ms</span></span>` +
        `<span>RTFx <span class="val">${rtfx}x</span></span>` +
        `<span>mel <span class="val">${mel}ms</span>  enc <span class="val">${enc}ms</span>  dec <span class="val">${dec}ms</span></span>`;
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
