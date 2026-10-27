#pragma once

namespace paraketto {
inline constexpr const char* web_ui = R"html(<!DOCTYPE html>
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
  .modelinfo { margin-top: 12px; font-size: 0.75rem; line-height: 1.6; }
  .modelinfo .badge { display: inline-block; color: #d4d4d8; background: #18181b;
                      border: 1px solid #27272a; border-radius: 6px; padding: 3px 9px;
                      font-weight: 600; font-variant-numeric: tabular-nums; }
  .modelinfo .sub { color: #3f3f46; margin-top: 5px; font-variant-numeric: tabular-nums; }
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
  .results { display: flex; flex-direction: column; gap: 24px; margin-top: 24px; }
  .result-header { display: flex; align-items: baseline; justify-content: space-between;
                   gap: 12px; margin: 0 4px 8px; font-size: 0.75rem; color: #71717a; }
  .recording-name { overflow-wrap: anywhere; min-width: 0; }
  .recorded-at { flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .transcript-wrap { position: relative; }
  .transcript { background: #0f0f11; border: 1px solid #1c1c1f; border-radius: 12px;
                padding: 20px 20px 56px; font-size: 1.05rem; line-height: 1.7;
                color: #f4f4f5; min-height: 100px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .result[aria-busy="true"] .transcript { color: #71717a; }
  .result.error .transcript { color: #f87171; }
  .result-actions { position: absolute; bottom: 10px; right: 10px; left: 10px;
                    display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
  .result-btn { width: 32px; height: 32px; flex-shrink: 0; border-radius: 8px;
                border: 1px solid #27272a; background: #18181b; color: #a1a1aa;
                cursor: pointer; display: flex; align-items: center; justify-content: center;
                padding: 0; transition: background 0.2s, color 0.2s; }
  .result-btn:hover:not(:disabled) { border-color: #3f3f46; background: #1c1c1f; color: #d4d4d8; }
  .result-btn:focus-visible { outline: 2px solid #a1a1aa; outline-offset: 3px; }
  .result-btn:disabled { opacity: 0.35; cursor: default; }
  .result-btn svg { width: 15px; height: 15px; }
  .playbtn { border-radius: 50%; }
  .playbtn svg { fill: currentColor; }
  .playbtn .pause-icon { display: none; }
  .playbtn.playing .play-icon { display: none; }
  .playbtn.playing .pause-icon { display: block; }
  .copy-status { color: #a1a1aa; font-size: 0.72rem; }
  .meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
  .meta span { font-size: 0.75rem; color: #3f3f46; font-variant-numeric: tabular-nums; }
  .meta .val { color: #71717a; }
  input[type="file"] { display: none; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>parakettő</h1>
    <p>speech-to-text</p>
    <div id="modelinfo" class="modelinfo"></div>
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

  <section class="results" id="results" aria-label="Recordings"></section>
  <template id="result-template">
    <article class="result" aria-busy="true">
      <div class="result-header">
        <span class="recording-name"></span>
        <time class="recorded-at"></time>
      </div>
      <div class="transcript-wrap">
        <div class="transcript" aria-live="polite">Transcribing…</div>
        <div class="result-actions">
          <span class="copy-status" role="status"></span>
          <button class="playbtn result-btn" type="button" title="Play recording" aria-label="Play recording">
            <svg class="play-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
            <svg class="pause-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
          </button>
          <button class="copybtn result-btn" type="button" title="Copy transcript" aria-label="Copy transcript" disabled>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>
          </button>
        </div>
      </div>
      <div class="meta"></div>
    </article>
  </template>
</div>
<script>
const $ = id => document.getElementById(id);

// --- Recording history: each request owns its transcript and playback ---
let activePlayer = null, recordingCount = 0;
function createResult(file, label) {
  const element = $('result-template').content.firstElementChild.cloneNode(true);
  const entry = {
    element, text: '',
    transcript: element.querySelector('.transcript'),
    meta: element.querySelector('.meta'),
    copyButton: element.querySelector('.copybtn'),
  };
  element.querySelector('.recording-name').textContent = label;
  const time = element.querySelector('.recorded-at'), now = new Date();
  time.dateTime = now.toISOString();
  time.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  time.title = now.toLocaleString();

  // Retain this URL for the lifetime of the card; later recordings must not revoke it.
  const player = new Audio(URL.createObjectURL(file));
  const playButton = element.querySelector('.playbtn');
  const feedback = element.querySelector('.copy-status');
  let feedbackTimer;
  function showFeedback(message) {
    clearTimeout(feedbackTimer);
    feedback.textContent = message;
    feedbackTimer = setTimeout(() => { feedback.textContent = ''; }, 2500);
  }
  function updatePlayback(playing) {
    playButton.classList.toggle('playing', playing);
    playButton.title = playing ? 'Pause recording' : 'Play recording';
    playButton.setAttribute('aria-label', playButton.title);
    if (!playing && activePlayer === player) activePlayer = null;
  }
  player.onplay = () => { activePlayer = player; updatePlayback(true); };
  player.onpause = player.onended = () => updatePlayback(false);
  playButton.addEventListener('click', async () => {
    if (!player.paused) { player.pause(); return; }
    if (activePlayer && activePlayer !== player) activePlayer.pause();
    activePlayer = player;
    player.currentTime = 0;
    try { await player.play(); }
    catch (e) {
      updatePlayback(false);
      if (e.name !== 'AbortError') showFeedback('Playback unavailable');
    }
  });
  entry.copyButton.addEventListener('click', async () => {
    try { await copyText(entry.text); showFeedback('Copied'); }
    catch { showFeedback('Copy failed'); }
  });
  $('results').prepend(element);
  return entry;
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; }
    catch { /* Try selection-based copying if clipboard access is denied. */ }
  }
  // Also support HTTP connections where the Clipboard API is unavailable.
  const field = document.createElement('textarea');
  field.value = text;
  field.readOnly = true;
  field.style.cssText = 'position:fixed;left:-9999px;top:0';
  const previousFocus = document.activeElement;
  document.body.append(field);
  try {
    field.select();
    if (!document.execCommand('copy')) throw new Error('Copy failed');
  } finally {
    field.remove();
    previousFocus?.focus({ preventScroll: true });
  }
}

// --- File upload / drop ---
$('dropzone').addEventListener('click', () => $('fileinput').click());
$('fileinput').addEventListener('change', e => {
  const file = e.target.files[0];
  e.target.value = '';  // Allow uploading the same file again as another entry.
  if (file) sendFile(file);
});
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
const REC_SKIP_S = 0.15;  // drop the first 150ms of capture to skip the mouse-click transient
let skipSamples = 0;
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
    node.port.onmessage = e => {
      if (!recording) return;
      if (skipSamples > 0) {                    // drop the mouse-click lead-in
        if (e.data.length <= skipSamples) { skipSamples -= e.data.length; return; }
        recPcm.push(e.data.subarray(skipSamples));
        skipSamples = 0;
        return;
      }
      recPcm.push(e.data);
    };
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
  skipSamples = Math.round(micCtx.sampleRate * REC_SKIP_S);
  rmsLen = 0; rmsHead = 0; lastRmsPush = 0;
  recording = true;
  recStart = Date.now();
  $('recbtn').classList.add('recording');
  $('waveform').classList.add('visible');
  $('rectime').classList.add('visible');
  if (activePlayer) activePlayer.pause();
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
  const entry = createResult(file, name ? `Recording ${++recordingCount}` : (file.name || 'Audio'));
  const fd = new FormData();
  fd.append('file', file, name || file.name || 'recording.wav');
  try {
    const r = await fetch('/transcribe', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) {
      entry.transcript.textContent = 'Error: ' + (j.error || r.status);
      entry.element.classList.add('error');
    }
    else {
      entry.text = j.text || '';
      entry.transcript.textContent = entry.text || 'No speech detected.';
      entry.copyButton.disabled = !entry.text;
      const dur = j.audio_duration_s.toFixed(1);
      const inf = (j.inference_time_s * 1000).toFixed(0);
      const rtfx = (j.audio_duration_s / j.inference_time_s).toFixed(0);
      entry.meta.innerHTML =
        `<span>${dur}s audio</span>` +
        `<span><span class="val">${inf}ms</span> inference</span>` +
        `<span><span class="val">${rtfx}x</span> RTFx</span>`;
    }
  } catch(e) {
    entry.transcript.textContent = 'Connection error';
    entry.element.classList.add('error');
  } finally {
    entry.element.setAttribute('aria-busy', 'false');
    refreshModelInfo();
  }
}

// --- Show which model/backend is running (GET /info) ---
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function refreshModelInfo() { fetch('/info').then(r => r.json()).then(j => {
  const wname = (j.weights || '').split('/').pop();
  const details = [wname];
  if (j.weights_mb != null) details.push(`${j.weights_mb} MB`);
  if (j.device) details.push(j.device);
  if (j.startup_ms != null) details.push(`${j.startup_ms} ms startup`);
  $('modelinfo').innerHTML =
    `<span class="badge">${esc(j.model)} · ${esc(j.backend)}</span>` +
    `<div class="sub">${details.map(esc).join(' · ')}</div>`;
}).catch(() => {}); }
refreshModelInfo();
</script>
</body>
</html>)html";
} // namespace paraketto
