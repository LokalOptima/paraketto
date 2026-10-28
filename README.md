# Paraketto

Minimal push-to-talk dictation for macOS. Hold a hotkey, speak, release: the text is
pasted into whatever app has focus. Transcription runs on the Apple Neural Engine using
[Parakeet Ultra](https://huggingface.co/FluidInference/parakeet-ultra-coreml) through
[FluidAudio](https://github.com/FluidInference/FluidAudio). Nothing leaves the Mac.

- Menu bar only, no Dock icon
- Hotkey: hold `⌃⌥Space` (change `Hotkey.Binding.default` in `Sources/Paraketto/Hotkey.swift`)
- 25 European languages with automatic detection, punctuation and capitalization included
- Model download on first launch (613 MB to `~/Library/Application Support/FluidAudio/Models/parakeet-ultra`)

## Build and run

Requires macOS 14+, Apple Silicon, and Xcode 16 or later.

```bash
make run
```

This builds with SwiftPM, assembles `Paraketto.app`, ad-hoc signs it, and launches it.

## Permissions

- **Microphone**: prompted on the first recording.
- **Accessibility**: needed to send the Cmd+V keystroke. Prompted at launch, granted in
  System Settings → Privacy & Security → Accessibility. Without it the transcript is only
  copied to the clipboard.

### Accessibility and rebuilds

macOS ties the Accessibility grant to the app's code signature. With ad-hoc signing every
rebuild is a different signature, so the toggle in System Settings looks on but the new build
is not trusted and text is only copied. Two ways out:

- **Per build:** `make trust` removes the stale entry and opens the Accessibility pane. Grant
  the app again.
- **Once:** create a self-signed code-signing certificate named `Paraketto Dev`. The Makefile
  picks it up automatically and the grant survives rebuilds.

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout key.pem -out cert.pem \
    -subj "/CN=Paraketto Dev" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=critical,codeSigning" \
    -addext "basicConstraints=critical,CA:false"
  openssl pkcs12 -export -legacy -out dev.p12 -inkey key.pem -in cert.pem -passout pass:x -name "Paraketto Dev"
  security import dev.p12 -k ~/Library/Keychains/login.keychain-db -P x -T /usr/bin/codesign
  rm key.pem dev.p12
  make trust   # one last re-grant for the newly signed build
  ```

  The first `codesign` with the new key may show a keychain dialog; choose "Always Allow".

## Layout

| File | Role |
|---|---|
| `App.swift` | `MenuBarExtra` shell |
| `DictationController.swift` | record → transcribe → paste state machine |
| `Hotkey.swift` | Carbon global hotkey with press and release |
| `AudioRecorder.swift` | `AVAudioEngine` mic capture to 16 kHz mono Float32 |
| `Transcriber.swift` | FluidAudio Parakeet Ultra wrapper |
| `Paster.swift` | pasteboard write, synthetic Cmd+V, pasteboard restore |
