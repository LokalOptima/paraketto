import AppKit
import Combine
import Foundation

/// Owns the record → transcribe → paste loop and exposes state for the menu bar, overlay and history.
@MainActor
final class DictationController: ObservableObject {
    enum State: Equatable {
        case downloadingModel(Double)
        case loadingModel
        case idle
        case recording
        case transcribing
        case notice(String)
        case failed(String)

        var symbolName: String {
            switch self {
            case .downloadingModel: return "arrow.down.circle"
            case .loadingModel: return "hourglass"
            case .idle, .notice: return "mic"
            case .recording: return "mic.fill"
            case .transcribing: return "waveform"
            case .failed: return "exclamationmark.triangle"
            }
        }

        var label: String {
            switch self {
            case .downloadingModel(let fraction): return "Downloading model… \(Int(fraction * 100))%"
            case .loadingModel: return "Loading model…"
            case .idle: return "Ready"
            case .recording: return "Recording…"
            case .transcribing: return "Transcribing…"
            case .notice(let message): return message
            case .failed(let message): return "Error: \(message)"
            }
        }

        var canStartRecording: Bool {
            switch self {
            case .idle, .notice, .failed: return true
            default: return false
            }
        }

        var isCancellable: Bool {
            self == .recording || self == .transcribing
        }
    }

    @Published private(set) var state: State = .loadingModel {
        didSet { if state != oldValue { Log.info("state: \(state.label)") } }
    }
    @Published private(set) var inputLevel: Float = 0
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var destination: Destination?

    let history = HistoryStore()
    let shortcuts = ShortcutSettings()

    private static let noticeDuration: Duration = .seconds(2)
    private static let meterInterval: Duration = .milliseconds(50)

    private let transcriber = Transcriber()
    private let recorder = AudioRecorder()
    private var hotkey: Hotkey?
    private var modifierTap: ModifierTap?
    private var shortcutSubscription: AnyCancellable?
    private var escapeMonitor: Any?
    private var overlay: Overlay?
    private var smoother = LevelSmoother()
    private var meterTask: Task<Void, Never>?
    private var recordingStarted = Date()
    private var transcription: Task<Void, Never>?

    var lastTranscript: String? { history.entries.first?.text }

    func start() {
        Log.startSession()
        Log.info("launch, log at \(Log.fileURL.path)")
        overlay = Overlay(controller: self)
        recorder.onRMS = { [weak self] rms in
            Task { @MainActor in self?.smoother.update(rms: rms) }
        }
        applyShortcuts()
        shortcutSubscription = shortcuts.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.applyShortcuts() }
        if !Paster.isAccessibilityTrusted {
            Paster.requestAccessibility()
        }
        Task {
            await loadModel()
            if CommandLine.arguments.contains("--demo-pill") { await demoPill() }
        }
    }

    /// (Re)arms the shortcut from settings. Nothing is armed while the Settings recorder is
    /// capturing a key, so the current shortcut can be pressed to replace itself.
    private func applyShortcuts() {
        hotkey = nil
        modifierTap = nil
        guard !shortcuts.isRecording else { return }
        switch shortcuts.shortcut {
        case .chord(let chord):
            let hotkey = Hotkey(binding: Hotkey.Binding(chord), onPress: { [weak self] in self?.toggleRecording() })
            self.hotkey = hotkey
            if let error = hotkey.registrationError, state == .idle {
                show(.failed("\(chord.label) is taken by another app (\(error))"))
            }
        case .tap(let key):
            modifierTap = ModifierTap(key) { [weak self] in self?.toggleRecording() }
        }
    }

    /// `--demo-pill`: drive the overlay through recording and transcribing with synthetic
    /// levels, without touching the microphone. Used to check the pill without a hotkey press.
    private func demoPill() async {
        destination = Destination(name: "Demo", bundleIdentifier: nil)
        startSession()
        state = .recording
        overlay?.show()
        for step in 0..<60 {
            try? await Task.sleep(for: .milliseconds(50))
            smoother.update(rms: step % 20 < 10 ? 0.05 : 0.002)
        }
        state = .transcribing
        try? await Task.sleep(for: .seconds(1))
        finishSession()
        show(.notice("Demo finished"))
    }

    func copyLastTranscript() {
        guard let text = lastTranscript else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Escape while recording or transcribing: discard the audio, paste nothing.
    func cancel() {
        guard state.isCancellable else { return }
        Log.info("cancelled")
        transcription?.cancel()
        transcription = nil
        if state == .recording { _ = recorder.stop() }
        finishSession()
        state = .idle
        overlay?.hide()
    }

    private func loadModel() async {
        do {
            let started = Date()
            try await transcriber.load { phase in
                Task { @MainActor [weak self] in
                    switch phase {
                    case .downloading(let fraction): self?.state = .downloadingModel(fraction)
                    case .loading: self?.state = .loadingModel
                    }
                }
            }
            state = .idle
            Log.info("Parakeet Ultra ready in \(Int(Date().timeIntervalSince(started) * 1000)) ms")
        } catch {
            state = .failed(error.localizedDescription)
            Log.info("Model load failed: \(error)")
        }
    }

    /// One press starts a recording, the next press stops it and transcribes.
    private func toggleRecording() {
        Log.info("shortcut")
        if state == .recording {
            stopRecording()
        } else if state.canStartRecording {
            beginRecording()
        }
    }

    private func beginRecording() {
        Task {
            guard await AudioRecorder.requestMicrophoneAccess() else {
                show(.failed("Microphone access denied"))
                return
            }
            guard state.canStartRecording else { return }
            do {
                destination = Destination.frontmost
                try recorder.start()
                startSession()
                state = .recording
                overlay?.show()
            } catch {
                Log.info("Recorder start failed: \(error)")
                show(.failed(error.localizedDescription))
            }
        }
    }

    private func stopRecording() {
        guard state == .recording else { return }
        let samples = recorder.stop()
        guard samples.count >= Transcriber.minimumSamples else {
            finishSession()
            state = .idle
            overlay?.hide()
            return
        }
        state = .transcribing
        transcription = Task {
            let started = Date()
            do {
                let text = try await transcriber.transcribe(samples)
                guard !Task.isCancelled else { return }
                let inferenceMilliseconds = Int(Date().timeIntervalSince(started) * 1000)
                let duration = Double(samples.count) / AudioRecorder.sampleRate
                Log.info("transcribed \(String(format: "%.1f", duration))s audio in \(inferenceMilliseconds) ms, \(text.count) characters")
                finishSession()
                guard !text.isEmpty else {
                    state = .idle
                    overlay?.hide()
                    return
                }
                let outcome = await Paster.deliver(text, to: destination)
                Log.info("delivery: \(outcome)")
                history.add(
                    HistoryEntry(
                        id: UUID(), date: recordingStarted, text: text,
                        destination: destination?.name ?? "", duration: duration,
                        inferenceMilliseconds: inferenceMilliseconds, delivery: delivery(for: outcome)
                    ),
                    samples: samples
                )
                switch outcome {
                case .pasted:
                    state = .idle
                    overlay?.hide()
                case .copiedOnly:
                    show(.notice("Copied"))
                case .destinationChanged:
                    show(.notice("Kept in history"))
                }
            } catch is CancellationError {
                return
            } catch {
                Log.info("Transcription failed: \(error)")
                finishSession()
                show(.failed(error.localizedDescription))
            }
        }
    }

    private func delivery(for outcome: Paster.Outcome) -> HistoryEntry.Delivery {
        switch outcome {
        case .pasted: return .pasted
        case .copiedOnly: return .copied
        case .destinationChanged: return .keptInHistory
        }
    }

    /// Watches for Escape and starts the meter/elapsed ticker for the duration of a dictation.
    private func startSession() {
        recordingStarted = Date()
        elapsed = 0
        smoother.reset()
        inputLevel = 0
        // A passive monitor rather than a Carbon hotkey: registering a hotkey while ⌃⌥Space is
        // physically held makes Carbon drop that key's release event, which left recordings stuck.
        escapeMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard event.keyCode == 53 else { return }
            Task { @MainActor in self?.cancel() }
        }
        meterTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: Self.meterInterval)
                guard let self, !Task.isCancelled else { return }
                if self.state == .recording {
                    self.smoother.tick(interval: 0.05)
                    self.inputLevel = self.smoother.level
                    self.elapsed = Date().timeIntervalSince(self.recordingStarted)
                }
            }
        }
    }

    private func finishSession() {
        if let escapeMonitor { NSEvent.removeMonitor(escapeMonitor) }
        escapeMonitor = nil
        meterTask?.cancel()
        meterTask = nil
    }

    /// Shows a transient message in the pill, then returns to idle so the hotkey keeps working.
    private func show(_ transient: State) {
        state = transient
        overlay?.show()
        overlay?.hide(after: Self.noticeDuration)
        Task { [weak self] in
            try? await Task.sleep(for: Self.noticeDuration)
            guard let self, self.state == transient else { return }
            self.state = .idle
        }
    }
}
