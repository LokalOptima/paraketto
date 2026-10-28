import AppKit
import Foundation

/// Owns the record → transcribe → paste loop and exposes state for the menu bar and overlay.
@MainActor
final class DictationController: ObservableObject {
    enum State: Equatable {
        case downloadingModel(Double)
        case loadingModel
        case idle
        case recording
        case transcribing
        case failed(String)

        var symbolName: String {
            switch self {
            case .downloadingModel: return "arrow.down.circle"
            case .loadingModel: return "hourglass"
            case .idle: return "mic"
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
            case .failed(let message): return "Error: \(message)"
            }
        }

        var canStartRecording: Bool {
            switch self {
            case .idle, .failed: return true
            default: return false
            }
        }
    }

    @Published private(set) var state: State = .loadingModel {
        didSet { if state != oldValue { Log.info("state: \(state.label)") } }
    }
    @Published private(set) var inputLevel: Float = 0
    @Published private(set) var lastTranscript = ""
    @Published private(set) var lastTiming = ""

    let hotkeyLabel = Hotkey.Binding.default.label

    private static let errorDisplayDuration: Duration = .seconds(2)

    private let transcriber = Transcriber()
    private let recorder = AudioRecorder()
    private var hotkey: Hotkey?
    private var overlay: Overlay?
    private var isKeyHeld = false

    func start() {
        Log.startSession()
        Log.info("launch, log at \(Log.fileURL.path)")
        overlay = Overlay(controller: self)
        recorder.onLevel = { [weak self] level in
            Task { @MainActor in self?.inputLevel = level }
        }
        let hotkey = Hotkey(
            onPress: { [weak self] in self?.beginRecording() },
            onRelease: { [weak self] in self?.endRecording() }
        )
        self.hotkey = hotkey
        if !Paster.isAccessibilityTrusted {
            Paster.requestAccessibility()
        }
        Task {
            await loadModel()
            if let error = hotkey.registrationError {
                state = .failed("Hotkey \(hotkeyLabel) unavailable (\(error))")
            }
        }
    }

    func copyLastTranscript() {
        guard !lastTranscript.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(lastTranscript, forType: .string)
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

    private func beginRecording() {
        Log.info("hotkey pressed")
        isKeyHeld = true
        guard state.canStartRecording else { return }
        Task {
            guard await AudioRecorder.requestMicrophoneAccess() else {
                fail("Microphone access denied")
                return
            }
            // The permission prompt can outlast the key press; do not start a recording nobody is holding.
            guard state.canStartRecording, isKeyHeld else { return }
            do {
                try recorder.start()
                state = .recording
                overlay?.show()
            } catch {
                Log.info("Recorder start failed: \(error)")
                fail(error.localizedDescription)
            }
        }
    }

    private func endRecording() {
        Log.info("hotkey released")
        isKeyHeld = false
        guard state == .recording else { return }
        let samples = recorder.stop()
        guard samples.count >= Transcriber.minimumSamples else {
            state = .idle
            overlay?.hide()
            return
        }
        state = .transcribing
        Task {
            let started = Date()
            do {
                let text = try await transcriber.transcribe(samples)
                let elapsed = Date().timeIntervalSince(started)
                let audioSeconds = Double(samples.count) / AudioRecorder.sampleRate
                lastTiming = String(format: "%.1fs audio in %.0f ms", audioSeconds, elapsed * 1000)
                Log.info("transcribed \(lastTiming), \(text.count) characters")
                if !text.isEmpty {
                    lastTranscript = text
                    let outcome = await Paster.deliver(text)
                    Log.info("delivery: \(outcome)")
                }
                state = .idle
                overlay?.hide()
            } catch {
                Log.info("Transcription failed: \(error)")
                fail(error.localizedDescription)
            }
        }
    }

    /// Shows the error in the pill for a moment, then returns to idle so the hotkey keeps working.
    private func fail(_ message: String) {
        state = .failed(message)
        overlay?.show()
        overlay?.hide(after: Self.errorDisplayDuration)
        Task { [weak self] in
            try? await Task.sleep(for: Self.errorDisplayDuration)
            guard let self, case .failed = self.state else { return }
            self.state = .idle
        }
    }
}
