import AVFoundation
import Foundation

/// Captures the default microphone and writes 16 kHz mono Float32 samples to disk,
/// the format Parakeet expects. Reports the raw RMS of each buffer for the level meter.
final class AudioRecorder {
    static let sampleRate: Double = 16_000

    /// Linear RMS amplitude relative to full scale, per captured buffer. Called off the main thread.
    var onRMS: ((Float) -> Void)?
    /// Total samples safely written, allowing transcription to catch up from disk.
    var onProgress: ((Int) -> Void)?

    private let engine = AVAudioEngine()
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false
    )!
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var recording: RecordingFile?
    private var recordingError: Error?

    static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    func start() throws -> RecordingFile {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw RecorderError.noInputDevice
        }
        let recording = try RecordingFile()
        lock.withLock {
            self.converter = converter
            self.recording = recording
            recordingError = nil
        }

        engine.prepare()
        do { try engine.start() }
        catch {
            lock.withLock { self.converter = nil; self.recording = nil }
            throw error
        }
        // Installed only after the engine is running, so a failed start never leaves a tap
        // behind (a second tap on the same bus is a fatal AVFoundation exception).
        input.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
            self?.append(buffer)
        }
        Log.info("Recording started: \(Int(inputFormat.sampleRate)) Hz, \(inputFormat.channelCount) ch")
        return recording
    }

    /// Stops capture, finalizes the WAV header and transfers ownership of the file.
    func stop() throws -> RecordingFile {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        return try lock.withLock {
            defer { recording = nil; recordingError = nil }
            converter = nil
            if let recordingError { throw recordingError }
            guard let recording else { throw RecordingFile.RecordingError.closed }
            try recording.finish()
            return recording
        }
    }

    private func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard let converter, let recording, recordingError == nil else { return }
        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        if let error {
            Log.info("Audio conversion failed: \(error)")
            recordingError = error
            return
        }
        guard let channel = output.floatChannelData?[0] else { return }
        let converted = Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
        do { onProgress?(try recording.append(converted)) }
        catch { recordingError = error }
        onRMS?(Self.rms(of: converted))
    }

    private static func rms(of samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        return (samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count)).squareRoot()
    }

    enum RecorderError: LocalizedError {
        case noInputDevice

        var errorDescription: String? {
            switch self {
            case .noInputDevice: return "No microphone input device is available."
            }
        }
    }
}

/// Maps raw RMS to a display level on a logarithmic scale (-65 to -25 dBFS) with a fast
/// attack and slow release, so quiet speech is visible and loud speech keeps its dynamics.
struct LevelSmoother {
    private static let floorDecibels: Float = -65
    private static let ceilingDecibels: Float = -25
    private static let attackSeconds: Float = 0.03
    private static let releaseSeconds: Float = 0.16

    private(set) var level: Float = 0
    private var pendingPeak: Float = 0

    mutating func update(rms: Float) {
        let decibels = 20 * log10(max(rms, 1e-6))
        // Retain brief syllables between the display's 50 ms ticks.
        pendingPeak = max(pendingPeak, min(max((decibels - Self.floorDecibels) / (Self.ceilingDecibels - Self.floorDecibels), 0), 1))
    }

    /// Advances the smoothed level by `interval` seconds toward the latest target.
    mutating func tick(interval: Float) {
        let target = pendingPeak
        pendingPeak = 0
        let tau = target > level ? Self.attackSeconds : Self.releaseSeconds
        let coefficient = 1 - exp(-interval / tau)
        level += (target - level) * coefficient
    }

    mutating func reset() {
        level = 0
        pendingPeak = 0
    }
}
