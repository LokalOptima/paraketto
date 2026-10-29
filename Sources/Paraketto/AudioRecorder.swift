import AVFoundation
import Foundation

/// Captures the default microphone and accumulates 16 kHz mono Float32 samples,
/// the format Parakeet expects. Reports the raw RMS of each buffer for the level meter.
final class AudioRecorder {
    static let sampleRate: Double = 16_000

    /// Linear RMS amplitude relative to full scale, per captured buffer. Called off the main thread.
    var onRMS: ((Float) -> Void)?
    /// Each converted 16 kHz chunk as it arrives, in order. Called off the main thread.
    var onSamples: (([Float]) -> Void)?

    private let engine = AVAudioEngine()
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false
    )!
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var samples: [Float] = []

    static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    func start() throws {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw RecorderError.noInputDevice
        }
        lock.withLock {
            self.converter = converter
            samples.removeAll(keepingCapacity: true)
        }

        engine.prepare()
        try engine.start()
        // Installed only after the engine is running, so a failed start never leaves a tap
        // behind (a second tap on the same bus is a fatal AVFoundation exception).
        input.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
            self?.append(buffer)
        }
        Log.info("Recording started: \(Int(inputFormat.sampleRate)) Hz, \(inputFormat.channelCount) ch")
    }

    /// Stops capture and returns everything recorded since `start()`.
    func stop() -> [Float] {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        return lock.withLock {
            converter = nil
            return samples
        }
    }

    private func append(_ buffer: AVAudioPCMBuffer) {
        guard let converter = lock.withLock({ converter }) else { return }
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
            return
        }
        guard let channel = output.floatChannelData?[0] else { return }
        let converted = Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
        lock.withLock { samples.append(contentsOf: converted) }
        onSamples?(converted)
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

/// Maps raw RMS to a display level on a logarithmic scale (-50 to -10 dBFS) with a fast
/// attack and slow release, so quiet speech is visible and loud speech keeps its dynamics.
struct LevelSmoother {
    private static let floorDecibels: Float = -50
    private static let ceilingDecibels: Float = -10
    private static let attackSeconds: Float = 0.03
    private static let releaseSeconds: Float = 0.16

    private(set) var level: Float = 0
    private var target: Float = 0

    mutating func update(rms: Float) {
        let decibels = 20 * log10(max(rms, 1e-6))
        target = min(max((decibels - Self.floorDecibels) / (Self.ceilingDecibels - Self.floorDecibels), 0), 1)
    }

    /// Advances the smoothed level by `interval` seconds toward the latest target.
    mutating func tick(interval: Float) {
        let tau = target > level ? Self.attackSeconds : Self.releaseSeconds
        let coefficient = 1 - exp(-interval / tau)
        level += (target - level) * coefficient
    }

    mutating func reset() {
        level = 0
        target = 0
    }
}
