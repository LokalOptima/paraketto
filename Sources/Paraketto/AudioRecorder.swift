import AVFoundation
import Foundation

/// Captures the default microphone and accumulates 16 kHz mono Float32 samples,
/// the format Parakeet expects. Reports a smoothed input level for the overlay.
final class AudioRecorder {
    static let sampleRate: Double = 16_000

    var onLevel: ((Float) -> Void)?

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
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
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
        onLevel?(Self.level(of: converted))
    }

    /// Root mean square mapped to 0...1 on a rough decibel scale so quiet speech still moves the meter.
    private static func level(of samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let meanSquare = samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count)
        let decibels = 10 * log10(max(meanSquare, 1e-10))
        return min(max((decibels + 50) / 50, 0), 1)
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
