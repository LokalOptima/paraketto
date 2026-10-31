import Foundation

/// A disk-backed recording shared by capture and transcription. Float32 WAV preserves
/// the exact samples for recognition and can be moved straight into History for replay.
/// The temporary file is removed when its last owner releases it (including cancellation).
final class RecordingFile: @unchecked Sendable {
    let url: URL
    private let lock = NSLock()
    private var handle: FileHandle?
    private var count = 0
    private var finished = false

    var sampleCount: Int { lock.withLock { count } }
    var duration: TimeInterval { Double(sampleCount) / AudioRecorder.sampleRate }

    init(directory: URL = FileManager.default.temporaryDirectory) throws {
        url = directory.appendingPathComponent("Paraketto-\(UUID().uuidString).wav")
        try Self.header(sampleCount: 0).write(to: url, options: .withoutOverwriting)
        do { handle = try FileHandle(forUpdating: url) }
        catch { try? FileManager.default.removeItem(at: url); throw error }
    }

    deinit {
        try? handle?.close()
        try? FileManager.default.removeItem(at: url)
    }

    @discardableResult
    func append(_ samples: [Float]) throws -> Int {
        try lock.withLock {
            guard !finished, let handle else { throw RecordingError.closed }
            guard count + samples.count <= (Int(UInt32.max) - 36) / 4 else {
                throw RecordingError.tooLarge
            }
            try handle.seekToEnd()
            try samples.withUnsafeBufferPointer { try handle.write(contentsOf: Data(buffer: $0)) }
            count += samples.count
            return count
        }
    }

    /// Read only published samples, in small batches; never materialize the full recording.
    func readSamples(from offset: Int, count requested: Int) throws -> [Float] {
        try lock.withLock {
            guard let handle else { throw RecordingError.closed }
            guard offset >= 0, requested >= 0, offset <= count, requested <= count - offset else {
                throw RecordingError.invalidRange
            }
            try handle.seek(toOffset: UInt64(44 + offset * 4))
            let size = requested * 4
            guard let data = try handle.read(upToCount: size), data.count == size else {
                throw RecordingError.incompleteRead
            }
            var samples = [Float](repeating: 0, count: requested)
            _ = samples.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return samples
        }
    }

    func finish() throws {
        try lock.withLock {
            guard !finished, let handle else { return }
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: Self.header(sampleCount: count))
            try handle.synchronize()
            finished = true
        }
    }

    /// Called only after capture has stopped and transcription has drained the file.
    func move(to destination: URL) throws {
        try finish()
        try lock.withLock {
            try handle?.close()
            handle = nil
            try FileManager.default.moveItem(at: url, to: destination)
        }
    }

    private static func header(sampleCount: Int) -> Data {
        var data = Data()
        func integer<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8)
        integer(UInt32(36 + sampleCount * 4))
        data.append(contentsOf: "WAVEfmt ".utf8)
        integer(UInt32(16))
        integer(UInt16(3)) // IEEE Float32, little-endian on Apple silicon.
        integer(UInt16(1))
        integer(UInt32(AudioRecorder.sampleRate))
        integer(UInt32(AudioRecorder.sampleRate) * 4)
        integer(UInt16(4))
        integer(UInt16(32))
        data.append(contentsOf: "data".utf8)
        integer(UInt32(sampleCount * 4))
        return data
    }

    enum RecordingError: LocalizedError {
        case closed, tooLarge, invalidRange, incompleteRead
        var errorDescription: String? {
            switch self {
            case .closed: return "The recording file is closed."
            case .tooLarge: return "The recording exceeded the WAV file size limit."
            case .invalidRange, .incompleteRead: return "Could not read the recorded audio."
            }
        }
    }
}
