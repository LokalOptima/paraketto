import AVFoundation

/// Decode and resample a file into bounded 100 ms, 16 kHz mono batches.
/// AVAudioFile handles the formats supported by the installed macOS audio codecs.
final class AudioFileReader {
    private let file: AVAudioFile
    private let converter: AVAudioConverter
    private let input: AVAudioPCMBuffer
    private let output: AVAudioPCMBuffer
    private var ended = false

    var progress: Double {
        file.length > 0 ? min(1, Double(file.framePosition) / Double(file.length)) : 1
    }

    init(url: URL) throws {
        file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                   channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: file.processingFormat, to: target),
              let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 1600) else {
            throw ReaderError.unsupported
        }
        self.converter = converter
        self.input = input
        self.output = output
    }

    func next() throws -> [Float]? {
        guard !ended else { return nil }
        while true {
            try Task.checkCancellation()
            var conversionError: NSError?
            var readError: Error?
            output.frameLength = 0
            let status = converter.convert(to: output, error: &conversionError) { requested, status in
                guard self.file.framePosition < self.file.length else {
                    status.pointee = .endOfStream
                    return nil
                }
                do {
                    try self.file.read(into: self.input, frameCount: min(requested, self.input.frameCapacity))
                    status.pointee = self.input.frameLength == 0 ? .endOfStream : .haveData
                    return self.input.frameLength == 0 ? nil : self.input
                } catch {
                    readError = error
                    status.pointee = .endOfStream
                    return nil
                }
            }
            if let readError { throw readError }
            if let conversionError { throw conversionError }
            if status == .error { throw ReaderError.unsupported }
            ended = status == .endOfStream
            if output.frameLength > 0, let channel = output.floatChannelData?[0] {
                return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
            }
            if ended { return nil }
        }
    }

    enum ReaderError: LocalizedError {
        case unsupported
        var errorDescription: String? { "This file could not be read as an audio recording." }
    }
}
