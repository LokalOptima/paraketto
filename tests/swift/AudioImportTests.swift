import AVFoundation
import XCTest
@testable import Paraketto

final class AudioImportTests: XCTestCase {
    func testReaderPreservesMonoSamplesAndBoundsEveryBatch() throws {
        let source = try RecordingFile()
        let expected: [Float] = (0..<48_137).map { sin(Float($0) * 0.1) * 0.15 }
        try source.append(expected)
        try source.finish()
        let reader = try AudioFileReader(url: source.url)
        var actual: [Float] = []
        while let chunk = try reader.next() {
            XCTAssertLessThanOrEqual(chunk.count, 1600)
            actual += chunk
        }
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(reader.progress, 1)
        XCTAssertNil(try reader.next())
    }

    func testReaderResamplesStereoWAVAndCompressedAudio() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for ext in ["wav", "m4a"] {
            let url = root.appendingPathComponent("stereo.\(ext)")
            try writeStereoFixture(to: url)
            let original = try Data(contentsOf: url)
            let reader = try AudioFileReader(url: url)
            var count = 0
            var energy: Float = 0
            while let chunk = try reader.next() {
                XCTAssertLessThanOrEqual(chunk.count, 1600)
                count += chunk.count
                energy += chunk.reduce(0) { $0 + $1 * $1 }
            }
            XCTAssertEqual(Double(count) / 16_000, 3, accuracy: 0.08, ext)
            XCTAssertGreaterThan(sqrt(energy / Float(count)), 0.05, ext)
            XCTAssertEqual(try Data(contentsOf: url), original, "Import must not modify the source")
        }
    }

    func testInvalidAudioIsRejected() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("not an audio file".utf8).write(to: url)
        XCTAssertThrowsError(try AudioFileReader(url: url))
    }

    @MainActor
    func testImportedHistoryPersistsAndOldEntriesStillDecode() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = HistoryStore(directory: root)
        let recording = try RecordingFile()
        try recording.append([Float](repeating: 0, count: 16_000))
        let entry = HistoryEntry(id: UUID(), date: Date(), text: "Imported transcript", destination: "",
                                 duration: 1, inferenceMilliseconds: 10, delivery: .imported,
                                 sourceFilename: "Meeting.m4a")
        try store.add(entry, recording: recording)
        XCTAssertEqual(HistoryStore(directory: root).entries, [entry])
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.audioURL(for: entry).path))
        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any])
        legacy.removeValue(forKey: "sourceFilename")
        legacy["delivery"] = "pasted"
        let decoded = try JSONDecoder().decode(HistoryEntry.self, from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertNil(decoded.sourceFilename)
        XCTAssertEqual(decoded.delivery, .pasted)
    }

    private func writeStereoFixture(to url: URL) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100,
                                   channels: 2, interleaved: false)!
        let settings: [String: Any] = url.pathExtension == "m4a"
            ? [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100,
               AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 96_000]
            : format.settings
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44_100 * 3)!
        buffer.frameLength = buffer.frameCapacity
        for channel in 0..<2 {
            for frame in 0..<Int(buffer.frameLength) {
                buffer.floatChannelData![channel][frame] = sin(Float(frame) * 2 * .pi * 440 / 44_100) * 0.15
            }
        }
        try file.write(from: buffer)
    }
}
