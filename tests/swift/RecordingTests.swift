import AVFoundation
import XCTest
@testable import Paraketto

final class RecordingTests: XCTestCase {
    func testDiskRecordingPreservesSamplesAndPlaysAfterMove() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var recording: RecordingFile? = try RecordingFile(directory: root)
        let temporary = recording!.url
        let chunk: [Float] = (0..<1600).map { sin(Float($0) * 0.1) * 0.1 }
        for _ in 0..<20 { try recording!.append(chunk) }
        XCTAssertEqual(recording!.sampleCount, 32_000)
        XCTAssertEqual(try recording!.readSamples(from: 1600, count: 1600), chunk)
        XCTAssertThrowsError(try recording!.readSamples(from: 31_999, count: 2))
        try recording!.finish()
        XCTAssertThrowsError(try recording!.append(chunk))
        let saved = root.appendingPathComponent("saved.wav")
        try recording!.move(to: saved)
        recording = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: saved.path))
        let file = try AVAudioFile(forReading: saved)
        XCTAssertEqual(file.length, 32_000)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600)!
        try file.read(into: buffer)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: 1600)), chunk)
        let player = try AVAudioPlayer(contentsOf: saved)
        XCTAssertEqual(player.duration, 2, accuracy: 0.001)
    }

    func testAbandonedRecordingsAreRemoved() throws {
        for finalize in [false, true] {
            var recording: RecordingFile? = try RecordingFile()
            let url = recording!.url
            try recording!.append([0.1, -0.2])
            if finalize { try recording!.finish() }
            recording = nil
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        }
    }

    func testSlowConsumerCanRecoverEverySampleFromCoalescedProgress() async throws {
        let recording = try RecordingFile()
        let (stream, progress) = AsyncStream<Int>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let chunk: [Float] = (0..<1600).map { Float($0) / 1600 }
        for _ in 0..<600 { progress.yield(try recording.append(chunk)) }
        try recording.finish()
        progress.finish()
        var consumed = 0
        var notifications = 0
        for await available in stream {
            notifications += 1
            while consumed < available {
                let count = min(1600, available - consumed)
                XCTAssertEqual(try recording.readSamples(from: consumed, count: count), chunk)
                consumed += count
            }
        }
        XCTAssertEqual(notifications, 1)
        XCTAssertEqual(consumed, 960_000)
    }

    func testMeterRespondsToQuietSpeechAndRetainsBriefPeaks() {
        var meter = LevelSmoother()
        meter.update(rms: 0)
        meter.tick(interval: 0.05)
        XCTAssertEqual(meter.level, 0)
        meter.update(rms: pow(10, -45.0 / 20))
        meter.update(rms: 0) // A later quiet audio callback must not erase the syllable.
        meter.tick(interval: 0.05)
        XCTAssertGreaterThan(meter.level, 0.35)
        for _ in 0..<20 { meter.tick(interval: 0.05) }
        XCTAssertLessThan(meter.level, 0.01)
        meter.update(rms: 1)
        meter.reset()
        meter.tick(interval: 0.05)
        XCTAssertEqual(meter.level, 0)
    }
}
