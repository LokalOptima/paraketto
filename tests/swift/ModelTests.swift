import AVFoundation
import CoreML
import XCTest
@testable import ParakeetTDT
@testable import Paraketto

final class ModelTests: XCTestCase {
    func testVADFrameCountsStayAlignedAcrossLongRecordings() throws {
        try requireModels()
        let store = ModelStore(model: .eng)
        let models = try ParakeetModels(directory: XCTUnwrap(store.cachedDirectory), model: .eng, vadModel: nil)
        let recognizer = try ParakeetTranscriber(models: models)
        for seconds in [30, 180] {
            let samples = [Float](repeating: 0, count: seconds * 16_000)
            let expectedFrames = (samples.count + 1279) / 1280
            XCTAssertEqual(try recognizer.speechProbabilities(samples).count, expectedFrames,
                           "Rounding each 15 s block independently shifts later pause timestamps")
        }
    }

    func testIncompleteModelDirectoriesAreNotUsableCaches() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(model: .eng)
        for entry in ParakeetModel.eng.requiredEntries {
            let files = entry.hasSuffix(".mlmodelc")
                ? ["coremldata.bin", "model.mil", "weights/weight.bin"].map { entry + "/" + $0 }
                : [entry]
            for file in files {
                let url = root.appendingPathComponent(file)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data([1]).write(to: url)
            }
        }
        XCTAssertTrue(store.isComplete(root))
        XCTAssertFalse(ModelStore(model: .multi).isComplete(root), "A v2 cache is not an Ultra cache")
        let weights = root.appendingPathComponent("Encoder.mlmodelc/weights/weight.bin")
        try Data().write(to: weights)
        XCTAssertFalse(store.isComplete(root), "An interrupted weight download must not count as ready")
        try FileManager.default.removeItem(at: weights)
        XCTAssertFalse(store.isComplete(root))
    }

    func testEnglishNeverPicksUpInstalledUltraOptimizations() {
        let english = ModelStore(model: .eng)
        XCTAssertNotEqual(english.ownDirectory, ModelStore(model: .multi).ownDirectory)
        XCTAssertNil(english.ownEncoderURL)
        XCTAssertNil(english.ownFusedDecoderURL)
    }

    /// Opt in after installing both model caches; never downloads models as part of a unit test.
    func testRealModelSwitchRoundTrip() async throws {
        try requireModels()
        let recognizer = Transcriber(vadModelURL: projectRoot.appendingPathComponent(".build/VadHead.mlmodelc"))
        let samples = try fixtureSamples()
        var transcripts: [String] = []
        for mode in [ParakeetModel.multi, .eng, .multi] {
            try await recognizer.load(model: mode) { _ in }
            try await recognizer.startSession()
            for start in stride(from: 0, to: samples.count, by: 1600) {
                await recognizer.append(Array(samples[start..<min(start + 1600, samples.count)]))
            }
            let text = try await recognizer.finish()
            transcripts.append(text)
            XCTAssertTrue(text.lowercased().contains("john taylor"), "\(mode): \(text)")
            XCTAssertTrue(text.lowercased().contains("cotton"), "\(mode): \(text)")
        }
        XCTAssertEqual(transcripts[0], transcripts[2], "Switching to v2 must not contaminate Ultra's state")
    }

    func testSharedPauseDetectorWithBothPreprocessors() async throws {
        try requireModels()
        let samples = try fixtureSamples()
        let vad = projectRoot.appendingPathComponent(".build/VadHead.mlmodelc")
        _ = try XCTUnwrap(FileManager.default.fileExists(atPath: vad.path) ? vad : nil, "Run make vad first")
        let silence = [Float](repeating: 0, count: 16_000 * 2)
        let utterance = samples + silence + samples + silence
        for mode in ParakeetModel.allCases {
            let store = ModelStore(model: mode)
            let directory = try await store.locate { _ in }
            let models = try ParakeetModels(directory: directory, model: mode, vadModel: vad,
                                           encoderModel: store.ownEncoderURL, fusedDecoder: store.ownFusedDecoderURL)
            let recognizer = try ParakeetTranscriber(models: models)
            let silent = try recognizer.speechProbabilities(silence)
            XCTAssertLessThan(silent.reduce(0, +) / Float(silent.count), 0.1, "\(mode) silence")
            let spoken = try recognizer.speechProbabilities(samples)
            XCTAssertTrue(spoken.contains { $0 >= 0.5 }, "\(mode) speech")
            let mixed = try recognizer.speechProbabilities(utterance)
            let gapStart = (samples.count + 8_000) / ParakeetConstants.samplesPerFrame
            let gapEnd = (samples.count + 24_000) / ParakeetConstants.samplesPerFrame
            let gap = mixed[gapStart..<gapEnd]
            XCTAssertLessThan(gap.reduce(0, +) / Float(gap.count), 0.5, "\(mode) pause within speech")
            let session = recognizer.startSession()
            for start in stride(from: 0, to: utterance.count, by: 1600) {
                try session.append(Array(utterance[start..<min(start + 1600, utterance.count)]))
            }
            let result = try session.finish()
            XCTAssertGreaterThanOrEqual(result.segments, 2, "\(mode) should split at the pause")
            XCTAssertEqual(result.text.lowercased().components(separatedBy: "john taylor").count - 1, 2, result.text)
        }
    }

    private var projectRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testRollingAudioStaysBoundedAcrossLongSpeechAndSilence() async throws {
        try requireModels()
        let speech = try fixtureSamples()
        let silence = [Float](repeating: 0, count: 32_000)
        let vad = projectRoot.appendingPathComponent(".build/VadHead.mlmodelc")
        _ = try XCTUnwrap(FileManager.default.fileExists(atPath: vad.path) ? vad : nil, "Run make vad first")
        for mode in ParakeetModel.allCases {
            let store = ModelStore(model: mode)
            let models = try ParakeetModels(directory: XCTUnwrap(store.cachedDirectory), model: mode,
                                           vadModel: vad, encoderModel: store.ownEncoderURL,
                                           fusedDecoder: store.ownFusedDecoderURL)
            let recognizer = try ParakeetTranscriber(models: models)
            let session = recognizer.startSession()
            for _ in 0..<12 {
                try session.append(speech + silence)
                XCTAssertLessThan(session.bufferedSampleCount, 16_000 * 32)
            }
            // A large offline append must also be consumed in bounded batches.
            try session.append([Float](repeating: 0, count: 16_000 * 180))
            let result = try session.finish()
            XCTAssertEqual(result.text.lowercased().components(separatedBy: "john taylor").count - 1, 12, "\(mode): \(result.text)")
            XCTAssertLessThan(session.peakBufferedSampleCount, 16_000 * 32)
            XCTAssertEqual(session.bufferedSampleCount, 0, "finish must release the remaining audio")
        }
    }

    func testQuickStopSilenceIsRejectedWithoutRejectingQuietSpeech() async throws {
        try requireModels()
        let quietSpeech = try fixtureSamples().map { $0 * 0.1 }
        let vad = projectRoot.appendingPathComponent(".build/VadHead.mlmodelc")
        _ = try XCTUnwrap(FileManager.default.fileExists(atPath: vad.path) ? vad : nil, "Run make vad first")
        for mode in ParakeetModel.allCases {
            let store = ModelStore(model: mode)
            let models = try ParakeetModels(directory: XCTUnwrap(store.cachedDirectory), model: mode,
                                           vadModel: vad, encoderModel: store.ownEncoderURL,
                                           fusedDecoder: store.ownFusedDecoderURL)
            let recognizer = try ParakeetTranscriber(models: models)
            try recognizer.warmUp()
            for count in [6400, 11_200, 16_000, 32_000] {
                var seed: UInt32 = 42
                let noise: [Float] = (0..<count).map { _ in
                    seed = seed &* 1664525 &+ 1013904223
                    return (Float(seed) / Float(UInt32.max) * 2 - 1) * 0.0003
                }
                for samples in [[Float](repeating: 0, count: count), noise] {
                    let session = recognizer.startSession()
                    try session.append(samples)
                    let result = try session.finish()
                    XCTAssertEqual(result.text, "", "\(mode), \(count) samples")
                    if samples.allSatisfy({ $0 == 0 }) {
                        XCTAssertEqual(result.segments, 0, "Digital silence must never reach the decoder")
                    }
                }
            }
            let text = try recognizer.transcribe(quietSpeech).text.lowercased()
            XCTAssertTrue(text.contains("john taylor"), "\(mode): \(text)")
            XCTAssertTrue(text.contains("cotton"), "\(mode): \(text)")
            let shortText = try recognizer.transcribe(Array(quietSpeech.prefix(24_000))).text.lowercased()
            XCTAssertTrue(shortText.contains("john"), "\(mode) must retain speech shorter than the live VAD cadence: \(shortText)")
        }
    }

    private func requireModels() throws {
        guard ProcessInfo.processInfo.environment["PARAKETTO_MODEL_TESTS"] == "1" else {
            throw XCTSkip("Set PARAKETTO_MODEL_TESTS=1 to exercise installed Core ML models")
        }
        _ = try XCTUnwrap(ParakeetModel.allCases.allSatisfy({ ModelStore(model: $0).cachedDirectory != nil }) ? true : nil,
                          "Run make prepare-models before enabling model integration tests")
    }

    func testFileImportMatchesStreamingAndCleansUpCancellation() async throws {
        try requireModels()
        let samples = try fixtureSamples()
        let source = try RecordingFile()
        try source.append(samples)
        try source.finish()
        let original = try Data(contentsOf: source.url)
        let recognizer = Transcriber(vadModelURL: projectRoot.appendingPathComponent(".build/VadHead.mlmodelc"))
        for mode in ParakeetModel.allCases {
            try await recognizer.load(model: mode) { _ in }
            try await recognizer.startSession()
            await recognizer.append(samples)
            let expected = try await recognizer.finish()
            var imported: Transcriber.FileResult? = try await recognizer.transcribeFile(source.url) { _ in }
            XCTAssertEqual(imported?.text, expected, "\(mode)")
            XCTAssertEqual(imported?.recording.sampleCount, samples.count)
            let temporary = try XCTUnwrap(imported?.recording.url)
            imported = nil
            XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.path))
        }
        XCTAssertEqual(try Data(contentsOf: source.url), original)
        let task = Task {
            try await recognizer.transcribeFile(source.url) { _ in
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        do {
            _ = try await task.value
            XCTFail("Cancelled imports must not produce a result")
        } catch is CancellationError { }
        let next = try await recognizer.transcribeFile(source.url) { _ in }
        XCTAssertTrue(next.text.lowercased().contains("john taylor"))
    }

    private func fixtureSamples() throws -> [Float] {
        let path = ProcessInfo.processInfo.environment["PARAKETTO_TEST_AUDIO"]
            ?? projectRoot.appendingPathComponent("data/librispeech/1995-1826-0002.wav").path
        let url = URL(fileURLWithPath: path)
        _ = try XCTUnwrap(FileManager.default.fileExists(atPath: url.path) ? url : nil,
                          "Set PARAKETTO_TEST_AUDIO to the LibriSpeech 1995-1826-0002.wav fixture")
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }
}
