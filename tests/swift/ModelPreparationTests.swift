import XCTest
@testable import Paraketto

final class ModelPreparationTests: XCTestCase {
    func testPreparationIsRememberedAndInvalidatedByChangedWeightsOrOS() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let weights = model.appendingPathComponent("weights.bin")
        try Data([1]).write(to: weights)
        func record(system: String = "macOS test") -> ModelPreparationRecord {
            ModelPreparationRecord(directory: model, additionalModels: [],
                                   recordURL: root.appendingPathComponent("prepared.json"), systemVersion: system)
        }

        XCTAssertFalse(record().isPrepared)
        try record().markPrepared()
        XCTAssertTrue(record().isPrepared)
        XCTAssertFalse(record(system: "macOS updated").isPrepared)
        try Data([1, 2]).write(to: weights)
        XCTAssertFalse(record().isPrepared)
        try record().markPrepared()
        XCTAssertTrue(record().isPrepared)
        try FileManager.default.removeItem(at: model)
        XCTAssertFalse(record().isPrepared)
    }

    func testSwitchingToAnOptimizedEncoderRequiresPreparation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model")
        let encoder = root.appendingPathComponent("optimized-encoder")
        for directory in [model, encoder] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data([1]).write(to: directory.appendingPathComponent("weights.bin"))
        }
        let receipt = root.appendingPathComponent("prepared.json")
        try ModelPreparationRecord(directory: model, additionalModels: [], recordURL: receipt).markPrepared()
        let optimized = ModelPreparationRecord(directory: model, additionalModels: [encoder], recordURL: receipt)
        XCTAssertFalse(optimized.isPrepared)
        try optimized.markPrepared()
        XCTAssertTrue(optimized.isPrepared)
    }
}
