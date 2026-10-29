import CoreML
import Foundation

/// The encoder as one or more Core ML functions of different window lengths. A segment is
/// encoded by the shortest function whose window holds its mel frames, so short utterances
/// avoid the cost of a full 15 s pass. FluidAudio's export has only the 15 s function.
/// Multi-function models need macOS 15; on macOS 14 only a single-function export loads.
public final class EncoderSet {
    public struct Variant {
        public let melFrames: Int
        let model: MLModel
    }

    /// Sorted by window length, shortest first.
    public let variants: [Variant]

    /// Loads every `encoder_<N>s` function of a multi-function model, or the single `main`
    /// function of a plain export.
    init(url: URL, configuration: MLModelConfiguration) throws {
        let functionNames: [String]
        if #available(macOS 15.0, *) {
            // MLModelAsset exposes the function list; it is empty for single-function models.
            functionNames = (try? MLModelAsset(url: url).functionNamesSync()) ?? []
        } else {
            functionNames = []
        }
        var variants: [Variant] = []
        for name in functionNames.isEmpty ? [nil] : functionNames.map(Optional.some) {
            let config = MLModelConfiguration()
            config.computeUnits = configuration.computeUnits
            if let name, #available(macOS 15.0, *) { config.functionName = name }
            let model = try MLModel(contentsOf: url, configuration: config)
            guard let shape = model.modelDescription.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape,
                  shape.count == 3 else {
                throw ParakeetError.unexpectedShape("encoder mel input", [])
            }
            variants.append(Variant(melFrames: shape[2].intValue, model: model))
        }
        self.variants = variants.sorted { $0.melFrames < $1.melFrames }
    }

    var longestWindow: Int { variants.last?.melFrames ?? ParakeetConstants.melFrames }

    /// The shortest variant that holds `melLength` frames, else the longest.
    func variant(forMelLength melLength: Int) -> Variant {
        variants.first { $0.melFrames >= melLength } ?? variants[variants.count - 1]
    }
}

@available(macOS 15.0, *)
private extension MLModelAsset {
    /// Synchronous wrapper over the async `functionNames`; model loading is synchronous here anyway.
    func functionNamesSync() throws -> [String] {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[String], Error> = .success([])
        Task {
            do { result = .success(try await functionNames) } catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try result.get()
    }
}
