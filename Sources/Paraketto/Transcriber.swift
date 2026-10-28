import Foundation
import ParakeetTDT

/// Parakeet Ultra through the ParakeetTDT library: encoder on the Neural Engine, greedy decode on the GPU.
actor Transcriber {
    enum LoadPhase: Equatable {
        /// Fetching model files from Hugging Face; `fraction` is 0...1 of the download.
        case downloading(fraction: Double)
        /// Models are on disk and Core ML is loading or compiling them.
        case loading
    }

    /// Recordings shorter than this have nothing to decode, so callers skip them up front.
    static let minimumSamples = ParakeetConstants.minimumSamples

    private var transcriber: ParakeetTranscriber?

    /// Downloads the models on first run (about 613 MB), then loads them. Subsequent
    /// launches load from Application Support in well under a second.
    func load(progress: @escaping @Sendable (LoadPhase) -> Void) async throws {
        let directory = try await ModelStore.locate { fraction in progress(.downloading(fraction: fraction)) }
        progress(.loading)
        let vadModel = Bundle.main.url(forResource: "VadHead", withExtension: "mlmodelc")
        if vadModel == nil { Log.info("VadHead.mlmodelc not bundled; long recordings fall back to fixed windows") }
        let models = try ParakeetModels(directory: directory, vadModel: vadModel, encoderModel: ModelStore.ownEncoderURL, fusedDecoder: ModelStore.ownFusedDecoderURL)
        Log.info("encoder windows (mel frames): \(models.encoder.variants.map(\.melFrames))")
        // The GPU decoder needs the decoder and joint weights (36 MB, fetched once). Core ML decodes if that fails.
        do {
            let weights = try await ModelStore.locateDecoderWeights { fraction in progress(.downloading(fraction: fraction)) }
            progress(.loading)
            transcriber = try ParakeetTranscriber(models: models, decoderWeights: weights, backend: .metal)
            Log.info("decoder: Metal")
        } catch {
            Log.info("Metal decoder unavailable (\(error.localizedDescription)); decoding with Core ML")
            transcriber = try ParakeetTranscriber(models: models)
        }
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        guard let transcriber else { throw TranscriberError.notLoaded }
        return try transcriber.transcribe(samples).text
    }

    enum TranscriberError: LocalizedError {
        case notLoaded

        var errorDescription: String? {
            switch self {
            case .notLoaded: return "The speech model has not finished loading."
            }
        }
    }
}
