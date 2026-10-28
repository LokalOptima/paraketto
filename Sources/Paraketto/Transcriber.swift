import FluidAudio
import Foundation

/// Parakeet Ultra on the Apple Neural Engine via FluidAudio's Core ML models.
actor Transcriber {
    enum LoadPhase: Equatable {
        /// Fetching model files from Hugging Face; `fraction` is 0...1 of the download.
        case downloading(fraction: Double)
        /// Models are on disk and Core ML is loading or compiling them.
        case loading
    }

    /// Recordings shorter than this are rejected by FluidAudio, so callers skip them up front.
    static let minimumSamples = ASRConstants.minimumRequiredSamples(forSampleRate: Int(AudioRecorder.sampleRate))

    private var manager: AsrManager?

    /// Downloads the models on first run (about 613 MB), then loads them. Subsequent
    /// launches load from ~/Library/Application Support/FluidAudio/Models in well under a second.
    func load(progress: @escaping @Sendable (LoadPhase) -> Void) async throws {
        let models = try await AsrModels.downloadAndLoad(version: .ultra) { update in
            switch update.phase {
            case .listing: progress(.downloading(fraction: 0))
            case .downloading: progress(.downloading(fraction: update.fractionCompleted))
            case .compiling: progress(.loading)
            }
        }
        progress(.loading)
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        self.manager = manager
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        guard let manager else { throw TranscriberError.notLoaded }
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let result = try await manager.transcribe(samples, decoderState: &state)
        return result.text.trimmingCharacters(in: .whitespacesAndNewlines)
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
