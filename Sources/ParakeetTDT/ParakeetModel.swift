import Foundation

/// Recognition modes have separate weights, vocabularies and caches.
public enum ParakeetModel: String, CaseIterable, Identifiable, Sendable {
    case eng
    case multi

    public var id: String { rawValue }
    public var title: String { rawValue }
    public var description: String {
        self == .eng ? "English only · Parakeet v2" : "25 languages · Parakeet Ultra"
    }
    public var repository: String {
        self == .eng ? "FluidInference/parakeet-tdt-0.6b-v2-coreml" : "FluidInference/parakeet-ultra-coreml"
    }
    // The v2 single-step joint has the same interface as Ultra's JointDecisionv3.
    public var jointFilename: String { self == .eng ? "JointDecision.mlmodelc" : "JointDecisionv3.mlmodelc" }
    public var vocabularyFilename: String { self == .eng ? "parakeet_vocab.json" : "parakeet_v3_vocab.json" }
    public var blankToken: Int { self == .eng ? 1024 : 8192 }
    var cacheName: String { self == .eng ? "parakeet-v2" : "parakeet-ultra" }
    var revision: String { self == .eng ? "ee09c569f73759e6d44c9bd16766f477b2b36d39" : "main" }
    var requiredEntries: [String] {
        ["Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", jointFilename, vocabularyFilename]
    }
}
