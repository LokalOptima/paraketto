import CoreML
import Foundation

public enum ParakeetError: LocalizedError {
    case missingFile(String)
    case missingOutput(String)
    case unexpectedShape(String, [Int])
    case download(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let name): return "Model file missing: \(name)"
        case .missingOutput(let name): return "Model output missing: \(name)"
        case .unexpectedShape(let name, let shape): return "Unexpected shape for \(name): \(shape)"
        case .download(let message): return "Model download failed: \(message)"
        }
    }
}

/// Fixed properties of the Parakeet TDT 0.6B v3 family export used here (v3, Ultra, Redux, Phonon-2 share it).
public enum ParakeetConstants {
    public static let sampleRate = 16_000
    /// The exported preprocessor and encoder take exactly this many samples (15 s).
    public static let windowSamples = 240_000
    public static let melBins = 128
    public static let melFrames = 1501
    /// Audio samples per encoder frame: 10 ms mel hop times 8x subsampling.
    public static let samplesPerFrame = 1280
    public static let encoderHiddenSize = 1024
    public static let decoderHiddenSize = 640
    public static let decoderLayers = 2
    public static let blankToken = 8192
    public static let durationBins = [0, 1, 2, 3, 4]
    public static let maxSymbolsPerFrame = 10
    /// Shorter recordings are rejected: the model has nothing to decode and FluidAudio uses the same floor.
    public static let minimumSamples = sampleRate * 3 / 10
}

/// The four Core ML models of FluidAudio's export plus our VAD head and the vocabulary.
public final class ParakeetModels {
    let preprocessor: MLModel
    public let encoder: EncoderSet
    let decoder: MLModel
    let joint: MLModel
    /// Decoder update and joint decision in one model (our export); nil falls back to two calls per token.
    public let fusedStep: MLModel?
    let vad: MLModel?
    let vocabulary: Vocabulary

    /// The preprocessor runs on the CPU; everything else prefers the Neural Engine, as FluidAudio does.
    /// - Parameter encoderModel: an alternative encoder (our multi-window export); nil uses the directory's.
    public init(directory: URL, vadModel: URL?, encoderModel: URL? = nil, fusedDecoder: URL? = nil) throws {
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        let ane = MLModelConfiguration()
        ane.computeUnits = .cpuAndNeuralEngine
        // Experiment hook: PARAKEET_UNITS=cpu|gpu|all|ane picks the compute units for the decoder and joint.
        let small = MLModelConfiguration()
        switch ProcessInfo.processInfo.environment["PARAKEET_UNITS"] {
        case "cpu": small.computeUnits = .cpuOnly
        case "gpu": small.computeUnits = .cpuAndGPU
        case "all": small.computeUnits = .all
        default: small.computeUnits = .cpuAndNeuralEngine
        }

        preprocessor = try Self.load(directory.appendingPathComponent("Preprocessor.mlmodelc"), cpu)
        let encoderURL = encoderModel ?? directory.appendingPathComponent("Encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw ParakeetError.missingFile(encoderURL.lastPathComponent)
        }
        encoder = try EncoderSet(url: encoderURL, configuration: ane)
        decoder = try Self.load(directory.appendingPathComponent("Decoder.mlmodelc"), small)
        joint = try Self.load(directory.appendingPathComponent("JointDecisionv3.mlmodelc"), small)
        fusedStep = try fusedDecoder.map { try Self.load($0, small) }
        vad = try vadModel.map { try Self.load($0, ane) }
        vocabulary = try Vocabulary(url: directory.appendingPathComponent("parakeet_v3_vocab.json"))
    }

    private static func load(_ url: URL, _ configuration: MLModelConfiguration) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ParakeetError.missingFile(url.lastPathComponent)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }
}

/// SentencePiece vocabulary: token id to piece, with "▁" marking a word start.
struct Vocabulary {
    private let pieces: [String]

    init(url: URL) throws {
        let table = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
        var pieces = Array(repeating: "", count: table.count)
        for (id, piece) in table {
            guard let index = Int(id), index < pieces.count else { continue }
            pieces[index] = piece
        }
        self.pieces = pieces
    }

    func isWordStart(_ token: Int) -> Bool {
        token >= 0 && token < pieces.count && pieces[token].hasPrefix("▁")
    }

    func text(for tokens: [Int]) -> String {
        var text = ""
        for token in tokens where token >= 0 && token < pieces.count {
            let piece = pieces[token]
            if piece.hasPrefix("▁") {
                if !text.isEmpty { text += " " }
                text += piece.dropFirst()
            } else {
                text += piece
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
