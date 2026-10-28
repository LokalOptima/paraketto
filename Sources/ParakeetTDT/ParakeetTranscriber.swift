import CoreML
import Foundation

public struct Transcription {
    public let text: String
    /// Number of 15 s windows decoded.
    public let segments: Int
    /// Token ids with the global encoder frame (80 ms) at which each was emitted.
    public let tokens: [(id: Int, frame: Int)]
}

/// Parakeet TDT (Ultra) speech recognition on Apple silicon with Core ML.
///
/// Pipeline per segment: 16 kHz samples → Preprocessor (mel) → Encoder (ANE, int8)
/// → greedy TDT decode with the LSTM decoder and joint network → SentencePiece text.
/// Recordings longer than the encoder's 15 s window are first cut at pauses found by the
/// model's VAD head. Each segment is encoded with about a second of neighbouring audio on
/// both sides so the boundary is heard in context, tokens are emitted only for the segment
/// itself, and the decoder state is carried from segment to segment.
/// Which implementation runs the sequential prediction-network and joint loop.
public enum DecoderBackend: String, Sendable {
    /// Core ML decoder and joint models (and the fused step when exported). No extra weights needed.
    case coreML = "coreml"
    /// Accelerate on the CPU; a measured negative result kept for comparison.
    case cpu
    /// The greedy loop on the GPU with Metal.
    case metal
}

public final class ParakeetTranscriber {
    /// Acoustic context added on each side of a segment, in encoder frames (~1 s).
    static let contextFrames = 12
    private static let windowFrames = ParakeetConstants.windowSamples / ParakeetConstants.samplesPerFrame

    private let models: ParakeetModels
    private let decoder: TdtDecoder
    /// Accelerate-based prediction network and joint; used instead of the Core ML models when present.
    private let cpuDecoder: CPUTdtDecoder?
    private let metalDecoder: MetalTdtDecoder?
    private let segmenter: Segmenter
    private let audioBuffer = try! MLMultiArray.float32([1, ParakeetConstants.windowSamples])

    /// - Parameters:
    ///   - decoderWeights: safetensors with the decoder/joint tensors (see `ModelStore.locateDecoderWeights`),
    ///     required by the `cpu` and `metal` backends.
    ///   - backend: falls back to Core ML when its weights are missing.
    public init(models: ParakeetModels, decoderWeights: URL? = nil, backend: DecoderBackend = .coreML) throws {
        self.models = models
        let vocabulary = models.vocabulary
        decoder = TdtDecoder(decoder: models.decoder, joint: models.joint, fusedStep: models.fusedStep, isWordStart: { vocabulary.isWordStart($0) })
        var cpuDecoder: CPUTdtDecoder?
        var metalDecoder: MetalTdtDecoder?
        if let decoderWeights {
            switch backend {
            case .cpu:
                let network = try CPUDecoder(weights: try SafeTensors(url: decoderWeights))
                cpuDecoder = CPUTdtDecoder(network: network, isWordStart: { vocabulary.isWordStart($0) })
            case .metal:
                metalDecoder = try MetalTdtDecoder(weights: try SafeTensors(url: decoderWeights), isWordStart: { vocabulary.isWordStart($0) })
            case .coreML:
                break
            }
        }
        self.cpuDecoder = cpuDecoder
        self.metalDecoder = metalDecoder
        var segmenter = Segmenter()
        segmenter.maximumSegmentFrames = Self.windowFrames - 2 * Self.contextFrames
        self.segmenter = segmenter
    }

    public func transcribe(_ samples: [Float]) throws -> Transcription {
        guard samples.count >= ParakeetConstants.minimumSamples else { return Transcription(text: "", segments: 0, tokens: []) }

        // A recording that fits one window is decoded whole, exactly as FluidAudio does.
        guard samples.count > ParakeetConstants.windowSamples else {
            var state = try DecoderState()
            resetDecoders()
            let t0 = Date()
            let mel = try preprocess(samples[...])
            let t1 = Date()
            let frames = try encode(mel)
            let t2 = Date()
            let result = try decode(frames: frames, range: 0..<frames.count, state: &state)
            if ProcessInfo.processInfo.environment["PARAKEET_TIMING"] != nil {
                let ms = { (a: Date, b: Date) in Int(b.timeIntervalSince(a) * 1000) }
                FileHandle.standardError.write(Data("timing: mel \(ms(t0, t1)) ms, encoder \(ms(t1, t2)) ms, decode \(ms(t2, Date())) ms over \(frames.count) frames, \(result.tokens.count) tokens; \(DecodeProfile.summary)\n".utf8))
                if let summary = alternateTimingSummary { FileHandle.standardError.write(Data(summary.utf8 + [10])) }
            }
            return Transcription(
                text: models.vocabulary.text(for: result.tokens), segments: 1,
                tokens: Array(zip(result.tokens, result.frames)).map { (id: $0.0, frame: $0.1) }
            )
        }

        var state = try DecoderState()
        resetDecoders()
        let timing = ProcessInfo.processInfo.environment["PARAKEET_TIMING"] != nil
        let t0 = Date()
        var melNanos = 0.0, encodeNanos = 0.0, decodeNanos = 0.0
        var tokens: [Int] = []
        var emitted: [(id: Int, frame: Int)] = []
        let totalFrames = (samples.count + ParakeetConstants.samplesPerFrame - 1) / ParakeetConstants.samplesPerFrame
        let segments = try segments(for: samples)
        let tVad = Date()
        var resumeFrame = 0  // where the previous segment stopped emitting, in global frames
        for segment in segments {
            let startFrame = max(segment.startFrame, resumeFrame)
            guard startFrame < segment.endFrame else { continue }
            let lead = min(Self.contextFrames, startFrame)
            let tail = min(Self.contextFrames, totalFrames - segment.endFrame)
            let windowStart = startFrame - lead
            let start = windowStart * ParakeetConstants.samplesPerFrame
            let end = min((segment.endFrame + tail) * ParakeetConstants.samplesPerFrame, samples.count)
            guard end - start >= ParakeetConstants.minimumSamples else { continue }
            let s0 = Date()
            let mel = try preprocess(samples[start..<end])
            let s1 = Date()
            let frames = try encode(mel)
            let s2 = Date()
            let result = try decode(frames: frames, range: lead..<(lead + segment.endFrame - startFrame), state: &state)
            melNanos += s1.timeIntervalSince(s0); encodeNanos += s2.timeIntervalSince(s1); decodeNanos += Date().timeIntervalSince(s2)
            // Emission timing shifts a little with context, so the new segment can start by re-emitting
            // the words the previous one ended with. Drop that overlap.
            let overlap = Self.overlap(tail: tokens, head: result.tokens)
            tokens += result.tokens.dropFirst(overlap)
            emitted += zip(result.tokens, result.frames).dropFirst(overlap).map { (id: $0.0, frame: windowStart + $0.1) }
            resumeFrame = windowStart + result.stopFrame
        }
        if timing {
            let ms = { (s: Double) in Int(s * 1000) }
            FileHandle.standardError.write(Data("timing: vad+mel \(ms(tVad.timeIntervalSince(t0))) ms, \(segments.count) segments: mel \(ms(melNanos)) ms, encoder \(ms(encodeNanos)) ms, decode \(ms(decodeNanos)) ms over \(totalFrames) frames, \(tokens.count) tokens; \(DecodeProfile.summary)\n".utf8))
            if let summary = alternateTimingSummary { FileHandle.standardError.write(Data(summary.utf8 + [10])) }
        }
        return Transcription(text: models.vocabulary.text(for: tokens), segments: segments.count, tokens: emitted)
    }

    /// The first `frames` columns of a [1, 128, T] mel; the array itself when it already fits.
    private static func slice(_ mel: MLMultiArray, toFrames frames: Int) throws -> MLMultiArray {
        let shape = mel.shape.map(\.intValue)
        guard shape.count == 3, shape[2] != frames else { return mel }
        guard shape[2] > frames, mel.dataType == .float32 else { throw ParakeetError.unexpectedShape("mel", shape) }
        let sliced = try MLMultiArray.float32([1, shape[1], frames])
        let rowStride = mel.strides[1].intValue
        for row in 0..<shape[1] {
            (sliced.floatPointer + row * frames).update(from: mel.floatPointer + row * rowStride, count: frames)
        }
        return sliced
    }

    private func decode(frames: EncoderFrames, range: Range<Int>, state: inout DecoderState) throws -> TdtDecoder.Result {
        if let metalDecoder { return metalDecoder.decode(frames: frames, range: range) }
        if let cpuDecoder { return cpuDecoder.decode(frames: frames, range: range) }
        return try decoder.decode(frames: frames, range: range, state: &state)
    }

    private func resetDecoders() {
        cpuDecoder?.reset()
        metalDecoder?.reset()
        metalDecoder?.resetTiming()
        DecodeProfile.reset()
    }

    private var alternateTimingSummary: String? { metalDecoder?.timingSummary ?? cpuDecoder?.timingSummary }

    /// Longest suffix of `tail` (up to 8 tokens, starting at a word boundary) that `head` begins with.
    static func overlap(tail: [Int], head: [Int], isWordStart: ((Int) -> Bool)? = nil) -> Int {
        for length in stride(from: min(8, tail.count, head.count), through: 1, by: -1)
        where Array(tail.suffix(length)) == Array(head.prefix(length)) {
            return length
        }
        return 0
    }

    /// VAD-guided cuts, or fixed windows when no VAD model is available.
    private func segments(for samples: [Float]) throws -> [Segmenter.Segment] {
        let totalFrames = (samples.count + ParakeetConstants.samplesPerFrame - 1) / ParakeetConstants.samplesPerFrame
        guard models.vad != nil else {
            return stride(from: 0, to: totalFrames, by: segmenter.maximumSegmentFrames).map {
                Segmenter.Segment(startFrame: $0, endFrame: min($0 + segmenter.maximumSegmentFrames, totalFrames))
            }
        }
        var probabilities = try speechProbabilities(samples)
        if probabilities.count < totalFrames { probabilities += Array(repeating: 1, count: totalFrames - probabilities.count) }
        return segmenter.segments(speechProbabilities: Array(probabilities.prefix(totalFrames)))
    }

    /// Runs the VAD head over consecutive 15 s blocks and joins the per-frame probabilities.
    public func speechProbabilities(_ samples: [Float]) throws -> [Float] {
        guard let vad = models.vad else { return [] }
        var probabilities: [Float] = []
        var start = 0
        while start < samples.count {
            let end = min(start + ParakeetConstants.windowSamples, samples.count)
            let mel = try preprocess(samples[start..<end])
            let output = try vad.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: mel.mel)
            ]))
            let block = try output.array("speech_probability")
            let frames = min((end - start) / ParakeetConstants.samplesPerFrame, block.count)
            probabilities += Array(UnsafeBufferPointer(start: block.floatPointer, count: frames))
            start = end
        }
        return probabilities
    }

    private struct Mel {
        let mel: MLMultiArray
        let length: MLMultiArray
        /// Encoder frames the audio actually covers: ceil(samples / 1280).
        let frames: Int
    }

    private func preprocess(_ samples: ArraySlice<Float>) throws -> Mel {
        let pointer = audioBuffer.floatPointer
        memset(pointer, 0, ParakeetConstants.windowSamples * MemoryLayout<Float>.stride)
        samples.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: $0.count) }
        // Declare the length rounded up to a whole encoder frame (zero-padded), as FluidAudio does;
        // the last frame then normalises against the same mel columns and decodes identically.
        let frames = (samples.count + ParakeetConstants.samplesPerFrame - 1) / ParakeetConstants.samplesPerFrame
        let declaredLength = min(frames * ParakeetConstants.samplesPerFrame, ParakeetConstants.windowSamples)
        let output = try models.preprocessor.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioBuffer),
            "audio_length": MLFeatureValue(multiArray: try .int32(declaredLength)),
        ]))
        return Mel(mel: try output.array("mel"), length: try output.array("mel_length"), frames: frames)
    }

    private func encode(_ mel: Mel) throws -> EncoderFrames {
        let variant = models.encoder.variant(forMelLength: mel.length.firstInt)
        let input = try Self.slice(mel.mel, toFrames: variant.melFrames)
        let output = try variant.model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: input),
            "mel_length": MLFeatureValue(multiArray: mel.length),
        ]))
        // The encoder reports one frame more than the audio covers at some lengths; FluidAudio caps it
        // by the audio length too, and decoding that padding frame can add spurious punctuation.
        let validFrames = min(try output.array("encoder_length").firstInt, mel.frames)
        return try EncoderFrames(
            try output.array("encoder"), validFrames: validFrames, hiddenSize: ParakeetConstants.encoderHiddenSize
        )
    }
}
