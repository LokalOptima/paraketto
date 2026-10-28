import CoreML
import Foundation

public struct Transcription {
    public let text: String
    /// Number of encoder passes (segments) decoded.
    public let segments: Int
    /// Token ids with the global encoder frame (80 ms) at which each was emitted.
    public let tokens: [(id: Int, frame: Int)]
}

/// Parakeet TDT (Ultra) speech recognition on Apple silicon with Core ML.
///
/// Pipeline per segment: 16 kHz samples → Preprocessor (mel) → Encoder (ANE, int8)
/// → greedy TDT decode with the LSTM decoder and joint network → SentencePiece text.
/// Recordings longer than the encoder's 15 s window are cut at pauses found by the
/// model's VAD head. Each segment is encoded with about a second of neighbouring audio on
/// both sides so the boundary is heard in context, tokens are emitted only for the segment
/// itself, and the decoder state is carried from segment to segment.
///
/// A `Session` accepts audio while it is being recorded and transcribes each segment as soon
/// as the cut after it can be decided, so at the end only the last partial segment is left.
/// `transcribe(_:)` is the same thing with all the audio appended at once.
public final class ParakeetTranscriber {
    /// Acoustic context added on each side of a segment, in encoder frames (~1 s).
    static let contextFrames = 12
    static let windowFrames = ParakeetConstants.windowSamples / ParakeetConstants.samplesPerFrame
    static let timing = ProcessInfo.processInfo.environment["PARAKEET_TIMING"] != nil

    let models: ParakeetModels
    let decoder: TdtDecoder
    let segmenter: Segmenter
    private let audioBuffer = try! MLMultiArray.float32([1, ParakeetConstants.windowSamples])

    public init(models: ParakeetModels) throws {
        self.models = models
        let vocabulary = models.vocabulary
        decoder = TdtDecoder(decoder: models.decoder, joint: models.joint, fusedStep: models.fusedStep, isWordStart: { vocabulary.isWordStart($0) })
        var segmenter = Segmenter()
        segmenter.maximumSegmentFrames = Self.windowFrames - 2 * Self.contextFrames
        self.segmenter = segmenter
    }

    public func transcribe(_ samples: [Float]) throws -> Transcription {
        let session = Session(transcriber: self)
        try session.append(samples)
        return try session.finish()
    }

    public func startSession() -> Session {
        Session(transcriber: self)
    }

    /// Runs a short silent recording through every model so the first real one does not pay for
    /// Core ML's lazy setup (about 50 ms on an M1 Max). Each encoder window is exercised.
    public func warmUp() throws {
        for variant in models.encoder.variants {
            let samples = [Float](repeating: 0, count: min(variant.melFrames * 160, ParakeetConstants.windowSamples))
            _ = try transcribe(samples)
        }
        if models.vad != nil { _ = try speechProbabilities([Float](repeating: 0, count: ParakeetConstants.windowSamples)) }
    }

    /// Runs the VAD head over consecutive 15 s blocks and joins the per-frame probabilities.
    public func speechProbabilities(_ samples: [Float]) throws -> [Float] {
        var probabilities: [Float] = []
        var start = 0
        while start < samples.count {
            let end = min(start + ParakeetConstants.windowSamples, samples.count)
            probabilities += try speechProbabilities(block: samples[start..<end])
            start = end
        }
        return probabilities
    }

    // MARK: Session

    /// One recording, transcribed incrementally. Not thread-safe: call from one actor or queue.
    public final class Session {
        private let transcriber: ParakeetTranscriber
        private var samples: [Float] = []
        /// Speech probabilities of the completed 15 s VAD blocks, one per encoder frame.
        private var vadCache: [Float] = []
        private var cachedBlocks = 0
        /// Global frame where the next segment starts.
        private var segmentStart = 0
        /// Global frame where the previous segment stopped emitting (may run past its cut to finish a word).
        private var resumeFrame = 0
        private var state: DecoderState
        private var tokens: [Int] = []
        private var emitted: [(id: Int, frame: Int)] = []
        private var segments = 0
        private var finished = false
        private var melSeconds = 0.0, encodeSeconds = 0.0, decodeSeconds = 0.0, vadSeconds = 0.0

        init(transcriber: ParakeetTranscriber) {
            self.transcriber = transcriber
            state = try! DecoderState()
            DecodeProfile.reset()
        }

        private var totalFrames: Int { (samples.count + ParakeetConstants.samplesPerFrame - 1) / ParakeetConstants.samplesPerFrame }

        /// Adds recorded audio and transcribes every segment whose cut can now be decided.
        public func append(_ newSamples: [Float]) throws {
            precondition(!finished, "session already finished")
            samples += newSamples
            try advance(final: false)
        }

        /// Transcribes what is left and returns the whole recording's text.
        public func finish() throws -> Transcription {
            precondition(!finished, "session already finished")
            finished = true
            if samples.count >= ParakeetConstants.minimumSamples { try advance(final: true) }
            if ParakeetTranscriber.timing {
                let ms = { (s: Double) in Int(s * 1000) }
                FileHandle.standardError.write(Data("timing: vad \(ms(vadSeconds)) ms, \(segments) segments: mel \(ms(melSeconds)) ms, encoder \(ms(encodeSeconds)) ms, decode \(ms(decodeSeconds)) ms over \(totalFrames) frames, \(tokens.count) tokens; \(DecodeProfile.summary)\n".utf8))
            }
            return Transcription(text: transcriber.models.vocabulary.text(for: tokens), segments: segments, tokens: emitted)
        }

        private func advance(final: Bool) throws {
            let segmenter = transcriber.segmenter
            let context = ParakeetTranscriber.contextFrames
            while true {
                let total = totalFrames
                let remaining = total - segmentStart
                if final {
                    // The tail is decoded in one pass when it fits the window with its lead context.
                    // A recording that fits whole is decoded exactly as FluidAudio does, without VAD.
                    if remaining + min(context, segmentStart) <= ParakeetTranscriber.windowFrames {
                        guard remaining > 0 else { return }
                        let tailHasSpeech = try segmentStart == 0 || segmenter.hasSpeech(probabilities(upTo: total)[segmentStart..<total])
                        if tailHasSpeech { try process(segmentStart..<total) }
                        return
                    }
                } else if total < segmentStart + segmenter.maximumSegmentFrames + context {
                    return  // not enough audio yet to place the cut and hear its right context
                }
                let windowEnd = segmentStart + segmenter.maximumSegmentFrames
                let probabilities = try probabilities(upTo: min(total, windowEnd + context))
                let cut = segmenter.cut(in: probabilities, from: segmentStart, windowEnd: windowEnd)
                if segmenter.hasSpeech(probabilities[segmentStart..<cut]) { try process(segmentStart..<cut) }
                segmentStart = cut
            }
        }

        /// Speech probabilities for frames `0..<limit`: cached whole blocks, plus the current partial
        /// block evaluated on the spot. Frames the VAD did not cover count as speech.
        private func probabilities(upTo limit: Int) throws -> [Float] {
            let block = ParakeetConstants.windowSamples
            let t0 = Date()
            while (cachedBlocks + 1) * block <= samples.count {
                vadCache += try transcriber.speechProbabilities(block: samples[(cachedBlocks * block)..<((cachedBlocks + 1) * block)])
                cachedBlocks += 1
            }
            var result = vadCache
            if result.count < limit, cachedBlocks * block < samples.count {
                result += try transcriber.speechProbabilities(block: samples[(cachedBlocks * block)...])
            }
            vadSeconds += Date().timeIntervalSince(t0)
            if result.count < limit { result += Array(repeating: 1, count: limit - result.count) }
            return result
        }

        /// Encodes `range` with context on both sides and decodes it, carrying the decoder state.
        private func process(_ range: Range<Int>) throws {
            let context = ParakeetTranscriber.contextFrames
            let startFrame = max(range.lowerBound, resumeFrame)
            guard startFrame < range.upperBound else { return }
            let lead = min(context, startFrame)
            let tail = min(context, totalFrames - range.upperBound)
            let windowStart = startFrame - lead
            let start = windowStart * ParakeetConstants.samplesPerFrame
            let end = min((range.upperBound + tail) * ParakeetConstants.samplesPerFrame, samples.count)
            guard end - start >= ParakeetConstants.minimumSamples else { return }

            let t0 = Date()
            let mel = try transcriber.preprocess(samples[start..<end])
            let t1 = Date()
            let frames = try transcriber.encode(mel)
            let t2 = Date()
            let result = try transcriber.decoder.decode(frames: frames, range: lead..<(lead + range.upperBound - startFrame), state: &state)
            let t3 = Date()
            melSeconds += t1.timeIntervalSince(t0); encodeSeconds += t2.timeIntervalSince(t1); decodeSeconds += t3.timeIntervalSince(t2)

            // Emission timing shifts a little with context, so a segment can start by re-emitting the
            // words the previous one ended with. Drop that overlap.
            let overlap = ParakeetTranscriber.overlap(tail: tokens, head: result.tokens)
            tokens += result.tokens.dropFirst(overlap)
            emitted += zip(result.tokens, result.frames).dropFirst(overlap).map { (id: $0.0, frame: windowStart + $0.1) }
            resumeFrame = windowStart + result.stopFrame
            segments += 1
        }
    }

    /// Longest suffix of `tail` (up to 8 tokens) that `head` begins with.
    static func overlap(tail: [Int], head: [Int]) -> Int {
        for length in stride(from: min(8, tail.count, head.count), through: 1, by: -1)
        where Array(tail.suffix(length)) == Array(head.prefix(length)) {
            return length
        }
        return 0
    }

    // MARK: Models

    struct Mel {
        let mel: MLMultiArray
        let length: MLMultiArray
        /// Encoder frames the audio actually covers: ceil(samples / 1280).
        let frames: Int
    }

    func preprocess(_ samples: ArraySlice<Float>) throws -> Mel {
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

    func encode(_ mel: Mel) throws -> EncoderFrames {
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

    /// VAD probabilities for one block of at most 15 s; frames the block covers but the head did not report count as speech.
    func speechProbabilities(block: ArraySlice<Float>) throws -> [Float] {
        let frames = (block.count + ParakeetConstants.samplesPerFrame - 1) / ParakeetConstants.samplesPerFrame
        guard let vad = models.vad else { return Array(repeating: 1, count: frames) }
        let mel = try preprocess(block)
        let output = try vad.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": MLFeatureValue(multiArray: mel.mel)]))
        let probabilities = try output.array("speech_probability")
        var result = Array(UnsafeBufferPointer(start: probabilities.floatPointer, count: min(frames, probabilities.count)))
        if result.count < frames { result += Array(repeating: 1, count: frames - result.count) }
        return result
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
}
