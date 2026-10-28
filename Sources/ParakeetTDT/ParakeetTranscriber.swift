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
/// model's VAD head. Each segment is encoded with 2 s of its neighbours' audio on both sides and
/// decoded with a fresh decoder, as moondream's runtime does (an LSTM state carried over minutes of
/// speech drifts until whole segments decode to nothing), and neighbours are joined where their
/// transcripts of the shared audio agree.
///
/// A `Session` accepts audio while it is being recorded and transcribes each segment as soon
/// as a pause ends it, so at the end only the speech after the last pause is left.
/// `transcribe(_:)` is the same thing with all the audio appended at once.
public final class ParakeetTranscriber {
    static let windowFrames = ParakeetConstants.windowSamples / ParakeetConstants.samplesPerFrame
    static let timing = ProcessInfo.processInfo.environment["PARAKEET_TIMING"] != nil

    let models: ParakeetModels
    let decoder: TdtDecoder
    let segmenter: Segmenter
    private let audioBuffer = try! MLMultiArray.float32([1, ParakeetConstants.windowSamples])

    /// - Parameter commitPauseSeconds: silence that ends a segment while recording (0 waits for the window to fill).
    public init(models: ParakeetModels, commitPauseSeconds: Double = 0.8) throws {
        self.models = models
        let vocabulary = models.vocabulary
        decoder = TdtDecoder(decoder: models.decoder, joint: models.joint, fusedStep: models.fusedStep, isWordStart: { vocabulary.isWordStart($0) })
        var segmenter = Segmenter()
        segmenter.maximumSegmentFrames = Self.windowFrames - 2 * Self.overlapFrames
        segmenter.commitPauseFrames = Int((commitPauseSeconds / 0.08).rounded())
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
    ///
    /// Every segment is decoded whole with a fresh decoder. A cut at a pause lies in silence, so each
    /// side keeps the tokens on its side of it, with `contextFrames` of audio around for the encoder.
    /// A cut through speech (the window filled) is instead overlapped by `overlapFrames` on each side:
    /// both segments transcribe the audio around it, their tokens there are aligned with a longest
    /// common subsequence and joined at the matched token nearest the cut, so a word is neither lost
    /// nor repeated and the text near each window edge, where the encoder hears worst, is the other side's.
    public final class Session {
        private let transcriber: ParakeetTranscriber
        private var samples: [Float] = []
        /// Speech probabilities of the completed 15 s VAD blocks, one per encoder frame.
        private var vadCache: [Float] = []
        private var cachedBlocks = 0
        /// Global frame where the next segment starts.
        private var segmentStart = 0
        /// Final tokens with their global frames.
        private var committed: [Token] = []
        /// Tokens of the last segment from `overlapFrames` before its end on; settled by the next segment.
        private var provisional: [Token] = []
        /// End frame of the previous segment, and whether it ended in a cut through speech.
        private var boundary = 0
        private var boundaryIsOverlapped = false
        private var segments = 0
        private var finished = false
        /// Frames available when the VAD was last consulted; it is re-run every `vadCadenceFrames` of new audio.
        private var vadCheckedFrames = 0
        private static let vadCadenceFrames = 3  // 0.24 s
        private var melSeconds = 0.0, encodeSeconds = 0.0, decodeSeconds = 0.0, vadSeconds = 0.0

        init(transcriber: ParakeetTranscriber) {
            self.transcriber = transcriber
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
            let tokens = committed + provisional
            if ParakeetTranscriber.timing {
                let ms = { (s: Double) in Int(s * 1000) }
                FileHandle.standardError.write(Data("timing: vad \(ms(vadSeconds)) ms, \(segments) segments: mel \(ms(melSeconds)) ms, encoder \(ms(encodeSeconds)) ms, decode \(ms(decodeSeconds)) ms over \(totalFrames) frames, \(tokens.count) tokens; \(DecodeProfile.summary)\n".utf8))
            }
            return Transcription(text: transcriber.models.vocabulary.text(for: tokens.map(\.id)), segments: segments, tokens: tokens.map { (id: $0.id, frame: $0.frame) })
        }

        private func advance(final: Bool) throws {
            let segmenter = transcriber.segmenter
            let overlap = ParakeetTranscriber.overlapFrames
            while true {
                let leftContext = boundaryIsOverlapped ? overlap : ParakeetTranscriber.contextFrames
                let total = totalFrames
                let remaining = total - segmentStart
                if final {
                    // The tail is decoded in one pass when it fits the window with its left overlap.
                    // A recording that fits whole is decoded exactly as FluidAudio does, without VAD.
                    if remaining + min(leftContext, segmentStart) <= ParakeetTranscriber.windowFrames {
                        guard remaining > 0 else { return }
                        let tailHasSpeech = try segmentStart == 0 || segmenter.hasSpeech(probabilities(upTo: total)[segmentStart..<total])
                        if tailHasSpeech { try process(segmentStart..<total) }
                        return
                    }
                } else if total < segmentStart + segmenter.minimumSegmentFrames + segmenter.commitPauseFrames
                            || (total - vadCheckedFrames < Self.vadCadenceFrames && total < segmentStart + segmenter.maximumSegmentFrames + overlap) {
                    return  // too little new audio to look for a pause, and the window is not full
                }
                let windowEnd = segmentStart + segmenter.maximumSegmentFrames
                let probabilities = try probabilities(upTo: min(total, windowEnd + overlap))
                vadCheckedFrames = total
                // A pause ends the segment before it right away; its right overlap is whatever silence has been heard.
                if let cut = segmenter.pauseCut(in: probabilities, from: segmentStart), cut <= windowEnd {
                    try commit(segmentStart..<cut, probabilities: probabilities, overlapped: false)
                    continue
                }
                guard total >= windowEnd + overlap else { return }  // no pause yet; wait for the window and its overlap
                let cut = segmenter.cut(in: probabilities, from: segmentStart, windowEnd: windowEnd)
                try commit(segmentStart..<cut, probabilities: probabilities, overlapped: true)
            }
        }

        private func commit(_ range: Range<Int>, probabilities: [Float], overlapped: Bool) throws {
            let speech = transcriber.segmenter.hasSpeech(probabilities[range])
            if speech {
                try process(range, overlapped: overlapped)
            } else {
                // Nothing to reconcile with across a silent stretch.
                committed += provisional
                provisional = []
                boundary = range.upperBound
                boundaryIsOverlapped = false
            }
            if TdtDecoder.debug {
                FileHandle.standardError.write(Data("segment \(range) (\(overlapped ? "window" : "pause")) speech=\(speech) committed=\(committed.count) provisional=\(provisional.count)\n".utf8))
            }
            segmentStart = range.upperBound
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

        /// Encodes `range` with its neighbours' audio, decodes the whole window, and joins the result
        /// to the previous segment. `overlapped`: the segment ends in a cut through speech.
        private func process(_ range: Range<Int>, overlapped: Bool = false) throws {
            let overlap = ParakeetTranscriber.overlapFrames, context = ParakeetTranscriber.contextFrames
            let windowEnd = min(totalFrames, range.upperBound + (overlapped ? overlap : context))
            // The model occasionally returns nothing at all for a window that starts in the middle of a
            // word, so a window with speech that decodes empty is tried again from a later start.
            let starts = [boundaryIsOverlapped ? overlap : context, context, 0].map { max(0, range.lowerBound - $0) }
            var tokens: [Token] = []
            var windowStart = starts[0]
            for candidate in Array(NSOrderedSet(array: starts)) as! [Int] {
                windowStart = candidate
                tokens = try decodeWindow(candidate..<windowEnd)
                if !tokens.isEmpty { break }
            }
            if TdtDecoder.debug, boundaryIsOverlapped {
                FileHandle.standardError.write(Data("join at \(boundary): previous \(provisional.map { "\($0.id)@\($0.frame)" }) next \(tokens.prefix(40).map { "\($0.id)@\($0.frame)" })\n".utf8))
            }
            if boundaryIsOverlapped {
                let joined = ParakeetTranscriber.join(previous: provisional, next: tokens, boundary: boundary, nextStart: windowStart)
                committed += joined.settled
                tokens = joined.rest
            } else {
                tokens = tokens.filter { $0.frame >= range.lowerBound }
            }
            if overlapped {
                let split = tokens.firstIndex { $0.frame >= range.upperBound - overlap } ?? tokens.count
                committed += tokens[..<split]
                provisional = Array(tokens[split...])
            } else {
                committed += tokens.filter { $0.frame < range.upperBound }
                provisional = []
            }
            boundary = range.upperBound
            boundaryIsOverlapped = overlapped
            segments += 1
        }

        /// Encodes and greedily decodes the audio of `frames` with a fresh decoder; tokens carry global frames.
        private func decodeWindow(_ frames: Range<Int>) throws -> [Token] {
            let start = frames.lowerBound * ParakeetConstants.samplesPerFrame
            let end = min(frames.upperBound * ParakeetConstants.samplesPerFrame, samples.count)
            guard end - start >= ParakeetConstants.minimumSamples else { return [] }
            let t0 = Date()
            let mel = try transcriber.preprocess(samples[start..<end])
            let t1 = Date()
            let encoded = try transcriber.encode(mel)
            let t2 = Date()
            var state = try DecoderState()
            let result = try transcriber.decoder.decode(frames: encoded, range: 0..<encoded.count, state: &state)
            melSeconds += t1.timeIntervalSince(t0); encodeSeconds += t2.timeIntervalSince(t1); decodeSeconds += Date().timeIntervalSince(t2)
            return zip(result.tokens, result.frames).map { Token(id: $0.0, frame: frames.lowerBound + $0.1) }
        }
    }

    struct Token: Equatable {
        let id: Int
        let frame: Int
    }

    /// Overlap shared by neighbouring segments on each side of a cut through speech, in encoder frames (2 s).
    static let overlapFrames = 25
    /// Audio added on each side of a cut at a pause, for the encoder only, in encoder frames (~1 s).
    static let contextFrames = 12
    /// Tokens on both sides count as the same emission when their frames differ by at most this much.
    static let joinFrameTolerance = 4

    /// Joins the tail of one segment (`previous`, from `overlapFrames` before `boundary`) to the next
    /// segment's tokens. The longest common subsequence of the two in the overlap, matching ids at
    /// nearby frames, says where they agree; the join is at the matched pair nearest the boundary.
    /// Without agreement, each side keeps the tokens on its side of the boundary.
    static func join(previous: [Token], next: [Token], boundary: Int, nextStart: Int) -> (settled: [Token], rest: [Token]) {
        let head = next.prefix { $0.frame < boundary + overlapFrames }
        let a = previous, b = Array(head)
        var lengths = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in stride(from: a.count - 1, through: 0, by: -1) {
            for j in stride(from: b.count - 1, through: 0, by: -1) {
                let same = a[i].id == b[j].id && abs(a[i].frame - b[j].frame) <= joinFrameTolerance
                lengths[i][j] = same ? lengths[i + 1][j + 1] + 1 : max(lengths[i + 1][j], lengths[i][j + 1])
            }
        }
        var pairs: [(Int, Int)] = []
        var i = 0, j = 0
        while i < a.count, j < b.count {
            if a[i].id == b[j].id && abs(a[i].frame - b[j].frame) <= joinFrameTolerance && lengths[i][j] == lengths[i + 1][j + 1] + 1 {
                pairs.append((i, j)); i += 1; j += 1
            } else if lengths[i + 1][j] >= lengths[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        // A lone match far from the cut is more likely chance than agreement.
        if let (pi, pj) = pairs.min(by: { abs(a[$0.0].frame - boundary) < abs(a[$1.0].frame - boundary) }),
           abs(a[pi].frame - boundary) <= overlapFrames / 2 {
            return (Array(a[...pi]), Array(next[(pj + 1)...]))
        }
        // No agreement. If the next segment started too late to hear the overlap, or heard much less of
        // it than the previous one did, the previous one's words stand and the next continues after them.
        let previousPast = a.filter { $0.frame >= boundary }
        let nextBefore = head.filter { $0.frame >= nextStart }
        if nextStart >= boundary || nextBefore.count * 2 < previousPast.count {
            let last = a.last?.frame ?? boundary
            return (a, next.filter { $0.frame > last })
        }
        return (a.filter { $0.frame < boundary }, next.filter { $0.frame >= boundary })
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
