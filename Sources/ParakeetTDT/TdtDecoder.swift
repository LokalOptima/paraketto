import CoreML
import Foundation

/// LSTM state of the prediction network plus the last emitted token, for one segment.
public struct DecoderState {
    let hidden: MLMultiArray
    let cell: MLMultiArray
    /// The decoder's projection for `lastToken`, consumed by the joint network.
    let step: MLMultiArray
    var lastToken: Int?

    public init() throws {
        hidden = try .float32([ParakeetConstants.decoderLayers, 1, ParakeetConstants.decoderHiddenSize])
        cell = try .float32([ParakeetConstants.decoderLayers, 1, ParakeetConstants.decoderHiddenSize])
        step = try .float32([1, ParakeetConstants.decoderHiddenSize, 1])
    }
}

/// Greedy Token-and-Duration Transducer decoding over one encoder output.
///
/// For each encoder frame the joint network returns a token and a duration bin. A blank
/// token skips `duration` frames without touching the LSTM. A real token is emitted, fed
/// to the LSTM, and the frame pointer advances by its duration (at least one frame once
/// `maxSymbolsPerFrame` tokens were emitted in place, so decoding always terminates).
/// Wall-clock split of a decode, printed with `PARAKEET_TIMING`: time inside Core ML calls
/// against the loop around them.
enum DecodeProfile {
    static var enabled = ProcessInfo.processInfo.environment["PARAKEET_TIMING"] != nil
    static var jointCalls = 0, fusedCalls = 0, decoderCalls = 0
    static var jointNanos: UInt64 = 0, fusedNanos: UInt64 = 0, decoderNanos: UInt64 = 0

    static func reset() {
        jointCalls = 0; fusedCalls = 0; decoderCalls = 0
        jointNanos = 0; fusedNanos = 0; decoderNanos = 0
    }

    static func measure<T>(_ nanos: inout UInt64, _ calls: inout Int, _ body: () throws -> T) rethrows -> T {
        guard enabled else { return try body() }
        let start = DispatchTime.now().uptimeNanoseconds
        defer { nanos += DispatchTime.now().uptimeNanoseconds - start; calls += 1 }
        return try body()
    }

    static var summary: String {
        let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
        return "joint \(jointCalls) calls \(ms(jointNanos)) ms, fused \(fusedCalls) calls \(ms(fusedNanos)) ms, decoder \(decoderCalls) calls \(ms(decoderNanos)) ms"
    }
}

struct TdtDecoder {
    static let debug = ProcessInfo.processInfo.environment["PARAKEET_DEBUG"] != nil
    let decoder: MLModel
    let joint: MLModel
    let blankToken: Int
    /// Optional fused model: one call advances the LSTM with a token and decides the next frame.
    let fusedStep: MLModel?
    /// Whether a token starts a new word ("▁" prefix). Used to finish a word that straddles a segment end.
    let isWordStart: (Int) -> Bool

    struct Result {
        let tokens: [Int]
        /// Encoder frame at which each token was emitted, parallel to `tokens`.
        let frames: [Int]
        /// The frame where decoding stopped: `range.upperBound`, or later if a word was finished past it.
        let stopFrame: Int
    }

    private let targets = try! MLMultiArray.int32(0, shape: [1, 1])
    private let fusedToken = try! MLMultiArray.int32(0, shape: [1, 1])
    private let targetLength = try! MLMultiArray.int32(1)
    private let encoderStep = try! MLMultiArray.float32([1, ParakeetConstants.encoderHiddenSize, 1])

    /// Decodes `range` of the encoder frames. Frames before the range are acoustic context only.
    /// Past the range, decoding continues only to finish the word in progress: it stops at the
    /// first token that starts a new word (without emitting it) or at the first blank, so the
    /// next segment can resume exactly there and no word is split across segments.
    func decode(frames: EncoderFrames, range: Range<Int>, state: inout DecoderState) throws -> Result {
        var tokens: [Int] = []
        var emitted: [Int] = []
        let range = range.clamped(to: 0..<frames.count)
        guard !range.isEmpty else { return Result(tokens: tokens, frames: emitted, stopFrame: range.lowerBound) }

        // A fresh utterance primes the prediction network with blank as start-of-sequence. A continued
        // one already holds the projection of its last token; stepping the LSTM again would drift it.
        if state.lastToken == nil {
            try runDecoder(token: blankToken, state: &state)
        }

        var frame = range.lowerBound
        var symbolsAtFrame = 0
        let jointInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: state.step),
        ])
        // A decision for `frame` already produced by the previous fused call, if any.
        var pending: (token: Int, bin: Int)?
        while frame < frames.count {
            let pastRange = frame >= range.upperBound
            let token: Int, bin: Int
            if let decision = pending {
                (token, bin) = decision
                pending = nil
            } else {
                frames.copyFrame(frame, into: encoderStep.floatPointer)
                let output = try DecodeProfile.measure(&DecodeProfile.jointNanos, &DecodeProfile.jointCalls) { try joint.prediction(from: jointInput) }
                token = try output.array("token_id").firstInt
                bin = try output.array("duration").firstInt
            }
            var duration = ParakeetConstants.durationBins[min(max(bin, 0), ParakeetConstants.durationBins.count - 1)]
            if Self.debug {
                FileHandle.standardError.write(Data("f=\(frame) tok=\(token) dur=\(duration)\n".utf8))
            }

            if token == blankToken {
                if pastRange { break }
                frame += max(duration, 1)
                symbolsAtFrame = 0
                continue
            }
            if pastRange && isWordStart(token) { break }
            tokens.append(token)
            emitted.append(frame)
            state.lastToken = token
            symbolsAtFrame += 1
            if duration == 0 && symbolsAtFrame >= ParakeetConstants.maxSymbolsPerFrame { duration = 1 }
            if duration > 0 { symbolsAtFrame = 0 }
            frame += duration
            if let fusedStep, frame < frames.count {
                // Advance the LSTM and decide the next frame in one call.
                pending = try runFused(fusedStep, token: token, frame: frame, frames: frames, state: &state)
            } else {
                try runDecoder(token: token, state: &state)
            }
        }
        return Result(tokens: tokens, frames: emitted, stopFrame: max(frame, range.upperBound))
    }

    /// Fused token step: LSTM update with `token`, then the joint decision for `frame`.
    private func runFused(_ model: MLModel, token: Int, frame: Int, frames: EncoderFrames, state: inout DecoderState) throws -> (token: Int, bin: Int) {
        fusedToken.dataPointer.assumingMemoryBound(to: Int32.self).pointee = Int32(token)
        frames.copyFrame(frame, into: encoderStep.floatPointer)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: fusedToken),
            "h_in": MLFeatureValue(multiArray: state.hidden),
            "c_in": MLFeatureValue(multiArray: state.cell),
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
        ])
        let options = MLPredictionOptions()
        options.outputBackings = ["h_out": state.hidden, "c_out": state.cell, "decoder": state.step]
        let output = try DecodeProfile.measure(&DecodeProfile.fusedNanos, &DecodeProfile.fusedCalls) { try model.prediction(from: input, options: options) }
        let h = try output.array("h_out")
        if h.dataPointer != state.hidden.dataPointer { h.copyFloats(count: state.hidden.count, to: state.hidden.floatPointer) }
        let c = try output.array("c_out")
        if c.dataPointer != state.cell.dataPointer { c.copyFloats(count: state.cell.count, to: state.cell.floatPointer) }
        let step = try output.array("decoder")
        if step.dataPointer != state.step.dataPointer { step.copyFloats(count: ParakeetConstants.decoderHiddenSize, to: state.step.floatPointer) }
        return (try output.array("token_id").firstInt, try output.array("duration").firstInt)
    }

    /// One LSTM step: updates hidden/cell state in place and refreshes `state.step` for the joint.
    private func runDecoder(token: Int, state: inout DecoderState) throws {
        targets.dataPointer.assumingMemoryBound(to: Int32.self).pointee = Int32(token)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_length": MLFeatureValue(multiArray: targetLength),
            "h_in": MLFeatureValue(multiArray: state.hidden),
            "c_in": MLFeatureValue(multiArray: state.cell),
        ])
        let options = MLPredictionOptions()
        options.outputBackings = ["h_out": state.hidden, "c_out": state.cell]
        let output = try DecodeProfile.measure(&DecodeProfile.decoderNanos, &DecodeProfile.decoderCalls) { try decoder.prediction(from: input, options: options) }
        // Output backings are a hint; copy if Core ML allocated fresh arrays instead.
        let h = try output.array("h_out")
        if h.dataPointer != state.hidden.dataPointer { h.copyFloats(count: state.hidden.count, to: state.hidden.floatPointer) }
        let c = try output.array("c_out")
        if c.dataPointer != state.cell.dataPointer { c.copyFloats(count: state.cell.count, to: state.cell.floatPointer) }
        try output.array("decoder").copyFloats(count: ParakeetConstants.decoderHiddenSize, to: state.step.floatPointer)
    }
}
