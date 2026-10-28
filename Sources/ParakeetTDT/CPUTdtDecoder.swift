import Foundation

/// Greedy TDT decoding driven by `CPUDecoder`. Same contract as `TdtDecoder.decode`, with the
/// LSTM state kept inside the `CPUDecoder` across segments of one recording.
final class CPUTdtDecoder {
    private let network: CPUDecoder
    private let isWordStart: (Int) -> Bool
    private var started = false

    init(network: CPUDecoder, isWordStart: @escaping (Int) -> Bool) {
        self.network = network
        self.isWordStart = isWordStart
    }

    /// Call once per recording before the first segment.
    func reset() {
        network.reset()
        network.resetTiming()
        started = true
    }

    var timingSummary: String { network.timingSummary }

    func decode(frames: EncoderFrames, range: Range<Int>) -> TdtDecoder.Result {
        precondition(started, "reset() before decoding a recording")
        var tokens: [Int] = []
        var emitted: [Int] = []
        let range = range.clamped(to: 0..<frames.count)
        guard !range.isEmpty else { return .init(tokens: [], frames: [], stopFrame: range.lowerBound) }

        let projected = network.projectEncoder(frames)
        var frame = range.lowerBound
        var symbolsAtFrame = 0
        // Decisions computed ahead for the current decoder state; invalidated when a token is fed.
        var batchStart = 0
        var batch: [CPUDecoder.Decision] = []

        while frame < frames.count {
            if frame < batchStart || frame >= batchStart + batch.count {
                batchStart = frame
                batch = network.decide(projectedFrames: projected, start: frame, count: min(network.lookahead, frames.count - frame))
            }
            let decision = batch[frame - batchStart]
            let pastRange = frame >= range.upperBound
            var duration = ParakeetConstants.durationBins[min(max(decision.durationBin, 0), ParakeetConstants.durationBins.count - 1)]
            if TdtDecoder.debug {
                FileHandle.standardError.write(Data("f=\(frame) tok=\(decision.token) dur=\(duration) p=\(decision.probability)\n".utf8))
            }
            if decision.token == ParakeetConstants.blankToken {
                if pastRange { break }
                frame += max(duration, 1)
                symbolsAtFrame = 0
                continue
            }
            if pastRange && isWordStart(decision.token) { break }
            tokens.append(decision.token)
            emitted.append(frame)
            network.feed(decision.token)
            batch = []  // state changed
            symbolsAtFrame += 1
            if duration == 0 && symbolsAtFrame >= ParakeetConstants.maxSymbolsPerFrame { duration = 1 }
            if duration > 0 { symbolsAtFrame = 0 }
            frame += duration
        }
        return .init(tokens: tokens, frames: emitted, stopFrame: max(frame, range.upperBound))
    }
}
