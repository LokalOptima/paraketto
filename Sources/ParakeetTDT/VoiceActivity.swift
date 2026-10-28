import Foundation

/// Places cuts in a recording longer than one 15 s window at pauses found by the model's
/// own voice-activity head, following the rules moondream describes for Photon (as
/// reconstructed in parakeet.cpp), with a 15 s cap instead of 30 s because that is the
/// exported encoder's window.
///
/// Input is one speech probability per 80 ms encoder frame.
struct Segmenter {
    var threshold: Float = 0.5
    /// Dips shorter than this inside speech are bridged, blips shorter than this are dropped.
    var smoothingFrames = 1  // 0.08 s
    /// A pause is a silent run at least this long.
    var minimumPauseFrames = 3  // 0.24 s
    /// Never cut closer than this to a segment's start.
    var minimumSegmentFrames = 13  // ~1 s
    /// Without a proper pause, the cut goes to the least speech-like frame within this many frames of the window end.
    var fallbackSearchFrames = 38  // ~3 s
    var maximumSegmentFrames = ParakeetConstants.windowSamples / ParakeetConstants.samplesPerFrame  // 187

    /// Where to end the segment starting at `start`, given probabilities for at least
    /// `0..<windowEnd`. Prefers the midpoint of the last pause fully inside the window, then
    /// the last pause whose midpoint is inside, then the quietest frame near the window end.
    /// Only frames up to `probabilities.count` are consulted, so a decision made while
    /// recording is the same one made afterwards with the whole recording.
    func cut(in probabilities: [Float], from start: Int, windowEnd: Int) -> Int {
        let speech = smoothed(probabilities[start...])
        let earliest = start + minimumSegmentFrames
        let pauses = runs(of: false, in: speech).map { ($0.lowerBound + start)..<($0.upperBound + start) }
            .filter { $0.count >= minimumPauseFrames }
        if let pause = pauses.last(where: { $0.lowerBound >= earliest && $0.upperBound <= windowEnd }) {
            return (pause.lowerBound + pause.upperBound) / 2
        }
        if let pause = pauses.last(where: { (earliest..<windowEnd).contains(($0.lowerBound + $0.upperBound) / 2) }) {
            return (pause.lowerBound + pause.upperBound) / 2
        }
        let searchStart = max(earliest, windowEnd - fallbackSearchFrames)
        return (searchStart..<windowEnd).min { probabilities[$0] < probabilities[$1] } ?? windowEnd
    }

    /// Whether any frame of the slice is speech after smoothing. Segments without speech are skipped.
    func hasSpeech(_ probabilities: ArraySlice<Float>) -> Bool {
        smoothed(probabilities).contains(true)
    }

    private func smoothed(_ probabilities: ArraySlice<Float>) -> [Bool] {
        var speech = probabilities.map { $0 >= threshold }
        bridge(&speech, shorterThan: smoothingFrames + 1, value: false)  // fill one-frame dips
        bridge(&speech, shorterThan: smoothingFrames + 1, value: true)   // drop one-frame blips
        return speech
    }

    private func runs(of value: Bool, in flags: [Bool]) -> [Range<Int>] {
        var runs: [Range<Int>] = []
        var index = 0
        while index < flags.count {
            guard flags[index] == value else { index += 1; continue }
            let start = index
            while index < flags.count, flags[index] == value { index += 1 }
            runs.append(start..<index)
        }
        return runs
    }

    /// Flips runs of `value` shorter than `length` to the opposite value.
    private func bridge(_ flags: inout [Bool], shorterThan length: Int, value: Bool) {
        for run in runs(of: value, in: flags) where run.count < length {
            for i in run { flags[i] = !value }
        }
    }
}
