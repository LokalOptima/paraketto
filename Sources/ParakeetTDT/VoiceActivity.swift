import Foundation

/// Cuts a recording longer than one 15 s window into segments at pauses found by the
/// model's own voice-activity head, following the rules moondream describes for Photon
/// (as reconstructed in parakeet.cpp), with a 15 s cap instead of 30 s because that is
/// the exported encoder's window.
///
/// Input is one speech probability per 80 ms encoder frame over the whole recording.
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

    struct Segment: Equatable {
        let startFrame: Int
        let endFrame: Int  // exclusive
    }

    func segments(speechProbabilities: [Float]) -> [Segment] {
        let total = speechProbabilities.count
        guard total > maximumSegmentFrames else { return [Segment(startFrame: 0, endFrame: total)] }

        var speech = speechProbabilities.map { $0 >= threshold }
        bridge(&speech, shorterThan: smoothingFrames + 1, value: false)  // fill one-frame dips
        bridge(&speech, shorterThan: smoothingFrames + 1, value: true)   // drop one-frame blips
        let pauses = runs(of: false, in: speech).filter { $0.count >= minimumPauseFrames }

        var result: [Segment] = []
        var start = 0
        while start < total {
            let remaining = total - start
            if remaining <= maximumSegmentFrames {
                append(Segment(startFrame: start, endFrame: total), speech: speech, to: &result)
                break
            }
            let windowEnd = start + maximumSegmentFrames
            let earliest = start + minimumSegmentFrames
            // Prefer the last pause fully inside the window, then the last whose midpoint is inside,
            // then the quietest frame near the window end (a dip too short to count as a pause).
            let cut: Int
            if let pause = pauses.last(where: { $0.lowerBound >= earliest && $0.upperBound <= windowEnd }) {
                cut = (pause.lowerBound + pause.upperBound) / 2
            } else if let pause = pauses.last(where: { (earliest..<windowEnd).contains(($0.lowerBound + $0.upperBound) / 2) }) {
                cut = (pause.lowerBound + pause.upperBound) / 2
            } else {
                let searchStart = max(earliest, windowEnd - fallbackSearchFrames)
                cut = (searchStart..<windowEnd).min { speechProbabilities[$0] < speechProbabilities[$1] } ?? windowEnd
            }
            append(Segment(startFrame: start, endFrame: cut), speech: speech, to: &result)
            start = cut
        }
        return result
    }

    /// Segments without any speech frame are dropped.
    private func append(_ segment: Segment, speech: [Bool], to result: inout [Segment]) {
        guard segment.endFrame > segment.startFrame, speech[segment.startFrame..<segment.endFrame].contains(true) else { return }
        result.append(segment)
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
