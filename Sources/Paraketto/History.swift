import AVFoundation
import Foundation

/// One dictation: transcript, where it went, and the recording that produced it.
struct HistoryEntry: Codable, Identifiable, Equatable {
    enum Delivery: String, Codable {
        case pasted
        case copied
        case keptInHistory
    }

    let id: UUID
    let date: Date
    var text: String
    let destination: String
    let duration: TimeInterval
    var inferenceMilliseconds: Int
    var delivery: Delivery

    var wordCount: Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }
}

/// Persists entries as `<id>.json` plus `<id>.wav` under Application Support.
@MainActor
final class HistoryStore: ObservableObject {
    @Published private(set) var entries: [HistoryEntry] = []

    let directory: URL

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        directory = support.appendingPathComponent("Paraketto/history", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        entries = loadAll()
    }

    func add(_ entry: HistoryEntry, samples: [Float]) {
        do {
            try WAVFile.data(samples: samples, sampleRate: Int(AudioRecorder.sampleRate)).write(to: audioURL(for: entry))
            try JSONEncoder().encode(entry).write(to: jsonURL(for: entry))
        } catch {
            Log.info("History save failed: \(error)")
        }
        entries.insert(entry, at: 0)
    }

    func remove(_ entry: HistoryEntry) {
        try? FileManager.default.removeItem(at: jsonURL(for: entry))
        try? FileManager.default.removeItem(at: audioURL(for: entry))
        entries.removeAll { $0.id == entry.id }
    }

    func audioURL(for entry: HistoryEntry) -> URL {
        directory.appendingPathComponent("\(entry.id.uuidString).wav")
    }

    private func jsonURL(for entry: HistoryEntry) -> URL {
        directory.appendingPathComponent("\(entry.id.uuidString).json")
    }

    private func loadAll() -> [HistoryEntry] {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in (try? Data(contentsOf: url)).flatMap { try? decoder.decode(HistoryEntry.self, from: $0) } }
            .sorted { $0.date > $1.date }
    }

    // MARK: Stats

    var totalWords: Int { entries.reduce(0) { $0 + $1.wordCount } }

    /// Words per minute of recorded speech across all entries.
    var wordsPerMinute: Int {
        let minutes = entries.reduce(0) { $0 + $1.duration } / 60
        return minutes > 0 ? Int((Double(totalWords) / minutes).rounded()) : 0
    }
}

/// Minimal PCM16 WAV writer and reader for the recorder's 16 kHz mono float samples.
enum WAVFile {
    static func data(samples: [Float], sampleRate: Int) -> Data {
        let pcm = samples.map { Int16(max(-1, min(1, $0)) * Float(Int16.max)) }
        let dataSize = UInt32(pcm.count * 2)
        var header = Data(capacity: 44)
        header.append(contentsOf: Array("RIFF".utf8))
        header.append(le(36 + dataSize))
        header.append(contentsOf: Array("WAVEfmt ".utf8))
        header.append(le(UInt32(16)))
        header.append(le(UInt16(1)))                       // PCM
        header.append(le(UInt16(1)))                       // mono
        header.append(le(UInt32(sampleRate)))
        header.append(le(UInt32(sampleRate * 2)))          // byte rate
        header.append(le(UInt16(2)))                       // block align
        header.append(le(UInt16(16)))                      // bits per sample
        header.append(contentsOf: Array("data".utf8))
        header.append(le(dataSize))
        return pcm.withUnsafeBufferPointer { header + Data(buffer: $0) }
    }

    private static func le<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }
}

/// Plays one history recording at a time.
@MainActor
final class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var playingID: UUID?
    private var player: AVAudioPlayer?

    func toggle(_ entry: HistoryEntry, url: URL) {
        if playingID == entry.id {
            stop()
            return
        }
        stop()
        guard let player = try? AVAudioPlayer(contentsOf: url) else { return }
        player.delegate = self
        self.player = player
        playingID = entry.id
        player.play()
    }

    func stop() {
        player?.stop()
        player = nil
        playingID = nil
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stop() }
    }
}
