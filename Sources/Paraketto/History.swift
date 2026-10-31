import Combine
import Foundation

/// One dictation: transcript, where it went, and the recording that produced it.
struct HistoryEntry: Codable, Identifiable, Equatable {
    enum Delivery: String, Codable {
        case pasted
        case copied
        case keptInHistory
        case imported
    }

    let id: UUID
    let date: Date
    var text: String
    let destination: String
    let duration: TimeInterval
    var inferenceMilliseconds: Int
    var delivery: Delivery
    var sourceFilename: String? = nil

    var wordCount: Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }
}

/// Persists entries as `<id>.json` plus `<id>.wav` under Application Support.
@MainActor
final class HistoryStore: ObservableObject {
    @Published private(set) var entries: [HistoryEntry] = []

    let directory: URL

    init(directory: URL? = nil) {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        self.directory = directory ?? support.appendingPathComponent("Paraketto/history", isDirectory: true)
        try? FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true)
        entries = loadAll()
    }

    func add(_ entry: HistoryEntry, recording: RecordingFile) throws {
        let metadata = try JSONEncoder().encode(entry)
        try recording.move(to: audioURL(for: entry))
        do {
            try metadata.write(to: jsonURL(for: entry), options: .atomic)
        } catch {
            try? FileManager.default.removeItem(at: audioURL(for: entry))
            throw error
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
}
