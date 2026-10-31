import Foundation

/// Remembers successful preparation, not Core ML's private cache contents. macOS
/// may purge that cache independently, so a later "Loading" can still take longer.
struct ModelPreparationRecord {
    private let url: URL
    private let signature: Data?

    init(directory: URL, additionalModels: [URL], recordURL: URL,
         systemVersion: String = ProcessInfo.processInfo.operatingSystemVersionString) {
        url = recordURL
        signature = try? Self.signature(for: [directory] + additionalModels, systemVersion: systemVersion)
    }

    var isPrepared: Bool {
        guard let signature, let previous = try? Data(contentsOf: url) else { return false }
        return signature == previous
    }

    func markPrepared() throws {
        guard let signature else { return }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try signature.write(to: url, options: .atomic)
    }

    private static func signature(for directories: [URL], systemVersion: String) throws -> Data {
        let keys: [URLResourceKey] = [.isRegularFileKey, .contentModificationDateKey, .fileSizeKey]
        var entries = ["preparation-v1", systemVersion,
                       ProcessInfo.processInfo.environment["PARAKEET_UNITS"] ?? "ane"]
        for directory in directories {
            guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: keys) else {
                throw CocoaError(.fileReadNoSuchFile)
            }
            entries.append(directory.standardizedFileURL.path)
            for case let file as URL in files {
                let values = try file.resourceValues(forKeys: Set(keys))
                guard values.isRegularFile == true else { continue }
                entries.append("\(file.path)|\(values.fileSize ?? 0)|\(values.contentModificationDate?.timeIntervalSince1970 ?? 0)")
            }
        }
        return try JSONEncoder().encode(entries.sorted())
    }
}
