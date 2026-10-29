import Foundation

/// Appends timestamped lines to ~/Library/Logs/Paraketto.log so a run launched from Finder
/// or `open` can still be inspected. The file is truncated at every launch and never
/// receives dictated text, only timings, counts and errors.
enum Log {
    static let fileURL = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/Paraketto.log")

    private static let queue = DispatchQueue(label: "io.lfrati.paraketto.log")
    private static let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()

    static func startSession() {
        queue.async { try? Data().write(to: fileURL) }
    }

    static func info(_ message: String) {
        let line = "\(formatter.string(from: Date())) \(message)\n"
        NSLog("%@", message)
        queue.async {
            guard let data = line.data(using: .utf8) else { return }
            if let handle = try? FileHandle(forWritingTo: fileURL) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
            } else {
                try? data.write(to: fileURL)
            }
        }
    }
}
