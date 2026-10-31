import Foundation

/// Locates one mode's assets, downloading into a staging directory before installing them.
/// Existing Ultra caches remain usable. No recognition weights are shared between modes.
public struct ModelStore {
    public let model: ParakeetModel

    public init(model: ParakeetModel = .multi) { self.model = model }

    public var ownDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Paraketto/models/\(model.cacheName)", isDirectory: true)
    }

    private var fluidAudioDirectories: [URL] {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models", isDirectory: true)
        let names = model == .eng
            ? ["parakeet-tdt-0.6b-v2-coreml", "parakeet-tdt-0.6b-v2"]
            : ["parakeet-ultra", "parakeet-ultra-coreml"]
        return names.map { root.appendingPathComponent($0, isDirectory: true) }
    }

    /// Check actual model files, not just directories left by an interrupted download.
    public func isComplete(_ directory: URL) -> Bool {
        let fm = FileManager.default
        return model.requiredEntries.allSatisfy { entry in
            let leaves = entry.hasSuffix(".mlmodelc")
                ? ["coremldata.bin", "model.mil", "weights/weight.bin"].map { entry + "/" + $0 }
                : [entry]
            return leaves.allSatisfy { file in
                let url = directory.appendingPathComponent(file)
                guard let attributes = try? fm.attributesOfItem(atPath: url.path),
                      let size = attributes[.size] as? NSNumber else { return false }
                return size.intValue > 0
            }
        }
    }

    public var cachedDirectory: URL? {
        ([ownDirectory] + fluidAudioDirectories).first(where: isComplete)
    }

    public func locate(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        if let cachedDirectory { return cachedDirectory }
        try await download(into: ownDirectory, progress: progress)
        return ownDirectory
    }

    /// These optimized exports contain Ultra weights and must never be used with English v2.
    public var ownEncoderURL: URL? {
        guard model == .multi else { return nil }
        let url = ownDirectory.deletingLastPathComponent().appendingPathComponent("parakeet-ultra-encoder/Encoder.mlmodelc")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    public var ownFusedDecoderURL: URL? {
        guard model == .multi else { return nil }
        let url = ownDirectory.deletingLastPathComponent().appendingPathComponent("parakeet-ultra-decoder/DecoderJoint.mlmodelc")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private struct Sibling: Decodable {
        let rfilename: String
        let size: Int64
    }

    private struct RepositoryInfo: Decodable {
        let sha: String
        let siblings: [Sibling]
    }

    func download(into directory: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        let repository = model.repository
        let api = URL(string: "https://huggingface.co/api/models/\(repository)/revision/\(model.revision)?blobs=true")!
        let (data, response) = try await URLSession.shared.data(from: api)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw ParakeetError.download("Could not list \(repository)")
        }
        let info = try JSONDecoder().decode(RepositoryInfo.self, from: data)
        let files = info.siblings.filter { file in
            model.requiredEntries.contains { file.rfilename == $0 || file.rfilename.hasPrefix($0 + "/") }
        }
        guard !files.isEmpty else { throw ParakeetError.download("Repository contains no model files") }
        let totalBytes = Double(files.reduce(Int64(0)) { $0 + $1.size })
        var doneBytes = 0.0
        let fm = FileManager.default
        let staging = directory.deletingLastPathComponent().appendingPathComponent(".\(model.cacheName)-\(UUID().uuidString).download")
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        progress(0)

        for file in files {
            try Task.checkCancellation()
            guard !file.rfilename.split(separator: "/").contains(".."), !file.rfilename.hasPrefix("/") else {
                throw ParakeetError.download("Invalid model file path")
            }
            let destination = staging.appendingPathComponent(file.rfilename)
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Resolve every file at one revision, even if the upstream branch moves mid-download.
            let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(info.sha)/\(file.rfilename)")!
            let completedBytes = doneBytes
            let reporter = ModelFileDownload(expectedBytes: file.size) { bytesWritten in
                guard totalBytes > 0 else { return }
                progress((completedBytes + Double(min(bytesWritten, file.size))) / totalBytes)
            }
            let (temporary, fileResponse) = try await reporter.download(from: url)
            defer { try? fm.removeItem(at: temporary) }
            guard (fileResponse as? HTTPURLResponse)?.statusCode == 200,
                  (try fm.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value == file.size else {
                throw ParakeetError.download("Incomplete download: \(file.rfilename)")
            }
            try fm.moveItem(at: temporary, to: destination)
            doneBytes += Double(file.size)
            if totalBytes > 0 { progress(doneBytes / totalBytes) }
        }
        try Task.checkCancellation()
        guard isComplete(staging) else { throw ParakeetError.download("Repository is missing expected files") }
        // Only replace this mode's incomplete cache once its entire replacement is ready.
        if fm.fileExists(atPath: directory.path) { try fm.removeItem(at: directory) }
        try fm.moveItem(at: staging, to: directory)
    }
}
