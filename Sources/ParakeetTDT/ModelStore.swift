import Foundation

/// Finds or downloads the Core ML export of Parakeet Ultra.
///
/// Looks in this app's own Application Support folder first, then in FluidAudio's cache
/// (so a machine that already has the models does not download 613 MB again), and otherwise
/// fetches every file of the Hugging Face repository with plain HTTPS.
public enum ModelStore {
    public static let repository = "FluidInference/parakeet-ultra-coreml"

    static let requiredEntries = [
        "Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecisionv3.mlmodelc",
        "parakeet_v3_vocab.json",
    ]

    public static var ownDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Paraketto/models/parakeet-ultra", isDirectory: true)
    }

    static var fluidAudioDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models/parakeet-ultra", isDirectory: true)
    }

    public static func isComplete(_ directory: URL) -> Bool {
        requiredEntries.allSatisfy { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// Returns a directory with all model files, downloading into `ownDirectory` when needed.
    /// `progress` receives the downloaded fraction, 0...1, by bytes.
    public static func locate(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        for candidate in [ownDirectory, fluidAudioDirectory] where isComplete(candidate) {
            return candidate
        }
        try await download(into: ownDirectory, progress: progress)
        return ownDirectory
    }

    /// Our multi-window encoder export (built with `make encoder-export`), if present.
    public static var ownEncoderURL: URL? {
        let url = ownDirectory.deletingLastPathComponent().appendingPathComponent("parakeet-ultra-encoder/Encoder.mlmodelc")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Our fused decoder+joint export (built with `make decoder-export`), if present.
    public static var ownFusedDecoderURL: URL? {
        let url = ownDirectory.deletingLastPathComponent().appendingPathComponent("parakeet-ultra-decoder/DecoderJoint.mlmodelc")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private struct Sibling: Decodable {
        let rfilename: String
        let size: Int?
    }

    private struct RepositoryInfo: Decodable {
        let siblings: [Sibling]
    }

    static func download(into directory: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        let api = URL(string: "https://huggingface.co/api/models/\(repository)?blobs=true")!
        let (data, response) = try await URLSession.shared.data(from: api)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw ParakeetError.download("Could not list \(repository)")
        }
        let files = try JSONDecoder().decode(RepositoryInfo.self, from: data).siblings
            .filter { file in requiredEntries.contains { file.rfilename.hasPrefix($0) } }
        let totalBytes = Double(files.reduce(0) { $0 + ($1.size ?? 0) })
        var doneBytes = 0.0

        for file in files {
            let destination = directory.appendingPathComponent(file.rfilename)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            let url = URL(string: "https://huggingface.co/\(repository)/resolve/main/\(file.rfilename)")!
            let (temporary, fileResponse) = try await URLSession.shared.download(from: url)
            guard (fileResponse as? HTTPURLResponse)?.statusCode == 200 else {
                throw ParakeetError.download("HTTP error fetching \(file.rfilename)")
            }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: temporary, to: destination)
            doneBytes += Double(file.size ?? 0)
            if totalBytes > 0 { progress(doneBytes / totalBytes) }
        }
        guard isComplete(directory) else {
            throw ParakeetError.download("Repository is missing expected files")
        }
    }
}
