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

    // MARK: Decoder weights for the CPU decoder

    public static let checkpointRepository = "moondream/parakeet-ultra"
    static let checkpointURL = URL(string: "https://huggingface.co/\(checkpointRepository)/resolve/main/model.safetensors")!
    static let decoderTensorPrefixes = ["decoder.", "joint.", "encoder_projector"]

    public static var decoderWeightsURL: URL {
        ownDirectory.deletingLastPathComponent().appendingPathComponent("parakeet-ultra-decoder.safetensors")
    }

    /// Returns a local safetensors file holding only the prediction network, projections and joint head
    /// (about 36 MB), fetched once from the 1.2 GB checkpoint with HTTP range requests.
    public static func locateDecoderWeights(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let destination = decoderWeightsURL
        if FileManager.default.fileExists(atPath: destination.path) { return destination }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let headerLength = Int(try await fetch(range: 0..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        let header = try JSONSerialization.jsonObject(with: try await fetch(range: 8..<(8 + headerLength))) as? [String: Any] ?? [:]
        let base = 8 + headerLength
        var selected: [(name: String, entry: [String: Any], offsets: [Int])] = []
        for (name, value) in header where name != "__metadata__" && decoderTensorPrefixes.contains(where: name.hasPrefix) {
            guard let entry = value as? [String: Any], let offsets = entry["data_offsets"] as? [Int] else { continue }
            selected.append((name, entry, offsets))
        }
        selected.sort { $0.offsets[0] < $1.offsets[0] }
        let totalBytes = Double(selected.reduce(0) { $0 + $1.offsets[1] - $1.offsets[0] })

        var outHeader: [String: Any] = [:]
        var blobs = Data()
        for (name, entry, offsets) in selected {
            let blob = try await fetch(range: (base + offsets[0])..<(base + offsets[1]))
            outHeader[name] = ["dtype": entry["dtype"]!, "shape": entry["shape"]!, "data_offsets": [blobs.count, blobs.count + blob.count]]
            blobs.append(blob)
            progress(Double(blobs.count) / totalBytes)
        }
        var encoded = try JSONSerialization.data(withJSONObject: outHeader)
        encoded.append(Data(repeating: 0x20, count: (8 - encoded.count % 8) % 8))
        var file = Data()
        file.append(withUnsafeBytes(of: UInt64(encoded.count).littleEndian) { Data($0) })
        file.append(encoded)
        file.append(blobs)
        try file.write(to: destination, options: .atomic)
        return destination
    }

    private static func fetch(range: Range<Int>) async throws -> Data {
        var request = URLRequest(url: checkpointURL)
        request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 206, data.count == range.count else {
            throw ParakeetError.download("Range request to \(checkpointRepository) failed")
        }
        return data
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
