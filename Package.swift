// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Paraketto",
    platforms: [.macOS(.v14)],
    targets: [
        // Parakeet TDT inference with Core ML: model download, mel, encoder, greedy TDT decode, VAD segmentation.
        .target(
            name: "ParakeetTDT",
            path: "Sources/ParakeetTDT",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The menu bar dictation app.
        .executableTarget(
            name: "Paraketto",
            dependencies: ["ParakeetTDT"],
            path: "Sources/Paraketto",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Command-line transcription for testing and benchmarking against reference transcripts.
        .executableTarget(
            name: "paraketto-cli",
            dependencies: ["ParakeetTDT"],
            path: "Sources/ParakettoCLI",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
