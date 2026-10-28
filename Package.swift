// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Paraketto",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "2faffc237b7b78ae80d938c52cf91700d6709bc5"
        )
    ],
    targets: [
        .executableTarget(
            name: "Paraketto",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/Paraketto",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
