import Foundation
import ParakeetTDT

/// Run through the signed app executable so Core ML prepares the same models, URLs,
/// configuration and application context used by interactive dictation.
enum ModelPreparation {
    static func run(cachedOnly: Bool) async throws {
        for mode in ParakeetModel.allCases {
            if cachedOnly, ModelStore(model: mode).cachedDirectory == nil {
                report("Skipping \(mode.title): not downloaded (use make prepare-models to download both modes).")
                continue
            }
            report("Preparing \(mode.title) for this Mac; the first preparation can take a minute…")
            let start = Date()
            // Release each recognizer before loading the next mode. This warms disk caches;
            // it does not leave a second app or resident model running after the build.
            try await prepare(mode)
            report(String(format: "%@ prepared in %.2f s", mode.title, Date().timeIntervalSince(start)))
        }
    }

    private static func prepare(_ mode: ParakeetModel) async throws {
        let recognizer = Transcriber()
        try await recognizer.load(model: mode) { phase in
            switch phase {
            case .downloading(let fraction):
                report(String(format: "%@ downloading %.0f%%", mode.title, fraction * 100))
            case .optimizing: report("Optimizing \(mode.title.uppercased())…")
            case .loading: report("Loading \(mode.title.uppercased())…")
            }
        }
    }

    private static func report(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
