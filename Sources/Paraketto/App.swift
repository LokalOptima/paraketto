import SwiftUI

@main
enum ParakettoMain {
    @MainActor
    static func main() {
        if CommandLine.arguments.contains("--prepare-models") {
            let cachedOnly = CommandLine.arguments.contains("--cached-only")
            Task.detached {
                do {
                    try await ModelPreparation.run(cachedOnly: cachedOnly)
                    exit(EXIT_SUCCESS)
                } catch {
                    FileHandle.standardError.write(Data("Model preparation failed: \(error.localizedDescription)\n".utf8))
                    exit(EXIT_FAILURE)
                }
            }
            dispatchMain()
        } else {
            ParakettoApp.main()
        }
    }
}

struct ParakettoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra {
            MenuContent(controller: appDelegate.controller)
        } label: {
            MenuBarLabel(controller: appDelegate.controller)
        }
        .menuBarExtraStyle(.window)

        Window("Transcripts", id: "history") {
            TranscriptsView(controller: appDelegate.controller)
        }
        .defaultSize(width: 560, height: 480)

        Settings {
            SettingsView(settings: appDelegate.controller.shortcuts)
        }
    }
}

/// Starts dictation at launch. The menu content only exists while the menu is open,
/// so the hotkey and model load cannot be tied to it.
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    @MainActor let controller = DictationController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        controller.start()
    }
}
