import SwiftUI

@main
struct ParakettoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra {
            MenuContent(controller: appDelegate.controller)
        } label: {
            MenuBarLabel(controller: appDelegate.controller)
        }
        .menuBarExtraStyle(.menu)
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

private struct MenuBarLabel: View {
    @ObservedObject var controller: DictationController

    var body: some View {
        Image(systemName: controller.state.symbolName)
    }
}

private struct MenuContent: View {
    @ObservedObject var controller: DictationController

    var body: some View {
        Text(controller.state.label)
        Text("Hold \(controller.hotkeyLabel) to dictate")
            .foregroundStyle(.secondary)

        if !Paster.isAccessibilityTrusted {
            Divider()
            Button("Grant Accessibility (needed to paste)…") {
                Paster.openAccessibilitySettings()
            }
        }

        if !controller.lastTranscript.isEmpty {
            Divider()
            Text(controller.lastTranscript)
                .lineLimit(3)
            Text(controller.lastTiming)
                .foregroundStyle(.secondary)
            Button("Copy Last Transcript") {
                controller.copyLastTranscript()
            }
        }

        Divider()
        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}
