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

        Window("History", id: "history") {
            HistoryView(store: appDelegate.controller.history, shortcuts: appDelegate.controller.shortcuts)
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

private struct MenuBarLabel: View {
    @ObservedObject var controller: DictationController

    var body: some View {
        Image(systemName: controller.state.symbolName)
    }
}

private struct MenuContent: View {
    @ObservedObject var controller: DictationController
    @ObservedObject private var shortcuts: ShortcutSettings
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    init(controller: DictationController) {
        self.controller = controller
        shortcuts = controller.shortcuts
    }

    var body: some View {
        Text(controller.state.label)
        Text("Start and stop: \(shortcuts.shortcut.label). Esc discards.")
            .foregroundStyle(.secondary)

        if !Paster.isAccessibilityTrusted {
            Divider()
            Button("Grant Accessibility (needed to paste)…") {
                Paster.openAccessibilitySettings()
            }
        }

        Divider()
        Button("History…") {
            openWindow(id: "history")
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
        .keyboardShortcut("h")
        Button("Copy Last Transcript") {
            controller.copyLastTranscript()
        }
        .disabled(controller.lastTranscript == nil)
        Button("Settings…") {
            openSettings()
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
        .keyboardShortcut(",")

        Divider()
        Button("Quit Paraketto") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}
