import AppKit
import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: ShortcutSettings

    var body: some View {
        Form {
            Section {
                LabeledContent("Start and stop dictation") {
                    ShortcutRecorder(settings: settings)
                }
            } footer: {
                Text("Tap a modifier key such as Right ⌥ on its own, or press a key combination. Esc always discards the current recording.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 440)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// A button that captures the next lone modifier tap or key combination as the shortcut.
private struct ShortcutRecorder: View {
    @ObservedObject var settings: ShortcutSettings
    @State private var monitors: [Any] = []
    @State private var pendingModifier: UInt16?
    @State private var rejected = false

    var body: some View {
        Button {
            settings.isRecording ? stop() : start()
        } label: {
            Text(settings.isRecording ? "Press a key…" : settings.shortcut.label)
                .frame(minWidth: 140)
        }
        .foregroundStyle(rejected ? .red : .primary)
        .help(rejected ? "A plain key would block typing. Add a modifier, or use a function key." : "Click, then tap a modifier or press a combination.")
        .onDisappear(perform: stop)
    }

    private func start() {
        settings.isRecording = true
        rejected = false
        pendingModifier = nil
        let keys = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            pendingModifier = nil
            if event.keyCode == 53, event.modifierFlags.intersection([.command, .option, .control, .shift]).isEmpty {
                stop()
            } else if let chord = Chord.from(event) {
                settings.shortcut = .chord(chord)
                stop()
            } else {
                rejected = true
            }
            return nil
        }
        let flags = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
            guard let key = ModifierKey(keyCode: event.keyCode) else { return event }
            if event.modifierFlags.contains(key.flag) {
                pendingModifier = key.keyCode
            } else if pendingModifier == key.keyCode {
                settings.shortcut = .tap(key)
                stop()
            }
            return event
        }
        monitors = [keys, flags].compactMap { $0 }
    }

    private func stop() {
        for monitor in monitors { NSEvent.removeMonitor(monitor) }
        monitors = []
        settings.isRecording = false
    }
}
