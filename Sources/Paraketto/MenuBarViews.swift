import SwiftUI
import ParakeetTDT

struct MenuBarLabel: View {
    @ObservedObject var controller: DictationController
    @StateObject private var secondaryClick = MenuBarSecondaryClick()
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Group {
            switch controller.state {
            case .idle, .notice:
                Image(nsImage: ParakeetArtwork.menuIcon)
            case .recording:
                Image(nsImage: ParakeetArtwork.recordingIcon)
            default:
                Image(systemName: controller.state.symbolName)
            }
        }
        .accessibilityLabel("Paraketto: \(controller.state.label)")
        .onAppear {
            secondaryClick.start {
                openWindow(id: "history")
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
        }
        .onDisappear { secondaryClick.stop() }
    }
}

struct MenuContent: View {
    @ObservedObject var controller: DictationController
    @ObservedObject private var shortcuts: ShortcutSettings
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    init(controller: DictationController) {
        self.controller = controller
        shortcuts = controller.shortcuts
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                if controller.state == .loadingModel {
                    ProgressView()
                        .controlSize(.mini)
                        .accessibilityLabel(statusLabel)
                }
                Text(statusLabel)
                    .fontWeight(.medium)
                    .foregroundStyle(controller.state == .idle ? Color(nsColor: .systemGreen) : .primary)
                    .lineLimit(1)
                    .layoutPriority(1)
                Spacer(minLength: 8)
                Text(shortcuts.shortcut.label)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)

            if case .downloadingModel(let fraction) = controller.state {
                HStack(spacing: 8) {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                        .accessibilityLabel(statusLabel)
                    Text(fraction, format: .percent.precision(.fractionLength(0)))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.top, 8)
            }

            RecognitionModePicker(selection: Binding(get: { controller.selectedModel }, set: { controller.selectModel($0) }),
                                  isEnabled: controller.canChangeModel)
                .padding(8)

            if let error = controller.modelError {
                Text(error).font(.caption).foregroundStyle(.red).padding(8)
                Button("Retry Mode Change") { controller.retryModelLoad() }
                    .disabled(!controller.canChangeModel)
            }

            if !Paster.isAccessibilityTrusted {
                Divider().padding(.vertical, 4)
                Button("Grant Accessibility…") { Paster.openAccessibilitySettings() }
            }

            Divider().padding(.vertical, 4)
            Button {
                openWindow(id: "history")
                NSApplication.shared.activate(ignoringOtherApps: true)
            } label: {
                Text("Transcripts")
            }
            Button {
                openSettings()
                NSApplication.shared.activate(ignoringOtherApps: true)
            } label: {
                MenuActionLabel(title: "Settings", shortcut: "⌘,")
            }
            .keyboardShortcut(",")

            Divider().padding(.vertical, 4)
            Button {
                NSApplication.shared.terminate(nil)
            } label: {
                MenuActionLabel(title: "Quit Paraketto", shortcut: "⌘Q")
            }
            .keyboardShortcut("q")
        }
        .buttonStyle(MenuActionStyle())
        .padding(6)
        .frame(width: 260)
    }

    private var statusLabel: String {
        let mode = controller.selectedModel.title.uppercased()
        switch controller.state {
        case .downloadingModel: return "Downloading \(mode)…"
        case .loadingModel: return "Loading \(mode)…"
        default: return controller.state.label
        }
    }
}
