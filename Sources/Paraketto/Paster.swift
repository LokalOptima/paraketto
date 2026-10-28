import AppKit
import ApplicationServices
import Carbon.HIToolbox

/// Delivers text into the frontmost app: pasteboard write, synthetic Cmd+V,
/// then the previous pasteboard contents are restored once the target has had time to read it.
enum Paster {
    private static let restoreDelay: Duration = .seconds(1)

    static var isAccessibilityTrusted: Bool {
        AXIsProcessTrusted()
    }

    /// Shows the one-time system prompt if Accessibility has not been granted yet.
    static func requestAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    /// Opens the Accessibility pane. Unlike the prompt, this works every time.
    static func openAccessibilitySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
    }

    enum Outcome {
        case pasted
        case copiedOnly
    }

    @MainActor
    static func deliver(_ text: String) async -> Outcome {
        let pasteboard = NSPasteboard.general
        let previous = snapshot(pasteboard)

        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        let ourChangeCount = pasteboard.changeCount

        guard isAccessibilityTrusted else { return .copiedOnly }

        await waitForModifiersReleased()
        sendCommandV()

        Task {
            try? await Task.sleep(for: restoreDelay)
            // Leave the pasteboard alone if the user or another app wrote to it in the meantime.
            guard pasteboard.changeCount == ourChangeCount else { return }
            restore(previous, to: pasteboard)
        }
        return .pasted
    }

    /// The hotkey's modifiers are usually still held when the recording key is released.
    /// Pasting while they are down would turn Cmd+V into Cmd+Ctrl+Opt+V in the target app.
    private static func waitForModifiersReleased() async {
        let modifiers: CGEventFlags = [.maskCommand, .maskControl, .maskAlternate, .maskShift]
        for _ in 0..<50 {
            if CGEventSource.flagsState(.combinedSessionState).intersection(modifiers).isEmpty { return }
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    private static func sendCommandV() {
        let source = CGEventSource(stateID: .combinedSessionState)
        let keyV = CGKeyCode(kVK_ANSI_V)
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyV, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: keyV, keyDown: false)
        else { return }
        // Carry the character explicitly so non-ANSI layouts (Dvorak, Colemak…) still see Cmd+V.
        var character: [UniChar] = [0x76]
        for event in [down, up] {
            event.flags = .maskCommand
            event.keyboardSetUnicodeString(stringLength: 1, unicodeString: &character)
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }

    private static func snapshot(_ pasteboard: NSPasteboard) -> [[NSPasteboard.PasteboardType: Data]] {
        (pasteboard.pasteboardItems ?? []).map { item in
            var entry: [NSPasteboard.PasteboardType: Data] = [:]
            for type in item.types {
                if let data = item.data(forType: type) { entry[type] = data }
            }
            return entry
        }
    }

    private static func restore(_ items: [[NSPasteboard.PasteboardType: Data]], to pasteboard: NSPasteboard) {
        pasteboard.clearContents()
        guard !items.isEmpty else { return }
        let restored = items.map { entry -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in entry { item.setData(data, forType: type) }
            return item
        }
        pasteboard.writeObjects(restored)
    }
}
