import AppKit
import ApplicationServices
import Carbon.HIToolbox

/// The app that had focus when recording started, and therefore should receive the text.
struct Destination: Equatable {
    let name: String
    let bundleIdentifier: String?

    @MainActor
    static var frontmost: Destination? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        return Destination(name: app.localizedName ?? "Unknown app", bundleIdentifier: app.bundleIdentifier)
    }
}

/// Delivers text into the destination app: pasteboard write, then a synthetic Cmd+V.
/// The transcript stays on the pasteboard afterwards; restoring the previous contents on a
/// timer races slow readers, and the old contents are one undo away in most apps anyway.
enum Paster {
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
        /// Accessibility is not granted; the text is on the clipboard.
        case copiedOnly
        /// Focus moved to another app since recording started; nothing was pasted.
        case destinationChanged
    }

    @MainActor
    static func deliver(_ text: String, to destination: Destination?) async -> Outcome {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        if let destination, let current = Destination.frontmost, current != destination {
            return .destinationChanged
        }
        guard isAccessibilityTrusted else { return .copiedOnly }

        await waitForModifiersReleased()
        sendCommandV()
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
}
