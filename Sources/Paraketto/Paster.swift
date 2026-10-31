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
/// timer races slow readers.
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

    enum Outcome: Equatable {
        case pasted
        /// Automatic delivery is unavailable; the text is on the clipboard.
        case copiedOnly
        /// Focus moved to another app since recording started; nothing was pasted.
        case destinationChanged
    }

    /// Keep system effects injectable so focus changes and cancellation can be tested without typing.
    @MainActor
    struct Operations {
        var copy: (String) -> Bool
        var destination: () -> Destination?
        var accessibilityTrusted: () -> Bool
        var waitForModifiers: () async throws -> Bool
        var paste: () -> Bool

        static var live: Operations {
            Operations(copy: { text in
                NSPasteboard.general.clearContents()
                return NSPasteboard.general.setString(text, forType: .string)
            }, destination: { Destination.frontmost }, accessibilityTrusted: { isAccessibilityTrusted },
                       waitForModifiers: waitForModifiersReleased, paste: sendCommandV)
        }
    }

    @MainActor
    static func deliver(_ text: String, to destination: Destination?, using operations: Operations? = nil) async throws -> Outcome {
        let operations = operations ?? .live
        try Task.checkCancellation()
        guard operations.copy(text) else { throw DeliveryError.clipboardUnavailable }
        guard let destination else { return .copiedOnly }
        guard operations.destination() == destination else { return .destinationChanged }
        guard operations.accessibilityTrusted() else { return .copiedOnly }

        let released = try await operations.waitForModifiers()
        try Task.checkCancellation()
        // Focus may change during the wait. Recheck immediately before sending any keys.
        guard operations.destination() == destination else { return .destinationChanged }
        guard released, operations.paste() else { return .copiedOnly }
        return .pasted
    }

    /// The hotkey's modifiers are usually still held when the recording key is released.
    /// Pasting while they are down would turn Cmd+V into Cmd+Ctrl+Opt+V in the target app.
    private static func waitForModifiersReleased() async throws -> Bool {
        let modifiers: CGEventFlags = [.maskCommand, .maskControl, .maskAlternate, .maskShift]
        for _ in 0..<50 {
            try Task.checkCancellation()
            if CGEventSource.flagsState(.combinedSessionState).intersection(modifiers).isEmpty { return true }
            try await Task.sleep(for: .milliseconds(20))
        }
        return false
    }

    enum DeliveryError: LocalizedError {
        case clipboardUnavailable
        var errorDescription: String? { "Could not copy the transcript to the clipboard." }
    }

    private static func sendCommandV() -> Bool {
        let source = CGEventSource(stateID: .combinedSessionState)
        let keyV = CGKeyCode(kVK_ANSI_V)
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyV, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: keyV, keyDown: false)
        else { return false }
        // Leave character translation to macOS for shortcuts. Forcing Unicode "v" breaks
        // Cmd+V in Notes and Signal.
        for event in [down, up] {
            event.flags = .maskCommand
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        return true
    }
}
