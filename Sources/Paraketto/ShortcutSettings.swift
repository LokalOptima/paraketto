import AppKit
import Carbon.HIToolbox
import Combine
import Foundation

/// A modifier key that can be tapped on its own to start and stop dictation.
enum ModifierKey: String, CaseIterable, Codable {
    case rightOption, leftOption
    case rightCommand, leftCommand
    case rightControl, leftControl
    case rightShift, leftShift
    case function

    init?(keyCode: UInt16) {
        guard let key = Self.allCases.first(where: { $0.keyCode == keyCode }) else { return nil }
        self = key
    }

    var label: String {
        switch self {
        case .rightOption: return "Right ⌥"
        case .leftOption: return "Left ⌥"
        case .rightCommand: return "Right ⌘"
        case .leftCommand: return "Left ⌘"
        case .rightControl: return "Right ⌃"
        case .leftControl: return "Left ⌃"
        case .rightShift: return "Right ⇧"
        case .leftShift: return "Left ⇧"
        case .function: return "Fn / Globe"
        }
    }

    var keyCode: UInt16 {
        switch self {
        case .rightOption: return 61
        case .leftOption: return 58
        case .rightCommand: return 54
        case .leftCommand: return 55
        case .rightControl: return 62
        case .leftControl: return 59
        case .rightShift: return 60
        case .leftShift: return 56
        case .function: return 63
        }
    }

    var flag: NSEvent.ModifierFlags {
        switch self {
        case .rightOption, .leftOption: return .option
        case .rightCommand, .leftCommand: return .command
        case .rightControl, .leftControl: return .control
        case .rightShift, .leftShift: return .shift
        case .function: return .function
        }
    }
}

/// A key plus modifiers, registered as a Carbon hotkey.
struct Chord: Codable, Equatable {
    let keyCode: UInt16
    let modifiers: NSEvent.ModifierFlags.RawValue
    let label: String

    var carbonModifiers: UInt32 {
        let flags = NSEvent.ModifierFlags(rawValue: modifiers)
        var result: UInt32 = 0
        if flags.contains(.command) { result |= UInt32(cmdKey) }
        if flags.contains(.option) { result |= UInt32(optionKey) }
        if flags.contains(.control) { result |= UInt32(controlKey) }
        if flags.contains(.shift) { result |= UInt32(shiftKey) }
        return result
    }

    /// Builds a chord from a key event. Returns nil for bare keys that would hijack typing;
    /// function keys are allowed without modifiers.
    static func from(_ event: NSEvent) -> Chord? {
        let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
        let name = Self.keyName(for: event)
        let isFunctionKey = name.hasPrefix("F") && Int(name.dropFirst()) != nil
        guard !flags.isEmpty || isFunctionKey else { return nil }
        var label = ""
        if flags.contains(.control) { label += "⌃" }
        if flags.contains(.option) { label += "⌥" }
        if flags.contains(.shift) { label += "⇧" }
        if flags.contains(.command) { label += "⌘" }
        return Chord(keyCode: event.keyCode, modifiers: flags.rawValue, label: label + name)
    }

    private static func keyName(for event: NSEvent) -> String {
        let special: [UInt16: String] = [
            UInt16(kVK_Space): "Space", UInt16(kVK_Return): "Return", UInt16(kVK_Tab): "Tab",
            UInt16(kVK_Delete): "Delete", UInt16(kVK_ForwardDelete): "⌦", UInt16(kVK_Escape): "Esc",
            UInt16(kVK_LeftArrow): "←", UInt16(kVK_RightArrow): "→", UInt16(kVK_UpArrow): "↑", UInt16(kVK_DownArrow): "↓",
            UInt16(kVK_Home): "Home", UInt16(kVK_End): "End", UInt16(kVK_PageUp): "Page Up", UInt16(kVK_PageDown): "Page Down",
            UInt16(kVK_F1): "F1", UInt16(kVK_F2): "F2", UInt16(kVK_F3): "F3", UInt16(kVK_F4): "F4", UInt16(kVK_F5): "F5",
            UInt16(kVK_F6): "F6", UInt16(kVK_F7): "F7", UInt16(kVK_F8): "F8", UInt16(kVK_F9): "F9", UInt16(kVK_F10): "F10",
            UInt16(kVK_F11): "F11", UInt16(kVK_F12): "F12", UInt16(kVK_F13): "F13", UInt16(kVK_F14): "F14", UInt16(kVK_F15): "F15",
        ]
        if let name = special[event.keyCode] { return name }
        return (event.charactersIgnoringModifiers ?? "?").uppercased()
    }
}

/// What starts and stops dictation: a lone modifier tapped, or a key combination.
enum Shortcut: Codable, Equatable {
    case tap(ModifierKey)
    case chord(Chord)

    static let `default` = Shortcut.tap(.rightOption)

    var label: String {
        switch self {
        case .tap(let key): return "tap \(key.label)"
        case .chord(let chord): return chord.label
        }
    }
}

/// The user's shortcut, persisted in UserDefaults and observed live by the controller.
@MainActor
final class ShortcutSettings: ObservableObject {
    @Published var shortcut: Shortcut { didSet { save() } }
    /// True while the recorder in Settings is waiting for a key; the shortcut is disarmed meanwhile.
    @Published var isRecording = false

    private static let key = "shortcut"

    init() {
        shortcut = UserDefaults.standard.data(forKey: Self.key)
            .flatMap { try? JSONDecoder().decode(Shortcut.self, from: $0) } ?? .default
    }

    private func save() {
        UserDefaults.standard.set(try? JSONEncoder().encode(shortcut), forKey: Self.key)
    }
}
