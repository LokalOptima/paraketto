import Carbon.HIToolbox
import Foundation

/// A system-wide hotkey built on Carbon's RegisterEventHotKey.
/// Delivers press and release; the dictation shortcut only uses the press.
final class Hotkey {
    struct Binding {
        let keyCode: UInt32
        let carbonModifiers: UInt32
        let label: String

        init(keyCode: UInt32, carbonModifiers: UInt32, label: String) {
            self.keyCode = keyCode
            self.carbonModifiers = carbonModifiers
            self.label = label
        }

        init(_ chord: Chord) {
            self.init(keyCode: UInt32(chord.keyCode), carbonModifiers: chord.carbonModifiers, label: chord.label)
        }
    }

    /// Non-nil when the system refused the registration, typically because another app owns the chord.
    private(set) var registrationError: OSStatus?

    private static var nextID: UInt32 = 1
    private let id: UInt32
    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private let onPress: () -> Void
    private let onRelease: () -> Void

    init(binding: Binding, onPress: @escaping () -> Void, onRelease: @escaping () -> Void = {}) {
        self.onPress = onPress
        self.onRelease = onRelease
        id = Self.nextID
        Self.nextID += 1
        installHandler()
        register(binding)
    }

    deinit {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
        if let handlerRef { RemoveEventHandler(handlerRef) }
    }

    private func installHandler() {
        var eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]
        let selfPointer = Unmanaged.passUnretained(self).toOpaque()
        let status = InstallEventHandler(
            GetApplicationEventTarget(),
            { _, event, userData in
                guard let event, let userData else { return noErr }
                let hotkey = Unmanaged<Hotkey>.fromOpaque(userData).takeUnretainedValue()
                // Every handler sees every hotkey event; act only on this instance's own id.
                var hotKeyID = EventHotKeyID()
                GetEventParameter(
                    event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                    nil, MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID
                )
                guard hotKeyID.id == hotkey.id else { return noErr }
                switch GetEventKind(event) {
                case UInt32(kEventHotKeyPressed): hotkey.onPress()
                case UInt32(kEventHotKeyReleased): hotkey.onRelease()
                default: break
                }
                return noErr
            },
            eventTypes.count,
            &eventTypes,
            selfPointer,
            &handlerRef
        )
        if status != noErr {
            Log.info("InstallEventHandler failed: \(status)")
        }
    }

    private func register(_ binding: Binding) {
        let hotKeyID = EventHotKeyID(signature: fourCharCode("PKTU"), id: id)
        let status = RegisterEventHotKey(
            binding.keyCode, binding.carbonModifiers, hotKeyID,
            GetApplicationEventTarget(), 0, &hotKeyRef
        )
        if status == noErr {
            Log.info("Hotkey \(binding.label) registered")
        } else {
            registrationError = status
            Log.info("RegisterEventHotKey failed for \(binding.label): \(status)")
        }
    }

    private func fourCharCode(_ string: String) -> OSType {
        string.utf8.reduce(0) { ($0 << 8) | OSType($1) }
    }
}
