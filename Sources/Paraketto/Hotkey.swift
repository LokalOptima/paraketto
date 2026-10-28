import Carbon.HIToolbox
import Foundation

/// A system-wide push-to-talk hotkey built on Carbon's RegisterEventHotKey.
/// Delivers both press and release so the caller can record while the key is held.
final class Hotkey {
    struct Binding {
        let keyCode: UInt32
        let carbonModifiers: UInt32
        let label: String

        /// Control + Option + Space.
        static let `default` = Binding(
            keyCode: UInt32(kVK_Space),
            carbonModifiers: UInt32(controlKey | optionKey),
            label: "⌃⌥Space"
        )
    }

    /// Non-nil when the system refused the registration, typically because another app owns the chord.
    private(set) var registrationError: OSStatus?

    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private let onPress: () -> Void
    private let onRelease: () -> Void

    init(binding: Binding = .default, onPress: @escaping () -> Void, onRelease: @escaping () -> Void) {
        self.onPress = onPress
        self.onRelease = onRelease
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
        let hotKeyID = EventHotKeyID(signature: fourCharCode("PKTU"), id: 1)
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
