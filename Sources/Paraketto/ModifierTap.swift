import AppKit
import Carbon.HIToolbox

/// Fires when a single modifier key is tapped on its own: pressed and released within
/// `maxHold` seconds without any other key in between, so Option+letter typing is ignored.
/// Carbon hotkeys cannot bind a bare modifier, so this uses global event monitors, which
/// need the Accessibility permission the app already has for pasting.
@MainActor
final class ModifierTap {
    private static let maxHold: TimeInterval = 0.6

    private let keyCode: UInt16
    private let flag: NSEvent.ModifierFlags
    private let onTap: () -> Void
    private var monitors: [Any] = []
    private var pressedAt: Date?
    private var interrupted = false

    init(_ key: ModifierKey, onTap: @escaping () -> Void) {
        keyCode = key.keyCode
        flag = key.flag
        self.onTap = onTap
        if let flags = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged, handler: { [weak self] event in
            Task { @MainActor in self?.flagsChanged(event) }
        }) { monitors.append(flags) }
        if let keys = NSEvent.addGlobalMonitorForEvents(matching: [.keyDown, .leftMouseDown, .rightMouseDown], handler: { [weak self] _ in
            Task { @MainActor in self?.interrupted = true }
        }) { monitors.append(keys) }
        Log.info("Modifier tap \(key.label) armed")
    }

    deinit {
        for monitor in monitors { NSEvent.removeMonitor(monitor) }
    }

    private func flagsChanged(_ event: NSEvent) {
        guard event.keyCode == keyCode else {
            // Another modifier joined or left while ours is down: treat it as a chord, not a tap.
            if pressedAt != nil { interrupted = true }
            return
        }
        if event.modifierFlags.contains(flag) {
            pressedAt = Date()
            interrupted = false
        } else if let pressedAt {
            self.pressedAt = nil
            if !interrupted, Date().timeIntervalSince(pressedAt) <= Self.maxHold {
                onTap()
            }
        }
    }
}
