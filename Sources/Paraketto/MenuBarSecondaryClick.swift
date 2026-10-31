import AppKit
import Combine

/// Intercepts only secondary clicks on this app's status item, before SwiftUI opens its menu.
@MainActor
final class MenuBarSecondaryClick: ObservableObject {
    private var monitor: Any?
    private weak var pressedWindow: NSWindow?
    private var action: (() -> Void)?

    func start(action: @escaping () -> Void) {
        self.action = action
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDown, .rightMouseUp]) { [weak self] event in
            guard let self else { return event }
            return self.handle(event)
        }
    }

    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        pressedWindow = nil
        action = nil
    }

    func handle(_ event: NSEvent) -> NSEvent? {
        guard event.type == .rightMouseDown || event.type == .rightMouseUp else { return event }
        if event.type == .rightMouseUp, let pressedWindow {
            self.pressedWindow = nil
            if event.window === pressedWindow, Self.isOnStatusButton(event) {
                action?()
            }
            return nil
        }
        guard Self.isOnStatusButton(event) else { return event }
        if event.type == .rightMouseDown { pressedWindow = event.window }
        return nil
    }

    private static func isOnStatusButton(_ event: NSEvent) -> Bool {
        guard let window = event.window, window.level == .statusBar,
              let content = window.contentView else { return false }
        func containsButton(_ view: NSView) -> Bool {
            if let button = view as? NSStatusBarButton {
                return button.bounds.contains(button.convert(event.locationInWindow, from: nil))
            }
            return view.subviews.contains(where: containsButton)
        }
        return containsButton(content)
    }
}
