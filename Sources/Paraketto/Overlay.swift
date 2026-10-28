import AppKit
import SwiftUI

/// A floating, click-through pill above the Dock on the screen the cursor is on.
/// Shows the input level while recording and an error message when something fails.
@MainActor
final class Overlay {
    private let panel: NSPanel
    private var hideTask: Task<Void, Never>?
    private var resizeObserver: NSObjectProtocol?

    init(controller: DictationController) {
        let hostingView = NSHostingView(rootView: PillView(controller: controller))
        hostingView.sizingOptions = [.preferredContentSize]
        panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: hostingView.fittingSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.contentView = hostingView

        // SwiftUI resizes the panel to its content; keep it centred when that happens.
        resizeObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResizeNotification, object: panel, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.position() }
        }
    }

    func show() {
        hideTask?.cancel()
        position()
        panel.orderFrontRegardless()
    }

    func hide(after delay: Duration = .zero) {
        hideTask?.cancel()
        guard delay > .zero else {
            panel.orderOut(nil)
            return
        }
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.panel.orderOut(nil)
        }
    }

    private func position() {
        let mouse = NSEvent.mouseLocation
        let screen = NSScreen.screens.first { NSMouseInRect(mouse, $0.frame, false) } ?? NSScreen.main
        guard let visible = screen?.visibleFrame else { return }
        let size = panel.frame.size
        panel.setFrameOrigin(NSPoint(x: visible.midX - size.width / 2, y: visible.minY + 16))
    }
}

private struct PillView: View {
    @ObservedObject var controller: DictationController

    var body: some View {
        HStack(spacing: 10) {
            switch controller.state {
            case .recording, .transcribing:
                SpectrumMeter(level: controller.inputLevel)
                    .opacity(controller.state == .recording ? 1 : 0.5)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                Text(message)
                    .lineLimit(1)
                    .frame(maxWidth: 260, alignment: .leading)
            case .idle, .downloadingModel, .loadingModel:
                Text(controller.state.label)
            }
        }
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(Color.white.opacity(0.95))
        .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
        .background(PillBackground())
        .fixedSize()
        .padding(12)
    }
}

enum PillStyle {
    static let fill = Color(red: 0.078, green: 0.078, blue: 0.11).opacity(0.82)
    static let bar = Color.white.opacity(0.92)
}

/// Dark capsule with a hairline border.
private struct PillBackground: View {
    var body: some View {
        Capsule()
            .fill(PillStyle.fill)
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5))
    }
}

/// 24 thin white bars. Each new level pushes the history left,
/// so the meter scrolls like a waveform rather than bouncing in place.
private struct SpectrumMeter: View {
    let level: Float

    private static let barCount = 24
    private static let barWidth: CGFloat = 2
    private static let spacing: CGFloat = 1.5
    private static let silenceFraction: CGFloat = 0.14
    private static let peakFraction: CGFloat = 0.86

    @State private var history: [CGFloat] = []

    var body: some View {
        Canvas { context, size in
            let padded = Array(repeating: CGFloat(0), count: max(0, Self.barCount - history.count)) + history
            for (index, value) in padded.suffix(Self.barCount).enumerated() {
                let barHeight = size.height * (Self.silenceFraction + Self.peakFraction * value)
                let rect = CGRect(
                    x: CGFloat(index) * (Self.barWidth + Self.spacing),
                    y: (size.height - barHeight) / 2,
                    width: Self.barWidth,
                    height: barHeight
                )
                context.fill(
                    Path(roundedRect: rect, cornerRadius: Self.barWidth / 2),
                    with: .color(PillStyle.bar)
                )
            }
        }
        .frame(
            width: CGFloat(Self.barCount) * Self.barWidth + CGFloat(Self.barCount - 1) * Self.spacing,
            height: 16
        )
        .onChange(of: level) { _, newValue in
            history.append(CGFloat(min(max(newValue, 0), 1)))
            if history.count > Self.barCount { history.removeFirst(history.count - Self.barCount) }
        }
    }
}
