import AppKit
import SwiftUI

/// A floating, click-through pill above the Dock on the screen the cursor is on.
/// While recording it shows only a scrolling level meter; notices and errors share the same card.
///
/// The panel has a fixed, transparent size and the pill draws centred inside it, so
/// state changes never resize the window.
@MainActor
final class Overlay {
    static let panelSize = NSSize(width: 420, height: 84)
    /// The visible card never changes size; text inside it is truncated instead.
    static let pillSize = CGSize(width: 160, height: 40)

    private let panel: NSPanel
    private var hideTask: Task<Void, Never>?

    init(controller: DictationController) {
        panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: Self.panelSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        let hostingView = NSHostingView(rootView: PillView(controller: controller))
        hostingView.frame = NSRect(origin: .zero, size: Self.panelSize)
        hostingView.autoresizingMask = [.width, .height]
        panel.contentView = hostingView
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
        panel.setFrameOrigin(NSPoint(x: visible.midX - Self.panelSize.width / 2, y: visible.minY + 12))
    }
}

private struct PillView: View {
    @ObservedObject var controller: DictationController

    var body: some View {
        content
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(PillStyle.text)
            .lineLimit(1)
            .truncationMode(.tail)
            .padding(.horizontal, 14)
            .frame(width: Overlay.pillSize.width, height: Overlay.pillSize.height)
            .background(PillBackground())
            .shadow(color: .black.opacity(0.3), radius: 10, y: 3)
            .frame(width: Overlay.panelSize.width, height: Overlay.panelSize.height)
    }

    @ViewBuilder
    private var content: some View {
        switch controller.state {
        case .recording, .transcribing:
            SpectrumMeter(level: controller.inputLevel, sampleTime: controller.elapsed)
                .opacity(controller.state == .recording ? 1 : 0.4)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(PillStyle.error)
        case .notice(let message):
            Label(message, systemImage: "doc.on.clipboard")
        case .idle, .downloadingModel, .loadingModel:
            Text(controller.state.label)
        }
    }

}

enum PillStyle {
    static let fill = Color(red: 0.078, green: 0.078, blue: 0.11).opacity(0.82)
    static let text = Color.white.opacity(0.95)
    static let dimText = Color.white.opacity(0.5)
    static let bar = Color.white.opacity(0.92)
    static let error = Color(red: 1.0, green: 0.42, blue: 0.42)
}

/// Dark capsule with a hairline border.
private struct PillBackground: View {
    var body: some View {
        Capsule()
            .fill(PillStyle.fill)
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5))
    }
}

/// Thin white bars. Each new level pushes the history left,
/// so the meter scrolls like a waveform rather than bouncing in place.
private struct SpectrumMeter: View {
    let level: Float
    let sampleTime: TimeInterval

    private static let barCount = 36
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
            height: 24
        )
        .onChange(of: sampleTime) { _, _ in
            history.append(CGFloat(min(max(level, 0), 1)))
            if history.count > Self.barCount { history.removeFirst(history.count - Self.barCount) }
        }
    }
}
