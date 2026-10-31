import AppKit
import SwiftUI

struct TranscriptRow: View {
    let entry: HistoryEntry
    @ObservedObject var store: HistoryStore
    @ObservedObject var player: AudioPlayer
    @State private var expanded = false
    @State private var isTruncated = false
    @State private var isHovered = false
    @State private var copied = false
    @State private var copySequence = 0
    @FocusState private var focusedAction: Action?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled

    private enum Action: Hashable { case expand, play, copy, delete }
    private var showsActions: Bool { isHovered || focusedAction != nil || voiceOverEnabled }

    private var transcript: String { entry.text.isEmpty ? "No speech detected" : entry.text }
    private var canExpand: Bool { isTruncated || expanded }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                Text(transcript)
                    .lineLimit(expanded ? nil : 4)
                    .truncationMode(.tail)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background {
                        GeometryReader { geometry in
                            VStack(spacing: 0) {
                                measureText(width: geometry.size.width, lineLimit: nil)
                                measureText(width: geometry.size.width, lineLimit: 4)
                            }
                            .hidden()
                            .accessibilityHidden(true)
                            .allowsHitTesting(false)
                        }
                    }
                    .onPreferenceChange(TranscriptTextHeights.self) { heights in
                        guard let full = heights[0], let collapsed = heights[4] else { return }
                        isTruncated = full > collapsed + 0.5
                    }
                Button(action: toggleExpansion) {
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .frame(width: 24, height: 18)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .focused($focusedAction, equals: .expand)
                .disabled(!canExpand)
                .accessibilityHidden(!canExpand)
                .accessibilityLabel(expanded ? "Collapse transcript" : "Expand transcript")
                .help(expanded ? "Collapse transcript" : "Expand transcript")
                .opacity(canExpand && showsActions ? 1 : 0)
                .allowsHitTesting(canExpand && showsActions)
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.12), value: showsActions)
            }
            HStack(spacing: 8) {
                Text(entry.date.formatted(.dateTime.hour().minute()))
                    .monospacedDigit()
                if let filename = entry.sourceFilename {
                    Text("·")
                    Text(filename).lineLimit(1).truncationMode(.middle)
                } else if !entry.destination.isEmpty {
                    Text("·")
                    Text(entry.destination)
                }
                if entry.delivery == .copied || entry.delivery == .keptInHistory {
                    Text("·")
                    Text(entry.delivery == .copied ? "copied" : "not pasted")
                }
                Spacer()
                actions
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 10)
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
        .task(id: copySequence) {
            guard copied else { return }
            do { try await Task.sleep(for: .seconds(1.5)) }
            catch { return }
            copied = false
        }
    }

    private func measureText(width: CGFloat, lineLimit: Int?) -> some View {
        Text(transcript)
            .lineLimit(lineLimit)
            .frame(width: width, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
            .background {
                GeometryReader { geometry in
                    Color.clear.preference(key: TranscriptTextHeights.self,
                                           value: [lineLimit ?? 0: geometry.size.height])
                }
            }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            Button {
                player.toggle(entry, url: store.audioURL(for: entry))
            } label: {
                Image(systemName: player.playingID == entry.id ? "stop.fill" : "play.fill")
                    .frame(width: 28, height: 24)
                    .contentShape(Rectangle())
            }
            .focused($focusedAction, equals: .play)
            .accessibilityLabel(player.playingID == entry.id ? "Stop recording" : "Play recording")
            .help(player.playingID == entry.id ? "Stop" : "Play recording")

            Button(action: copyTranscript) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .contentTransition(reduceMotion ? .identity : .symbolEffect(.replace))
                    .foregroundStyle(copied ? Color(nsColor: .systemGreen) : .secondary)
                    .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: copied)
                    .frame(width: 28, height: 24)
                    .contentShape(Rectangle())
            }
            .focused($focusedAction, equals: .copy)
            .accessibilityLabel(copied ? "Copied to clipboard" : "Copy transcript")
            .help(copied ? "Copied!" : "Copy transcript")

            Button(role: .destructive) {
                if player.playingID == entry.id { player.stop() }
                store.remove(entry)
            } label: {
                Image(systemName: "trash")
                    .frame(width: 28, height: 24)
                    .contentShape(Rectangle())
            }
            .focused($focusedAction, equals: .delete)
            .accessibilityLabel("Delete transcript and recording")
            .help("Delete transcript and recording")
        }
        .font(.system(size: 13))
        // Keep the reserved width so revealing controls never shifts the metadata.
        .opacity(showsActions ? 1 : 0)
        .allowsHitTesting(showsActions)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.12), value: showsActions)
    }

    private func copyTranscript() {
        NSPasteboard.general.clearContents()
        if NSPasteboard.general.setString(entry.text, forType: .string) {
            copied = true
            copySequence += 1
        }
    }

    private func toggleExpansion() {
        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.15)) { expanded.toggle() }
    }

}

private struct TranscriptTextHeights: PreferenceKey {
    static let defaultValue: [Int: CGFloat] = [:]

    static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}
