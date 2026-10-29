import AppKit
import SwiftUI

/// Past dictations grouped by day, with playback, copy and delete per entry.
struct HistoryView: View {
    @ObservedObject var store: HistoryStore
    @ObservedObject var shortcuts: ShortcutSettings
    @StateObject private var player = AudioPlayer()

    private var sections: [(title: String, entries: [HistoryEntry])] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: store.entries) { calendar.startOfDay(for: $0.date) }
        return grouped.keys.sorted(by: >).map { day in
            (title: Self.dayTitle(day, calendar: calendar), entries: grouped[day] ?? [])
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if store.entries.isEmpty {
                ContentUnavailableView(
                    "No dictations yet",
                    systemImage: "mic",
                    description: Text("Start and stop with \(shortcuts.shortcut.label).")
                )
            } else {
                List {
                    ForEach(sections, id: \.title) { section in
                        Section(section.title) {
                            ForEach(section.entries) { entry in
                                HistoryRow(entry: entry, store: store, player: player)
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
            Divider()
            footer
        }
        .frame(minWidth: 460, minHeight: 320)
    }

    private var footer: some View {
        HStack {
            Text("\(store.entries.count) dictations")
            Text("·").foregroundStyle(.tertiary)
            Text("\(store.totalWords) words")
            Text("·").foregroundStyle(.tertiary)
            Text("\(store.wordsPerMinute) wpm")
            Spacer()
            Text("Start and stop: \(shortcuts.shortcut.label)")
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private static func dayTitle(_ day: Date, calendar: Calendar) -> String {
        if calendar.isDateInToday(day) { return "Today" }
        if calendar.isDateInYesterday(day) { return "Yesterday" }
        return day.formatted(.dateTime.weekday(.wide).month(.wide).day())
    }
}

private struct HistoryRow: View {
    let entry: HistoryEntry
    @ObservedObject var store: HistoryStore
    @ObservedObject var player: AudioPlayer
    @State private var expanded = false

    private static let previewLimit = 240
    private var isLong: Bool { entry.text.count > Self.previewLimit || entry.text.contains("\n") }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                Text(entry.text)
                    .lineLimit(expanded ? nil : 4)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if isLong {
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
                    } label: {
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    }
                    .buttonStyle(.borderless)
                    .help(expanded ? "Collapse" : "Expand")
                }
            }
            HStack(spacing: 8) {
                Text(entry.date.formatted(.dateTime.hour().minute()))
                    .monospacedDigit()
                if !entry.destination.isEmpty {
                    Text("·")
                    Text(entry.destination)
                }
                Text("·")
                Text(duration(entry.duration))
                if entry.delivery != .pasted {
                    Text("·")
                    Text(entry.delivery == .copied ? "copied" : "not pasted")
                }
                Spacer()
                Button {
                    player.toggle(entry, url: store.audioURL(for: entry))
                } label: {
                    Image(systemName: player.playingID == entry.id ? "stop.fill" : "play.fill")
                }
                .help(player.playingID == entry.id ? "Stop" : "Play recording")
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(entry.text, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .help("Copy transcript")
                Button(role: .destructive) {
                    if player.playingID == entry.id { player.stop() }
                    store.remove(entry)
                } label: {
                    Image(systemName: "trash")
                }
                .help("Delete transcript and recording")
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func duration(_ seconds: TimeInterval) -> String {
        seconds < 60 ? String(format: "%.0fs", seconds) : String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }
}
