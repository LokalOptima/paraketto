import AppKit
import SwiftUI

/// Past dictations grouped by day, with playback, copy and delete per entry.
struct TranscriptsView: View {
    @ObservedObject var controller: DictationController
    @ObservedObject var store: HistoryStore
    @ObservedObject var shortcuts: ShortcutSettings
    @StateObject private var player = AudioPlayer()
    @State private var isDropTargeted = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.displayScale) private var displayScale

    init(controller: DictationController) {
        self.controller = controller
        store = controller.history
        shortcuts = controller.shortcuts
    }

    private var sections: [(day: Date, title: String, entries: [HistoryEntry])] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: store.entries) { calendar.startOfDay(for: $0.date) }
        return grouped.keys.sorted(by: >).map { day in
            (day: day, title: Self.dayTitle(day, calendar: calendar), entries: grouped[day] ?? [])
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if let job = controller.fileImport {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(job.filename).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            Text(job.mode.title).foregroundStyle(.secondary)
                        }
                        ProgressView(value: job.progress)
                            .accessibilityLabel("Transcribing \(job.filename)")
                    }
                    Button("Cancel") { controller.cancelFileImport() }
                }
                .padding()
                Divider()
            }
            if !controller.importErrors.isEmpty {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(controller.importErrors, id: \.self) { error in
                            Text(error).foregroundStyle(.red).textSelection(.enabled)
                        }
                    }
                    Spacer()
                    Button("Dismiss") { controller.dismissImportErrors() }
                }
                .padding()
                Divider()
            }
            if store.entries.isEmpty {
                ContentUnavailableView(
                    "No transcripts yet",
                    systemImage: "mic",
                    description: Text("Start and stop with \(shortcuts.shortcut.label).")
                )
            } else {
                ScrollView {
                    LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                        ForEach(sections, id: \.day) { section in
                            Section {
                                ForEach(section.entries) { entry in
                                    TranscriptRow(entry: entry, store: store, player: player)
                                        .padding(.horizontal, 16)
                                    if entry.id != section.entries.last?.id {
                                        separator.padding(.horizontal, 16)
                                    }
                                }
                            } header: {
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(section.title)
                                        .font(.callout.weight(.semibold))
                                        .foregroundStyle(Color(nsColor: .systemTeal))
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 8)
                                    separator
                                }
                                .background(Color(nsColor: .textBackgroundColor))
                            }
                        }
                    }
                }
                // Transparent title bars must not reveal scroll content outside this viewport.
                .clipped()
            }
            Divider()
            footer
        }
        .frame(minWidth: 460, minHeight: 320)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor).ignoresSafeArea())
        .toolbarBackground(.hidden, for: .windowToolbar)
        .overlay {
            if isDropTargeted {
                AudioFileDropOverlay(isAvailable: controller.canImportFiles)
                    .padding(12)
                    .transition(reduceMotion ? .opacity : .opacity.combined(with: .scale(scale: 0.97)))
                    .allowsHitTesting(false)
            }
        }
        .animation(.easeOut(duration: reduceMotion ? 0 : 0.16), value: isDropTargeted)
        .dropDestination(for: URL.self) { urls, _ in
            let accepted = controller.importFiles(urls)
            if accepted { player.stop() }
            return accepted
        } isTargeted: { isDropTargeted = $0 }
        .onExitCommand { if controller.fileImport != nil { controller.cancelFileImport() } }
    }

    private var separator: some View {
        Color(nsColor: .separatorColor).frame(height: 1 / displayScale)
    }

    private var footer: some View {
        HStack {
            Text("\(store.entries.count) transcripts")
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

/// Appears only while dragging over History; there is no persistent import prompt.
struct AudioFileDropOverlay: View {
    let isAvailable: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(.regularMaterial)
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(isAvailable ? Color.accentColor : Color.secondary,
                                  style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
            }
            .overlay {
                VStack(spacing: 12) {
                    Image(systemName: isAvailable ? "arrow.down.doc" : "hourglass")
                        .font(.system(size: 38, weight: .light))
                    Text(isAvailable ? "Drop audio files here" : "Paraketto is busy")
                        .font(.title3.weight(.semibold))
                }
                .foregroundStyle(isAvailable ? Color.accentColor : Color.secondary)
            }
    }
}
