import AVFoundation
import XCTest
@testable import Paraketto

final class AudioPlayerTests: XCTestCase {
    @MainActor
    func testAnOldCompletionDoesNotStopCurrentPlayback() async throws {
        let recording = try RecordingFile()
        try recording.append([Float](repeating: 0, count: 160_000))
        try recording.finish()
        let entry = HistoryEntry(id: UUID(), date: Date(), text: "", destination: "", duration: 10,
                                 inferenceMilliseconds: 0, delivery: .imported)
        let player = AudioPlayer()
        defer { player.stop() }
        player.toggle(entry, url: recording.url)
        let finishedPlayer = try AVAudioPlayer(contentsOf: recording.url)
        player.audioPlayerDidFinishPlaying(finishedPlayer, successfully: true)
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(player.playingID, entry.id)
    }
}
