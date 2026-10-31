import XCTest
@testable import Paraketto

@MainActor
final class PasteTests: XCTestCase {
    private let original = Destination(name: "Original", bundleIdentifier: "test.original")
    private let other = Destination(name: "Other", bundleIdentifier: "test.other")

    func testFocusChangeWhileWaitingDoesNotPasteIntoAnotherApp() async throws {
        var current = original
        var clipboard = ""
        var pasteCount = 0
        let operations = Paster.Operations(copy: { clipboard = $0; return true }, destination: { current },
            accessibilityTrusted: { true }, waitForModifiers: { current = self.other; return true },
            paste: { pasteCount += 1; return true })
        let outcome = try await Paster.deliver("transcript", to: original, using: operations)
        XCTAssertEqual(outcome, .destinationChanged)
        XCTAssertEqual(clipboard, "transcript")
        XCTAssertEqual(pasteCount, 0)
    }

    func testHeldModifiersLeaveTextOnClipboardWithoutSendingKeys() async throws {
        var pasteCount = 0
        let operations = Paster.Operations(copy: { _ in true }, destination: { self.original },
            accessibilityTrusted: { true }, waitForModifiers: { false },
            paste: { pasteCount += 1; return true })
        let outcome = try await Paster.deliver("transcript", to: original, using: operations)
        XCTAssertEqual(outcome, .copiedOnly)
        XCTAssertEqual(pasteCount, 0)
    }

    func testCancellationDuringWaitDoesNotPaste() async throws {
        var pasteCount = 0
        let operations = Paster.Operations(copy: { _ in true }, destination: { self.original },
            accessibilityTrusted: { true }, waitForModifiers: {
                withUnsafeCurrentTask { $0?.cancel() }
                return true
            }, paste: { pasteCount += 1; return true })
        let task = Task { try await Paster.deliver("transcript", to: original, using: operations) }
        do { _ = try await task.value; XCTFail("Cancelled delivery must stop") }
        catch is CancellationError { }
        XCTAssertEqual(pasteCount, 0)
    }

    func testUnavailableClipboardIsNotReportedAsCopiedOrPasted() async throws {
        let operations = Paster.Operations(copy: { _ in false }, destination: { self.original },
            accessibilityTrusted: { true }, waitForModifiers: { XCTFail("Must stop before waiting"); return true },
            paste: { XCTFail("Must not send keys"); return true })
        do { _ = try await Paster.deliver("transcript", to: original, using: operations); XCTFail("Expected clipboard error") }
        catch Paster.DeliveryError.clipboardUnavailable { }
    }

    func testUnchangedDestinationPastesExactlyOnce() async throws {
        var pasteCount = 0
        let operations = Paster.Operations(copy: { _ in true }, destination: { self.original },
            accessibilityTrusted: { true }, waitForModifiers: { true },
            paste: { pasteCount += 1; return true })
        let outcome = try await Paster.deliver("transcript", to: original, using: operations)
        XCTAssertEqual(outcome, .pasted)
        XCTAssertEqual(pasteCount, 1)
    }
}
