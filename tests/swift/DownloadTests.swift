import Foundation
import Network
import XCTest
@testable import ParakeetTDT

final class DownloadTests: XCTestCase {
    func testDownloadReportsIntermediateBytesAndRetainsCompleteFile() async throws {
        let server = try await DownloadFixture.start()
        defer { server.stop() }
        let events = DownloadEvents()
        let download = ModelFileDownload(expectedBytes: Int64(DownloadFixture.size)) { _ = events.append($0) }
        let (url, response) = try await download.download(from: server.url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual(try Data(contentsOf: url), Data(repeating: 120, count: DownloadFixture.size))
        let counts = events.values
        XCTAssertGreaterThan(counts.count, 2, "Progress must arrive before the file finishes")
        XCTAssertEqual(counts.last, Int64(DownloadFixture.size))
        XCTAssertTrue(zip(counts, counts.dropFirst()).allSatisfy { $0 < $1 })
    }

    func testDownloadCanBeCancelledBeforeAndDuringTransfer() async throws {
        let server = try await DownloadFixture.start()
        defer { server.stop() }
        for cancelImmediately in [true, false] {
            let started = cancelImmediately ? nil : expectation(description: "Download started")
            let events = DownloadEvents()
            let download = ModelFileDownload(expectedBytes: Int64(DownloadFixture.size)) {
                if events.append($0) { started?.fulfill() }
            }
            let task = Task { try await download.download(from: server.url) }
            if let started { await fulfillment(of: [started], timeout: 5) }
            task.cancel()
            do { _ = try await task.value; XCTFail("Cancelled transfer returned a file") }
            catch is CancellationError { }
        }
    }
}

private final class DownloadEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [Int64] = []
    func append(_ bytes: Int64) -> Bool {
        lock.withLock {
            counts.append(bytes)
            return counts.count == 1
        }
    }
    var values: [Int64] { lock.withLock { counts } }
}

/// A small, deliberately chunked HTTP response on loopback; no internet or model downloads.
private final class DownloadFixture: @unchecked Sendable {
    static let size = 2 * 1024 * 1024
    private let listener: NWListener
    private let queue = DispatchQueue(label: "Paraketto.download-test")
    private var connections: [NWConnection] = [] // Accessed only on queue.
    var url: URL { URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/model")! }

    private init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    static func start() async throws -> DownloadFixture {
        let server = try DownloadFixture()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            server.listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    server.listener.stateUpdateHandler = nil
                    continuation.resume()
                case .failed(let error):
                    server.listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default: break
                }
            }
            server.listener.newConnectionHandler = { [weak server] connection in
                guard let server else { connection.cancel(); return }
                server.connections.append(connection)
                connection.start(queue: server.queue)
                connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { _, _, _, error in
                    guard error == nil else { connection.cancel(); return }
                    let header = "HTTP/1.1 200 OK\r\nContent-Length: \(size)\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
                    connection.send(content: Data(header.utf8), completion: .contentProcessed { error in
                        guard error == nil else { return }
                        server.sendBody(connection, remaining: size)
                    })
                }
            }
            server.listener.start(queue: server.queue)
        }
        return server
    }

    private func sendBody(_ connection: NWConnection, remaining: Int) {
        guard remaining > 0 else { connection.cancel(); return }
        let count = min(32_768, remaining)
        connection.send(content: Data(repeating: 120, count: count), completion: .contentProcessed { error in
            guard error == nil else { return }
            self.queue.asyncAfter(deadline: .now() + .milliseconds(8)) {
                self.sendBody(connection, remaining: remaining - count)
            }
        })
    }

    func stop() {
        listener.cancel()
        queue.async {
            self.connections.forEach { $0.cancel() }
            self.connections.removeAll()
        }
    }
}
