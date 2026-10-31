import Foundation

/// One cancellable file transfer. Uses the delegate API because URLSession's async
/// download convenience method doesn't forward intermediate download callbacks.
final class ModelFileDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedBytes: Int64
    private let report: @Sendable (Int64) -> Void
    private let lock = NSLock()
    private var lastReportedBytes: Int64 = 0
    private var cancelled = false
    private var task: URLSessionDownloadTask?
    private var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    // Only accessed on the session's serial delegate queue.
    private var downloadedFile: URL?
    private var fileError: Error?

    init(expectedBytes: Int64, report: @escaping @Sendable (Int64) -> Void) {
        self.expectedBytes = expectedBytes
        self.report = report
    }

    func download(from url: URL) async throws -> (URL, URLResponse) {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
                let task = session.downloadTask(with: url)
                let shouldCancel = lock.withLock {
                    self.continuation = continuation
                    self.task = task
                    return cancelled
                }
                if shouldCancel { task.cancel() }
                task.resume()
            }
        } onCancel: {
            let task = self.lock.withLock {
                self.cancelled = true
                return self.task
            }
            task?.cancel()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        // Bound UI updates to roughly 200 per file, including the final chunk.
        guard totalBytesWritten > lastReportedBytes,
              totalBytesWritten - lastReportedBytes >= max(1, expectedBytes / 200)
                || totalBytesWritten >= expectedBytes else { return }
        lastReportedBytes = totalBytesWritten
        report(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // URLSession removes its temporary file as soon as this callback returns.
        let retained = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".download")
        do {
            try FileManager.default.moveItem(at: location, to: retained)
            downloadedFile = retained
        } catch { fileError = error }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let (continuation, cancelled) = lock.withLock {
            let continuation = self.continuation
            self.continuation = nil
            self.task = nil
            return (continuation, self.cancelled)
        }
        session.finishTasksAndInvalidate()
        let result: Result<(URL, URLResponse), Error>
        if cancelled {
            result = .failure(CancellationError())
        } else if let error = error ?? fileError {
            result = .failure(error)
        } else if let downloadedFile, let response = task.response {
            result = .success((downloadedFile, response))
        } else {
            result = .failure(URLError(.badServerResponse))
        }
        if case .failure = result, let downloadedFile {
            try? FileManager.default.removeItem(at: downloadedFile)
        }
        continuation?.resume(with: result)
    }
}
