import Accelerate
import Foundation

/// Prediction network and joint of Parakeet TDT on the CPU with Accelerate, replacing the two
/// Core ML models for the sequential part of decoding.
///
/// Core ML runs these tiny models on the CPU anyway (LSTM, argmax and top-k have no ANE
/// support), one framework call per step, each re-reading every weight. Two things make this
/// implementation cheaper per step:
///
/// 1. The first LSTM layer's input projection is precomputed for all 8193 tokens, so a token
///    step is a row lookup instead of a 2560×640 matrix-vector product.
/// 2. While the decoder state is unchanged (runs of blanks), the joint head is evaluated for
///    several upcoming encoder frames in one matrix product, reading the 8198×640 weight once.
///
/// Numerics follow the Core ML models: fp16 weights widened to fp32, fp32 arithmetic.
final class CPUDecoder {
    static let vocabulary = ParakeetConstants.blankToken + 1          // 8193 incl. blank
    static let outputs = vocabulary + ParakeetConstants.durationBins.count  // 8198
    private static let hidden = ParakeetConstants.decoderHiddenSize    // 640
    private static let gates = 4 * hidden                              // 2560
    private static let encoderSize = ParakeetConstants.encoderHiddenSize

    /// Frames of joint logits computed per batch while the decoder state is fixed.
    var lookahead = 8

    /// Accumulated costs for PARAKEET_TIMING.
    private(set) var feedCalls = 0, feedSeconds = 0.0, decideCalls = 0, decideFrames = 0, decideSeconds = 0.0, projectSeconds = 0.0
    func resetTiming() { feedCalls = 0; feedSeconds = 0; decideCalls = 0; decideFrames = 0; decideSeconds = 0; projectSeconds = 0 }
    var timingSummary: String {
        String(format: "cpu decoder: project %.1f ms | feed %d calls %.1f ms (%.0f µs each) | decide %d calls / %d frames %.1f ms (%.0f µs per call)",
               projectSeconds * 1000, feedCalls, feedSeconds * 1000, feedCalls > 0 ? feedSeconds * 1e6 / Double(feedCalls) : 0,
               decideCalls, decideFrames, decideSeconds * 1000, decideCalls > 0 ? decideSeconds * 1e6 / Double(decideCalls) : 0)
    }

    // Weights (row-major, PyTorch layout [out, in]).
    private let inputTable: [Float]        // [vocabulary, gates]: W_ih0 · embedding(token) + b_ih0 + b_hh0
    private let weightHH0: [Float]         // [gates, hidden]
    private let weightIH1: [Float]         // [gates, hidden]
    private let weightHH1: [Float]         // [gates, hidden]
    private let bias1: [Float]             // [gates]: b_ih1 + b_hh1
    private let decoderProjection: [Float] // [hidden, hidden]
    private let decoderProjectionBias: [Float]
    private let encoderProjection: [Float] // [hidden, encoderSize]
    private let encoderProjectionBias: [Float]
    private let head: [Float]              // [outputs, hidden]
    private let headBias: [Float]

    // State.
    private var h0 = [Float](repeating: 0, count: hidden)
    private var c0 = [Float](repeating: 0, count: hidden)
    private var h1 = [Float](repeating: 0, count: hidden)
    private var c1 = [Float](repeating: 0, count: hidden)
    /// decoderProjection · h1 + bias, the decoder's contribution to the joint.
    private var projectedState = [Float](repeating: 0, count: hidden)
    private(set) var lastToken: Int?

    // Scratch.
    private var gateBuffer = [Float](repeating: 0, count: gates)
    private var logits: [Float] = []

    init(weights: SafeTensors) throws {
        func matrix(_ name: String, _ rows: Int, _ cols: Int) throws -> [Float] {
            guard weights.shape(of: name) == [rows, cols] || weights.shape(of: name) == [rows, cols, 1] else {
                throw ParakeetError.unexpectedShape(name, weights.shape(of: name) ?? [])
            }
            return try weights.floats(name)
        }
        let embedding = try matrix("decoder.embedding.weight", Self.vocabulary, Self.hidden)
        let weightIH0 = try matrix("decoder.lstm.weight_ih_l0", Self.gates, Self.hidden)
        let biasIH0 = try weights.floats("decoder.lstm.bias_ih_l0")
        let biasHH0 = try weights.floats("decoder.lstm.bias_hh_l0")
        // inputTable[t] = W_ih0 · E[t] + b, computed as E · W_ih0ᵀ in one GEMM.
        var table = [Float](repeating: 0, count: Self.vocabulary * Self.gates)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, Int32(Self.vocabulary), Int32(Self.gates), Int32(Self.hidden),
            1, embedding, Int32(Self.hidden), weightIH0, Int32(Self.hidden), 0, &table, Int32(Self.gates)
        )
        let bias0 = vDSP.add(biasIH0, biasHH0)
        for token in 0..<Self.vocabulary {
            let row = token * Self.gates
            vDSP.add(bias0, table[row..<(row + Self.gates)], result: &table[row..<(row + Self.gates)])
        }
        inputTable = table
        weightHH0 = try matrix("decoder.lstm.weight_hh_l0", Self.gates, Self.hidden)
        weightIH1 = try matrix("decoder.lstm.weight_ih_l1", Self.gates, Self.hidden)
        weightHH1 = try matrix("decoder.lstm.weight_hh_l1", Self.gates, Self.hidden)
        bias1 = vDSP.add(try weights.floats("decoder.lstm.bias_ih_l1"), try weights.floats("decoder.lstm.bias_hh_l1"))
        decoderProjection = try matrix("decoder.decoder_projector.weight", Self.hidden, Self.hidden)
        decoderProjectionBias = try weights.floats("decoder.decoder_projector.bias")
        encoderProjection = try matrix("encoder_projector.weight", Self.hidden, Self.encoderSize)
        encoderProjectionBias = try weights.floats("encoder_projector.bias")
        head = try matrix("joint.head.weight", Self.outputs, Self.hidden)
        headBias = try weights.floats("joint.head.bias")
        reset()
    }

    func reset() {
        for array in [h0, c0, h1, c1] { _ = array }
        h0 = [Float](repeating: 0, count: Self.hidden)
        c0 = h0; h1 = h0; c1 = h0
        lastToken = nil
        feed(ParakeetConstants.blankToken)  // start-of-sequence, as the Core ML path primes with blank
        lastToken = nil
    }

    /// Advances the LSTM with `token` and refreshes the projected state.
    func feed(_ token: Int) {
        let started = Date()
        defer { feedCalls += 1; feedSeconds += Date().timeIntervalSince(started) }
        // Layer 0: gates = table[token] + W_hh0 · h0
        let row = token * Self.gates
        inputTable.withUnsafeBufferPointer { table in
            gateBuffer.withUnsafeMutableBufferPointer { gatesOut in
                gatesOut.baseAddress!.update(from: table.baseAddress! + row, count: Self.gates)
            }
        }
        Self.parallelGEMV(weightHH0, rows: Self.gates, cols: Self.hidden, x: h0, y: &gateBuffer)
        Self.lstmCell(gates: gateBuffer, h: &h0, c: &c0)
        // Layer 1: gates = W_ih1 · h0 + W_hh1 · h1 + bias1
        gateBuffer = bias1
        Self.parallelGEMV(weightIH1, rows: Self.gates, cols: Self.hidden, x: h0, y: &gateBuffer)
        Self.parallelGEMV(weightHH1, rows: Self.gates, cols: Self.hidden, x: h1, y: &gateBuffer)
        Self.lstmCell(gates: gateBuffer, h: &h1, c: &c1)
        // Joint's decoder side.
        projectedState = decoderProjectionBias
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(Self.hidden), Int32(Self.hidden), 1, decoderProjection, Int32(Self.hidden), h1, 1, 1, &projectedState, 1)
        lastToken = token
    }

    /// PyTorch gate order i, f, g, o.
    private static func lstmCell(gates: [Float], h: inout [Float], c: inout [Float]) {
        let n = hidden
        gates.withUnsafeBufferPointer { g in
            for j in 0..<n {
                let i = sigmoid(g[j]), f = sigmoid(g[n + j]), gg = tanh(g[2 * n + j]), o = sigmoid(g[3 * n + j])
                c[j] = f * c[j] + i * gg
                h[j] = o * tanh(c[j])
            }
        }
    }

    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    /// Number of row chunks matrix products are split into across cores; the weights are
    /// memory-bound, so this multiplies the usable bandwidth.
    static let chunks = max(2, min(8, ProcessInfo.processInfo.activeProcessorCount))

    /// y += A · x for a row-major [rows, cols] matrix, rows split across cores.
    private static func parallelGEMV(_ a: [Float], rows: Int, cols: Int, x: [Float], y: inout [Float]) {
        let per = (rows + chunks - 1) / chunks
        a.withUnsafeBufferPointer { ap in
            x.withUnsafeBufferPointer { xp in
                y.withUnsafeMutableBufferPointer { yp in
                    DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                        let start = chunk * per
                        let count = min(per, rows - start)
                        guard count > 0 else { return }
                        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(count), Int32(cols), 1,
                                    ap.baseAddress! + start * cols, Int32(cols), xp.baseAddress!, 1, 1, yp.baseAddress! + start, 1)
                    }
                }
            }
        }
    }

    /// C[m, n] += X[m, k] · W[n, k]ᵀ with the n (weight rows) dimension split across cores.
    private static func parallelGEMMTransposed(_ w: [Float], n: Int, k: Int, x: UnsafePointer<Float>, m: Int, c: UnsafeMutablePointer<Float>) {
        let per = (n + chunks - 1) / chunks
        w.withUnsafeBufferPointer { wp in
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                let start = chunk * per
                let count = min(per, n - start)
                guard count > 0 else { return }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(count), Int32(k),
                            1, x, Int32(k), wp.baseAddress! + start * k, Int32(k), 1, c + start, Int32(n))
            }
        }
    }

    /// Projects all encoder frames of a segment once: [frames, hidden] = frames · W_encᵀ + b.
    func projectEncoder(_ frames: EncoderFrames) -> [Float] {
        let started = Date()
        defer { projectSeconds += Date().timeIntervalSince(started) }
        var input = [Float](repeating: 0, count: frames.count * Self.encoderSize)
        input.withUnsafeMutableBufferPointer { buffer in
            for t in 0..<frames.count { frames.copyFrame(t, into: buffer.baseAddress! + t * Self.encoderSize) }
        }
        var output = [Float](repeating: 0, count: frames.count * Self.hidden)
        for t in 0..<frames.count {
            output.replaceSubrange((t * Self.hidden)..<((t + 1) * Self.hidden), with: encoderProjectionBias)
        }
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, Int32(frames.count), Int32(Self.hidden), Int32(Self.encoderSize),
            1, input, Int32(Self.encoderSize), encoderProjection, Int32(Self.encoderSize), 1, &output, Int32(Self.hidden)
        )
        return output
    }

    struct Decision {
        let token: Int
        let durationBin: Int
        let probability: Float
    }

    /// Joint decisions for `count` consecutive frames starting at `start`, with the current
    /// decoder state: head · ReLU(f_t + g) for each frame, as one matrix product.
    func decide(projectedFrames: [Float], start: Int, count: Int) -> [Decision] {
        let started = Date()
        defer { decideCalls += 1; decideFrames += count; decideSeconds += Date().timeIntervalSince(started) }
        let n = Self.hidden
        var hidden = [Float](repeating: 0, count: count * n)
        projectedFrames.withUnsafeBufferPointer { f in
            hidden.withUnsafeMutableBufferPointer { h in
                for k in 0..<count {
                    vDSP_vadd(f.baseAddress! + (start + k) * n, 1, projectedState, 1, h.baseAddress! + k * n, 1, vDSP_Length(n))
                }
                var zero: Float = 0
                vDSP_vthres(h.baseAddress!, 1, &zero, h.baseAddress!, 1, vDSP_Length(count * n))  // ReLU
            }
        }
        if logits.count < count * Self.outputs { logits = [Float](repeating: 0, count: count * Self.outputs) }
        for k in 0..<count {
            logits.replaceSubrange((k * Self.outputs)..<((k + 1) * Self.outputs), with: headBias)
        }
        hidden.withUnsafeBufferPointer { h in
            logits.withUnsafeMutableBufferPointer { l in
                Self.parallelGEMMTransposed(head, n: Self.outputs, k: n, x: h.baseAddress!, m: count, c: l.baseAddress!)
            }
        }
        var decisions: [Decision] = []
        decisions.reserveCapacity(count)
        logits.withUnsafeBufferPointer { l in
            for k in 0..<count {
                let row = l.baseAddress! + k * Self.outputs
                var tokenMax: Float = 0, durationMax: Float = 0
                var tokenIndex: vDSP_Length = 0, durationIndex: vDSP_Length = 0
                vDSP_maxvi(row, 1, &tokenMax, &tokenIndex, vDSP_Length(Self.vocabulary))
                vDSP_maxvi(row + Self.vocabulary, 1, &durationMax, &durationIndex, vDSP_Length(ParakeetConstants.durationBins.count))
                // Softmax probability is only computed when debugging: it costs 8193 exps per frame.
                var probability: Float = -1
                if TdtDecoder.debug {
                    var sum: Float = 0
                    for i in 0..<Self.vocabulary { sum += exp(row[i] - tokenMax) }
                    probability = 1 / sum
                }
                decisions.append(Decision(token: Int(tokenIndex), durationBin: Int(durationIndex), probability: probability))
            }
        }
        return decisions
    }
}
