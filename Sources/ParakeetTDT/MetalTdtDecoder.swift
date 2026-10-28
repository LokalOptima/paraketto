import Foundation
import Metal

/// Greedy TDT decoding with the prediction network and joint on the GPU.
///
/// The per-token work is a handful of matrix-vector products over about 26 MB of fp16 weights,
/// which on the CPU is bound by one core's memory bandwidth (0.7 ms per token in Core ML). The
/// GPU reads the same fp16 weights at several hundred GB/s, and the whole greedy loop is encoded
/// as one chain of dependent dispatches: the argmax kernel picks the token, applies the TDT
/// duration rule, advances the frame pointer and gathers the next embedding, so the CPU is not
/// consulted between steps. Kernels of a finished or blank step exit at once (a few µs each).
///
/// Numerics match the Core ML models: fp16 weights, fp32 arithmetic. Summation order differs, so
/// a near-tie in the argmax can in principle resolve differently.
final class MetalTdtDecoder {
    static let hidden = ParakeetConstants.decoderHiddenSize            // 640
    static let gates = 4 * hidden                                      // 2560
    static let vocabulary = ParakeetConstants.blankToken + 1           // 8193 incl. blank
    static let outputs = vocabulary + ParakeetConstants.durationBins.count  // 8198
    static let encoderSize = ParakeetConstants.encoderHiddenSize       // 1024
    /// Decode steps encoded per command buffer; the chain is resumed if the segment is not done.
    static let stepsPerCommandBuffer = 48
    static let maxTokensPerSegment = 1024

    private struct State {  // mirrors the Metal struct
        var frame: UInt32 = 0, symbolsAtFrame: UInt32 = 0, done: UInt32 = 0, hasPending: UInt32 = 0, count: UInt32 = 0
    }
    private struct Params { var frames: UInt32, rangeEnd: UInt32, maxTokens: UInt32 }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let projectEncoder, matvec, cell, joint, decide: MTLComputePipelineState

    // Weights, fp16 unless noted.
    private let embedding, weightIH0, weightHH0, weightIH1, weightHH1, decoderProjection, encoderProjection, head: MTLBuffer
    private let bias0, bias1, decoderProjectionBias, encoderProjectionBias, headBias: MTLBuffer  // fp32
    private let wordStart: MTLBuffer  // uint8 per token
    private let blankEmbedding: [Float]

    // State carried across segments of one recording.
    private let h0, c0, h1, c1, x, projectedState: MTLBuffer  // fp32 [hidden]
    private let gates0, gates1: MTLBuffer                     // fp32 [gates]
    private let logits: MTLBuffer                             // fp32 [outputs]
    private let state, params, tokens, frames: MTLBuffer
    private var primed = false

    // Per-segment inputs.
    private var encoderBuffer: MTLBuffer
    private var projectedEncoder: MTLBuffer

    private(set) var commandBuffers = 0, steps = 0, seconds = 0.0
    func resetTiming() { commandBuffers = 0; steps = 0; seconds = 0 }
    var timingSummary: String {
        String(format: "metal decoder: %d command buffers, %d steps encoded, %.1f ms", commandBuffers, steps, seconds * 1000)
    }

    init(weights: SafeTensors, isWordStart: (Int) -> Bool) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw ParakeetError.missingFile("Metal device")
        }
        self.device = device
        self.queue = queue
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw ParakeetError.missingFile("kernel \(name)") }
            return try device.makeComputePipelineState(function: function)
        }
        projectEncoder = try pipeline("project_encoder")
        matvec = try pipeline("matvec")
        cell = try pipeline("lstm_cell")
        joint = try pipeline("joint")
        decide = try pipeline("decide")

        func tensor(_ name: String, _ shape: [Int]) throws -> [Float] {
            let actual = weights.shape(of: name) ?? []
            guard actual == shape || actual == shape + [1] else { throw ParakeetError.unexpectedShape(name, actual) }
            return try weights.floats(name)
        }
        func half(_ values: [Float]) -> MTLBuffer {
            let buffer = device.makeBuffer(length: values.count * 2, options: .storageModeShared)!
            let out = buffer.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<values.count { out[i] = Float16(values[i]) }
            return buffer
        }
        func float(_ values: [Float]) -> MTLBuffer {
            device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)!
        }
        let H = Self.hidden, G = Self.gates
        let embeddingValues = try tensor("decoder.embedding.weight", [Self.vocabulary, H])
        embedding = half(embeddingValues)
        blankEmbedding = Array(embeddingValues[(ParakeetConstants.blankToken * H)..<((ParakeetConstants.blankToken + 1) * H)])
        weightIH0 = half(try tensor("decoder.lstm.weight_ih_l0", [G, H]))
        weightHH0 = half(try tensor("decoder.lstm.weight_hh_l0", [G, H]))
        weightIH1 = half(try tensor("decoder.lstm.weight_ih_l1", [G, H]))
        weightHH1 = half(try tensor("decoder.lstm.weight_hh_l1", [G, H]))
        bias0 = float(zip(try tensor("decoder.lstm.bias_ih_l0", [G]), try tensor("decoder.lstm.bias_hh_l0", [G])).map(+))
        bias1 = float(zip(try tensor("decoder.lstm.bias_ih_l1", [G]), try tensor("decoder.lstm.bias_hh_l1", [G])).map(+))
        decoderProjection = half(try tensor("decoder.decoder_projector.weight", [H, H]))
        decoderProjectionBias = float(try tensor("decoder.decoder_projector.bias", [H]))
        encoderProjection = half(try tensor("encoder_projector.weight", [H, Self.encoderSize]))
        encoderProjectionBias = float(try tensor("encoder_projector.bias", [H]))
        head = half(try tensor("joint.head.weight", [Self.outputs, H]))
        headBias = float(try tensor("joint.head.bias", [Self.outputs]))
        var starts = [UInt8](repeating: 0, count: Self.vocabulary)
        for token in 0..<ParakeetConstants.blankToken where isWordStart(token) { starts[token] = 1 }
        wordStart = device.makeBuffer(bytes: starts, length: starts.count, options: .storageModeShared)!

        func zeros(_ count: Int) -> MTLBuffer { device.makeBuffer(length: count * 4, options: .storageModeShared)! }
        h0 = zeros(H); c0 = zeros(H); h1 = zeros(H); c1 = zeros(H); x = zeros(H); projectedState = zeros(H)
        gates0 = zeros(G); gates1 = zeros(G); logits = zeros(Self.outputs)
        state = zeros(MemoryLayout<State>.size / 4); params = zeros(MemoryLayout<Params>.size / 4)
        tokens = zeros(Self.maxTokensPerSegment); frames = zeros(Self.maxTokensPerSegment)
        let windowFrames = ParakeetConstants.windowSamples / ParakeetConstants.samplesPerFrame + 1
        encoderBuffer = zeros(windowFrames * Self.encoderSize)
        projectedEncoder = zeros(windowFrames * H)
        warmUp()
    }

    /// Runs every kernel once on zeros so the first real decode does not pay for pipeline and
    /// buffer setup (about 90 ms on an M1 Max).
    private func warmUp() {
        state.contents().storeBytes(of: State(hasPending: 1), as: State.self)
        params.contents().storeBytes(of: Params(frames: 1, rangeEnd: 1, maxTokens: 1), as: Params.self)
        let commandBuffer = queue.makeCommandBuffer()!
        let pass = commandBuffer.makeComputeCommandEncoder()!
        encodeProjection(pass, frames: 1)
        encodePredictionStep(pass)
        encodeJointStep(pass)
        encodePredictionStep(pass)
        pass.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        reset()
    }

    /// Call once per recording before the first segment.
    func reset() {
        for buffer in [h0, c0, h1, c1, projectedState] { memset(buffer.contents(), 0, buffer.length) }
        primed = false
    }

    func decode(frames encoder: EncoderFrames, range: Range<Int>) -> TdtDecoder.Result {
        let range = range.clamped(to: 0..<encoder.count)
        guard !range.isEmpty else { return .init(tokens: [], frames: [], stopFrame: range.lowerBound) }
        let started = DispatchTime.now().uptimeNanoseconds
        let H = Self.hidden

        if encoderBuffer.length < encoder.count * Self.encoderSize * 4 {
            encoderBuffer = device.makeBuffer(length: encoder.count * Self.encoderSize * 4, options: .storageModeShared)!
            projectedEncoder = device.makeBuffer(length: encoder.count * H * 4, options: .storageModeShared)!
        }
        let encoderPointer = encoderBuffer.contents().assumingMemoryBound(to: Float.self)
        for frame in 0..<encoder.count { encoder.copyFrame(frame, into: encoderPointer + frame * Self.encoderSize) }

        var initial = State(frame: UInt32(range.lowerBound))
        if !primed {
            // A fresh utterance primes the prediction network with blank as start-of-sequence.
            x.contents().copyMemory(from: blankEmbedding, byteCount: H * 4)
            initial.hasPending = 1
            primed = true
        }
        state.contents().storeBytes(of: initial, as: State.self)
        let parameters = Params(frames: UInt32(encoder.count), rangeEnd: UInt32(range.upperBound), maxTokens: UInt32(Self.maxTokensPerSegment))
        params.contents().storeBytes(of: parameters, as: Params.self)

        var first = true
        repeat {
            let commandBuffer = queue.makeCommandBuffer()!
            let encoderPass = commandBuffer.makeComputeCommandEncoder()!
            if first {
                encodeProjection(encoderPass, frames: encoder.count)
                encodePredictionStep(encoderPass)  // consumes the priming token, if any
                first = false
            }
            for _ in 0..<Self.stepsPerCommandBuffer {
                encodeJointStep(encoderPass)
                encodePredictionStep(encoderPass)
            }
            encoderPass.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            commandBuffers += 1
            steps += Self.stepsPerCommandBuffer
            if TdtDecoder.debug { debugDump() }
        } while state.contents().load(as: State.self).done == 0

        let final = state.contents().load(as: State.self)
        let count = Int(final.count)
        let emittedTokens = Array(UnsafeBufferPointer(start: tokens.contents().assumingMemoryBound(to: UInt32.self), count: count)).map(Int.init)
        let emittedFrames = Array(UnsafeBufferPointer(start: frames.contents().assumingMemoryBound(to: UInt32.self), count: count)).map(Int.init)
        seconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        return .init(tokens: emittedTokens, frames: emittedFrames, stopFrame: max(Int(final.frame), range.upperBound))
    }

    private func debugDump() {
        func stats(_ buffer: MTLBuffer, _ count: Int) -> String {
            let p = UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count)
            let bad = p.filter { !$0.isFinite }.count
            return String(format: "max %.3g%@", p.map { abs($0) }.max() ?? 0, bad > 0 ? " (\(bad) non-finite)" : "")
        }
        let s = state.contents().load(as: State.self)
        let last = s.count > 0 ? Int(tokens.contents().assumingMemoryBound(to: UInt32.self)[Int(s.count) - 1]) : -1
        FileHandle.standardError.write(Data("metal: frame \(s.frame) count \(s.count) last \(last) done \(s.done) pending \(s.hasPending) | x \(stats(x, Self.hidden)) h0 \(stats(h0, Self.hidden)) c0 \(stats(c0, Self.hidden)) h1 \(stats(h1, Self.hidden)) c1 \(stats(c1, Self.hidden)) proj \(stats(projectedState, Self.hidden)) logits \(stats(logits, Self.outputs))\n".utf8))
    }

    // MARK: Encoding

    private func encodeProjection(_ encoder: MTLComputeCommandEncoder, frames: Int) {
        encoder.setComputePipelineState(projectEncoder)
        encoder.setBuffers([encoderBuffer, encoderProjection, encoderProjectionBias, projectedEncoder], offsets: [0, 0, 0, 0], range: 0..<4)
        var count = UInt32(frames)
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: frames, height: Self.hidden, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
    }

    /// joint → decide: logits for the current frame, then token choice and TDT bookkeeping.
    private func encodeJointStep(_ encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(joint)
        encoder.setBuffers([head, headBias, projectedEncoder, projectedState, state, logits], offsets: [0, 0, 0, 0, 0, 0], range: 0..<6)
        encoder.dispatchThreads(MTLSize(width: Self.outputs * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(decide)
        encoder.setBuffers([logits, embedding, wordStart, state, params, x, tokens, frames], offsets: [Int](repeating: 0, count: 8), range: 0..<8)
        encoder.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Two LSTM layers and the decoder projection for the pending token; no-ops when none is pending.
    private func encodePredictionStep(_ encoder: MTLComputeCommandEncoder) {
        encodeMatvec(encoder, weightIH0, x, weightHH0, h0, bias0, gates0, rows: Self.gates, columns: Self.hidden)
        encodeCell(encoder, gates0, h0, c0)
        encodeMatvec(encoder, weightIH1, h0, weightHH1, h1, bias1, gates1, rows: Self.gates, columns: Self.hidden)
        encodeCell(encoder, gates1, h1, c1)
        encodeMatvec(encoder, decoderProjection, h1, nil, nil, decoderProjectionBias, projectedState, rows: Self.hidden, columns: Self.hidden)
    }

    private func encodeMatvec(_ encoder: MTLComputeCommandEncoder, _ weightA: MTLBuffer, _ inputA: MTLBuffer, _ weightB: MTLBuffer?, _ inputB: MTLBuffer?,
                              _ bias: MTLBuffer, _ output: MTLBuffer, rows: Int, columns: Int) {
        encoder.setComputePipelineState(matvec)
        encoder.setBuffers([weightA, inputA, weightB ?? weightA, inputB ?? inputA, bias, output, state], offsets: [Int](repeating: 0, count: 7), range: 0..<7)
        var shape = SIMD4<UInt32>(UInt32(columns), UInt32(rows), weightB == nil ? 0 : 1, 0)
        encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.size, index: 7)
        encoder.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeCell(_ encoder: MTLComputeCommandEncoder, _ gates: MTLBuffer, _ h: MTLBuffer, _ c: MTLBuffer) {
        encoder.setComputePipelineState(cell)
        encoder.setBuffers([gates, h, c, state], offsets: [0, 0, 0, 0], range: 0..<4)
        encoder.dispatchThreads(MTLSize(width: Self.hidden, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: Kernels

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint H = \(hidden);
    constant uint V = \(vocabulary);
    constant uint OUT = \(outputs);
    constant uint ENC = \(encoderSize);
    constant uint BLANK = \(ParakeetConstants.blankToken);
    constant uint MAX_SYMBOLS = \(ParakeetConstants.maxSymbolsPerFrame);
    constant uint DURATIONS[5] = {\(ParakeetConstants.durationBins.map(String.init).joined(separator: ", "))};

    struct State { uint frame; uint symbolsAtFrame; uint done; uint hasPending; uint count; };
    struct Params { uint frames; uint rangeEnd; uint maxTokens; };

    // projected[t, row] = We[row, :] . enc[t, :] + be[row]; one thread per (t, row), rows of We are
    // shared across the 32 lanes of a simdgroup (consecutive t), so the weight is read once per row.
    kernel void project_encoder(device const float* enc [[buffer(0)]], device const half* We [[buffer(1)]],
                                device const float* be [[buffer(2)]], device float* projected [[buffer(3)]],
                                constant uint& frames [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
        uint t = id.x, row = id.y;
        if (t >= frames || row >= H) return;
        device const half4* w = (device const half4*)(We + row * ENC);
        device const float4* e = (device const float4*)(enc + t * ENC);
        float acc = 0;
        for (uint k = 0; k < ENC / 4; k++) acc += dot(float4(w[k]), e[k]);
        projected[t * H + row] = acc + be[row];
    }

    // y = Wa . a (+ Wb . b) + bias; one simdgroup per row, fp16 weights, fp32 accumulation.
    // Runs only while a token is pending. Nothing in the chain writes the flag: threadgroups of
    // one dispatch run in any order, so a flag cleared mid-dispatch would skip part of the output.
    kernel void matvec(device const half* Wa [[buffer(0)]], device const float* a [[buffer(1)]],
                       device const half* Wb [[buffer(2)]], device const float* b [[buffer(3)]],
                       device const float* bias [[buffer(4)]], device float* y [[buffer(5)]],
                       device State* state [[buffer(6)]], constant uint4& shape [[buffer(7)]],
                       uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
        if (state->hasPending == 0) return;
        uint K = shape.x, N = shape.y, row = tid / 32;
        if (row >= N) return;
        device const half4* wa = (device const half4*)(Wa + row * K);
        device const float4* av = (device const float4*)a;
        float acc = 0;
        for (uint k = lane; k < K / 4; k += 32) acc += dot(float4(wa[k]), av[k]);
        if (shape.z) {
            device const half4* wb = (device const half4*)(Wb + row * K);
            device const float4* bv = (device const float4*)b;
            for (uint k = lane; k < K / 4; k += 32) acc += dot(float4(wb[k]), bv[k]);
        }
        acc = simd_sum(acc);
        if (lane == 0) y[row] = acc + bias[row];
    }

    // PyTorch gate order i, f, g, o.
    kernel void lstm_cell(device const float* gates [[buffer(0)]], device float* h [[buffer(1)]], device float* c [[buffer(2)]],
                          device const State* state [[buffer(3)]], uint j [[thread_position_in_grid]]) {
        if (state->hasPending == 0 || j >= H) return;
        float i = 1 / (1 + exp(-gates[j]));
        float f = 1 / (1 + exp(-gates[H + j]));
        // Fast-math tanh overflows to NaN past about 44; fp32 tanh is exactly ±1 from 10 on.
        float g = tanh(clamp(gates[2 * H + j], -15.0f, 15.0f));
        float o = 1 / (1 + exp(-gates[3 * H + j]));
        float cn = f * c[j] + i * g;
        c[j] = cn;
        h[j] = o * tanh(clamp(cn, -15.0f, 15.0f));
    }

    // logits = Wh . relu(projectedEncoder[frame] + projectedState) + bh
    kernel void joint(device const half* Wh [[buffer(0)]], device const float* bh [[buffer(1)]],
                      device const float* projected [[buffer(2)]], device const float* projectedState [[buffer(3)]],
                      device const State* state [[buffer(4)]], device float* logits [[buffer(5)]],
                      uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
        if (state->done) return;
        uint row = tid / 32;
        if (row >= OUT) return;
        device const half4* w = (device const half4*)(Wh + row * H);
        device const float4* f = (device const float4*)(projected + state->frame * H);
        device const float4* g = (device const float4*)projectedState;
        float acc = 0;
        for (uint k = lane; k < H / 4; k += 32) acc += dot(float4(w[k]), max(f[k] + g[k], 0.0f));
        acc = simd_sum(acc);
        if (lane == 0) logits[row] = acc + bh[row];
    }

    // One threadgroup: argmax over tokens and durations, TDT bookkeeping, next embedding.
    kernel void decide(device const float* logits [[buffer(0)]], device const half* E [[buffer(1)]],
                       device const uchar* wordStart [[buffer(2)]], device State* state [[buffer(3)]],
                       constant Params& params [[buffer(4)]], device float* x [[buffer(5)]],
                       device uint* tokens [[buffer(6)]], device uint* frames [[buffer(7)]],
                       uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
        if (state->done) {
            if (t == 0) state->hasPending = 0;  // the last token was fed once; do not feed it again
            return;
        }
        threadgroup float bestValue[32];
        threadgroup uint bestIndex[32];
        threadgroup uint emitted;
        float v = -INFINITY;
        uint idx = 0;
        for (uint n = t; n < V; n += 1024) { float l = logits[n]; if (l > v) { v = l; idx = n; } }
        float mv = simd_max(v);
        uint mi = simd_min(v == mv ? idx : 0xffffffffu);  // first index among ties, as reduce_argmax
        if (lane == 0) { bestValue[sg] = mv; bestIndex[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float bv = bestValue[0]; uint token = bestIndex[0];
            for (uint s = 1; s < 32; s++) {
                if (bestValue[s] > bv || (bestValue[s] == bv && bestIndex[s] < token)) { bv = bestValue[s]; token = bestIndex[s]; }
            }
            uint bin = 0;
            for (uint d = 1; d < OUT - V; d++) if (logits[V + d] > logits[V + bin]) bin = d;
            uint duration = DURATIONS[bin];
            State s = *state;
            s.hasPending = 0;  // the previous token's prediction step has run by now
            bool pastRange = s.frame >= params.rangeEnd;
            uint emit = 0;
            if (token == BLANK) {
                if (pastRange) s.done = 1;
                else { s.frame += max(duration, 1u); s.symbolsAtFrame = 0; }
            } else if (pastRange && wordStart[token]) {
                s.done = 1;  // the next segment starts with this word
            } else {
                tokens[s.count] = token; frames[s.count] = s.frame; s.count += 1;
                s.symbolsAtFrame += 1;
                if (duration == 0 && s.symbolsAtFrame >= MAX_SYMBOLS) duration = 1;
                if (duration > 0) s.symbolsAtFrame = 0;
                s.frame += duration;
                s.hasPending = 1;
                emit = token + 1;
                if (s.count >= params.maxTokens) s.done = 1;
            }
            if (s.frame >= params.frames) s.done = 1;
            *state = s;
            emitted = emit;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint e = emitted;
        if (e == 0) return;
        for (uint k = t; k < H; k += 1024) x[k] = float(E[(e - 1) * H + k]);
    }
    """
}
