import AVFoundation
import CoreML
import Foundation
import ParakeetTDT

/// Transcribes WAV files with the ParakeetTDT library and optionally checks the text
/// against a reference TSV (file name, tab, expected text), for comparison with FluidAudio.
///
///   paraketto-cli [--vad VadHead.mlmodelc] [--compare reference.tsv] [--vad-probe] file.wav ...

var files: [String] = []
var vadPath: String?
var comparePath: String?
var vadProbe = false
var showTokens = false
var showPlan = false
var encoderPath: String?
var chunkMilliseconds = 0  // > 0 feeds each file to a session in chunks, as the app does while recording
var useFused = true
var fusedPath: String?
var arguments = Array(CommandLine.arguments.dropFirst())
while !arguments.isEmpty {
    let argument = arguments.removeFirst()
    switch argument {
    case "--vad": vadPath = arguments.removeFirst()
    case "--compare": comparePath = arguments.removeFirst()
    case "--vad-probe": vadProbe = true
    case "--tokens": showTokens = true
    case "--plan": showPlan = true
    case "--encoder": encoderPath = arguments.removeFirst()
    case "--chunk-ms": chunkMilliseconds = Int(arguments.removeFirst()) ?? 0
    case "--fused":
        let value = arguments.removeFirst()
        useFused = value != "off"
        fusedPath = ["on", "off"].contains(value) ? nil : value
    default: files.append(argument)
    }
}
/// Prints, per model, how Core ML assigns its operations to compute devices (its `MLComputePlan`).
@available(macOS 14.4, *)
func printComputePlans(directory: URL, vad: URL?) async {
    var models: [(String, URL, MLComputeUnits)] = [
        ("Preprocessor", directory.appendingPathComponent("Preprocessor.mlmodelc"), .cpuOnly),
        ("Encoder", directory.appendingPathComponent("Encoder.mlmodelc"), .cpuAndNeuralEngine),
        ("Decoder", directory.appendingPathComponent("Decoder.mlmodelc"), .cpuAndNeuralEngine),
        ("JointDecisionv3", directory.appendingPathComponent("JointDecisionv3.mlmodelc"), .cpuAndNeuralEngine),
    ]
    if let vad { models.append(("VadHead", vad, .cpuAndNeuralEngine)) }
    if let own = ModelStore.ownEncoderURL { models.append(("Encoder (ours, default function)", own, .cpuAndNeuralEngine)) }
    for (name, url, units) in models {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        guard let plan = try? await MLComputePlan.load(contentsOf: url, configuration: configuration),
              case .program(let program) = plan.modelStructure,
              let main = program.functions["main"]
        else { print("\(name): no compute plan"); continue }
        var byDevice: [String: (count: Int, weight: Double)] = [:]
        var unsupported: [String: Int] = [:]
        for operation in main.block.operations {
            let cost = plan.estimatedCost(of: operation)?.weight ?? 0
            guard let usage = plan.deviceUsage(for: operation) else { continue }
            let device: String
            switch usage.preferred {
            case .cpu: device = "CPU"
            case .gpu: device = "GPU"
            case .neuralEngine: device = "ANE"
            @unknown default: device = "other"
            }
            byDevice[device, default: (0, 0)].count += 1
            byDevice[device, default: (0, 0)].weight += cost
            if device.contains("CPU") && usage.supported.count == 1 { unsupported[operation.operatorName, default: 0] += 1 }
        }
        let summary = byDevice.sorted { $0.value.weight > $1.value.weight }
            .map { String(format: "%@ %d ops %.0f%%", $0.key, $0.value.count, $0.value.weight * 100) }
            .joined(separator: ", ")
        print("\(name): \(main.block.operations.count) ops → \(summary)")
        if !unsupported.isEmpty {
            print("  CPU-only operators: " + unsupported.sorted { $0.value > $1.value }.map { "\($0.key)×\($0.value)" }.joined(separator: " "))
        }
    }
}

guard !files.isEmpty || showPlan else {
    FileHandle.standardError.write(Data("usage: paraketto-cli [--vad DIR.mlmodelc] [--compare ref.tsv] [--vad-probe] file.wav ...\n".utf8))
    exit(2)
}

func loadSamples(_ path: String) throws -> [Float] {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
        throw ParakeetError.missingFile(path)
    }
    try file.read(into: input)
    if file.processingFormat == target {
        return Array(UnsafeBufferPointer(start: input.floatChannelData![0], count: Int(input.frameLength)))
    }
    let converter = AVAudioConverter(from: file.processingFormat, to: target)!
    let ratio = 16_000 / file.processingFormat.sampleRate
    let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024)!
    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, status in
        if consumed { status.pointee = .endOfStream; return nil }
        consumed = true
        status.pointee = .haveData
        return input
    }
    if let error { throw error }
    return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
}

let started = Date()
let directory = try await ModelStore.locate { fraction in
    FileHandle.standardError.write(Data(String(format: "downloading %.0f%%\n", fraction * 100).utf8))
}
if showPlan {
    if #available(macOS 14.4, *) {
        await printComputePlans(directory: directory, vad: vadPath.map { URL(fileURLWithPath: $0) })
    } else {
        print("MLComputePlan needs macOS 14.4")
    }
    if files.isEmpty { exit(0) }
}
let encoderURL = encoderPath.map { URL(fileURLWithPath: $0) } ?? ModelStore.ownEncoderURL
let fusedURL = useFused ? (fusedPath.map { URL(fileURLWithPath: $0) } ?? ModelStore.ownFusedDecoderURL) : nil
let models = try ParakeetModels(directory: directory, vadModel: vadPath.map { URL(fileURLWithPath: $0) }, encoderModel: encoderURL, fusedDecoder: fusedURL)
FileHandle.standardError.write(Data("fused decoder step: \(fusedURL == nil ? "off" : "on")\n".utf8))
FileHandle.standardError.write(Data("encoder: \(encoderURL?.path ?? "FluidAudio 15 s") windows \(models.encoder.variants.map(\.melFrames))\n".utf8))
let transcriber = try ParakeetTranscriber(models: models)
FileHandle.standardError.write(Data(String(format: "models loaded in %.0f ms from %@\n", Date().timeIntervalSince(started) * 1000, directory.path).utf8))

var reference: [String: String] = [:]
if let comparePath {
    for line in try String(contentsOfFile: comparePath, encoding: .utf8).split(separator: "\n") where !line.hasPrefix("#") {
        let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
        if parts.count == 2 { reference[parts[0]] = parts[1] }
    }
}

// Warm up once so timings reflect steady state.
if let first = files.first { _ = try transcriber.transcribe(try loadSamples(first)) }

var mismatches = 0
var totalAudio = 0.0
var totalTime = 0.0
for path in files {
    let samples = try loadSamples(path)
    let name = URL(fileURLWithPath: path).lastPathComponent
    if vadProbe {
        let probabilities = try transcriber.speechProbabilities(samples)
        print("\(name): " + probabilities.map { $0 >= 0.5 ? "#" : "." }.joined())
        continue
    }
    let clock = Date()
    let result: Transcription
    var finishElapsed = 0.0
    if chunkMilliseconds > 0 {
        let session = transcriber.startSession()
        let chunk = chunkMilliseconds * 16
        for start in stride(from: 0, to: samples.count, by: chunk) {
            try session.append(Array(samples[start..<min(start + chunk, samples.count)]))
        }
        let finishClock = Date()
        result = try session.finish()
        finishElapsed = Date().timeIntervalSince(finishClock)
    } else {
        result = try transcriber.transcribe(samples)
    }
    let elapsed = Date().timeIntervalSince(clock)
    let seconds = Double(samples.count) / 16_000
    totalAudio += seconds
    totalTime += elapsed
    var status = ""
    if let expected = reference[name] {
        status = expected == result.text ? "  [match]" : "  [DIFF]"
        if expected != result.text {
            mismatches += 1
            print("  expected: \(expected)")
        }
    }
    let finishNote = chunkMilliseconds > 0 ? String(format: " (%.0f ms after the last chunk)", finishElapsed * 1000) : ""
    print(String(format: "%@\t%.1fs audio, %.0f ms%@, %.0fx, %d segment(s)%@\n\t%@", name, seconds, elapsed * 1000, finishNote, seconds / elapsed, result.segments, status, result.text))
    if showTokens {
        print("\t" + result.tokens.map { "\($0.id)@\($0.frame)" }.joined(separator: " "))
    }
}
if !vadProbe {
    print(String(format: "TOTAL %.1fs audio in %.0f ms = %.0fx RTFx", totalAudio, totalTime * 1000, totalAudio / totalTime))
    if comparePath != nil { print("mismatches: \(mismatches) of \(files.count)") }
}
exit(mismatches == 0 ? 0 : 1)
