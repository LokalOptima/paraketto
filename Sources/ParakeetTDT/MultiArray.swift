import CoreML
import Foundation

extension MLMultiArray {
    /// A zero-filled Float32 array of the given shape.
    static func float32(_ shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.stride)
        return array
    }

    /// An Int32 array holding one value.
    static func int32(_ value: Int, shape: [Int] = [1]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        array.dataPointer.assumingMemoryBound(to: Int32.self).initialize(repeating: Int32(value), count: array.count)
        return array
    }

    var floatPointer: UnsafeMutablePointer<Float> {
        dataPointer.assumingMemoryBound(to: Float.self)
    }

    var firstInt: Int {
        switch dataType {
        case .int32: return Int(dataPointer.assumingMemoryBound(to: Int32.self).pointee)
        default: return Int(self[0].doubleValue)
        }
    }

    var firstFloat: Float {
        switch dataType {
        case .float32: return floatPointer.pointee
        default: return self[0].floatValue
        }
    }

    /// Copies `count` contiguous Float32 values from this array's start into `destination`,
    /// regardless of the array's shape (used for the decoder output, [1,640,1] or [1,1,640]).
    func copyFloats(count: Int, to destination: UnsafeMutablePointer<Float>) {
        precondition(dataType == .float32 && self.count >= count)
        destination.update(from: floatPointer, count: count)
    }
}

extension MLFeatureProvider {
    func array(_ name: String) throws -> MLMultiArray {
        guard let value = featureValue(for: name)?.multiArrayValue else {
            throw ParakeetError.missingOutput(name)
        }
        return value
    }
}

/// Zero-copy view over the encoder output, which Core ML may lay out as [1, hidden, time]
/// or [1, time, hidden]. Resolves the axes from the hidden size and uses the array's strides.
struct EncoderFrames {
    let count: Int
    let hiddenSize: Int
    private let base: UnsafeMutablePointer<Float>
    private let timeStride: Int
    private let hiddenStride: Int
    private let array: MLMultiArray  // keeps the storage alive

    init(_ array: MLMultiArray, validFrames: Int, hiddenSize: Int) throws {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, array.dataType == .float32 else {
            throw ParakeetError.unexpectedShape("encoder", shape)
        }
        let hiddenAxis: Int
        if shape[1] == hiddenSize { hiddenAxis = 1 } else if shape[2] == hiddenSize { hiddenAxis = 2 } else {
            throw ParakeetError.unexpectedShape("encoder", shape)
        }
        let timeAxis = hiddenAxis == 1 ? 2 : 1
        self.array = array
        self.hiddenSize = hiddenSize
        self.count = min(validFrames, shape[timeAxis])
        self.base = array.floatPointer
        self.timeStride = strides[timeAxis]
        self.hiddenStride = strides[hiddenAxis]
    }

    func copyFrame(_ index: Int, into destination: UnsafeMutablePointer<Float>) {
        let source = base + index * timeStride
        if hiddenStride == 1 {
            destination.update(from: source, count: hiddenSize)
        } else {
            for i in 0..<hiddenSize { destination[i] = source[i * hiddenStride] }
        }
    }
}
