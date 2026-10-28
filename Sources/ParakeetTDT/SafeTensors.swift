import Accelerate
import Foundation

/// Minimal safetensors reader: header JSON plus raw little-endian tensor data.
/// F16 and F32 tensors are returned as Float32 arrays.
struct SafeTensors {
    private struct Entry: Decodable {
        let dtype: String
        let shape: [Int]
        let data_offsets: [Int]
    }

    private let data: Data
    private let entries: [String: Entry]
    private let base: Int

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        var header = try JSONDecoder().decode([String: Entry].self, from: data.subdata(in: 8..<(8 + headerLength)))
        header.removeValue(forKey: "__metadata__")
        entries = header
        base = 8 + headerLength
    }

    var names: Set<String> { Set(entries.keys) }

    func shape(of name: String) -> [Int]? { entries[name]?.shape }

    func floats(_ name: String) throws -> [Float] {
        guard let entry = entries[name] else { throw ParakeetError.missingFile("tensor \(name)") }
        let range = (base + entry.data_offsets[0])..<(base + entry.data_offsets[1])
        let count = entry.shape.reduce(1, *)
        switch entry.dtype {
        case "F32":
            return data.subdata(in: range).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case "F16":
            var output = [Float](repeating: 0, count: count)
            data.subdata(in: range).withUnsafeBytes { raw in
                var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress!), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
                output.withUnsafeMutableBufferPointer { out in
                    var destination = vImage_Buffer(data: out.baseAddress!, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
                    vImageConvert_Planar16FtoPlanarF(&source, &destination, 0)
                }
            }
            return output
        default:
            throw ParakeetError.unexpectedShape("\(name) dtype \(entry.dtype)", entry.shape)
        }
    }
}
