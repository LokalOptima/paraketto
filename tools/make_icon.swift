import AppKit
import CoreImage

@main
struct MakeIcon {
    static func main() throws {
        let sourceURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        guard let source = NSImage(contentsOf: sourceURL),
              let original = source.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            fatalError("Cannot load icon at \(sourceURL.path)")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for points in [16, 32, 128, 256, 512] {
            for scale in [1, 2] {
                let suffix = scale == 2 ? "@2x" : ""
                let name = "icon_\(points)x\(points)\(suffix).png"
                try write(original, pixels: points * scale, rounded: true, to: directory.appendingPathComponent(name))
            }
        }

        // The supplied white-on-black artwork becomes an alpha mask for macOS tinting.
        let mask = CIImage(cgImage: original).applyingFilter("CIMaskToAlpha")
        let menuImage = CIContext().createCGImage(mask, from: mask.extent)!
        try write(menuImage, pixels: 40, to: directory.deletingLastPathComponent().appendingPathComponent("ParakettoMenu.png"))
    }

    private static func write(_ image: CGImage, pixels: Int, rounded: Bool = false, to url: URL) throws {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        let context = NSGraphicsContext(bitmapImageRep: bitmap)!.cgContext
        context.interpolationQuality = .high
        let side = CGFloat(pixels)
        let canvas = CGRect(x: 0, y: 0, width: side, height: side)
        // Give the app icon transparent margins and rounded corners at every size.
        let bounds = rounded ? canvas.insetBy(dx: side / 16, dy: side / 16) : canvas
        if rounded {
            let radius = side * 196 / 1024
            context.addPath(CGPath(roundedRect: bounds, cornerWidth: radius, cornerHeight: radius, transform: nil))
            context.clip()
        }
        let scale = min(bounds.width / CGFloat(image.width), bounds.height / CGFloat(image.height))
        let width = CGFloat(image.width) * scale
        let height = CGFloat(image.height) * scale
        context.draw(image, in: CGRect(x: (side - width) / 2, y: (side - height) / 2, width: width, height: height))
        try bitmap.representation(using: .png, properties: [:])!.write(to: url)
    }
}
