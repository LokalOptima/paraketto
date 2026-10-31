import AppKit

/// Menu-bar images derived from the supplied Resources/paraketto.png artwork during bundling.
enum ParakeetArtwork {
    static let menuIcon: NSImage = {
        guard let url = Bundle.main.url(forResource: "ParakettoMenu", withExtension: "png"),
              let image = NSImage(contentsOf: url) else {
            return NSImage(systemSymbolName: "mic", accessibilityDescription: "Paraketto")!
        }
        image.size = NSSize(width: 20, height: 20)
        image.isTemplate = true
        return image
    }()

    static let recordingIcon: NSImage = {
        let image = NSImage(size: NSSize(width: 20, height: 20), flipped: false) { rect in
            menuIcon.draw(in: rect)
            NSColor.black.setFill()
            NSBezierPath(ovalIn: CGRect(x: 16.5, y: 16.5, width: 3.5, height: 3.5)).fill()
            return true
        }
        image.isTemplate = true
        return image
    }()
}
