import SwiftUI
import ParakeetTDT

/// Normal window content, so macOS renders both text labels and its selected segment.
struct RecognitionModePicker: View {
    @Binding var selection: ParakeetModel
    var isEnabled = true

    var body: some View {
        Picker("Mode", selection: $selection) {
            ForEach(ParakeetModel.allCases) { model in
                Text(isEnabled ? model.title : " ")
                    .accessibilityLabel(model.title)
                    .tag(model)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .disabled(!isEnabled)
        .overlay {
            // Keep native disabled interaction and segment styling, but draw readable
            // labels ourselves: macOS dims the selected label almost to invisibility.
            if !isEnabled {
                HStack(spacing: 0) {
                    ForEach(ParakeetModel.allCases) { model in
                        Text(model.title)
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity)
                    }
                }
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            }
        }
    }
}

struct MenuActionLabel: View {
    let title: String
    let shortcut: String

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(shortcut).foregroundStyle(.secondary)
        }
    }
}

struct MenuActionStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        MenuActionBody(isPressed: configuration.isPressed, label: configuration.label)
    }

    private struct MenuActionBody<Label: View>: View {
        let isPressed: Bool
        let label: Label
        @State private var isHovered = false
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            label
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .contentShape(Rectangle())
                .background {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(isEnabled && (isHovered || isPressed)
                              ? Color.primary.opacity(isPressed ? 0.15 : 0.08) : .clear)
                }
                .onHover { isHovered = $0 }
        }
    }
}
