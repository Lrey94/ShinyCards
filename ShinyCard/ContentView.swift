import SwiftUI

// MARK: - Card collections

/// Pluggable provider — currently uses pokemontcg.io's free image CDN for both
/// the original Base Set and Prismatic Evolutions. (Scrydex requires API key
/// + paid tier despite advertising "free" endpoints, so it's not used here.)
struct CardCollection: Identifiable, Hashable {
    let id: String
    let displayName: String
    let setID: String
    let cardNumbers: [Int]

    func imageURL(for number: Int) -> URL? {
        URL(string: "https://images.pokemontcg.io/\(setID)/\(number)_hires.png")
    }

    func cacheKey(for number: Int) -> String { "\(setID)/\(number)" }

    static let baseSet = CardCollection(
        id: "base1",
        displayName: "Base Set",
        setID: "base1",
        cardNumbers: Array(1...12)
    )

    static let prismaticEvolutions = CardCollection(
        id: "sv8pt5",
        displayName: "Prismatic Evolutions",
        setID: "sv8pt5",
        cardNumbers: Array(1...12)
    )

    static let all: [CardCollection] = [.baseSet, .prismaticEvolutions]
}

// MARK: - Content view

struct ContentView: View {
    @State private var holoEnabled = true
    @State private var resetTrigger = 0
    @State private var selectedCollection: CardCollection = .baseSet
    @State private var selectedCard: Int = 1
    @State private var isLoading = false

    var body: some View {
        ZStack {
            StarfieldBackground()
                .ignoresSafeArea()

            CardView(
                holoEnabled: $holoEnabled,
                resetTrigger: resetTrigger,
                cardURL: selectedCollection.imageURL(for: selectedCard),
                cardCacheKey: selectedCollection.cacheKey(for: selectedCard),
                onLoadingChange: { loading in isLoading = loading }
            )
            .frame(width: 450, height: 650)
            .offset(y: -40)
            .opacity(isLoading ? 0 : 1)
            .scaleEffect(isLoading ? 0.92 : 1.0)
            .animation(.easeInOut(duration: 0.25), value: isLoading)

            if isLoading {
                ProgressView()
                    .progressViewStyle(.circular)
                    .scaleEffect(1.6)
                    .tint(.white.opacity(0.85))
                    .offset(y: -40)
                    .transition(.opacity)
            }

            VStack {
                CollectionPicker(selected: $selectedCollection) { newCollection in
                    selectedCollection = newCollection
                    selectedCard = newCollection.cardNumbers.first ?? 1
                    resetTrigger &+= 1
                }
                .padding(.top, 8)

                Spacer()

                DeckRow(
                    collection: selectedCollection,
                    selectedCard: selectedCard
                ) { newCard in
                    guard newCard != selectedCard else { return }
                    selectedCard = newCard
                    resetTrigger &+= 1
                }
                .padding(.bottom, 12)

                HStack(spacing: 16) {
                    Button {
                        holoEnabled.toggle()
                    } label: {
                        Label(holoEnabled ? "Holo: On" : "Holo: Off",
                              systemImage: holoEnabled ? "sparkles" : "sparkle")
                            .frame(minWidth: 110)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(holoEnabled ? .pink : .gray)

                    Button {
                        resetTrigger &+= 1
                    } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                            .frame(minWidth: 90)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                }
                .font(.headline)
                .padding(.bottom, 30)
            }
        }
    }
}

// MARK: - Collection picker

private struct CollectionPicker: View {
    @Binding var selected: CardCollection
    let onChange: (CardCollection) -> Void

    var body: some View {
        Menu {
            ForEach(CardCollection.all) { collection in
                Button {
                    onChange(collection)
                } label: {
                    HStack {
                        Text(collection.displayName)
                        if collection == selected {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "rectangle.stack.fill")
                Text(selected.displayName)
                    .fontWeight(.semibold)
                Image(systemName: "chevron.down")
                    .font(.caption.weight(.bold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.2), lineWidth: 1))
        }
    }
}

// MARK: - Deck row

struct DeckRow: View {
    let collection: CardCollection
    let selectedCard: Int
    let onSelect: (Int) -> Void

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(collection.cardNumbers, id: \.self) { n in
                        Thumbnail(
                            url: collection.imageURL(for: n),
                            isSelected: n == selectedCard
                        )
                        .id(n)
                        .onTapGesture {
                            onSelect(n)
                            withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                                proxy.scrollTo(n, anchor: .center)
                            }
                        }
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
            }
            .onChange(of: selectedCard) { _, newValue in
                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                    proxy.scrollTo(newValue, anchor: .center)
                }
            }
            .onChange(of: collection) { _, _ in
                // Reset scroll to start when switching collections.
                withAnimation(.easeOut(duration: 0.3)) {
                    proxy.scrollTo(collection.cardNumbers.first ?? 1, anchor: .leading)
                }
            }
        }
    }
}

private struct Thumbnail: View {
    let url: URL?
    let isSelected: Bool

    var body: some View {
        AsyncImage(url: url) { phase in
            switch phase {
            case .success(let image):
                image.resizable().scaledToFill()
            case .empty:
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.05))
                    ProgressView().tint(.white.opacity(0.5))
                }
            case .failure:
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.red.opacity(0.2))
                    .overlay(Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.white.opacity(0.6)))
            @unknown default:
                EmptyView()
            }
        }
        .frame(width: isSelected ? 78 : 60,
               height: isSelected ? 110 : 84)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected
                        ? Color.yellow
                        : Color.white.opacity(0.18),
                        lineWidth: isSelected ? 2 : 1)
        )
        .shadow(color: isSelected ? .yellow.opacity(0.55) : .clear,
                radius: isSelected ? 10 : 0)
        .blur(radius: isSelected ? 0 : 0.6)
        .opacity(isSelected ? 1.0 : 0.55)
        .animation(.spring(response: 0.4, dampingFraction: 0.75),
                   value: isSelected)
    }
}

// MARK: - Starfield

private struct Star {
    let x: Double          // 0...1
    let baseY: Double      // 0...1
    let radius: Double     // pixels
    let speed: Double      // fraction of height per second
    let phase: Double      // 0...2π — twinkle offset
    let hue: Double        // 0...1
}

struct StarfieldBackground: View {
    private let stars: [Star]

    init(count: Int = 140) {
        var rng = SystemRandomNumberGenerator()
        self.stars = (0..<count).map { _ in
            Star(
                x: .random(in: 0...1, using: &rng),
                baseY: .random(in: 0...1, using: &rng),
                radius: .random(in: 0.4...2.2, using: &rng),
                speed: .random(in: 0.005...0.03, using: &rng),
                phase: .random(in: 0...(2 * .pi), using: &rng),
                hue: .random(in: 0.55...0.85, using: &rng)
            )
        }
    }

    var body: some View {
        TimelineView(.animation) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate

            ZStack {
                // Deep space gradient
                LinearGradient(
                    colors: [
                        Color(red: 0.02, green: 0.02, blue: 0.06),
                        Color(red: 0.06, green: 0.03, blue: 0.12),
                        Color(red: 0.01, green: 0.01, blue: 0.04)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )

                // Subtle nebula glow that drifts
                RadialGradient(
                    colors: [
                        Color(hue: 0.78, saturation: 0.6, brightness: 0.18, opacity: 0.55),
                        .clear
                    ],
                    center: UnitPoint(x: 0.3 + 0.05 * sin(t * 0.1),
                                      y: 0.4 + 0.05 * cos(t * 0.13)),
                    startRadius: 50,
                    endRadius: 500
                )
                .blendMode(.screen)

                Canvas { ctx, size in
                    for star in stars {
                        // Drift downward, wrapping at the bottom.
                        let y = (star.baseY + t * star.speed)
                            .truncatingRemainder(dividingBy: 1.0)
                        let twinkle = 0.35 + 0.65 *
                            (sin(t * 1.6 + star.phase) * 0.5 + 0.5)

                        let px = star.x * size.width
                        let py = y * size.height
                        let r  = star.radius

                        let rect = CGRect(x: px - r, y: py - r, width: r * 2, height: r * 2)
                        let color = Color(hue: star.hue,
                                          saturation: 0.15,
                                          brightness: 1.0,
                                          opacity: twinkle)
                        ctx.fill(Path(ellipseIn: rect), with: .color(color))
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
