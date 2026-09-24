# ShinyCard

An iOS app that renders holographic Pokémon cards in real time using Metal, with an AR mode that drops them onto real-world surfaces.

Built as a Metal / ARKit playground: quaternion rotation, a Fresnel-driven foil shader, a GPU particle system, real-time bloom post-processing, and passthrough AR with light-estimation-driven shading.

## Features

- **Holographic foil shader** — Fresnel-gated rainbow shift, per-pixel sparkle sampled from a noise texture, moving specular sheen, and a gold rim that glows brightest when the card is edge-on. Toggle on/off at runtime.
- **Interactive card** — pan to rotate with inertia + damping, pinch to zoom, single-tap to flip (smooth 180° quaternion slerp with cubic ease in/out), Reset button to return to identity rotation.
- **Ambient motion** — CoreMotion gyro tilt drives a subtle live response to how the phone is held; when idle, a two-frequency sine wobble makes the card "float" as if waiting to be handled. Gyro is auto-calibrated on first sample so launch pose is neutral.
- **Rounded corners** — signed-distance-field discard on every face of the card, so silhouettes never show a giveaway 90° corner.
- **GPU particle system** — 300 sparkles updated in a compute kernel, rendered as billboarded instanced quads with additive blending. Emitted only while the holo effect is on; per-particle warm/pink/cyan hue variation.
- **Real-time bloom** — six-pass pipeline: particle compute → scene render → bright extract → separable Gaussian blur (H then V, half-res) → composite. Holo highlights bleed convincingly outside the card silhouette.
- **AR mode** — full ARKit passthrough. Tap a real-world surface to place the card lying flat at real Pokémon-card scale (~6.3 cm wide). ARWorldTrackingConfiguration's light estimation drives holo intensity so the card responds to actual room lighting.
- **Card collections** — pick between the original Base Set and Prismatic Evolutions from a top menu. Deck row of thumbnails, tap to swap card (Metal view crossfades, texture is fetched async and cached per collection).
- **Animated starfield background** — SwiftUI Canvas + TimelineView with ~140 drifting, twinkling stars, a deep-space gradient, and a slowly drifting nebula glow. The MTKView is transparent so this shows through around the card edges.

## Architecture

```
ShinyCard/
├── ShinyCardApp.swift          # App entry point
├── ContentView.swift           # Main UI, collection menu, deck row, buttons
├── Metal/
│   ├── Card.swift              # Card state: rotation, inertia, tilt, idle wobble, flip
│   ├── CardView.swift          # UIViewRepresentable wrapping MTKView + gestures + CoreMotion
│   ├── Renderer.swift          # Main renderer: card, particles, bloom
│   ├── ARCardView.swift        # UIViewRepresentable for AR mode
│   ├── ARRenderer.swift        # AR renderer: camera passthrough + card + light estimate
│   └── MathUtils.swift         # matrix_float4x4 helpers (perspective, translation, quaternion)
└── Shaders.metal               # All Metal shaders (card, particles, bloom, AR camera)
```

## Requirements

- iOS 17+
- Xcode 15+
- A physical device (AR mode requires a real camera + gyro; simulator won't run AR)
- Add `NSCameraUsageDescription` to `Info.plist` before using AR

## How to try it

1. Clone and open `ShinyCard.xcodeproj` in Xcode.
2. Select your development team, then build & run on a device.
3. **Pan** the card to rotate it, **pinch** to zoom, **tap** to flip.
4. Toggle the **Holo** button to see the foil effect turn on/off.
5. Tap **AR** to enter AR mode, then tap a flat surface to place the card in your room.
6. Switch collections from the menu at the top to load different cards.

## Card images

Card fronts are fetched on demand from [pokemontcg.io](https://pokemontcg.io)'s free image CDN:

```
https://images.pokemontcg.io/{setId}/{cardNumber}_hires.png
```

Two collections are wired up out of the box:

- `base1` — Original Base Set (Charizard, Blastoise, Venusaur, etc.)
- `sv8pt5` — Prismatic Evolutions

Add more by extending `CardCollection.all` in `ContentView.swift`.

## Rendering pipeline notes

- **Vertex layout** — `Vertex { position, normal, uv, faceID }` uses `SIMD3<Float>` which is **16-byte aligned** in Swift (not 12). The vertex descriptor computes offsets via `MemoryLayout<Vertex>.offset(of:)` to stay correct.
- **Face IDs** — the card is a 6-face box. Front (0), back (1), and four edge faces (2–5). The fragment shader routes each face to a different code path (base art + holo, back art, or gold trim).
- **SDF corners** — every face runs the same `roundedCardSDF(localPos.xy)` discard, so rounded corners are consistent across the whole silhouette.
- **Bloom** — sceneTex is full-res `bgra8Unorm` with a `depth32Float` companion. Bright and blur textures are half-res for cost savings. Composite preserves the scene's alpha so the SwiftUI starfield still shows behind transparent regions.
- **AR** — the camera image is a biplanar YCbCr `CVPixelBuffer`; the fragment shader does the BT.601 full-range conversion. The card's real-world model matrix lays it flat on the detected surface and scales it to physical Pokémon-card dimensions.

## License / attribution

Pokémon card artwork is © Nintendo / Creatures Inc. / GAME FREAK inc. and served here from the community pokemontcg.io CDN. This project is for personal / educational use only — do not ship the card imagery in a distributed app.
