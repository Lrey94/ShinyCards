import SwiftUI
import ARKit
import MetalKit

struct ARCardView: UIViewRepresentable {
    @Binding var holoEnabled: Bool
    let cardURL: URL?
    let cardCacheKey: String
    var onLoadingChange: ((Bool) -> Void)? = nil

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black
        view.isOpaque = true

        let session = ARSession()
        let renderer = ARRenderer(view: view, session: session)
        view.delegate = renderer

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        config.isLightEstimationEnabled = true
        session.run(config)

        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        view.addGestureRecognizer(tap)

        context.coordinator.session = session
        context.coordinator.renderer = renderer
        context.coordinator.view = view
        context.coordinator.lastCacheKey = cardCacheKey

        renderer.holoEnabled = holoEnabled
        renderer.onLoadingChange = onLoadingChange
        renderer.loadFrontTexture(url: cardURL, cacheKey: cardCacheKey)

        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.holoEnabled = holoEnabled
        context.coordinator.renderer?.onLoadingChange = onLoadingChange
        if context.coordinator.lastCacheKey != cardCacheKey {
            context.coordinator.lastCacheKey = cardCacheKey
            context.coordinator.renderer?.loadFrontTexture(url: cardURL, cacheKey: cardCacheKey)
        }
    }

    static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.session?.pause()
    }

    class Coordinator: NSObject {
        var session: ARSession?
        weak var view: MTKView?
        var renderer: ARRenderer?
        var lastCacheKey: String = ""

        @objc func handleTap(_ g: UITapGestureRecognizer) {
            guard let view, let session,
                  let frame = session.currentFrame else { return }
            let loc = g.location(in: view)
            let normalized = CGPoint(
                x: loc.x / view.bounds.width,
                y: loc.y / view.bounds.height
            )
            // Existing planes first; fall back to estimated planes for
            // smoother tap-to-place before plane detection has converged.
            let alignment: ARRaycastQuery.TargetAlignment = .horizontal
            let target: ARRaycastQuery.Target = .estimatedPlane

            let query = frame.raycastQuery(
                from: normalized,
                allowing: target,
                alignment: alignment
            )

            if let hit = session.raycast(query).first {
                renderer?.placeCard(at: hit.worldTransform)
            }
        }
    }
}
