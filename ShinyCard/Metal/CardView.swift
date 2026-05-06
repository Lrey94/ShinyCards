import SwiftUI
import MetalKit
import CoreMotion

struct CardView: UIViewRepresentable {
    @Binding var holoEnabled: Bool
    let resetTrigger: Int
    let cardURL: URL?
    let cardCacheKey: String
    var onLoadingChange: ((Bool) -> Void)? = nil

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        let renderer = Renderer(view: view)
        view.delegate = renderer
        view.preferredFramesPerSecond = 60
        view.isPaused = false
        view.enableSetNeedsDisplay = false

        context.coordinator.renderer = renderer
        context.coordinator.view = view
        context.coordinator.lastResetTrigger = resetTrigger
        context.coordinator.lastCacheKey = cardCacheKey
        context.coordinator.startMotion()
        renderer.card.holoEnabled = holoEnabled
        renderer.onLoadingChange = onLoadingChange
        renderer.loadFrontTexture(url: cardURL, cacheKey: cardCacheKey)

        let pan = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        view.addGestureRecognizer(pan)

        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        tap.numberOfTapsRequired = 1
        view.addGestureRecognizer(tap)

        let pinch = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        view.addGestureRecognizer(pinch)

        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.card.holoEnabled = holoEnabled
        context.coordinator.renderer?.onLoadingChange = onLoadingChange
        if context.coordinator.lastResetTrigger != resetTrigger {
            context.coordinator.lastResetTrigger = resetTrigger
            context.coordinator.renderer?.card.reset()
        }
        if context.coordinator.lastCacheKey != cardCacheKey {
            context.coordinator.lastCacheKey = cardCacheKey
            context.coordinator.renderer?.loadFrontTexture(url: cardURL, cacheKey: cardCacheKey)
        }
    }

    class Coordinator: NSObject {
        var renderer: Renderer?
        weak var view: MTKView?
        var lastResetTrigger: Int = 0
        var lastCacheKey: String = ""
        private var pinchStartScale: Float = 1.0
        private let motionManager = CMMotionManager()

        private let sensitivity: Float = 0.01

        deinit { motionManager.stopDeviceMotionUpdates() }

        func startMotion() {
            guard motionManager.isDeviceMotionAvailable else { return }
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
                guard let self, let motion, let renderer = self.renderer else { return }
                // attitude.pitch and attitude.roll are in radians
                renderer.card.updateTilt(
                    pitch: Float(motion.attitude.pitch),
                    roll:  Float(motion.attitude.roll)
                )
            }
        }

        @objc func handlePan(_ g: UIPanGestureRecognizer) {
            guard let renderer, let view else { return }

            switch g.state {
            case .began:
                renderer.card.isInteracting = true
                renderer.card.stopVelocity()

            case .changed:
                let t = g.translation(in: view)
                renderer.card.applyRotation(
                    yaw: Float(t.x) * sensitivity,
                    pitch: Float(t.y) * sensitivity
                )
                g.setTranslation(.zero, in: view)

            case .ended, .cancelled:
                renderer.card.isInteracting = false
                let v = g.velocity(in: view)
                renderer.card.setVelocity(
                    yawPerSec: Float(v.x) * sensitivity,
                    pitchPerSec: Float(v.y) * sensitivity
                )

            default: break
            }
        }

        @objc func handleTap(_ g: UITapGestureRecognizer) {
            renderer?.card.startFlip()
        }

        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            guard let renderer else { return }
            switch g.state {
            case .began:
                renderer.card.isInteracting = true
                pinchStartScale = renderer.card.scale
            case .changed:
                renderer.card.setScale(pinchStartScale * Float(g.scale))
            case .ended, .cancelled:
                renderer.card.isInteracting = false
            default: break
            }
        }
    }
}
