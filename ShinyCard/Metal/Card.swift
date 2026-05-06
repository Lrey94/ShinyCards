import simd
import Metal

struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var uv: SIMD2<Float>
    var faceID: Float
}

enum CardGeometry {
    static let w: Float = 1.25
    static let h: Float = 1.75
    static let d: Float = 0.025

    static func makeVertices() -> ([Vertex], [UInt16]) {
        let verts: [Vertex] = [
            // Front (+Z) faceID 0
            Vertex(position: [-w,-h, d], normal: [0,0,1], uv: [0,1], faceID: 0),
            Vertex(position: [ w,-h, d], normal: [0,0,1], uv: [1,1], faceID: 0),
            Vertex(position: [ w, h, d], normal: [0,0,1], uv: [1,0], faceID: 0),
            Vertex(position: [-w, h, d], normal: [0,0,1], uv: [0,0], faceID: 0),
            // Back (-Z) faceID 1
            Vertex(position: [ w,-h,-d], normal: [0,0,-1], uv: [0,1], faceID: 1),
            Vertex(position: [-w,-h,-d], normal: [0,0,-1], uv: [1,1], faceID: 1),
            Vertex(position: [-w, h,-d], normal: [0,0,-1], uv: [1,0], faceID: 1),
            Vertex(position: [ w, h,-d], normal: [0,0,-1], uv: [0,0], faceID: 1),
            // Right (+X) faceID 2
            Vertex(position: [ w,-h, d], normal: [1,0,0], uv: [0,1], faceID: 2),
            Vertex(position: [ w,-h,-d], normal: [1,0,0], uv: [1,1], faceID: 2),
            Vertex(position: [ w, h,-d], normal: [1,0,0], uv: [1,0], faceID: 2),
            Vertex(position: [ w, h, d], normal: [1,0,0], uv: [0,0], faceID: 2),
            // Left (-X) faceID 3
            Vertex(position: [-w,-h,-d], normal: [-1,0,0], uv: [0,1], faceID: 3),
            Vertex(position: [-w,-h, d], normal: [-1,0,0], uv: [1,1], faceID: 3),
            Vertex(position: [-w, h, d], normal: [-1,0,0], uv: [1,0], faceID: 3),
            Vertex(position: [-w, h,-d], normal: [-1,0,0], uv: [0,0], faceID: 3),
            // Top (+Y) faceID 4
            Vertex(position: [-w, h, d], normal: [0,1,0], uv: [0,1], faceID: 4),
            Vertex(position: [ w, h, d], normal: [0,1,0], uv: [1,1], faceID: 4),
            Vertex(position: [ w, h,-d], normal: [0,1,0], uv: [1,0], faceID: 4),
            Vertex(position: [-w, h,-d], normal: [0,1,0], uv: [0,0], faceID: 4),
            // Bottom (-Y) faceID 5
            Vertex(position: [-w,-h,-d], normal: [0,-1,0], uv: [0,1], faceID: 5),
            Vertex(position: [ w,-h,-d], normal: [0,-1,0], uv: [1,1], faceID: 5),
            Vertex(position: [ w,-h, d], normal: [0,-1,0], uv: [1,0], faceID: 5),
            Vertex(position: [-w,-h, d], normal: [0,-1,0], uv: [0,0], faceID: 5),
        ]
        var indices: [UInt16] = []
        for face in 0..<6 {
            let b = UInt16(face * 4)
            indices += [b, b+1, b+2, b, b+2, b+3]
        }
        return (verts, indices)
    }
}

class Card {
    var rotation: simd_quatf = simd_quatf(angle: 0, axis: [0,1,0])
    var tiltOffset: simd_quatf = simd_quatf(angle: 0, axis: [0,1,0])
    var position: SIMD3<Float> = [0, 0, 0]
    var scale: Float = 1.0
    var holoEnabled: Bool = true
    var isInteracting: Bool = false

    static let minScale: Float = 0.5
    static let maxScale: Float = 4.0

    var angularVelocity: SIMD2<Float> = .zero
    private let damping: Float = 0.94
    private let stopThreshold: Float = 0.001

    // Smoothed gyro state
    private var smoothPitch: Float = 0
    private var smoothRoll: Float = 0
    // Attitude captured on the first gyro sample — establishes the neutral
    // (zero-tilt) pose. Whatever angle the user is holding the phone at when
    // the app launches becomes "facing them squarely".
    private var calibrationPitch: Float?
    private var calibrationRoll: Float?

    // Idle "floating" animation — ticks only while idle.
    private var idleTime: Float = 0
    private var idleAmplitude: Float = 0    // eased 0…1, fades in when idle

    // Flip animation state.
    private var flipFromQuat: simd_quatf?
    private var flipToQuat: simd_quatf?
    private var flipElapsed: Float = 0
    private let flipDuration: Float = 0.7

    var isFlipping: Bool { flipFromQuat != nil }

    private var idleRotation: simd_quatf {
        // No idle wobble during a flip — keeps the rotation curve clean.
        if isFlipping { return simd_quatf(angle: 0, axis: [0,1,0]) }
        // Two slightly different frequencies so the wobble feels organic, not
        // metronomic. ±~2.5° on each axis at full amplitude.
        let yaw = sin(idleTime * 0.9) * 0.065 * idleAmplitude
        let pitch = sin(idleTime * 0.7 + 1.3) * 0.055 * idleAmplitude
        let qy = simd_quatf(angle: yaw, axis: [0,1,0])
        let qx = simd_quatf(angle: pitch, axis: [1,0,0])
        return simd_normalize(qy * qx)
    }

    var modelMatrix: matrix_float4x4 {
        // tilt + idle wobble are ambient; user rotation composes on top
        let combinedRot = tiltOffset * idleRotation * rotation
        return matrix_float4x4.translation(position)
            * matrix_float4x4.from(quaternion: combinedRot)
            * matrix_float4x4.uniformScale(scale)
    }

    /// Update ambient tilt from device attitude (radians).
    /// Pitch is roughly forward/back tilt, roll is left/right.
    func updateTilt(pitch: Float, roll: Float) {
        // First sample becomes the rest orientation. All subsequent readings
        // are deltas from this baseline, so the card starts square-on no
        // matter how the user is holding the phone.
        if calibrationPitch == nil {
            calibrationPitch = pitch
            calibrationRoll = roll
        }
        let dPitch = pitch - (calibrationPitch ?? 0)
        let dRoll  = roll  - (calibrationRoll  ?? 0)

        let alpha: Float = 0.18             // smoothing factor
        smoothPitch += (dPitch - smoothPitch) * alpha
        smoothRoll  += (dRoll  - smoothRoll)  * alpha

        let amplitude: Float = 0.22         // radians of max tilt
        let p = max(-1, min(1, smoothPitch * 1.6)) * amplitude
        let r = max(-1, min(1, smoothRoll  * 1.6)) * amplitude

        let qy = simd_quatf(angle: r, axis: [0,1,0])
        let qx = simd_quatf(angle: -p, axis: [1,0,0])
        tiltOffset = simd_normalize(qy * qx)
    }

    /// Force a recalibration on the next gyro sample. Useful if you ever add
    /// a "level the card" gesture.
    func recalibrateTilt() {
        calibrationPitch = nil
        calibrationRoll = nil
        tiltOffset = simd_quatf(angle: 0, axis: [0, 1, 0])
        smoothPitch = 0
        smoothRoll = 0
    }

    func reset() {
        rotation = simd_quatf(angle: 0, axis: [0,1,0])
        angularVelocity = .zero
        scale = 1.0
        // Cancel any flip in progress.
        flipFromQuat = nil
        flipToQuat = nil
        flipElapsed = 0
    }

    /// Begin a smooth 180° flip around world Y. Ignored if a flip is already
    /// running, so rapid taps don't desync the animation.
    func startFlip() {
        if isFlipping { return }
        stopVelocity()
        flipFromQuat = rotation
        let half = simd_quatf(angle: .pi, axis: [0, 1, 0])
        flipToQuat = simd_normalize(half * rotation)
        flipElapsed = 0
    }

    private func updateFlip(dt: Float) {
        guard let from = flipFromQuat, let to = flipToQuat else { return }
        flipElapsed += dt
        let raw = min(flipElapsed / flipDuration, 1.0)
        // Cubic ease in/out — slow start, fast middle, slow finish.
        let eased: Float = raw < 0.5
            ? 4 * raw * raw * raw
            : 1 - powf(-2 * raw + 2, 3) / 2
        rotation = simd_slerp(from, to, eased)
        if raw >= 1.0 {
            rotation = to
            flipFromQuat = nil
            flipToQuat = nil
        }
    }

    func setScale(_ s: Float) {
        scale = max(Self.minScale, min(Self.maxScale, s))
    }

    /// Apply rotation deltas in radians directly (used during drag).
    func applyRotation(yaw: Float, pitch: Float) {
        let qy = simd_quatf(angle: yaw,   axis: [0,1,0])
        let qx = simd_quatf(angle: pitch, axis: [1,0,0])
        rotation = simd_normalize(qy * qx * rotation)
    }

    /// Set the velocity (radians/sec). Called when drag ends.
    func setVelocity(yawPerSec: Float, pitchPerSec: Float) {
        angularVelocity = SIMD2<Float>(yawPerSec, pitchPerSec)
    }

    func stopVelocity() {
        angularVelocity = .zero
    }

    /// Step inertia and idle animation forward by `dt` seconds.
    func updateInertia(dt: Float) {
        // A running flip drives rotation directly; skip inertia + idle ticks.
        if isFlipping {
            updateFlip(dt: dt)
            // Force the idle wobble down so it doesn't re-emerge mid-flip.
            idleAmplitude += (0 - idleAmplitude) * min(1, 4.0 * dt)
            return
        }

        // Idle wobble: fade amplitude in when idle (no touch + no inertia),
        // fade it out when the user is interacting or the card is spinning.
        let isIdle = !isInteracting && length(angularVelocity) < stopThreshold
        let target: Float = isIdle ? 1.0 : 0.0
        // ~0.6 sec ease in/out, frame-rate independent.
        let easeRate: Float = 1.8
        idleAmplitude += (target - idleAmplitude) * min(1, easeRate * dt)
        if idleAmplitude > 0.001 {
            idleTime += dt
        }

        guard length(angularVelocity) > stopThreshold else {
            angularVelocity = .zero
            return
        }
        applyRotation(
            yaw: angularVelocity.x * dt,
            pitch: angularVelocity.y * dt
        )
        // Frame-rate-independent damping: damping^(dt*60) so it decays
        // at the same rate regardless of refresh rate.
        let decay = powf(damping, dt * 60.0)
        angularVelocity *= decay
    }
}
