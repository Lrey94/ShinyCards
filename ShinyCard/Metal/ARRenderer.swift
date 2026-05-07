import MetalKit
import ARKit
import simd

/// Renders an ARKit camera feed plus the holographic card placed at a
/// world-space anchor. Bloom and particles are skipped here — we render
/// straight to the drawable for predictable performance over passthrough.
class ARRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let session: ARSession

    // Pipelines
    var cardPipeline: MTLRenderPipelineState!
    var cameraPipeline: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState!
    var noDepthState: MTLDepthStencilState!

    // Card buffers
    var vertexBuffer: MTLBuffer!
    var indexBuffer: MTLBuffer!
    var indexCount: Int = 0

    // Card textures
    var frontTex: MTLTexture!
    var backTex:  MTLTexture!
    var rainbowTex: MTLTexture!
    var noiseTex: MTLTexture!
    var sampler: MTLSamplerState!

    // Camera image plumbing
    var capturedImageTextureCache: CVMetalTextureCache!
    private var yPlaneTexRef: CVMetalTexture?
    private var cbcrPlaneTexRef: CVMetalTexture?

    // Async front-texture loader
    private var frontTexCache: [String: MTLTexture] = [:]
    private var pendingCacheKey: String?
    var onLoadingChange: ((Bool) -> Void)?

    // Card placement (world transform from raycast hit). nil = unplaced.
    var cardWorldTransform: matrix_float4x4?

    // Holo flag (mirrors main app's toggle).
    var holoEnabled: Bool = true

    // Real-world Pokemon card width. Local geometry has half-width 1.25 so
    // total local width is 2.5 — this scale maps that to ~6.3 cm.
    private let cardRealWidth: Float = 0.063

    private var viewportSize: CGSize = .zero
    private let startTime = CACurrentMediaTime()

    init(view: MTKView, session: ARSession) {
        self.device = view.device!
        self.queue = device.makeCommandQueue()!
        self.session = session
        super.init()

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        capturedImageTextureCache = cache

        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float

        buildCardPipeline(view: view)
        buildCameraPipeline(view: view)
        buildBuffers()
        buildTextures()
        buildSampler()
        buildDepth()

        viewportSize = view.bounds.size
    }

    // MARK: - Pipelines

    private func buildCardPipeline(view: MTKView) {
        let lib = device.makeDefaultLibrary()!
        let vfn = lib.makeFunction(name: "vertex_main")!
        let ffn = lib.makeFunction(name: "fragment_main")!

        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = MemoryLayout<Vertex>.offset(of: \.position)!
        vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float3
        vd.attributes[1].offset = MemoryLayout<Vertex>.offset(of: \.normal)!
        vd.attributes[1].bufferIndex = 0
        vd.attributes[2].format = .float2
        vd.attributes[2].offset = MemoryLayout<Vertex>.offset(of: \.uv)!
        vd.attributes[2].bufferIndex = 0
        vd.attributes[3].format = .float
        vd.attributes[3].offset = MemoryLayout<Vertex>.offset(of: \.faceID)!
        vd.attributes[3].bufferIndex = 0
        vd.layouts[0].stride = MemoryLayout<Vertex>.stride

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.vertexDescriptor = vd
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        cardPipeline = try! device.makeRenderPipelineState(descriptor: desc)
    }

    private func buildCameraPipeline(view: MTKView) {
        let lib = device.makeDefaultLibrary()!
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "camera_vertex")
        desc.fragmentFunction = lib.makeFunction(name: "camera_fragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        cameraPipeline = try! device.makeRenderPipelineState(descriptor: desc)
    }

    private func buildBuffers() {
        let (verts, idx) = CardGeometry.makeVertices()
        vertexBuffer = device.makeBuffer(
            bytes: verts,
            length: MemoryLayout<Vertex>.stride * verts.count
        )
        indexBuffer = device.makeBuffer(
            bytes: idx,
            length: MemoryLayout<UInt16>.size * idx.count
        )
        indexCount = idx.count
    }

    private func buildTextures() {
        let loader = MTKTextureLoader(device: device)
        let opts: [MTKTextureLoader.Option: Any] = [
            .SRGB: false, .generateMipmaps: true
        ]
        if let tex = try? loader.newTexture(name: "card_front", scaleFactor: 1, bundle: nil, options: opts) {
            frontTex = tex
        } else {
            frontTex = makeFlatPlaceholder(color: SIMD3<Float>(0.85, 0.15, 0.20))
        }
        if let tex = try? loader.newTexture(name: "card_back", scaleFactor: 1, bundle: nil, options: opts) {
            backTex = tex
        } else {
            backTex = makeFlatPlaceholder(color: SIMD3<Float>(0.10, 0.25, 0.65))
        }
        rainbowTex = makeRainbowTexture()
        noiseTex   = makeNoiseTexture()
    }

    private func buildSampler() {
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.mipFilter = .notMipmapped
        sd.sAddressMode = .repeat
        sd.tAddressMode = .repeat
        sampler = device.makeSamplerState(descriptor: sd)
    }

    private func buildDepth() {
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dd)

        let nd = MTLDepthStencilDescriptor()
        nd.depthCompareFunction = .always
        nd.isDepthWriteEnabled = false
        noDepthState = device.makeDepthStencilState(descriptor: nd)
    }

    // MARK: - Procedural fallbacks

    private func makeFlatPlaceholder(color: SIMD3<Float>) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        var pixels = [UInt8](repeating: 0, count: 4 * 4 * 4)
        for i in 0..<(4 * 4) {
            pixels[i*4+0] = UInt8(color.x * 255)
            pixels[i*4+1] = UInt8(color.y * 255)
            pixels[i*4+2] = UInt8(color.z * 255)
            pixels[i*4+3] = 255
        }
        tex.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0,
                    withBytes: pixels, bytesPerRow: 16)
        return tex
    }

    private func makeRainbowTexture() -> MTLTexture {
        let width = 256
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: 1, mipmapped: false)
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        var pixels = [UInt8](repeating: 0, count: width * 4)
        for x in 0..<width {
            let h = Float(x) / Float(width)
            let i = floor(h * 6); let f = h * 6 - i
            let v: Float = 1, s: Float = 1
            let p = v * (1 - s); let q = v * (1 - f * s); let t = v * (1 - (1 - f) * s)
            let rgb: SIMD3<Float>
            switch Int(i) % 6 {
            case 0: rgb = [v, t, p]
            case 1: rgb = [q, v, p]
            case 2: rgb = [p, v, t]
            case 3: rgb = [p, q, v]
            case 4: rgb = [t, p, v]
            default: rgb = [v, p, q]
            }
            pixels[x*4+0] = UInt8(rgb.x * 255)
            pixels[x*4+1] = UInt8(rgb.y * 255)
            pixels[x*4+2] = UInt8(rgb.z * 255)
            pixels[x*4+3] = 255
        }
        tex.replace(region: MTLRegionMake2D(0, 0, width, 1), mipmapLevel: 0,
                    withBytes: pixels, bytesPerRow: width * 4)
        return tex
    }

    private func makeNoiseTexture() -> MTLTexture {
        let size = 256
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        for i in 0..<(size * size) {
            let n = UInt8.random(in: 0...255)
            pixels[i*4+0] = n; pixels[i*4+1] = n; pixels[i*4+2] = n; pixels[i*4+3] = 255
        }
        tex.replace(region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                    withBytes: pixels, bytesPerRow: size * 4)
        return tex
    }

    // MARK: - Public API

    func placeCard(at transform: matrix_float4x4) {
        cardWorldTransform = transform
    }

    /// Async fetch + swap, mirroring the main Renderer's loader.
    func loadFrontTexture(url: URL?, cacheKey: String) {
        if pendingCacheKey == cacheKey { return }
        pendingCacheKey = cacheKey

        if let cached = frontTexCache[cacheKey] {
            frontTex = cached
            pendingCacheKey = nil
            DispatchQueue.main.async { self.onLoadingChange?(false) }
            return
        }
        guard let url else {
            DispatchQueue.main.async { self.onLoadingChange?(false) }
            return
        }
        DispatchQueue.main.async { self.onLoadingChange?(true) }

        let device = self.device
        URLSession.shared.dataTask(with: url) { [weak self] data, _, error in
            guard let self else { return }
            guard let data, error == nil else {
                DispatchQueue.main.async {
                    self.pendingCacheKey = nil
                    self.onLoadingChange?(false)
                }
                return
            }
            let loader = MTKTextureLoader(device: device)
            let opts: [MTKTextureLoader.Option: Any] = [
                .SRGB: false, .generateMipmaps: true
            ]
            if let tex = try? loader.newTexture(data: data, options: opts) {
                DispatchQueue.main.async {
                    self.frontTexCache[cacheKey] = tex
                    if self.pendingCacheKey == cacheKey {
                        self.frontTex = tex
                        self.pendingCacheKey = nil
                    }
                    self.onLoadingChange?(false)
                }
            } else {
                DispatchQueue.main.async {
                    self.pendingCacheKey = nil
                    self.onLoadingChange?(false)
                }
            }
        }.resume()
    }

    // MARK: - Camera image textures

    private func updateCapturedImageTextures(frame: ARFrame) {
        let pb = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pb) >= 2 else { return }
        yPlaneTexRef    = createCVTexture(from: pb, format: .r8Unorm,  plane: 0)
        cbcrPlaneTexRef = createCVTexture(from: pb, format: .rg8Unorm, plane: 1)
    }

    private func createCVTexture(from pb: CVPixelBuffer, format: MTLPixelFormat, plane: Int) -> CVMetalTexture? {
        let w = CVPixelBufferGetWidthOfPlane(pb, plane)
        let h = CVPixelBufferGetHeightOfPlane(pb, plane)
        var texRef: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, capturedImageTextureCache, pb, nil,
            format, w, h, plane, &texRef
        )
        return status == kCVReturnSuccess ? texRef : nil
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = view.bounds.size
    }

    func draw(in view: MTKView) {
        viewportSize = view.bounds.size

        guard let frame = session.currentFrame,
              let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        updateCapturedImageTextures(frame: frame)

        // Resolve UI orientation for camera + display matrices.
        var uiOrientation: UIInterfaceOrientation = .portrait
        if let scene = view.window?.windowScene {
            uiOrientation = scene.interfaceOrientation
        }

        let viewMatrix = frame.camera.viewMatrix(for: uiOrientation)
        let projection = frame.camera.projectionMatrix(
            for: uiOrientation,
            viewportSize: viewportSize,
            zNear: 0.001, zFar: 1000
        )

        // displayTransform maps captured-image UVs to viewport-normalized
        // coords; its inverse is what we need in the camera vertex shader.
        let dt = frame.displayTransform(for: uiOrientation, viewportSize: viewportSize).inverted()
        var displayMatrix = matrix_float3x3(columns: (
            SIMD3<Float>(Float(dt.a),  Float(dt.b),  0),
            SIMD3<Float>(Float(dt.c),  Float(dt.d),  0),
            SIMD3<Float>(Float(dt.tx), Float(dt.ty), 1)
        ))

        // Light estimate → holo intensity. Default ARLightEstimate ambient is
        // ~1000 lumens. Clamp to a useful range so super-dark and super-bright
        // rooms still produce visible (but proportional) holo.
        let ambient = frame.lightEstimate?.ambientIntensity ?? 1000.0
        let lightFactor = Float(min(1.8, max(0.4, ambient / 1000.0)))

        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        // ---- 1. Camera background ----
        if let yRef = yPlaneTexRef,    let yTex = CVMetalTextureGetTexture(yRef),
           let cbRef = cbcrPlaneTexRef, let cbTex = CVMetalTextureGetTexture(cbRef) {
            enc.setRenderPipelineState(cameraPipeline)
            enc.setDepthStencilState(noDepthState)
            enc.setVertexBytes(&displayMatrix,
                               length: MemoryLayout<matrix_float3x3>.stride,
                               index: 0)
            enc.setFragmentTexture(yTex, index: 0)
            enc.setFragmentTexture(cbTex, index: 1)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }

        // ---- 2. Card (only once placed) ----
        if let worldTransform = cardWorldTransform {
            // Local card is 2.5 × 3.5 units; scale to ~6.3 cm wide.
            let localHalfWidth: Float = CardGeometry.w   // 1.25
            let scale = cardRealWidth / (2.0 * localHalfWidth)
            let scaleM   = matrix_float4x4.uniformScale(scale)
            // Lay flat: rotate so the card's +Z (front) points up (+Y).
            let lieFlat  = matrix_float4x4.from(quaternion: simd_quatf(angle: -.pi / 2, axis: [1, 0, 0]))
            // Lift 1mm above the surface to dodge z-fighting if there are any
            // ARKit-rendered helpers later.
            let lift     = matrix_float4x4.translation([0, 0.001, 0])
            let model    = worldTransform * lift * lieFlat * scaleM
            let normalM  = model.upperLeft3x3

            // Camera world-space position = inverse(view).columns.3.
            let camInv = simd_inverse(viewMatrix)
            let cameraPos = SIMD3<Float>(camInv.columns.3.x,
                                         camInv.columns.3.y,
                                         camInv.columns.3.z)

            var u = Uniforms(
                model: model,
                view: viewMatrix,
                projection: projection,
                normalMatrix: normalM,
                cameraPos: cameraPos,
                time: Float(CACurrentMediaTime() - startTime),
                holoEnabled: holoEnabled ? lightFactor : 0.0
            )

            enc.setRenderPipelineState(cardPipeline)
            enc.setDepthStencilState(depthState)
            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(frontTex,   index: 0)
            enc.setFragmentTexture(backTex,    index: 1)
            enc.setFragmentTexture(rainbowTex, index: 2)
            enc.setFragmentTexture(noiseTex,   index: 3)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint16,
                indexBuffer: indexBuffer,
                indexBufferOffset: 0
            )
        }

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}
