import MetalKit
import simd

struct Uniforms {
    var model: matrix_float4x4
    var view: matrix_float4x4
    var projection: matrix_float4x4
    var normalMatrix: matrix_float3x3
    var cameraPos: SIMD3<Float>
    var time: Float
    var holoEnabled: Float
}

// Layout-compatible with Metal's `Particle` struct (see Shaders.metal).
// SIMD3<Float> is 16-byte aligned in both Swift and Metal so the offsets match.
struct Particle {
    var position: SIMD3<Float>
    var velocity: SIMD3<Float>
    var life: Float    // 1 = just born, 0 = dead
    var seed: Float    // per-particle randomness (color/size hash)
}

struct ParticleUniforms {
    var dt: Float
    var time: Float
    var emit: Float    // 1 if emitting (holo on), 0 if not
    var _pad: Float = 0
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var pipeline: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState!
    var vertexBuffer: MTLBuffer!
    var indexBuffer: MTLBuffer!
    var indexCount: Int = 0
    var sampler: MTLSamplerState!

    // Particle system
    var particlePipeline: MTLRenderPipelineState!
    var particleComputePipeline: MTLComputePipelineState!
    var particleDepthState: MTLDepthStencilState!
    var particleBuffer: MTLBuffer!
    let particleCount: Int = 300

    var frontTex: MTLTexture!
    var backTex:  MTLTexture!
    var rainbowTex: MTLTexture!
    var noiseTex: MTLTexture!

    // Bloom post-processing pipelines + offscreen textures.
    var brightPipeline: MTLRenderPipelineState!
    var blurPipeline: MTLRenderPipelineState!
    var compositePipeline: MTLRenderPipelineState!
    var postSampler: MTLSamplerState!
    var sceneTex: MTLTexture?
    var sceneDepth: MTLTexture?
    var brightTex: MTLTexture?
    var blurHTex: MTLTexture?
    var blurVTex: MTLTexture?
    private var offscreenSize: CGSize = .zero
    private let bloomDownscale: CGFloat = 0.5

    // Front-texture cache so tapping a previously viewed card is instant.
    // Keyed by `setID/number` so different collections don't collide.
    private var frontTexCache: [String: MTLTexture] = [:]
    private var pendingCacheKey: String?

    /// Fired on the main queue whenever a remote front-texture starts or
    /// finishes loading. SwiftUI uses this to drive a spinner.
    var onLoadingChange: ((Bool) -> Void)?

    let card = Card()
    var aspect: Float = 1
    private let startTime = CACurrentMediaTime()
    private var lastFrameTime = CACurrentMediaTime()

    init(view: MTKView) {
        self.device = view.device!
        self.queue = device.makeCommandQueue()!
        super.init()
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        // Transparent clear so the SwiftUI background shows through.
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.isOpaque = false
        view.backgroundColor = .clear
        buildPipeline(view: view)
        buildBuffers()
        buildTextures()
        buildSampler()
        buildDepth()
        buildParticles(view: view)
        buildPostPipelines(view: view)
    }

    // MARK: - Bloom

    func buildPostPipelines(view: MTKView) {
        let lib = device.makeDefaultLibrary()!

        let bright = MTLRenderPipelineDescriptor()
        bright.vertexFunction = lib.makeFunction(name: "fullscreen_vertex")
        bright.fragmentFunction = lib.makeFunction(name: "bright_extract")
        bright.colorAttachments[0].pixelFormat = view.colorPixelFormat
        brightPipeline = try! device.makeRenderPipelineState(descriptor: bright)

        let blur = MTLRenderPipelineDescriptor()
        blur.vertexFunction = lib.makeFunction(name: "fullscreen_vertex")
        blur.fragmentFunction = lib.makeFunction(name: "blur_separable")
        blur.colorAttachments[0].pixelFormat = view.colorPixelFormat
        blurPipeline = try! device.makeRenderPipelineState(descriptor: blur)

        let comp = MTLRenderPipelineDescriptor()
        comp.vertexFunction = lib.makeFunction(name: "fullscreen_vertex")
        comp.fragmentFunction = lib.makeFunction(name: "composite_bloom")
        comp.colorAttachments[0].pixelFormat = view.colorPixelFormat
        // Composite renders into MTKView's drawable pass, which carries the
        // view's depth attachment — declare it so the formats match.
        comp.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        // No blending: composite outputs final color (with the scene's alpha).
        compositePipeline = try! device.makeRenderPipelineState(descriptor: comp)

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        postSampler = device.makeSamplerState(descriptor: sd)
    }

    /// (Re)create offscreen textures whenever the drawable size changes.
    func ensureOffscreenTextures(size: CGSize) {
        if size == offscreenSize, sceneTex != nil { return }
        offscreenSize = size

        let w = max(1, Int(size.width))
        let h = max(1, Int(size.height))
        let bw = max(1, Int(size.width * bloomDownscale))
        let bh = max(1, Int(size.height * bloomDownscale))

        let scene = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: w, height: h, mipmapped: false
        )
        scene.usage = [.renderTarget, .shaderRead]
        scene.storageMode = .private
        sceneTex = device.makeTexture(descriptor: scene)

        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: w, height: h, mipmapped: false
        )
        depth.usage = [.renderTarget]
        depth.storageMode = .private
        sceneDepth = device.makeTexture(descriptor: depth)

        let bloom = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: bw, height: bh, mipmapped: false
        )
        bloom.usage = [.renderTarget, .shaderRead]
        bloom.storageMode = .private
        brightTex = device.makeTexture(descriptor: bloom)
        blurHTex  = device.makeTexture(descriptor: bloom)
        blurVTex  = device.makeTexture(descriptor: bloom)
    }

    func buildParticles(view: MTKView) {
        // Initialize all particles dead — they spawn over the first ~2 sec.
        let dead = Particle(position: SIMD3<Float>(0, -10, 0),
                            velocity: .zero,
                            life: 0,
                            seed: 0)
        var particles = [Particle](repeating: dead, count: particleCount)
        // Stagger initial life so they don't all spawn on frame 1.
        for i in 0..<particleCount {
            particles[i].life = -Float(i) / Float(particleCount) * 2.0
            particles[i].seed = Float(i)
        }
        particleBuffer = device.makeBuffer(
            bytes: particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared
        )

        let lib = device.makeDefaultLibrary()!
        particleComputePipeline = try! device.makeComputePipelineState(
            function: lib.makeFunction(name: "particle_update")!
        )

        let pdesc = MTLRenderPipelineDescriptor()
        pdesc.vertexFunction = lib.makeFunction(name: "particle_vertex")
        pdesc.fragmentFunction = lib.makeFunction(name: "particle_fragment")
        // No vertex descriptor — vertex shader uses [[vertex_id]] + [[instance_id]].
        pdesc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pdesc.colorAttachments[0].isBlendingEnabled = true
        pdesc.colorAttachments[0].rgbBlendOperation = .add
        pdesc.colorAttachments[0].alphaBlendOperation = .add
        // Additive blending — particles glow.
        pdesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pdesc.colorAttachments[0].destinationRGBBlendFactor = .one
        pdesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        pdesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        pdesc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        particlePipeline = try! device.makeRenderPipelineState(descriptor: pdesc)

        // Particles read depth (so card occludes the ones behind it) but
        // don't write it.
        let pd = MTLDepthStencilDescriptor()
        pd.depthCompareFunction = .less
        pd.isDepthWriteEnabled = false
        particleDepthState = device.makeDepthStencilState(descriptor: pd)
    }
    
    func buildPipeline(view: MTKView) {
        let lib = device.makeDefaultLibrary()!
        let vfn = lib.makeFunction(name: "vertex_main")!
        let ffn = lib.makeFunction(name: "fragment_main")!
        
        // SIMD3<Float> is 16 bytes in Swift (alignment), not 12 — must use MemoryLayout.offset
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
        // Card itself blends so its rounded edges anti-alias against the
        // transparent background / shadow underneath.
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        pipeline = try! device.makeRenderPipelineState(descriptor: desc)
    }

    
    func buildBuffers() {
        let (verts, idx) = CardGeometry.makeVertices()
        vertexBuffer = device.makeBuffer(bytes: verts, length: MemoryLayout<Vertex>.stride * verts.count)
        indexBuffer = device.makeBuffer(bytes: idx, length: MemoryLayout<UInt16>.size * idx.count)
        indexCount = idx.count
    }
    
    func buildTextures() {
        let loader = MTKTextureLoader(device: device)
        let opts: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .generateMipmaps: true
        ]
        
        // Try to load card images from assets, fall back to placeholders
        frontTex = loadOrGenerate(loader: loader, name: "card_front", opts: opts) {
            Self.makePlaceholderCard(device: self.device,
                                     color: SIMD3<Float>(0.85, 0.15, 0.20))
        }
        backTex = loadOrGenerate(loader: loader, name: "card_back", opts: opts) {
            Self.makePlaceholderCard(device: self.device,
                                     color: SIMD3<Float>(0.10, 0.25, 0.65))
        }
        
        // Always generate these procedurally — no need to ship them as assets
        rainbowTex = Self.makeRainbowTexture(device: device)
        noiseTex   = Self.makeNoiseTexture(device: device)
    }
    
    /// Asynchronously fetch a card front from `url` and swap it in. The cache
    /// key disambiguates cards from different collections (e.g. `base1/1` vs
    /// `sv8pt5/1`).
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
                print("⚠️ \(cacheKey) download failed: \(error?.localizedDescription ?? "?")")
                DispatchQueue.main.async {
                    self.pendingCacheKey = nil
                    self.onLoadingChange?(false)
                }
                return
            }
            let loader = MTKTextureLoader(device: device)
            let opts: [MTKTextureLoader.Option: Any] = [
                .SRGB: false,
                .generateMipmaps: true
            ]
            do {
                let tex = try loader.newTexture(data: data, options: opts)
                DispatchQueue.main.async {
                    self.frontTexCache[cacheKey] = tex
                    if self.pendingCacheKey == cacheKey {
                        self.frontTex = tex
                        self.pendingCacheKey = nil
                    }
                    self.onLoadingChange?(false)
                }
            } catch {
                print("⚠️ \(cacheKey) decode failed: \(error)")
                DispatchQueue.main.async {
                    self.pendingCacheKey = nil
                    self.onLoadingChange?(false)
                }
            }
        }.resume()
    }

    private func loadOrGenerate(loader: MTKTextureLoader,
                                name: String,
                                opts: [MTKTextureLoader.Option: Any],
                                fallback: () -> MTLTexture) -> MTLTexture {
        if let tex = try? loader.newTexture(name: name, scaleFactor: 1,
                                            bundle: nil, options: opts) {
            return tex
        }
        print("⚠️ Asset '\(name)' not found in Assets.xcassets — using procedural placeholder")
        return fallback()
    }
    
    func buildSampler() {
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.mipFilter = .notMipmapped
        sd.sAddressMode = .repeat
        sd.tAddressMode = .repeat
        sampler = device.makeSamplerState(descriptor: sd)
    }
    
    func buildDepth() {
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dd)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(size.width / size.height)
        ensureOffscreenTextures(size: size)
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = Float(min(now - lastFrameTime, 1.0 / 30.0))
        lastFrameTime = now
        card.updateInertia(dt: dt)

        let drawableSize = view.drawableSize
        guard drawableSize.width > 1, drawableSize.height > 1 else { return }
        ensureOffscreenTextures(size: drawableSize)

        guard let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let sceneTex = sceneTex,
              let sceneDepth = sceneDepth,
              let brightTex = brightTex,
              let blurH = blurHTex,
              let blurV = blurVTex
        else { return }

        // ---- 1. Particle compute pass ----
        if let comp = cmd.makeComputeCommandEncoder() {
            comp.setComputePipelineState(particleComputePipeline)
            comp.setBuffer(particleBuffer, offset: 0, index: 0)
            var pu = ParticleUniforms(
                dt: dt,
                time: Float(now - startTime),
                emit: card.holoEnabled ? 1.0 : 0.0
            )
            comp.setBytes(&pu, length: MemoryLayout<ParticleUniforms>.stride, index: 1)
            let tpg = MTLSize(width: 32, height: 1, depth: 1)
            let tg  = MTLSize(width: (particleCount + 31) / 32, height: 1, depth: 1)
            comp.dispatchThreadgroups(tg, threadsPerThreadgroup: tpg)
            comp.endEncoding()
        }

        // ---- 2. Scene pass: card + particles → sceneTex ----
        let scenePass = MTLRenderPassDescriptor()
        scenePass.colorAttachments[0].texture = sceneTex
        scenePass.colorAttachments[0].loadAction = .clear
        scenePass.colorAttachments[0].storeAction = .store
        scenePass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        scenePass.depthAttachment.texture = sceneDepth
        scenePass.depthAttachment.loadAction = .clear
        scenePass.depthAttachment.storeAction = .dontCare
        scenePass.depthAttachment.clearDepth = 1.0

        let proj = matrix_float4x4.perspective(fovyRadians: .pi/4, aspect: aspect, near: 0.1, far: 100)
        let cameraPos = SIMD3<Float>(0, 0, 7)
        let viewM = matrix_float4x4.translation(-cameraPos)
        let model = card.modelMatrix
        let normalM = model.upperLeft3x3

        var u = Uniforms(model: model, view: viewM, projection: proj,
                         normalMatrix: normalM, cameraPos: cameraPos,
                         time: Float(now - startTime),
                         holoEnabled: card.holoEnabled ? 1.0 : 0.0)

        if let enc = cmd.makeRenderCommandEncoder(descriptor: scenePass) {
            // Card
            enc.setRenderPipelineState(pipeline)
            enc.setDepthStencilState(depthState)
            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(frontTex,   index: 0)
            enc.setFragmentTexture(backTex,    index: 1)
            enc.setFragmentTexture(rainbowTex, index: 2)
            enc.setFragmentTexture(noiseTex,   index: 3)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                      indexType: .uint16, indexBuffer: indexBuffer,
                                      indexBufferOffset: 0)
            // Particles
            enc.setRenderPipelineState(particlePipeline)
            enc.setDepthStencilState(particleDepthState)
            enc.setVertexBuffer(particleBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0,
                               vertexCount: 4, instanceCount: particleCount)
            enc.endEncoding()
        }

        // ---- 3. Bright extract: sceneTex → brightTex (half-res) ----
        runPostPass(cmd: cmd, target: brightTex, pipeline: brightPipeline) { enc in
            enc.setFragmentTexture(sceneTex, index: 0)
            enc.setFragmentSamplerState(self.postSampler, index: 0)
        }

        // ---- 4. Blur horizontal: brightTex → blurH ----
        runPostPass(cmd: cmd, target: blurH, pipeline: blurPipeline) { enc in
            enc.setFragmentTexture(brightTex, index: 0)
            enc.setFragmentSamplerState(self.postSampler, index: 0)
            var dir = SIMD2<Float>(1.0 / Float(brightTex.width), 0)
            enc.setFragmentBytes(&dir, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }

        // ---- 5. Blur vertical: blurH → blurV ----
        runPostPass(cmd: cmd, target: blurV, pipeline: blurPipeline) { enc in
            enc.setFragmentTexture(blurH, index: 0)
            enc.setFragmentSamplerState(self.postSampler, index: 0)
            var dir = SIMD2<Float>(0, 1.0 / Float(blurH.height))
            enc.setFragmentBytes(&dir, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }

        // ---- 6. Composite to drawable ----
        if let finalRPD = view.currentRenderPassDescriptor,
           let enc = cmd.makeRenderCommandEncoder(descriptor: finalRPD) {
            enc.setRenderPipelineState(compositePipeline)
            enc.setFragmentTexture(sceneTex, index: 0)
            enc.setFragmentTexture(blurV, index: 1)
            enc.setFragmentSamplerState(postSampler, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        cmd.present(drawable)
        cmd.commit()
    }

    /// Render a fullscreen-triangle post-process pass into `target`.
    private func runPostPass(cmd: MTLCommandBuffer,
                             target: MTLTexture,
                             pipeline: MTLRenderPipelineState,
                             setup: (MTLRenderCommandEncoder) -> Void) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .dontCare
        rpd.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(pipeline)
        setup(enc)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }
    
    // MARK: - Procedural textures
    
    static func makeRainbowTexture(device: MTLDevice) -> MTLTexture {
        let width = 256
        let height = 1
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        
        var pixels = [UInt8](repeating: 0, count: width * 4)
        for x in 0..<width {
            let hue = Float(x) / Float(width)
            let rgb = hsvToRgb(h: hue, s: 1.0, v: 1.0)
            pixels[x * 4 + 0] = UInt8(rgb.x * 255)
            pixels[x * 4 + 1] = UInt8(rgb.y * 255)
            pixels[x * 4 + 2] = UInt8(rgb.z * 255)
            pixels[x * 4 + 3] = 255
        }
        tex.replace(region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: pixels,
                    bytesPerRow: width * 4)
        return tex
    }
    
    static func makeNoiseTexture(device: MTLDevice) -> MTLTexture {
        let size = 256
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: size, height: size, mipmapped: false
        )
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        for i in 0..<(size * size) {
            let n = UInt8.random(in: 0...255)
            pixels[i * 4 + 0] = n
            pixels[i * 4 + 1] = n
            pixels[i * 4 + 2] = n
            pixels[i * 4 + 3] = 255
        }
        tex.replace(region: MTLRegionMake2D(0, 0, size, size),
                    mipmapLevel: 0,
                    withBytes: pixels,
                    bytesPerRow: size * 4)
        return tex
    }
    
    static func makePlaceholderCard(device: MTLDevice,
                                    color: SIMD3<Float>) -> MTLTexture {
        let w = 512, h = 720
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: w, height: h, mipmapped: false
        )
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                
                // Gold outer border (0–20 px)
                let isBorder = x < 20 || x > w - 20 || y < 20 || y > h - 20
                // Black inner frame (40–60 px from edge)
                let isInner = (x > 40 && x < w - 40 && y > 40 && y < h - 40) &&
                (x < 60 || x > w - 60 || y < 60 || y > h - 60)
                
                if isBorder {
                    pixels[i + 0] = 230; pixels[i + 1] = 200; pixels[i + 2] = 60
                } else if isInner {
                    pixels[i + 0] = 40;  pixels[i + 1] = 40;  pixels[i + 2] = 40
                } else {
                    pixels[i + 0] = UInt8(color.x * 255)
                    pixels[i + 1] = UInt8(color.y * 255)
                    pixels[i + 2] = UInt8(color.z * 255)
                }
                pixels[i + 3] = 255
            }
        }
        tex.replace(region: MTLRegionMake2D(0, 0, w, h),
                    mipmapLevel: 0,
                    withBytes: pixels,
                    bytesPerRow: w * 4)
        return tex
    }
    
    private static func hsvToRgb(h: Float, s: Float, v: Float) -> SIMD3<Float> {
        let i = floor(h * 6)
        let f = h * 6 - i
        let p = v * (1 - s)
        let q = v * (1 - f * s)
        let t = v * (1 - (1 - f) * s)
        switch Int(i) % 6 {
        case 0: return SIMD3<Float>(v, t, p)
        case 1: return SIMD3<Float>(q, v, p)
        case 2: return SIMD3<Float>(p, v, t)
        case 3: return SIMD3<Float>(p, q, v)
        case 4: return SIMD3<Float>(t, p, v)
        default: return SIMD3<Float>(v, p, q)
        }
    }
}
