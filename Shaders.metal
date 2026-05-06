#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
    float  faceID   [[attribute(3)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 localPos;     // pre-model card-local position (for SDF rounding)
    float2 uv;
    int    faceID [[flat]];
};

// Card-local rounded-rect SDF. Card half-extents are w=1.25, h=1.75.
// Returns signed distance: <0 inside, >0 outside.
static inline float roundedCardSDF(float2 p) {
    const float w = 1.25;
    const float h = 1.75;
    const float r = 0.16;       // corner radius
    float2 q = abs(p) - float2(w - r, h - r);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

struct Uniforms {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float3x3 normalMatrix;
    float3   cameraPos;
    float    time;
    float    holoEnabled;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant Uniforms& u [[buffer(1)]]) {
    VertexOut out;
    float4 worldPos = u.model * float4(in.position, 1.0);
    out.worldPos = worldPos.xyz;
    out.position = u.projection * u.view * worldPos;
    out.worldNormal = normalize(u.normalMatrix * in.normal);
    out.localPos = in.position;
    out.uv = in.uv;
    out.faceID = int(in.faceID);  // explicit, not derived from vid
    return out;
}

// Diagnostic mode — set to one of:
//   0 = textured (normal rendering)
//   1 = UV gradient   (red = u, green = v) → smooth corner gradient if UVs OK
//   2 = face ID color (front=red, back=green, edges=blue tints) → confirms face routing
//   3 = world normal  (xyz mapped to rgb)  → confirms normals + rotation
//   4 = solid magenta on front face        → confirms the front face is what you see
#define DEBUG_MODE 0

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant Uniforms& u [[buffer(1)]],
                              texture2d<float> frontTex [[texture(0)]],
                              texture2d<float> backTex  [[texture(1)]],
                              texture2d<float> rainbowTex [[texture(2)]],
                              texture2d<float> noiseTex   [[texture(3)]],
                              sampler s [[sampler(0)]]) {
#if DEBUG_MODE == 1
    // UV gradient: bottom-left=black, bottom-right=red, top-left=green, top-right=yellow
    return float4(in.uv.x, in.uv.y, 0.0, 1.0);

#elif DEBUG_MODE == 2
    // Face ID colors
    if (in.faceID == 0) return float4(1.0, 0.0, 0.0, 1.0); // front  → red
    if (in.faceID == 1) return float4(0.0, 1.0, 0.0, 1.0); // back   → green
    if (in.faceID == 2) return float4(0.0, 0.5, 1.0, 1.0); // right  → cyan
    if (in.faceID == 3) return float4(0.0, 0.2, 1.0, 1.0); // left   → blue
    if (in.faceID == 4) return float4(1.0, 0.0, 1.0, 1.0); // top    → magenta
    return float4(1.0, 1.0, 0.0, 1.0);                     // bottom → yellow

#elif DEBUG_MODE == 3
    // World normal as color (mapped from [-1,1] → [0,1])
    float3 N = normalize(in.worldNormal);
    return float4(N * 0.5 + 0.5, 1.0);

#elif DEBUG_MODE == 4
    // Solid magenta on front face only — easy to spot
    if (in.faceID == 0) return float4(1.0, 0.0, 1.0, 1.0);
    return float4(0.1, 0.1, 0.1, 1.0);

#else
    // Round the card's silhouette: discard any fragment outside the rounded SDF.
    // This makes the corners curved on every face (front, back, edges) so the
    // card never has the giveaway sharp 90° corners.
    float sdf = roundedCardSDF(in.localPos.xy);
    if (sdf > 0.0) discard_fragment();

    float3 N = normalize(in.worldNormal);
    float3 V = normalize(u.cameraPos - in.worldPos);
    // Per-face fresnel (used by holo / sparkle / spec).
    float fres = 1.0 - saturate(dot(N, V));

    // Whole-card "edge-on" factor: 1 when the card is perpendicular to the
    // camera (mid-flip), 0 when facing it. Computed from the card's +Z axis
    // after the model matrix, independent of which face the fragment is on.
    float3 cardN = normalize(u.normalMatrix * float3(0, 0, 1));
    float edgeOn = 1.0 - abs(dot(cardN, V));

    // Gold rim color, used both at the rounded silhouette and on the edge faces.
    float3 goldRim = float3(1.25, 1.0, 0.5);

    // Edge faces (top/bottom/left/right of the card body) — gold trim that
    // brightens dramatically when the card is edge-on, which is exactly when
    // these faces are visible during the flip. Sells the 3D thickness.
    if (in.faceID > 1) {
        float3 edgeColor = float3(0.95, 0.85, 0.4);
        float boost = 1.0 + edgeOn * 1.6;
        return float4(edgeColor * boost, 1.0);
    }

    // Soft gold rim hugging the rounded silhouette of front + back. Fades
    // toward 0 inside the card; ramps up sharply when the card is edge-on.
    float rimMask = smoothstep(-0.05, 0.0, sdf) * pow(edgeOn, 1.2);

    if (in.faceID == 1) {
        float4 b = backTex.sample(s, in.uv);
        return float4(mix(b.rgb, goldRim, saturate(rimMask)), b.a);
    }

    // Front face — base art, optionally with holographic foil overlay, plus
    // the gold rim on top.
    float4 base = frontTex.sample(s, in.uv);
    if (u.holoEnabled < 0.5) {
        float3 lit = mix(base.rgb, goldRim, saturate(rimMask));
        return float4(lit, base.a);
    }

    // Rainbow band shifts with view angle, uv, and time.
    float rainbowU = fract(in.uv.x * 1.4 + in.uv.y * 0.6
                           + fres * 1.2 + u.time * 0.08);
    float3 rainbow = rainbowTex.sample(s, float2(rainbowU, 0.5)).rgb;

    // Glittery sparkle that scrolls with view — pow() to keep only bright
    // pixels. Sparkle only appears off-axis (multiplied by fres²) so a
    // square-on card has almost none.
    float2 noiseUV = in.uv * 8.0 + V.xy * 0.6;
    float sparkle  = noiseTex.sample(s, noiseUV).r;
    sparkle = pow(sparkle, 8.0) * fres * fres * 1.4;

    // Specular highlight — a moving "sheen" you can chase by tilting.
    float3 L = normalize(float3(0.3, 0.7, 0.7));
    float3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 28.0) * 0.55;

    // Blend rainbow over the base; near zero at square-on, ramps up at
    // grazing angles. No constant base term so the card looks normal at rest.
    float holoStrength = fres * fres * 0.75;
    float3 lit = mix(base.rgb, base.rgb * 0.7 + rainbow * 0.7, holoStrength);
    lit += sparkle;
    lit += spec;

    // Gold rim painted last so it shows through the foil at edge-on angles.
    lit = mix(lit, goldRim, saturate(rimMask));

    return float4(lit, base.a);
#endif
}

// MARK: - Bloom post-processing
//
// Pass order:
//   1. Scene rendered to an offscreen color texture.
//   2. `bright_extract` keeps only pixels above a luminance threshold.
//   3. `blur_separable` runs twice (horizontal then vertical, half-res).
//   4. `composite_bloom` adds the blurred bright buffer back over the scene.

struct PostVertex {
    float4 position [[position]];
    float2 uv;
};

vertex PostVertex fullscreen_vertex(uint vid [[vertex_id]]) {
    // Single oversized triangle covering the whole screen.
    float2 pos[3] = { float2(-1, -1), float2( 3, -1), float2(-1,  3) };
    float2 uv[3]  = { float2( 0,  1), float2( 2,  1), float2( 0, -1) };
    PostVertex out;
    out.position = float4(pos[vid], 0, 1);
    out.uv = uv[vid];
    return out;
}

fragment float4 bright_extract(PostVertex in [[stage_in]],
                                texture2d<float> sceneTex [[texture(0)]],
                                sampler s [[sampler(0)]]) {
    float4 c = sceneTex.sample(s, in.uv);
    float luma = dot(c.rgb, float3(0.299, 0.587, 0.114));
    // Soft knee around the threshold so bloom ramps in smoothly rather than
    // stamping out a hard silhouette.
    const float threshold = 0.72;
    const float knee = 0.18;
    float soft = smoothstep(threshold, threshold + knee, luma);
    // Square the contribution so only genuinely bright pixels bloom strongly.
    return float4(c.rgb * soft * soft, 1.0);
}

fragment float4 blur_separable(PostVertex in [[stage_in]],
                                constant float2& dir [[buffer(0)]],
                                texture2d<float> tex [[texture(0)]],
                                sampler s [[sampler(0)]]) {
    // 9-tap Gaussian (5 unique weights, mirrored).
    const float w[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    float3 result = tex.sample(s, in.uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        float2 off = dir * float(i);
        result += tex.sample(s, in.uv + off).rgb * w[i];
        result += tex.sample(s, in.uv - off).rgb * w[i];
    }
    return float4(result, 1.0);
}

fragment float4 composite_bloom(PostVertex in [[stage_in]],
                                 texture2d<float> sceneTex [[texture(0)]],
                                 texture2d<float> bloomTex [[texture(1)]],
                                 sampler s [[sampler(0)]]) {
    float4 scene = sceneTex.sample(s, in.uv);
    float3 bloom = bloomTex.sample(s, in.uv).rgb;
    // Additive composite. Scene's alpha is preserved so the SwiftUI starfield
    // behind the MTKView remains visible at the corners.
    return float4(scene.rgb + bloom * 1.0, scene.a);
}

// MARK: - Particle system
//
// A pool of particles lives in a single device buffer. Every frame:
//   1. `particle_update` (compute) integrates positions, decrements life, and
//      respawns dead particles when emitting.
//   2. `particle_vertex` reads the buffer per-instance and emits a billboarded
//      quad facing the camera.
//   3. `particle_fragment` shades each quad as a soft additive sparkle.

struct Particle {
    float3 position;
    float3 velocity;
    float  life;       // 1=just born → 0=dead. Negative = waiting to spawn.
    float  seed;       // per-particle hash input
};

struct ParticleUniforms {
    float dt;
    float time;
    float emit;        // 1 = spawn dead particles, 0 = let them stay dead
    float _pad;
};

// Cheap hash for procedural randomness in the kernel.
static inline float hash11(float x) {
    return fract(sin(x * 12.9898) * 43758.5453);
}

kernel void particle_update(device Particle* particles [[buffer(0)]],
                            constant ParticleUniforms& u [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    Particle p = particles[id];

    // Tick down life. Dead particles (life <= 0) wait until emit==1, then
    // respawn at the bottom of the card with a randomized upward velocity.
    p.life -= u.dt * 0.45;

    if (p.life <= 0.0) {
        if (u.emit > 0.5) {
            float t = u.time + p.seed * 1.7;
            float r1 = hash11(p.seed * 1.0 + t);
            float r2 = hash11(p.seed * 2.7 + t * 1.3);
            float r3 = hash11(p.seed * 5.1 + t * 0.7);
            float r4 = hash11(p.seed * 7.3 + t * 0.4);

            // Spawn anywhere along the card's width, just below it.
            p.position = float3((r1 * 2.0 - 1.0) * 2.5,
                                -2.1 + r2 * 0.3,
                                (r3 * 2.0 - 1.0) * 0.7);
            // Mostly upward, with a little side drift.
            p.velocity = float3((r4 * 2.0 - 1.0) * 0.35,
                                0.55 + r1 * 0.4,
                                (r2 * 2.0 - 1.0) * 0.15);
            p.life = 0.7 + r3 * 0.4;   // 0.7…1.1 — varies how long it lives
        }
    } else {
        // Drift; soft horizontal wobble + slight upward acceleration.
        p.velocity.x += sin(u.time * 1.7 + p.seed) * 0.25 * u.dt;
        p.velocity.y += 0.05 * u.dt;
        p.position += p.velocity * u.dt;
    }
    particles[id] = p;
}

struct ParticleOut {
    float4 position [[position]];
    float2 uv;
    float  life;
    float  seed;
};

vertex ParticleOut particle_vertex(uint vid [[vertex_id]],
                                    uint iid [[instance_id]],
                                    device const Particle* particles [[buffer(0)]],
                                    constant Uniforms& u [[buffer(1)]]) {
    Particle p = particles[iid];

    // Quad corners as a triangle strip in [-1, 1].
    float2 corners[4] = {
        float2(-1, -1), float2( 1, -1),
        float2(-1,  1), float2( 1,  1)
    };
    float2 c = corners[vid];

    // Cull dead particles by collapsing them to a degenerate point off-screen.
    if (p.life <= 0.0) {
        ParticleOut o;
        o.position = float4(0, 0, -10, 1);
        o.uv = float2(0.5);
        o.life = 0;
        o.seed = p.seed;
        return o;
    }

    // Billboard: pull the camera's right and up axes from the inverse-view
    // basis (rows of the view matrix's upper-left 3×3, since view here is
    // pure translation — camera rotation is identity).
    float3 right = float3(1, 0, 0);
    float3 up    = float3(0, 1, 0);

    // Per-particle size + a "born" easing so they don't pop in.
    float sizeMul = 0.5 + 0.5 * fract(p.seed * 0.137);
    float born = smoothstep(0.0, 0.15, 1.0 - p.life);  // small at start, grows
    float size = 0.045 * sizeMul * (0.6 + 0.6 * born);

    float3 worldPos = p.position + (right * c.x + up * c.y) * size;

    ParticleOut o;
    o.position = u.projection * u.view * float4(worldPos, 1.0);
    o.uv   = c * 0.5 + 0.5;
    o.life = p.life;
    o.seed = p.seed;
    return o;
}

fragment float4 particle_fragment(ParticleOut in [[stage_in]]) {
    if (in.life <= 0.0) discard_fragment();

    // Soft circular sparkle with a 4-pointed star kicker.
    float2 d = in.uv - 0.5;
    float r = length(d) * 2.0;
    if (r > 1.0) discard_fragment();

    float core = pow(1.0 - r, 2.5);
    float star = pow(max(0.0, 1.0 - abs(d.x) * 6.0), 8.0)
               + pow(max(0.0, 1.0 - abs(d.y) * 6.0), 8.0);
    float intensity = core + star * 0.35;

    // Fade in at birth (life near 1), fade out as it dies.
    float lifeFade = smoothstep(0.0, 0.15, in.life)
                   * smoothstep(0.0, 0.30, 1.0 - max(0.0, in.life - 0.7) * 3.3);

    // Color varies per-particle: warm gold → pink → cyan tints.
    float h = fract(in.seed * 0.257);
    float3 warm = float3(1.0, 0.95, 0.65);
    float3 pink = float3(1.0, 0.55, 0.85);
    float3 cyan = float3(0.65, 0.85, 1.0);
    float3 c1 = mix(warm, pink, smoothstep(0.0, 0.5, h));
    float3 color = mix(c1, cyan, smoothstep(0.5, 1.0, h));

    return float4(color * intensity * lifeFade, intensity * lifeFade);
}
