import simd

extension matrix_float4x4 {
    static func perspective(fovyRadians fovy: Float, aspect: Float, near: Float, far: Float) -> matrix_float4x4 {
        let ys = 1 / tanf(fovy * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        return matrix_float4x4(columns: (
            SIMD4<Float>(xs, 0, 0, 0),
            SIMD4<Float>(0, ys, 0, 0),
            SIMD4<Float>(0, 0, zs, -1),
            SIMD4<Float>(0, 0, zs * near, 0)
        ))
    }

    static func translation(_ t: SIMD3<Float>) -> matrix_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
        return m
    }

    static func uniformScale(_ s: Float) -> matrix_float4x4 {
        matrix_float4x4(diagonal: SIMD4<Float>(s, s, s, 1))
    }

    static func from(quaternion q: simd_quatf) -> matrix_float4x4 {
        let m3 = matrix_float3x3(q)
        return matrix_float4x4(columns: (
            SIMD4<Float>(m3.columns.0, 0),
            SIMD4<Float>(m3.columns.1, 0),
            SIMD4<Float>(m3.columns.2, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }

    var upperLeft3x3: matrix_float3x3 {
        matrix_float3x3(columns: (
            SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z),
            SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z),
            SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z)
        ))
    }
}
