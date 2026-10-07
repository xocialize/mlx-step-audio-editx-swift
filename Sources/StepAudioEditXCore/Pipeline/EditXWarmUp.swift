// EditXWarmUp.swift — a one-second edit at load, so the first real edit pays no Metal kernel compile.
//
// MLX compiles each Metal kernel the first time a process uses it. Measured in ML[X] Audio Studio (AB-R-0420): the
// first edit after load took 12–14 s wall where the steady state is 3 s (RTF 2.9 against 0.5), and the cost came back
// in every process — it is per process, not per binary. One short edit through every stage (both tokenizers, CAM++,
// the LM, the flow, HiFT) compiles them all, and the kernels are shape-generic, so a synthetic one-second vowel does
// the same work as a real take without shipping a clip.

import Foundation

extension EditXPipeline {
    /// One second of a synthetic "ah" at 24 kHz: a 120 Hz harmonic series shaped by two formants, a touch of
    /// vibrato, 20 ms fades, peak 0.3. Deterministic — nothing is kept from it but the compiled kernels.
    public static func warmUpSignal(sampleRate: Int = 24_000, seconds: Double = 1.0) -> [Float] {
        let n = Int(Double(sampleRate) * seconds)
        let f0 = 120.0
        let formants: [(centre: Double, width: Double)] = [(700, 130), (1_200, 180)]
        let fadeSamples = 0.02 * Double(sampleRate)
        var out = [Float](repeating: 0, count: n)
        var peak: Float = 0
        for i in 0 ..< n {
            let t = Double(i) / Double(sampleRate)
            let f = f0 * (1 + 0.01 * sin(2 * .pi * 5 * t))
            var s = 0.0
            var k = 1
            while Double(k) * f < 4_000 {
                let fk = Double(k) * f
                let gain = formants.reduce(0.05) { $0 + exp(-pow((fk - $1.centre) / $1.width, 2)) }
                s += gain / Double(k) * sin(2 * .pi * fk * t)
                k += 1
            }
            let fade = min(1, Double(i) / fadeSamples, Double(n - 1 - i) / fadeSamples)
            out[i] = Float(s * max(0, fade))
            peak = max(peak, abs(out[i]))
        }
        if peak > 0 {
            let g = 0.3 / peak
            for i in 0 ..< n { out[i] *= g }
        }
        return out
    }

    /// The warm-up edit. The result is discarded; a caller ignores every error but cancellation.
    public func warmUp(checkpoint: (() throws -> Void)? = nil) throws {
        _ = try edit(Self.warmUpSignal(), sampleRate: 24_000, text: "Ah.", edit: .emotion("happy"), seed: 1,
                     maxNewTokens: 64, checkpoint: checkpoint)
    }
}
