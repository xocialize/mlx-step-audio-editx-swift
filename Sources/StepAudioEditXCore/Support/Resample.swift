// Resample.swift — torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), the
// kernel re-derived in Double exactly as Tools/oracle-capture/sinc_resample.py does (that numpy version matches
// torchaudio to 3e-7). The reference tokenizes 16 kHz audio, the CAM++ front end too; the Studio's takes arrive at
// 24 or 48 kHz. (A linear-interpolation resampler here perturbed every 16 kHz consumer — AB-L-0209.)

import Foundation

public enum Resample {
    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    public static func sinc(_ wav: [Float], from origRate: Int, to newRate: Int,
                            lowpassFilterWidth: Int = 6, rolloff: Double = 0.99) -> [Float] {
        if origRate == newRate || wav.isEmpty { return wav }
        let g = gcd(origRate, newRate), orig = origRate / g, new = newRate / g
        let baseFreq = Double(min(orig, new)) * rolloff
        let width = Int(ceil(Double(lowpassFilterWidth * orig) / baseFreq))
        let k = 2 * width + orig
        // kernels (new, K)
        var kernels = [Double](repeating: 0, count: new * k)
        for p in 0..<new {
            for j in 0..<k {
                let idx = Double(-width + j) / Double(orig)
                var t = (-Double(p) / Double(new) + idx) * baseFreq
                t = min(max(t, -Double(lowpassFilterWidth)), Double(lowpassFilterWidth))
                let window = pow(cos(t * Double.pi / Double(lowpassFilterWidth) / 2), 2)
                let tp = t * Double.pi
                let sincv = tp == 0 ? 1.0 : sin(tp) / tp
                kernels[p * k + j] = sincv * window * (baseFreq / Double(orig))
            }
        }
        let n = wav.count, target = Int(ceil(Double(new * n) / Double(orig)))
        // x = pad(wav, (width, width + orig)); frames every `orig` samples; out[frame, p] = Σ_j kernels[p, j] · x[frame·orig + j]
        var x = [Double](repeating: 0, count: width + n + width + orig)
        for i in 0..<n { x[width + i] = Double(wav[i]) }
        let nFrames = (x.count - k) / orig + 1
        var out = [Float](repeating: 0, count: nFrames * new)
        for f in 0..<nFrames {
            let base = f * orig
            for p in 0..<new {
                var acc = 0.0
                let row = p * k
                for j in 0..<k { acc += kernels[row + j] * x[base + j] }
                out[f * new + p] = Float(acc)
            }
        }
        return Array(out.prefix(target))
    }
}
