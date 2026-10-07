// Signal.swift — the small DSP the tokenizers need, translated from mlx-speech's
// `step_audio_tokenizer/processor.py` (`_build_slaney_mel_filters`, `_periodic_hann_window`,
// `_stft_power_reflect_padded`, the Whisper log-mel) with the tables built in Double before the fp32 cast.

import Foundation
import MLX
import MLXFFT

public enum Signal {
    // --- Slaney mel scale (librosa / Whisper), as in processor.py
    static let slaneyFSp = 200.0 / 3.0
    static let slaneyMinLogHz = 1000.0
    static let slaneyMinLogMel = 1000.0 / (200.0 / 3.0)
    static let slaneyLogStep = log(6.4) / 27.0

    static func hzToMel(_ f: Double) -> Double {
        f < slaneyMinLogHz ? f / slaneyFSp : slaneyMinLogMel + log(f / slaneyMinLogHz) / slaneyLogStep
    }
    static func melToHz(_ m: Double) -> Double {
        m < slaneyMinLogMel ? m * slaneyFSp : slaneyMinLogHz * exp(slaneyLogStep * (m - slaneyMinLogMel))
    }

    /// (n_mels, n_fft/2+1) Slaney-normalised triangular filters — `_build_slaney_mel_filters`.
    public static func slaneyMelFilters(sampleRate: Int, nFFT: Int, nMels: Int, fmin: Double = 0, fmax: Double? = nil) -> MLXArray {
        let nFreqs = nFFT / 2 + 1, nyquist = Double(sampleRate) / 2
        let fMax = fmax ?? nyquist
        let fftFreqs = (0..<nFreqs).map { Double($0) * nyquist / Double(nFreqs - 1) }
        let melMin = hzToMel(fmin), melMax = hzToMel(fMax)
        let hz = (0..<(nMels + 2)).map { melToHz(melMin + (melMax - melMin) * Double($0) / Double(nMels + 1)) }
        var filters = [Float](repeating: 0, count: nMels * nFreqs)
        for i in 0..<nMels {
            let enorm = 2.0 / (hz[i + 2] - hz[i])
            for j in 0..<nFreqs {
                let lower = (fftFreqs[j] - hz[i]) / (hz[i + 1] - hz[i])        // -ramps[i] / (hz[i+1]-hz[i])
                let upper = (hz[i + 2] - fftFreqs[j]) / (hz[i + 2] - hz[i + 1]) // ramps[i+2] / (hz[i+2]-hz[i+1])
                filters[i * nFreqs + j] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return MLXArray(filters, [nMels, nFreqs])
    }

    /// 0.5 − 0.5·cos(2πn/N) — numpy/torch "periodic" Hann.
    public static func periodicHann(_ n: Int) -> MLXArray {
        MLXArray((0..<n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) })
    }

    /// `np.pad(x, (p, p), mode="reflect")` for a 1-D signal (mlx-swift's PadMode has no reflect).
    public static func reflectPadded(_ x: MLXArray, _ p: Int) -> MLXArray {
        precondition(x.ndim == 1 && x.dim(0) > p, "reflect pad needs more than \(p) samples")
        let left = x[1 ... p][.stride(by: -1)]
        let n = x.dim(0)
        let right = x[(n - 1 - p) ..< (n - 1)][.stride(by: -1)]
        return concatenated([left, x, right])
    }

    /// Frames (nFrames, frameLength) with hop `hop` over a 1-D signal, no padding (numpy `1 + (N − L) // hop`).
    public static func framed(_ x: MLXArray, frameLength: Int, hop: Int) -> MLXArray {
        let n = 1 + (x.dim(0) - frameLength) / hop
        return asStrided(x, [n, frameLength], strides: [hop, 1])
    }

    /// `_stft_power_reflect_padded`: reflect-pad n_fft/2, periodic Hann, |rfft|², drop the last frame → (n_fft/2+1, T).
    public static func stftPowerReflectPadded(_ wav: MLXArray, nFFT: Int, hop: Int) -> MLXArray {
        let padded = reflectPadded(wav, nFFT / 2)
        let frames = framed(padded, frameLength: nFFT, hop: hop) * periodicHann(nFFT)   // (T, nFFT)
        let spec = MLXFFT.rfft(frames, n: nFFT, axis: -1)                                 // (T, nFFT/2+1) complex
        let power = abs(spec) ** 2
        let t = power.dim(0)
        return power[..<(t - 1)].transposed(1, 0)                                         // (F, T−1)
    }

    /// Whisper's log-mel as the S3 tokenizer front end expects it (processor.compute_vq06_log_mel_spectrogram):
    /// log10 with a 1e-10 floor, clamp to max − 8, (x + 4) / 4 → (n_mels, T).
    public static func whisperLogMel(_ wav16: MLXArray, filters: MLXArray, nFFT: Int, hop: Int) -> MLXArray {
        let power = stftPowerReflectPadded(wav16, nFFT: nFFT, hop: hop)
        let mel = filters.matmul(power)
        var logSpec = log10(maximum(mel, MLXArray(Float(1e-10))))
        logSpec = maximum(logSpec, logSpec.max() - 8)
        return (logSpec + 4) / 4
    }
}
