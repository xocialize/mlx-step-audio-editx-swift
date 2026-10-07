// MelFrontend.swift — the CosyVoice 24 kHz mel the flow is conditioned on (`frontend.py: mel_spectrogram`): reflect-pad
// (n_fft − hop)/2, periodic Hann over win_size, |rfft| as √(|X|² + 1e-9), Slaney mel 80 bins 0–8 kHz, log with a 1e-5
// floor → (T, 80). The prompt take is resampled to 24 kHz with the same sinc kernel when it arrives at another rate.

import Foundation
import MLX
import MLXFFT

public struct MelFrontend {
    public let config: MelFrontendConfig
    let filters: MLXArray        // (80, n_fft/2+1)
    let window: MLXArray         // (win_size,)

    public init(_ c: MelFrontendConfig) {
        config = c
        filters = Signal.slaneyMelFilters(sampleRate: c.sampling_rate, nFFT: c.n_fft, nMels: c.num_mels, fmin: c.fmin, fmax: c.fmax)
        window = Signal.periodicHann(c.win_size)
    }

    public static func load(bundle: EditXBundle) throws -> MelFrontend {
        MelFrontend(try ConfigIO.load(MelFrontendConfig.self, from: bundle.file("frontend-config.json")))
    }

    /// `_stft_magnitude_padded` → `mel_spectrogram`: 1-D 24 kHz waveform → (T, 80) log-mel.
    public func callAsFunction(_ wav24: MLXArray) -> MLXArray {
        let c = config
        let pad = (c.n_fft - c.hop_size) / 2
        let padded = Signal.reflectPadded(wav24, pad)
        var frames = Signal.framed(padded, frameLength: c.n_fft, hop: c.hop_size)          // (T, n_fft)
        if c.win_size < c.n_fft {
            let w = concatenated([window, MLXArray.zeros([c.n_fft - c.win_size])])
            frames = frames * w
        } else {
            frames = frames * window
        }
        let spec = MLXFFT.rfft(frames, n: c.n_fft, axis: -1)
        let magnitude = sqrt(abs(spec) ** 2 + 1e-9)                                           // (T, F)
        let mel = magnitude.matmul(filters.transposed(1, 0))                                   // (T, 80)
        return log(maximum(mel, MLXArray(Float(1e-5))))
    }

    /// Any-rate [Float] → (T, 80).
    public func features(_ wav: [Float], sampleRate: Int) -> MLXArray {
        let w = sampleRate == config.sampling_rate ? wav : Resample.sinc(wav, from: sampleRate, to: config.sampling_rate)
        return self(MLXArray(w))
    }
}
