// CampPlusFrontend.swift — the CAM++ speaker embedding end to end: 16 kHz audio → kaldi 80-fbank (povey window,
// 25/10 ms, 512-point FFT, torchaudio's mel banks baked in `kaldi_mel_banks.npy`) − time mean → CAMPPlus → (192).
// The fbank core and the CAMPPlus module are lifted from mlx-indextts2-swift (`Frontend/FbankFrontend.swift`,
// `Models/CampPlus.swift`); the checkpoint is that package's `campplus_cn_common.safetensors` — the same 3D-Speaker
// model EditX ships as `campplus.onnx` (AB-R-0413). The reference (`frontend.py: extract_spk_embedding`) resamples to
// 16 kHz with torchaudio's sinc and runs `kaldi.fbank(num_mel_bins=80, dither=0)` then subtracts the time mean.

import Foundation
import MLX
import MLXAudioDSP

enum CampPlusResources {
    static let melFloor: Float = 1.192092955078125e-07

    private static func loadNPY(_ name: String) -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy", subdirectory: "Resources"),
              let array = try? NPY.load(url)
        else { fatalError("missing baked resource \(name).npy") }
        return array
    }

    /// (257, 80) — torchaudio kaldi banks, transposed from (80, 257) at load (zero Nyquist column already appended).
    static let kaldiMelBanks = loadNPY("kaldi_mel_banks").transposed()

    /// Lifted `FrontendResources.kaldiLogMel`: 1-D waveform → (t, 80), povey window.
    static func kaldiLogMel(_ waveform: MLXArray, filters: MLXArray) -> MLXArray? {
        guard let frames = AudioDSP.framedSnipEdges(waveform, frameLength: 400, hop: 160) else { return nil }
        var x = AudioDSP.removeDCOffset(frames)
        x = AudioDSP.preEmphasized(x, coefficient: 0.97)
        let power = AudioDSP.powerSpectrum(x, window: AudioDSP.poveyWindow(400), fftLength: 512)   // (t, 257)
        let mel = power.matmul(filters)                                                              // (t, 80)
        return MLX.log(maximum(mel, melFloor))
    }
}

public enum CampPlusFbank {
    /// Raw kaldi log-mel: (t, 80) (no input scaling — CMN cancels it downstream).
    public static func fbank(_ waveform16k: MLXArray) -> MLXArray? {
        CampPlusResources.kaldiLogMel(waveform16k, filters: CampPlusResources.kaldiMelBanks)
    }
    /// fbank − time-mean (the CampPlus input): (t, 80).
    public static func fbankCMN(_ waveform16k: MLXArray) -> MLXArray? {
        guard let raw = fbank(waveform16k) else { return nil }
        return raw - raw.mean(axis: 0, keepDims: true)
    }
}

/// Loads CAMPPlus from the package resources and runs it on audio or on a precomputed fbank.
public final class SpeakerEncoder {
    public let model: CAMPPlus

    public init(model: CAMPPlus) { self.model = model }

    public static func load(dtype: DType = .float32) throws -> SpeakerEncoder {
        guard let url = Bundle.module.url(forResource: "campplus_cn_common", withExtension: "safetensors", subdirectory: "Resources") else {
            throw StepAudioEditXError.missingFile("Resources/campplus_cn_common.safetensors")
        }
        let model = CAMPPlus()
        let raw = try WeightIO.load(url, dtype: dtype)
        try WeightIO.apply(CAMPPlus.sanitize(raw), to: model, component: "campplus")
        return SpeakerEncoder(model: model)
    }

    /// (t, 80) CMN fbank → (192,)
    public func embed(fbankCMN: MLXArray) -> MLXArray {
        let y = model(fbankCMN.expandedDimensions(axis: 0))   // (1, 192)
        eval(y)
        return y[0]
    }

    /// Any-rate audio → (192,): resample to 16 kHz (sinc), fbank, CMN, CAMPPlus.
    public func embed(_ wav: [Float], sampleRate: Int) throws -> MLXArray {
        let w = sampleRate == 16000 ? wav : Resample.sinc(wav, from: sampleRate, to: 16000)
        guard let fb = CampPlusFbank.fbankCMN(MLXArray(w)) else { throw StepAudioEditXError.invalidInput("audio too short for a CAM++ fbank") }
        return embed(fbankCMN: fb)
    }
}
