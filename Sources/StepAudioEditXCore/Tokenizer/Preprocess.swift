// Preprocess.swift — what both tokenizers consume: the take at 16 kHz, peak-normalised, silence-trimmed with a
// fixed lead/tail and a length rounded to the vq02 hop. Translated 1:1 from processor.py
// (`preprocess_wav`, `_energy_normalize`, `_signal_to_frame_nonsilent`, `_trim_silence`), which matches the upstream
// `utils.{energy_norm_fn, trim_silence}` + torchaudio resample to 3e-7 (parity run 3).

import Foundation

public enum Preprocess {
    /// `_energy_normalize`: peak → 0.999, with the peak floored at 0.01.
    public static func energyNormalize(_ wav: [Float]) -> [Float] {
        guard let peak = wav.map({ abs($0) }).max(), peak > 0 else { return wav }
        let s = Float(0.999) / max(peak, 0.01)
        return wav.map { $0 * s }
    }

    /// `_signal_to_frame_nonsilent` — librosa.effects.trim's frame test: centred RMS frames, dB relative to the
    /// loudest frame, non-silent where > −top_db.
    public static func nonSilentFrames(_ wav: [Float], topDb: Double, frameLength: Int, hop: Int) -> [Bool] {
        if wav.isEmpty { return [] }
        let pad = frameLength / 2
        var padded = [Float](repeating: 0, count: wav.count + 2 * pad)
        for i in 0..<wav.count { padded[pad + i] = wav[i] }
        let frameCount = 1 + (padded.count - frameLength) / hop
        if frameCount <= 0 { return [] }
        var rms = [Float](repeating: 0, count: frameCount)
        for f in 0..<frameCount {
            var acc: Float = 0
            let start = f * hop
            for i in 0..<frameLength { let v = padded[start + i]; acc += v * v }
            rms[f] = (acc / Float(frameLength)).squareRoot()
        }
        let ref = rms.max() ?? 0
        if ref <= 0 { return [Bool](repeating: false, count: frameCount) }
        let refDb = 20 * log10(Double(max(ref, 1e-10)))
        return rms.map { 20 * log10(Double(max($0, 1e-10))) - refDb > -topDb }
    }

    /// `_trim_silence`: trim to the non-silent span, keep `keepLeft` / `keepRight` seconds around it, and round the
    /// span to whole `outputHop` frames (zero-padding when the take is shorter than the rounded length).
    public static func trimSilence(_ wav: [Float], sampleRate: Int, topDb: Double, frameLength: Int, hop: Int,
                                   keepLeft: Double, keepRight: Double, outputHop: Int) -> [Float] {
        let nonSilent = nonSilentFrames(wav, topDb: topDb, frameLength: frameLength, hop: hop)
        let idx = nonSilent.enumerated().filter { $0.element }.map(\.offset)
        let start = idx.first.map { $0 * hop } ?? 0
        let end = idx.last.map { min(wav.count, ($0 + 1) * hop) } ?? 0
        let numFrames = Int(ceil(Double(max(0, end - start)) / Double(outputHop)))
        let leftKeep = Int(keepLeft * Double(sampleRate))
        let startIdx = start - leftKeep
        var trimmed: [Float]
        if startIdx > 0 { trimmed = Array(wav[startIdx...]) }
        else { trimmed = [Float](repeating: 0, count: -startIdx) + wav }
        let outLen = Int(Double(numFrames * outputHop) + (keepLeft + keepRight) * Double(sampleRate))
        if outLen < trimmed.count { trimmed = Array(trimmed.prefix(outLen)) }
        else { trimmed += [Float](repeating: 0, count: outLen - trimmed.count) }
        return trimmed
    }

    /// `preprocess_wav`: mono → 16 kHz (sinc) → energy-normalise → trim.
    public static func run(_ wav: [Float], sampleRate: Int, config c: TokenizerConfig,
                           enableTrim: Bool = true, energyNorm: Bool = true) -> [Float] {
        var x = sampleRate == c.vq02_sample_rate ? wav : Resample.sinc(wav, from: sampleRate, to: c.vq02_sample_rate)
        if energyNorm { x = energyNormalize(x) }
        if enableTrim {
            x = trimSilence(x, sampleRate: c.vq02_sample_rate, topDb: c.trim_top_db, frameLength: c.trim_frame_length,
                            hop: c.trim_hop_length, keepLeft: c.trim_keep_left_seconds, keepRight: c.trim_keep_right_seconds,
                            outputHop: c.trim_output_hop_samples)
        }
        return x
    }
}
