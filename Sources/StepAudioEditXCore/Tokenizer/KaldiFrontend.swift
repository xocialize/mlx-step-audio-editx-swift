// KaldiFrontend.swift — the Paraformer (FunASR) streaming front end: kaldi fbank (hamming, 25/10 ms, 512-point FFT,
// pre-emphasis 0.97, DC removal, HTK mel 80 bins 20 Hz–Nyquist, log with the fp32-eps floor) → low-frame-rate stacking
// (LFR m=7, n=6) → CMVN, carried chunk to chunk through the same caches as mlx-speech's `StepAudioVQ02Frontend`
// (`vq02.py`), which reproduces FunASR's `WavFrontendOnline`. Dither is OFF: inference must be reproducible and the
// reference's dither is a noise draw no second implementation can match.

import Foundation
import MLX
import MLXFFT

/// kaldi fbank over a 1-D waveform (already × 2^15), as `_kaldi_fbank` with dither = 0 → (frames, n_mels).
public struct KaldiFbank {
    public let sampleRate: Int, frameLength: Int, frameShift: Int, paddedWindow: Int, nMels: Int
    public let preemphasis: Float, removeDC: Bool
    let window: MLXArray          // (frameLength,)
    let melBanks: MLXArray        // (paddedWindow/2 + 1, nMels) — zero Nyquist row appended, transposed for the matmul

    public init(_ f: VQ02Config.Frontend) {
        sampleRate = f.sample_rate
        frameShift = Int(Double(f.sample_rate) * f.frame_shift_ms / 1000)
        frameLength = Int(Double(f.sample_rate) * f.frame_length_ms / 1000)
        paddedWindow = f.round_to_power_of_two ? Self.nextPowerOfTwo(frameLength) : frameLength
        nMels = f.n_mels; preemphasis = Float(f.preemphasis_coefficient); removeDC = f.remove_dc_offset
        window = Self.window(f.window_type, frameLength)
        melBanks = Self.melBanks(numBins: f.n_mels, paddedWindow: paddedWindow, sampleRate: Double(f.sample_rate),
                                 lowFreq: f.low_freq, highFreq: f.high_freq)
    }

    static func nextPowerOfTwo(_ v: Int) -> Int { v == 0 ? 1 : 1 << (Int.bitWidth - (v - 1).leadingZeroBitCount) }

    static func window(_ type: String, _ n: Int) -> MLXArray {
        let w: [Double]
        switch type {
        case "hamming": w = (0..<n).map { 0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(n - 1)) }
        case "hanning": w = (0..<n).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n - 1)) }
        case "povey": w = (0..<n).map { pow(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n - 1)), 0.85) }
        default: w = [Double](repeating: 1, count: n)
        }
        return MLXArray(w.map { Float($0) })
    }

    /// `_get_mel_banks` in float32 arithmetic (the reference computes it in np.float32), then a zero Nyquist column.
    static func melBanks(numBins: Int, paddedWindow: Int, sampleRate: Double, lowFreq: Double, highFreq: Double) -> MLXArray {
        let numFFTBins = paddedWindow / 2
        let nyquist = sampleRate / 2
        let high = highFreq <= 0 ? nyquist + highFreq : highFreq
        let binWidth = Float(sampleRate / Double(paddedWindow))
        func mel(_ f: Float) -> Float { 1127 * log1p(f / 700) }
        let melLow = mel(Float(lowFreq)), melHigh = mel(Float(high))
        let melDelta = (melHigh - melLow) / Float(numBins + 1)
        var banks = [Float](repeating: 0, count: numBins * (numFFTBins + 1))
        for i in 0..<numBins {
            let left = melLow + Float(i) * melDelta, centre = melLow + Float(i + 1) * melDelta, right = melLow + Float(i + 2) * melDelta
            for j in 0..<numFFTBins {
                let m = mel(binWidth * Float(j))
                let up = (m - left) / (centre - left), down = (right - m) / (right - centre)
                banks[i * (numFFTBins + 1) + j] = max(0, min(up, down))
            }
        }
        return MLXArray(banks, [numBins, numFFTBins + 1]).transposed(1, 0)
    }

    public func frameCount(_ samples: Int) -> Int { samples < frameLength ? 0 : 1 + (samples - frameLength) / frameShift }

    /// (frames, nMels) log-mel energies of `wav` (1-D, already scaled by 2^15).
    public func callAsFunction(_ wav: MLXArray) -> MLXArray {
        let n = frameCount(wav.dim(0))
        precondition(n > 0, "fbank needs at least one frame")
        var frames = asStrided(wav, [n, frameLength], strides: [frameShift, 1])
        if removeDC { frames = frames - frames.mean(axis: 1, keepDims: true) }
        if preemphasis != 0 {
            let prev = concatenated([frames[0..., ..<1], frames[0..., ..<(frameLength - 1)]], axis: 1)   // pad mode "edge", then drop the last
            frames = frames - preemphasis * prev
        }
        frames = frames * window
        if paddedWindow != frameLength { frames = padded(frames, widths: [IntOrPair((0, 0)), IntOrPair((0, paddedWindow - frameLength))]) }
        let spectrum = abs(MLXFFT.rfft(frames, n: paddedWindow, axis: 1)) ** 2          // (n, paddedWindow/2+1)
        let energies = spectrum.matmul(melBanks)                                         // (n, nMels)
        return log(maximum(energies, MLXArray(Float.ulpOfOne)))
    }
}

/// The streaming LFR + CMVN front end with FunASR's caches (`StepAudioVQ02Frontend`): feed one chunk at a time.
public final class VQ02Frontend {
    public let config: VQ02Config.Frontend
    let fbank: KaldiFbank
    let cmvnAdd: [Float], cmvnMul: [Float]          // cmvn[0, :560], cmvn[1, :560]
    let frameLengthSamples: Int, frameShiftSamples: Int
    // cache
    var inputCache: [Float] = []
    var reserveWaveforms: [Float] = []
    var waveforms: [Float]? = nil
    var lfrSpliceCache: [[Float]]? = nil             // (frames, nMels)

    public init(config: VQ02Config, cmvn: MLXArray) {
        self.config = config.frontend; self.fbank = KaldiFbank(config.frontend)
        let dim = config.encoder.input_size
        let c = cmvn.asType(.float32); eval(c)
        let flat = c.asArray(Float.self); let w = c.dim(1)
        cmvnAdd = Array(flat[0 ..< dim]); cmvnMul = Array(flat[w ..< (w + dim)])
        frameLengthSamples = fbank.frameLength; frameShiftSamples = fbank.frameShift
    }

    public func reset() { inputCache = []; reserveWaveforms = []; waveforms = nil; lfrSpliceCache = nil }

    func computeFrameNum(_ samples: Int) -> Int { (samples - frameLengthSamples) / frameShiftSamples + 1 }

    /// `forward_fbank`: (used waveform, fbank frames (T, nMels)) or nil when the chunk yields no whole frame.
    func forwardFbank(_ chunk: [Float]) -> (waveform: [Float], feats: [[Float]])? {
        let input = inputCache + chunk
        let frameNum = computeFrameNum(input.count)
        let tail = input.count - frameNum * frameShiftSamples
        inputCache = tail > 0 ? Array(input.suffix(tail)) : []
        if frameNum <= 0 { return nil }
        let usedLength = (frameNum - 1) * frameShiftSamples + frameLengthSamples
        let waveform = Array(input.prefix(usedLength))
        let f = fbank(MLXArray(waveform.map { $0 * 32768 }))
        eval(f)
        let rows = f.dim(0), cols = f.dim(1); let flat = f.asArray(Float.self)
        let feats = (0..<rows).map { Array(flat[($0 * cols) ..< (($0 + 1) * cols)]) }
        return (waveform, feats)
    }

    func applyCMVN(_ row: [Float]) -> [Float] { (0..<row.count).map { (row[$0] + cmvnAdd[$0]) * cmvnMul[$0] } }

    /// `forward_lfr_cmvn` for one utterance: (stacked (T', m·nMels), splice index).
    func forwardLFRCMVN(_ feats: [[Float]], isFinal: Bool) -> (out: [[Float]], spliceIdx: Int) {
        let m = config.lfr_m, n = config.lfr_n, total = feats.count
        let outputFrames = Int(ceil(Double(total - (m - 1) / 2) / Double(n)))
        var spliceIdx = outputFrames
        var out = [[Float]]()
        for i in 0..<max(outputFrames, 0) {
            let start = i * n
            if m <= total - start {
                out.append(applyCMVN(feats[start ..< (start + m)].flatMap { $0 }))
            } else if isFinal {
                var frame = feats[start...].flatMap { $0 }
                for _ in 0 ..< (m - (total - start)) { frame += feats[total - 1] }
                out.append(applyCMVN(frame))
            } else { spliceIdx = i; break }
        }
        let splice = min(total - 1, spliceIdx * n)
        lfrSpliceCache = Array(feats[max(splice, 0)...])
        return (out, splice)
    }

    /// One chunk of 16 kHz samples → encoder input rows (T', 560), possibly empty.
    public func callAsFunction(_ chunk: [Float], isFinal: Bool) -> [[Float]] {
        let m = config.lfr_m
        if let (waveform, fb) = forwardFbank(chunk), !fb.isEmpty {
            waveforms = reserveWaveforms.isEmpty ? waveform : reserveWaveforms + waveform
            if lfrSpliceCache == nil { lfrSpliceCache = [[Float]](repeating: fb[0], count: (m - 1) / 2) }
            if fb.count + lfrSpliceCache!.count >= m {
                let feats = lfrSpliceCache! + fb
                let frameFromWaveforms = (waveforms!.count - frameLengthSamples) / frameShiftSamples + 1
                let minusFrame = reserveWaveforms.isEmpty ? (m - 1) / 2 : 0
                let (out, splice) = forwardLFRCMVN(feats, isFinal: isFinal)
                let reserveFrameIdx = splice - minusFrame
                reserveWaveforms = Array(waveforms![(reserveFrameIdx * frameShiftSamples) ..< (frameFromWaveforms * frameShiftSamples)])
                let sampleLength = (frameFromWaveforms - 1) * frameShiftSamples + frameLengthSamples
                waveforms = Array(waveforms!.prefix(sampleLength))
                return out
            } else {
                reserveWaveforms = Array(waveforms!.dropLast(frameLengthSamples - frameShiftSamples))
                lfrSpliceCache! += fb
                return []
            }
        } else if isFinal {
            waveforms = reserveWaveforms.isEmpty ? [] : reserveWaveforms
            guard let cache = lfrSpliceCache, !cache.isEmpty else { return [] }
            return forwardLFRCMVN(cache, isFinal: true).out
        }
        return []
    }
}
