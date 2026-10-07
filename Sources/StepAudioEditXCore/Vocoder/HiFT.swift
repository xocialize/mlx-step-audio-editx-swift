// HiFT.swift — the HiFT vocoder (CosyVoice lineage, StepFun-trained): a ConvRNN-free f0 predictor on the mel, an
// NSF harmonic source (8 harmonics + noise, through a 9→1 linear and tanh) whose 16-point STFT joins the upsampling
// stack at each stage, Snake-activated resblocks, and an iSTFT head (n_fft 16, hop 4) producing 24 kHz audio.
// Translated 1:1 from mlx-speech's `step_audio_editx/hift.py`; module paths equal the bundle keys (`conv_pre.weight`,
// `ups.N.weight`, `resblocks.N.convs1.N.weight`, `resblocks.N.activations1.N.alpha`, `source_downs.N.weight`,
// `f0_predictor.condnet.N.weight`, `m_source.l_linear.weight` …). Tensors run channels-first (B, C, T) between modules
// as the reference does, transposing around MLX's channels-last convs.

import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXRandom

func leakyReLU(_ x: MLXArray, _ slope: Float) -> MLXArray { maximum(x, 0) + slope * minimum(x, 0) }
func elu(_ x: MLXArray) -> MLXArray { MLX.where(x .> 0, x, exp(x) - 1) }
func applyConv(_ conv: Conv1d, _ x: MLXArray) -> MLXArray { conv(x.transposed(0, 2, 1)).transposed(0, 2, 1) }
func applyConvT(_ conv: ConvTransposed1d, _ x: MLXArray) -> MLXArray { conv(x.transposed(0, 2, 1)).transposed(0, 2, 1) }

public final class Snake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    public init(channels: Int) { self._alpha.wrappedValue = MLXArray.ones([channels]) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {        // x (B, C, T)
        let a = alpha.reshaped([1, -1, 1])
        return x + (1 / (a + 1e-9)) * sin(x * a) ** 2
    }
}

public final class HiFTResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "activations1") var activations1: [Snake]
    @ModuleInfo(key: "activations2") var activations2: [Snake]
    public init(channels: Int, kernelSize: Int, dilations: [Int]) {
        self._convs1.wrappedValue = dilations.map { Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize, stride: 1, padding: (kernelSize * $0 - $0) / 2, dilation: $0, bias: true) }
        self._convs2.wrappedValue = dilations.map { _ in Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize, stride: 1, padding: (kernelSize - 1) / 2, bias: true) }
        self._activations1.wrappedValue = dilations.map { _ in Snake(channels: channels) }
        self._activations2.wrappedValue = dilations.map { _ in Snake(channels: channels) }
    }
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for i in 0 ..< convs1.count {
            var xt = activations1[i](x)
            xt = applyConv(convs1[i], xt)
            xt = activations2[i](xt)
            xt = applyConv(convs2[i], xt)
            x = xt + x
        }
        return x
    }
}

public final class F0Predictor: Module {
    @ModuleInfo(key: "condnet") var condnet: [Conv1d]
    @ModuleInfo(key: "classifier") var classifier: Linear
    public init(_ c: HiFTConfig.F0Predictor) {
        self._condnet.wrappedValue = (0 ..< 5).map { Conv1d(inputChannels: $0 == 0 ? c.in_channels : c.cond_channels, outputChannels: c.cond_channels, kernelSize: 3, stride: 1, padding: 1, bias: true) }
        self._classifier.wrappedValue = Linear(c.cond_channels, c.num_class, bias: true)
    }
    /// mel (B, 80, T) → f0 (B, T) Hz
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel
        for conv in condnet { x = elu(applyConv(conv, x)) }
        return abs(classifier(x.transposed(0, 2, 1)).squeezed(axis: -1))
    }
}

/// Noise for the NSF source: MLXRandom by default; the gate replays the oracle's draws.
public protocol SourceNoise {
    func uniform(_ shape: [Int]) -> MLXArray          // [0, 1)
    func normal(_ shape: [Int]) -> MLXArray
}
public struct RandomSourceNoise: SourceNoise {
    public init() {}
    public func uniform(_ shape: [Int]) -> MLXArray { MLXRandom.uniform(low: 0, high: 1, shape) }
    public func normal(_ shape: [Int]) -> MLXArray { MLXRandom.normal(shape) }
}
public final class ReplaySourceNoise: SourceNoise {
    var uniforms: [MLXArray], normals: [MLXArray]
    public init(uniforms: [MLXArray], normals: [MLXArray]) { self.uniforms = uniforms; self.normals = normals }
    public func uniform(_ shape: [Int]) -> MLXArray { let x = uniforms.removeFirst(); precondition(x.shape == shape, "uniform draw \(x.shape) vs \(shape)"); return x }
    public func normal(_ shape: [Int]) -> MLXArray { let x = normals.removeFirst(); precondition(x.shape == shape, "normal draw \(x.shape) vs \(shape)"); return x }
}

/// `SourceModuleHnNSF2`: f0 at the sample rate (B, T, 1) → (B, T, 1) harmonic source, plus noise and u/v.
public final class SourceModuleHnNSF: Module {
    @ModuleInfo(key: "l_linear") var lLinear: Linear
    let sampleRate: Int, upsampleScale: Int, harmonicNum: Int, sineAmp: Float, noiseStd: Float, voicedThreshold: Float
    public var noise: SourceNoise = RandomSourceNoise()

    public init(sampleRate: Int, upsampleScale: Int, harmonicNum: Int, sineAmp: Float, noiseStd: Float, voicedThreshold: Float) {
        self.sampleRate = sampleRate; self.upsampleScale = upsampleScale; self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp; self.noiseStd = noiseStd; self.voicedThreshold = voicedThreshold
        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1, bias: true)
    }

    /// `torch.nn.functional.interpolate(mode="linear", align_corners=False)` along the time axis of (B, T, C).
    static func linearResize(_ x: MLXArray, to target: Int) -> MLXArray {
        let n = x.dim(1)
        if n == target { return x }
        let scale = Float(n) / Float(target)
        var i0 = [Int32](), i1 = [Int32](), w1 = [Float]()
        for t in 0 ..< target {
            let src = max((Float(t) + 0.5) * scale - 0.5, 0)
            let lo = min(Int(floor(src)), n - 1), hi = min(lo + 1, n - 1)
            i0.append(Int32(lo)); i1.append(Int32(hi)); w1.append(src - Float(lo))
        }
        let w = MLXArray(w1).reshaped([1, -1, 1])
        return x[0..., MLXArray(i0), 0...] * (1 - w) + x[0..., MLXArray(i1), 0...] * w
    }

    func f0ToSine(_ f0Values: MLXArray) -> MLXArray {                 // (B, T, H+1)
        var rad = (f0Values / Float(sampleRate)) % 1
        var randIni = noise.uniform([f0Values.dim(0), f0Values.dim(2)])
        randIni = concatenated([MLXArray.zeros([f0Values.dim(0), 1]), randIni[0..., 1...]], axis: 1)
        rad = concatenated([rad[0..., ..<1, 0...] + randIni.expandedDimensions(axis: 1), rad[0..., 1..., 0...]], axis: 1)
        let downLength = max(Int((Float(rad.dim(1)) / Float(upsampleScale)).rounded()), 1)
        let down = Self.linearResize(rad, to: downLength)
        let phase = cumsum(down, axis: 1) * (2 * Float.pi)
        let up = Self.linearResize(phase, to: f0Values.dim(1)) * Float(upsampleScale)
        return sin(up)
    }

    /// f0 (B, T, 1) → (sine_merge (B, T, 1), noise, uv)
    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let harmonics = MLXArray((1 ... harmonicNum + 1).map { Float($0) }).reshaped([1, 1, -1])
        let fn = f0 * harmonics
        let sineWaves = f0ToSine(fn) * sineAmp
        let uv = (f0 .> voicedThreshold).asType(.float32)
        let noiseAmp = uv * noiseStd + (1 - uv) * (sineAmp / 3)
        let nz = noiseAmp * noise.normal(sineWaves.shape)
        let merged = tanh(lLinear(sineWaves * uv + nz))
        return (merged, nz, uv)
    }
}

final class WindowBox: @unchecked Sendable { let window: MLXArray; init(_ w: MLXArray) { window = w } }

public final class HiFTGenerator: Module {
    @ModuleInfo(key: "m_source") public var mSource: SourceModuleHnNSF
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1d]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HiFTResBlock]
    @ModuleInfo(key: "resblocks") var resblocks: [HiFTResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    @ModuleInfo(key: "f0_predictor") public var f0Predictor: F0Predictor
    public let config: HiFTConfig
    let numKernels: Int, numUpsamples: Int, upsampleScale: Int
    let windowBox: WindowBox            // boxed: a stored MLXArray on a Module would be reflected as a parameter
    var stftWindow: MLXArray { windowBox.window }

    public init(_ c: HiFTConfig) {
        config = c; numKernels = c.resblock_kernel_sizes.count; numUpsamples = c.upsample_rates.count
        upsampleScale = c.upsample_rates.reduce(1, *) * c.istft_hop_len
        self._mSource.wrappedValue = SourceModuleHnNSF(sampleRate: c.sampling_rate, upsampleScale: upsampleScale, harmonicNum: c.nb_harmonics,
                                                       sineAmp: Float(c.nsf_alpha), noiseStd: Float(c.nsf_sigma), voicedThreshold: Float(c.nsf_voiced_threshold))
        self._convPre.wrappedValue = Conv1d(inputChannels: c.in_channels, outputChannels: c.base_channels, kernelSize: 7, stride: 1, padding: 3, bias: true)
        self._ups.wrappedValue = zip(c.upsample_rates, c.upsample_kernel_sizes).enumerated().map { i, rk in
            ConvTransposed1d(inputChannels: c.base_channels / (1 << i), outputChannels: c.base_channels / (1 << (i + 1)), kernelSize: rk.1, stride: rk.0, padding: (rk.1 - rk.0) / 2, bias: true)
        }
        var downsampleRates = [1] + Array(c.upsample_rates.reversed().dropLast())
        for i in 1 ..< downsampleRates.count { downsampleRates[i] *= downsampleRates[i - 1] }   // cumprod
        let cum = Array(downsampleRates.reversed())
        var downs = [Conv1d](), sres = [HiFTResBlock]()
        for (i, (rate, ks, dil)) in zip(cum, zip(c.source_resblock_kernel_sizes, c.source_resblock_dilation_sizes)).map({ ($0, $1.0, $1.1) }).enumerated() {
            let out = c.base_channels / (1 << (i + 1))
            if rate == 1 { downs.append(Conv1d(inputChannels: c.istft_n_fft + 2, outputChannels: out, kernelSize: 1, stride: 1, padding: 0, bias: true)) }
            else { downs.append(Conv1d(inputChannels: c.istft_n_fft + 2, outputChannels: out, kernelSize: rate * 2, stride: rate, padding: rate / 2, bias: true)) }
            sres.append(HiFTResBlock(channels: out, kernelSize: ks, dilations: dil))
        }
        self._sourceDowns.wrappedValue = downs; self._sourceResblocks.wrappedValue = sres
        var rbs = [HiFTResBlock]()
        for i in 0 ..< c.upsample_rates.count {
            let ch = c.base_channels / (1 << (i + 1))
            for (ks, dil) in zip(c.resblock_kernel_sizes, c.resblock_dilation_sizes) { rbs.append(HiFTResBlock(channels: ch, kernelSize: ks, dilations: dil)) }
        }
        self._resblocks.wrappedValue = rbs
        self._convPost.wrappedValue = Conv1d(inputChannels: c.base_channels / (1 << c.upsample_rates.count), outputChannels: c.istft_n_fft + 2, kernelSize: 7, stride: 1, padding: 3, bias: true)
        self._f0Predictor.wrappedValue = F0Predictor(c.f0_predictor)
        windowBox = WindowBox(Signal.periodicHann(c.istft_n_fft))
    }

    public static func load(bundle: EditXBundle, dtype: DType = .float32) throws -> HiFTGenerator {
        let cfg = try ConfigIO.load(HiFTConfig.self, from: bundle.file("hift-config.json"))
        let g = HiFTGenerator(cfg)
        try WeightIO.apply(try WeightIO.load(bundle.file("hift.safetensors"), dtype: dtype), to: g, component: "hift")
        return g
    }

    /// `_stft_real_imag` for (B, T): reflect-pad n_fft/2, hann, rfft → (B, n_fft+2, F) [real; imag].
    func sourceSTFT(_ x: MLXArray) -> MLXArray {
        let n = config.istft_n_fft, hop = config.istft_hop_len, pad = n / 2
        let b = x.dim(0)
        var rows = [MLXArray]()
        for i in 0 ..< b {
            let padded = Signal.reflectPadded(x[i], pad)
            let frames = Signal.framed(padded, frameLength: n, hop: hop) * stftWindow      // (F, n)
            let spec = MLXFFT.rfft(frames, n: n, axis: -1)                                   // (F, n/2+1)
            rows.append(concatenated([spec.realPart(), spec.imaginaryPart()], axis: 1).transposed(1, 0))  // (n+2, F)
        }
        return stacked(rows, axis: 0)
    }

    /// `_istft`: magnitude / phase (B, n/2+1, F) → (B, samples), overlap-add with window-sum normalisation.
    func istft(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        let n = config.istft_n_fft, hop = config.istft_hop_len, pad = n / 2
        let spec = magnitude * cos(phase) + MLXArray(real: 0, imaginary: 1) * (magnitude * sin(phase))
        let frames = MLXFFT.irfft(spec.transposed(0, 2, 1), n: n, axis: -1)                // (B, F, n), real
        let f = frames.dim(1), outLen = hop * (f - 1) + n
        let windowed = frames * stftWindow                                                   // (B, F, n)
        // overlap-add: n/hop shifted lanes of hop samples each
        var y = MLXArray.zeros([frames.dim(0), outLen])
        var wsum = MLXArray.zeros([outLen])
        let w2 = stftWindow * stftWindow
        for j in 0 ..< (n / hop) {
            let lane = windowed[0..., 0..., (j * hop) ..< ((j + 1) * hop)].reshaped([frames.dim(0), f * hop])   // samples j·hop … of every frame
            let left = j * hop, right = outLen - left - f * hop
            y = y + padded(lane, widths: [IntOrPair((0, 0)), IntOrPair((left, right))])
            let wl = broadcast(w2[(j * hop) ..< ((j + 1) * hop)].reshaped([1, hop]), to: [f, hop]).reshaped([f * hop])
            wsum = wsum + padded(wl, widths: [IntOrPair((left, right))])
        }
        let safe = MLX.where(wsum .> 1e-8, wsum, MLXArray.ones(like: wsum))
        y = MLX.where((wsum .> 1e-8).reshaped([1, -1]), y / safe.reshaped([1, -1]), y)
        return outLen > 2 * pad ? y[0..., pad ..< (outLen - pad)] : y
    }

    /// `decode_without_stft`: mel (B, 80, T) + source STFT (B, n+2, F) → (B, n+2, T·480/4)
    public func decodeWithoutSTFT(_ mel: MLXArray, _ sStft: MLXArray) -> MLXArray {
        var x = applyConv(convPre, mel)
        for i in 0 ..< numUpsamples {
            x = leakyReLU(x, Float(config.lrelu_slope))
            x = applyConvT(ups[i], x)
            if i == numUpsamples - 1 { x = concatenated([x[0..., 0..., 1 ..< 2], x], axis: 2) }   // reflection pad left 1
            var si = applyConv(sourceDowns[i], sStft)
            si = sourceResblocks[i](si)
            x = x + si
            var xs: MLXArray? = nil
            for k in 0 ..< numKernels {
                let out = resblocks[i * numKernels + k](x)
                xs = xs == nil ? out : xs! + out
            }
            x = xs! / Float(numKernels)
        }
        x = leakyReLU(x, 0.01)    // the torch reference uses F.leaky_relu's default slope here
        return applyConv(convPost, x)
    }

    /// `decode`: mel (B, 80, T) + source (B, 1, samples) → waveform (B, samples)
    public func decode(_ mel: MLXArray, source: MLXArray) -> MLXArray {
        let sStft = sourceSTFT(source[0..., 0, 0...]); eval(sStft)
        let decoded = decodeWithoutSTFT(mel, sStft); eval(decoded)
        let bins = config.istft_n_fft / 2 + 1
        let magnitude = minimum(exp(decoded[0..., ..<bins, 0...]), MLXArray(Float(1e2)))
        let phase = sin(decoded[0..., bins..., 0...])
        let y = istft(magnitude: magnitude, phase: phase); eval(y)
        return clip(y, min: -Float(config.audio_limit), max: Float(config.audio_limit))
    }

    /// `inference`: mel (B, 80, T) → (waveform (B, samples), source (B, 1, samples))
    public func callAsFunction(_ mel: MLXArray) -> (MLXArray, MLXArray) {
        let f0 = f0Predictor(mel)                                                         // (B, T)
        let f0Up = repeated(f0.expandedDimensions(axis: 1), count: upsampleScale, axis: 2).transposed(0, 2, 1)   // (B, T·s, 1)
        let (merged, _, _) = mSource(f0Up)
        let source = merged.transposed(0, 2, 1)                                          // (B, 1, T·s)
        return (decode(mel, source: source), source)
    }
}
