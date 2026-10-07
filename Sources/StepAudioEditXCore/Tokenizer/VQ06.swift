// VQ06.swift — the S3 v1 semantic tokenizer (25 Hz, 4 096 codes): Whisper-style conv stem + 6 attention blocks
// over 128-mel log-spectrograms, L2-normalised, nearest-codebook assignment. Translated 1:1 from mlx-speech's
// `step_audio_tokenizer/vq06.py`; module paths equal the bundle keys (`encoder.conv1.weight`, `encoder.blocks.N.attn.query.weight`,
// `encoder.positional_embedding`, `quantizer.codebook` …). Linear weights are stored (in, out) — `x @ W` — as upstream
// exported them, so `VQ06Linear` keeps that layout rather than remapping onto MLXNN.Linear.

import Foundation
import MLX
import MLXNN

public final class VQ06Linear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray          // (in, out)
    @ParameterInfo(key: "bias") var bias: MLXArray?

    public init(_ inputDim: Int, _ outputDim: Int, bias: Bool) {
        self._weight.wrappedValue = MLXArray.zeros([inputDim, outputDim])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputDim]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.asType(.float32).matmul(weight.asType(.float32))
        if let b = bias { y = y + b.asType(.float32) }
        return y.asType(x.dtype)
    }
}

/// Two-pass LayerNorm in fp32 (mean → centre → variance → rsqrt), as the reference computes it.
public final class VQ06LayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    public init(_ size: Int, eps: Double) {
        self._weight.wrappedValue = MLXArray.ones([size]); self._bias.wrappedValue = MLXArray.zeros([size]); self.eps = Float(eps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: -1, keepDims: true)
        let centred = xf - mean
        let variance = (centred * centred).mean(axis: -1, keepDims: true)
        return (centred * rsqrt(variance + eps) * weight + bias).asType(x.dtype)
    }
}

public final class VQ06Conv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray          // (out, k, in) — MLX layout, as in the bundle
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int, padding: Int

    public init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self._weight.wrappedValue = MLXArray.zeros([outputChannels, kernelSize, inputChannels])
        self._bias.wrappedValue = MLXArray.zeros([outputChannels]); self.stride = stride; self.padding = padding
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {   // x (N, L, C)
        let padded = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padding, padding)), IntOrPair((0, 0))])
        let y = conv1d(padded.asType(.float32), weight.asType(.float32), stride: stride, padding: 0)
        return (y + bias.asType(.float32)).asType(x.dtype)
    }
}

public final class VQ06Attention: Module {
    @ModuleInfo(key: "query") var query: VQ06Linear
    @ModuleInfo(key: "key") var key: VQ06Linear
    @ModuleInfo(key: "value") var value: VQ06Linear
    @ModuleInfo(key: "out") var out: VQ06Linear
    let hiddenSize: Int, numHeads: Int, headDim: Int, scale: Float

    public init(hiddenSize: Int, numHeads: Int) {
        self.hiddenSize = hiddenSize; self.numHeads = numHeads; self.headDim = hiddenSize / numHeads
        self.scale = Float(pow(Double(hiddenSize / numHeads), -0.25))
        self._query.wrappedValue = VQ06Linear(hiddenSize, hiddenSize, bias: true)
        self._key.wrappedValue = VQ06Linear(hiddenSize, hiddenSize, bias: false)
        self._value.wrappedValue = VQ06Linear(hiddenSize, hiddenSize, bias: true)
        self._out.wrappedValue = VQ06Linear(hiddenSize, hiddenSize, bias: true)
    }

    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = query(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = key(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = value(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        q = q * scale; k = k * scale
        var scores = q.asType(.float32).matmul(k.asType(.float32).transposed(0, 1, 3, 2))
        if let m = attentionMask { scores = scores + m.asType(.float32) }
        var weights = softmax(scores, axis: -1)
        if let m = attentionMask { weights = MLX.where(m .== 0, weights, MLXArray.zeros(like: weights)) }
        let hidden = weights.matmul(v.asType(.float32)).transposed(0, 2, 1, 3).reshaped([b, t, hiddenSize])
        return out(hidden.asType(x.dtype))
    }
}

public final class VQ06MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: VQ06Linear
    @ModuleInfo(key: "fc2") var fc2: VQ06Linear
    public init(hiddenSize: Int) {
        self._fc1.wrappedValue = VQ06Linear(hiddenSize, hiddenSize * 4, bias: true)
        self._fc2.wrappedValue = VQ06Linear(hiddenSize * 4, hiddenSize, bias: true)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }   // nn.gelu = the erf form
}

public final class VQ06Block: Module {
    @ModuleInfo(key: "attn_ln") var attnLn: VQ06LayerNorm
    @ModuleInfo(key: "attn") var attn: VQ06Attention
    @ModuleInfo(key: "mlp_ln") var mlpLn: VQ06LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VQ06MLP
    public init(hiddenSize: Int, numHeads: Int, eps: Double) {
        self._attnLn.wrappedValue = VQ06LayerNorm(hiddenSize, eps: eps)
        self._attn.wrappedValue = VQ06Attention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._mlpLn.wrappedValue = VQ06LayerNorm(hiddenSize, eps: eps)
        self._mlp.wrappedValue = VQ06MLP(hiddenSize: hiddenSize)
    }
    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var x = x + attn(attnLn(x), attentionMask: attentionMask)
        x = x + mlp(mlpLn(x))
        return x
    }
}

public final class VQ06Encoder: Module {
    @ModuleInfo(key: "conv1") var conv1: VQ06Conv1d
    @ModuleInfo(key: "conv2") var conv2: VQ06Conv1d
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [VQ06Block]
    let config: VQ06Config

    public init(_ c: VQ06Config) {
        self.config = c
        self._conv1.wrappedValue = VQ06Conv1d(c.num_mels, c.hidden_size, kernelSize: c.conv1_kernel_size, stride: c.conv1_stride, padding: c.conv1_padding)
        self._conv2.wrappedValue = VQ06Conv1d(c.hidden_size, c.hidden_size, kernelSize: c.conv2_kernel_size, stride: c.conv2_stride, padding: c.conv2_padding)
        self._positionalEmbedding.wrappedValue = MLXArray.zeros([c.max_positions, c.hidden_size])
        self._blocks.wrappedValue = (0..<c.num_layers).map { _ in VQ06Block(hiddenSize: c.hidden_size, numHeads: c.num_heads, eps: c.layer_norm_eps) }
    }

    func convOutputLength(_ n: Int) -> Int {
        let a = (n + 2 * config.conv1_padding - config.conv1_kernel_size) / config.conv1_stride + 1
        return (a + 2 * config.conv2_padding - config.conv2_kernel_size) / config.conv2_stride + 1
    }

    /// features (N, n_mels, T), lengths → (L2-normalised states (N, T', H), T')
    public func callAsFunction(_ features: MLXArray, length: Int) -> (MLXArray, Int) {
        var x = features.transposed(0, 2, 1)
        x = gelu(conv1(x)); x = gelu(conv2(x))
        let encodedLength = convOutputLength(length)
        let t = x.dim(1)
        x = x + positionalEmbedding[..<t].asType(x.dtype)
        // the padding mask — one full-length chunk ⇒ every position valid
        let positions = MLXArray(Int32(0) ..< Int32(t))
        let valid = positions .< Int32(encodedLength)
        let mask = MLX.where(valid.reshaped([1, 1, 1, t]), MLXArray.zeros([1]), MLXArray(-Float.greatestFiniteMagnitude))
        for block in blocks { x = block(x, attentionMask: mask) }
        let xf = x.asType(.float32)
        var norms = sqrt((xf * xf).sum(axis: -1, keepDims: true))
        norms = maximum(norms, MLXArray(Float(config.l2_norm_eps)))
        return (xf / norms, encodedLength)
    }
}

public final class VQ06Quantizer: Module {
    @ParameterInfo(key: "codebook") var codebook: MLXArray     // (hidden, codebook_size)
    public init(hiddenSize: Int, codebookSize: Int) { self._codebook.wrappedValue = MLXArray.zeros([hiddenSize, codebookSize]) }
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let f = features.asType(.float32), cb = codebook.asType(.float32)
        var distances = (f * f).sum(axis: -1, keepDims: true)
        distances = distances - 2 * f.matmul(cb)
        distances = distances + (cb * cb).sum(axis: 0, keepDims: true)
        return argMax(-distances, axis: -1)
    }
}

public final class VQ06Model: Module {
    @ModuleInfo(key: "encoder") var encoder: VQ06Encoder
    @ModuleInfo(key: "quantizer") var quantizer: VQ06Quantizer
    public let config: VQ06Config
    public init(_ c: VQ06Config) {
        self.config = c
        self._encoder.wrappedValue = VQ06Encoder(c)
        self._quantizer.wrappedValue = VQ06Quantizer(hiddenSize: c.hidden_size, codebookSize: c.codebook_size)
    }
    /// features (1, n_mels, T) → codes (T')
    public func encode(_ features: MLXArray, length: Int) -> [Int32] {
        let (encoded, encodedLength) = encoder(features, length: length)
        let codes = quantizer(encoded)[0, ..<encodedLength]
        eval(codes)
        return codes.asArray(Int32.self)
    }
}

/// The vq06 runtime: preprocessed 16 kHz audio → ≤ 30 s chunks → Whisper log-mel → codes.
public final class VQ06Tokenizer {
    public let model: VQ06Model
    public let config: TokenizerConfig
    let melFilters: MLXArray

    public init(model: VQ06Model, config: TokenizerConfig) {
        self.model = model; self.config = config
        self.melFilters = Signal.slaneyMelFilters(sampleRate: config.vq06_sample_rate, nFFT: config.vq06_n_fft, nMels: config.vq06_num_mels)
    }

    public static func load(bundle: EditXBundle, dtype: DType = .float32) throws -> VQ06Tokenizer {
        let cfg = try ConfigIO.load(VQ06Config.self, from: bundle.file("vq06-config.json"))
        let tcfg = try ConfigIO.load(TokenizerConfig.self, from: bundle.file("step-audio-tokenizer-config.json"))
        let model = VQ06Model(cfg)
        try WeightIO.apply(try WeightIO.load(bundle.file("vq06.safetensors"), dtype: dtype), to: model, component: "vq06")
        return VQ06Tokenizer(model: model, config: tcfg)
    }

    /// `split_vq06_audio`: ≤ max_chunk_seconds pieces, dropping a tail shorter than min_chunk_samples.
    public func chunks(_ wav16: [Float]) -> [[Float]] {
        let maxSamples = Int((config.vq06_max_chunk_seconds * Double(config.vq06_sample_rate)).rounded())
        if wav16.count <= maxSamples { return [wav16] }
        var out = [[Float]](); var start = 0
        while start < wav16.count {
            let end = min(start + maxSamples, wav16.count)
            let c = Array(wav16[start ..< end])
            if c.count >= config.vq06_min_chunk_samples { out.append(c) }
            start = end
        }
        return out
    }

    /// Codes for already-preprocessed 16 kHz audio (the oracle's `wav16_pre`).
    public func encodePreprocessed(_ wav16: [Float]) -> [Int32] {
        var tokens = [Int32]()
        for chunk in chunks(wav16) {
            let feats = Signal.whisperLogMel(MLXArray(chunk), filters: melFilters, nFFT: config.vq06_n_fft, hop: config.vq06_hop_length)
            tokens += model.encode(feats.expandedDimensions(axis: 0), length: feats.dim(1))
        }
        return tokens
    }

    public func encode(_ wav: [Float], sampleRate: Int) -> [Int32] {
        encodePreprocessed(Preprocess.run(wav, sampleRate: sampleRate, config: config))
    }
}
