// SANMEncoder.swift — the Paraformer streaming encoder (`SANMEncoderChunkOpt`): 50 pre-norm layers of SAN-M attention
// (self-attention + an FSMN memory block over the values) and ReLU feed-forwards, run chunk by chunk with per-layer
// K/V caches (look-back 4 chunks, right context 5 frames), 5 frames of input overlap, and a sinusoidal position
// encoding that counts from the utterance start. Translated 1:1 from mlx-speech's `vq02.py`; module paths equal the
// bundle keys (`encoder.encoders0.0.self_attn.linear_q_k_v.weight`, `encoder.encoders.N.feed_forward.w_1.weight`,
// `encoder.after_norm.weight` …). The k-means lives in `VQ02Tokenizer`.

import Foundation
import MLX
import MLXNN

/// Two-pass LayerNorm, eps 1e-12 (the FunASR default).
public final class SANMLayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float
    public init(_ size: Int, eps: Float = 1e-12) {
        self._weight.wrappedValue = MLXArray.ones([size]); self._bias.wrappedValue = MLXArray.zeros([size]); self.eps = eps
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centred = x - mean
        let variance = (centred * centred).mean(axis: -1, keepDims: true)
        return centred * rsqrt(variance + eps) * weight + bias
    }
}

/// Depthwise conv over time with symmetric padding ((k−1)/2 left, the rest right) — the FSMN memory block.
public final class FSMNBlock: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray          // (channels, k, 1)
    let left: Int, right: Int
    public init(channels: Int, kernelSize: Int) {
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
        left = (kernelSize - 1) / 2; right = kernelSize - 1 - left
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((left, right)), IntOrPair((0, 0))])
        return conv1d(p, weight, stride: 1, padding: 0, groups: x.dim(-1))
    }
}

public final class SANMFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear
    public init(_ inputDim: Int, _ hiddenDim: Int) {
        self._w1.wrappedValue = Linear(inputDim, hiddenDim); self._w2.wrappedValue = Linear(hiddenDim, inputDim)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { w2(maximum(w1(x), 0)) }
}

public struct SANMCache { var k: MLXArray; var v: MLXArray }

public final class SANMAttention: Module {
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_q_k_v") var linearQKV: Linear
    @ModuleInfo(key: "fsmn_block") var fsmn: FSMNBlock
    let h: Int, dK: Int

    public init(nHead: Int, inFeat: Int, nFeat: Int, kernelSize: Int) {
        h = nHead; dK = nFeat / nHead
        self._linearOut.wrappedValue = Linear(nFeat, nFeat)
        self._linearQKV.wrappedValue = Linear(inFeat, nFeat * 3)
        self._fsmn.wrappedValue = FSMNBlock(channels: nFeat, kernelSize: kernelSize)
    }

    /// `forward_chunk`: returns the layer output and the next K/V cache.
    public func forwardChunk(_ x: MLXArray, cache: SANMCache?, chunkSize: [Int], lookBack: Int) -> (MLXArray, SANMCache?) {
        let (b, t) = (x.dim(0), x.dim(1))
        let qkv = linearQKV(x)
        let split = h * dK
        let q = qkv[0..., 0..., ..<split], k = qkv[0..., 0..., split ..< (2 * split)], v = qkv[0..., 0..., (2 * split)...]
        var qH = q.reshaped([b, t, h, dK]).transposed(0, 2, 1, 3)
        var kH = k.reshaped([b, t, h, dK]).transposed(0, 2, 1, 3)
        var vH = v.reshaped([b, t, h, dK]).transposed(0, 2, 1, 3)
        var nextCache: SANMCache? = nil
        if lookBack > 0 || lookBack == -1 {
            let rc = chunkSize[2]
            let kStride = rc > 0 ? kH[0..., 0..., ..<(t - rc)] : kH, vStride = rc > 0 ? vH[0..., 0..., ..<(t - rc)] : vH
            if let c = cache {
                kH = concatenated([c.k, kH], axis: 2); vH = concatenated([c.v, vH], axis: 2)
                var nk = concatenated([c.k, kStride], axis: 2), nv = concatenated([c.v, vStride], axis: 2)
                if lookBack != -1 {
                    let keep = lookBack * chunkSize[1]
                    let n = nk.dim(2)
                    if n > keep { nk = nk[0..., 0..., (n - keep)...]; nv = nv[0..., 0..., (n - keep)...] }
                }
                nextCache = SANMCache(k: nk, v: nv)
            } else {
                nextCache = SANMCache(k: kStride, v: vStride)
            }
        }
        let fsmnMemory = fsmn(v) + v
        qH = qH * Float(pow(Double(dK), -0.5))
        let scores = qH.matmul(kH.transposed(0, 1, 3, 2))
        let attention = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attended = attention.matmul(vH).transposed(0, 2, 1, 3).reshaped([b, t, h * dK])
        return (linearOut(attended) + fsmnMemory, nextCache)
    }
}

public final class SANMEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SANMAttention
    @ModuleInfo(key: "feed_forward") var feedForward: SANMFeedForward
    @ModuleInfo(key: "norm1") var norm1: SANMLayerNorm
    @ModuleInfo(key: "norm2") var norm2: SANMLayerNorm
    let inSize: Int, size: Int, normalizeBefore: Bool

    public init(inSize: Int, size: Int, heads: Int, linearUnits: Int, kernelSize: Int, normalizeBefore: Bool) {
        self.inSize = inSize; self.size = size; self.normalizeBefore = normalizeBefore
        self._selfAttn.wrappedValue = SANMAttention(nHead: heads, inFeat: inSize, nFeat: size, kernelSize: kernelSize)
        self._feedForward.wrappedValue = SANMFeedForward(size, linearUnits)
        self._norm1.wrappedValue = SANMLayerNorm(inSize); self._norm2.wrappedValue = SANMLayerNorm(size)
    }

    public func forwardChunk(_ input: MLXArray, cache: SANMCache?, chunkSize: [Int], lookBack: Int) -> (MLXArray, SANMCache?) {
        var x = input
        var residual = x
        if normalizeBefore { x = norm1(x) }
        let (attnOut, next) = selfAttn.forwardChunk(x, cache: cache, chunkSize: chunkSize, lookBack: lookBack)
        x = inSize == size ? residual + attnOut : attnOut
        if !normalizeBefore { x = norm1(x) }
        residual = x
        if normalizeBefore { x = norm2(x) }
        x = residual + feedForward(x)
        if !normalizeBefore { x = norm2(x) }
        return (x, next)
    }
}

/// The encoder's per-utterance streaming state.
public final class SANMEncoderState {
    var startIdx = 0
    let chunkSize: [Int], lookBack: Int
    var feats: MLXArray                         // (1, chunk[0]+chunk[2], inputSize) overlap carried into the next chunk
    var layerCaches: [SANMCache?]
    init(chunkSize: [Int], lookBack: Int, inputSize: Int, layers: Int) {
        self.chunkSize = chunkSize; self.lookBack = lookBack
        feats = MLXArray.zeros([1, chunkSize[0] + chunkSize[2], inputSize]); layerCaches = [SANMCache?](repeating: nil, count: layers)
    }
}

public final class SANMEncoder: Module {
    @ModuleInfo(key: "encoders0") var encoders0: [SANMEncoderLayer]
    @ModuleInfo(key: "encoders") var encoders: [SANMEncoderLayer]
    @ModuleInfo(key: "after_norm") var afterNorm: SANMLayerNorm
    public let config: VQ02Config
    let outputScale: Float

    public init(_ c: VQ02Config) {
        config = c; let e = c.encoder
        outputScale = Float(Double(e.output_size).squareRoot())
        self._encoders0.wrappedValue = [SANMEncoderLayer(inSize: e.input_size, size: e.output_size, heads: e.attention_heads, linearUnits: e.linear_units, kernelSize: e.kernel_size, normalizeBefore: e.normalize_before)]
        self._encoders.wrappedValue = (0 ..< (e.num_blocks - 1)).map { _ in SANMEncoderLayer(inSize: e.output_size, size: e.output_size, heads: e.attention_heads, linearUnits: e.linear_units, kernelSize: e.kernel_size, normalizeBefore: e.normalize_before) }
        self._afterNorm.wrappedValue = SANMLayerNorm(e.output_size)
    }

    public func makeState(chunkSize: [Int], lookBack: Int) -> SANMEncoderState {
        SANMEncoderState(chunkSize: chunkSize, lookBack: lookBack, inputSize: config.encoder.input_size, layers: encoders0.count + encoders.count)
    }

    /// `StreamSinusoidalPositionEncoder`: positions count from the utterance start (float32, as the reference).
    static func positionEncoding(start: Int, timesteps: Int, dim: Int) -> MLXArray {
        let half = dim / 2
        let logIncrement = Float(log(10000.0)) / Float(half - 1)
        let inv = (0..<half).map { exp(Float($0) * -logIncrement) }
        var enc = [Float](repeating: 0, count: timesteps * dim)
        for t in 0..<timesteps {
            let pos = Float(start + t + 1)
            for i in 0..<half { let s = pos * inv[i]; enc[t * dim + i] = sin(s); enc[t * dim + half + i] = cos(s) }
        }
        return MLXArray(enc, [1, timesteps, dim])
    }

    /// `forward_chunk`: one chunk's encoder input (1, T, 560) → (1, overlap + T, 512); the caller keeps the last T rows.
    public func forwardChunk(_ input: MLXArray, state: SANMEncoderState) -> MLXArray {
        var x = input * outputScale
        let t = x.dim(1)
        x = x + Self.positionEncoding(start: state.startIdx, timesteps: t, dim: x.dim(2)).asType(x.dtype)
        state.startIdx += t
        let overlap = concatenated([state.feats, x], axis: 1)
        let keep = state.chunkSize[0] + state.chunkSize[2]
        state.feats = overlap[0..., (overlap.dim(1) - keep)...]
        x = overlap
        var idx = 0
        for layer in encoders0 { let (y, c) = layer.forwardChunk(x, cache: state.layerCaches[idx], chunkSize: state.chunkSize, lookBack: state.lookBack); x = y; state.layerCaches[idx] = c; idx += 1 }
        for layer in encoders { let (y, c) = layer.forwardChunk(x, cache: state.layerCaches[idx], chunkSize: state.chunkSize, lookBack: state.lookBack); x = y; state.layerCaches[idx] = c; idx += 1 }
        return afterNorm(x)
    }
}

public final class VQ02Model: Module {
    @ModuleInfo(key: "encoder") var encoder: SANMEncoder
    public init(_ c: VQ02Config) { self._encoder.wrappedValue = SANMEncoder(c) }
}

/// The vq02 runtime: preprocessed 16 kHz audio → 240 ms chunks → front end → encoder → k-means codes (16.7 Hz).
public final class VQ02Tokenizer {
    public let model: VQ02Model
    public let config: VQ02Config
    public let tokenizerConfig: TokenizerConfig
    public let frontend: VQ02Frontend
    let codebook: MLXArray                    // (1024, 512)
    let codebookNorm: MLXArray                // (1, 1024)

    public init(model: VQ02Model, config: VQ02Config, tokenizerConfig: TokenizerConfig, cmvn: MLXArray, codebook: MLXArray) {
        self.model = model; self.config = config; self.tokenizerConfig = tokenizerConfig
        self.frontend = VQ02Frontend(config: config, cmvn: cmvn)
        self.codebook = codebook.asType(.float32)
        self.codebookNorm = (self.codebook * self.codebook).sum(axis: 1).reshaped([1, -1])
    }

    public static func load(bundle: EditXBundle, dtype: DType = .float32) throws -> VQ02Tokenizer {
        let cfg = try ConfigIO.load(VQ02Config.self, from: bundle.file("vq02-config.json"))
        let tcfg = try ConfigIO.load(TokenizerConfig.self, from: bundle.file("step-audio-tokenizer-config.json"))
        let model = VQ02Model(cfg)
        try WeightIO.apply(try WeightIO.load(bundle.file("vq02.safetensors"), dtype: dtype), to: model, component: "vq02")
        let assets = try WeightIO.load(bundle.file("step-audio-tokenizer-assets.safetensors"), dtype: .float32)
        guard let cmvn = assets["cmvn"], let codebook = assets["linguistic_codebook"] else {
            throw StepAudioEditXError.keyContract(component: "tokenizer-assets", missing: ["cmvn", "linguistic_codebook"], unused: [])
        }
        return VQ02Tokenizer(model: model, config: cfg, tokenizerConfig: tcfg, cmvn: cmvn, codebook: codebook)
    }

    /// `extract_encoder_features` for a whole utterance (is_final): (T, 512), plus the per-chunk encoder inputs for gating.
    public func encoderFeatures(_ wav16: [Float], collectInputs: Bool = false) -> (features: MLXArray, inputs: [MLXArray]) {
        let chunkSize = tokenizerConfig.vq02_chunk_size, lookBack = tokenizerConfig.encoder_chunk_look_back
        let stride = chunkSize[1] * 960
        let numChunks = wav16.count / stride + 1
        frontend.reset()
        let state = model.encoder.makeState(chunkSize: chunkSize, lookBack: lookBack)
        var outs = [MLXArray](), inputs = [MLXArray]()
        for ci in 0..<numChunks {
            let chunkFinal = ci == numChunks - 1
            let lo = ci * stride, hi = min((ci + 1) * stride, wav16.count)
            let chunk = lo < hi ? Array(wav16[lo ..< hi]) : []
            if chunkFinal && chunk.count < 480 { break }
            let rows = frontend(chunk, isFinal: chunkFinal)
            if rows.isEmpty { if chunkFinal { break } else { continue } }
            let speech = MLXArray(rows.flatMap { $0 }, [1, rows.count, rows[0].count])
            if collectInputs { inputs.append(speech) }
            let out = model.encoder.forwardChunk(speech, state: state)
            let y = out[0..., (out.dim(1) - rows.count)...]
            eval(y)
            outs.append(y)
        }
        if outs.isEmpty { return (MLXArray.zeros([0, config.encoder.output_size]), inputs) }
        return (concatenated(outs, axis: 1)[0], inputs)
    }

    /// `cluster_linguistic_features`: nearest centroid by squared L2.
    public func cluster(_ features: MLXArray) -> [Int32] {
        let f = features.asType(.float32)
        let d = (f * f).sum(axis: 1, keepDims: true) + codebookNorm - 2 * f.matmul(codebook.transposed(1, 0))
        let codes = argMin(d, axis: 1).asType(.int32)
        eval(codes)
        return codes.asArray(Int32.self)
    }

    public func encodePreprocessed(_ wav16: [Float]) -> [Int32] { cluster(encoderFeatures(wav16).features) }

    public func encode(_ wav: [Float], sampleRate: Int) -> [Int32] {
        encodePreprocessed(Preprocess.run(wav, sampleRate: sampleRate, config: tokenizerConfig))
    }
}
