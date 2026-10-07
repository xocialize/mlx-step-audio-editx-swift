// FlowModel.swift — the token → mel flow (`CausalMaskedDiffWithXvec`): dual-codebook embedding → upsample conformer
// encoder (6 blocks at the token rate, a ×2 upsampler, 4 blocks at the mel rate; relative-position attention with
// the ESPnet rel-shift) → `encoder_proj` to 80 → a DiT conditional flow-matching decoder (16 adaLN blocks with causal
// conv branches, 10 Euler steps on a cosine schedule, classifier-free guidance 0.7) started from a FIXED noise buffer.
// Translated 1:1 from mlx-speech's `step_audio_editx/flow_model.py` + the inference path; module paths equal the
// `flow-model.safetensors` keys.

import Foundation
import MLX
import MLXNN

func mish(_ x: MLXArray) -> MLXArray { x * tanh(softplus(x)) }

/// Two-pass LayerNorm in fp32, optionally affine-free (`StepAudioLayerNorm`).
public final class FlowLayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let eps: Float
    public init(_ dim: Int, eps: Float, affine: Bool = true) {
        self._weight.wrappedValue = affine ? MLXArray.ones([dim]) : nil
        self._bias.wrappedValue = affine ? MLXArray.zeros([dim]) : nil
        self.eps = eps
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: -1, keepDims: true)
        let c = xf - mean
        var y = c * rsqrt((c * c).mean(axis: -1, keepDims: true) + eps)
        if let w = weight { y = y * w }
        if let b = bias { y = y + b }
        return y.asType(x.dtype)
    }
}

public final class LinearNoSubsampling: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: FlowLayerNorm
    public init(_ inputSize: Int, _ outputSize: Int) {
        self._linear.wrappedValue = Linear(inputSize, outputSize, bias: true); self._norm.wrappedValue = FlowLayerNorm(outputSize, eps: 1e-5)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { norm(linear(x)) }
}

/// ESPnet relative positional encoding: x·√d and the (1, 2T−1, d) table slice.
public final class RelPositionalEncoding: @unchecked Sendable {
    let dModel: Int, xscale: Float
    var pe: MLXArray, maxLen: Int
    public init(dModel: Int, maxLen: Int = 5000) {
        self.dModel = dModel; self.xscale = Float(Double(dModel).squareRoot()); self.maxLen = maxLen; self.pe = Self.build(maxLen, dModel)
    }
    static func build(_ maxLen: Int, _ d: Int) -> MLXArray {
        // float32 as the reference builds it
        var pos = [Float](repeating: 0, count: maxLen * d), neg = [Float](repeating: 0, count: maxLen * d)
        for p in 0 ..< maxLen {
            for i in stride(from: 0, to: d, by: 2) {
                let div = expf(Float(i) * -(logf(10000) / Float(d)))
                let a = Float(p) * div
                pos[p * d + i] = sinf(a); pos[p * d + i + 1] = cosf(a)
                neg[p * d + i] = sinf(-a); neg[p * d + i + 1] = cosf(-a)
            }
        }
        let positive = MLXArray(pos, [maxLen, d])[.stride(by: -1)]          // flipped
        let negative = MLXArray(neg, [maxLen, d])[1...]
        return concatenated([positive, negative], axis: 0).expandedDimensions(axis: 0)   // (1, 2·maxLen−1, d)
    }
    func ensure(_ size: Int) { if pe.dim(1) < size * 2 - 1 { maxLen = max(size, maxLen * 2); pe = Self.build(maxLen, dModel) } }
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let size = x.dim(1); ensure(size)
        let centre = pe.dim(1) / 2
        return (x * xscale, pe[0..., (centre - size + 1) ..< (centre + size)])
    }
}

public final class RelPositionAttention: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray
    let h: Int, dK: Int
    public init(nHead: Int, nFeat: Int, keyBias: Bool) {
        h = nHead; dK = nFeat / nHead
        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: true); self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: true); self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: true)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead]); self._posBiasV.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead])
    }
    func relShift(_ x: MLXArray) -> MLXArray {           // (B, H, T, 2T−1) → (B, H, T, T)
        let (b, hh, t, p) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padded = concatenated([MLXArray.zeros([b, hh, t, 1]), x], axis: -1).reshaped([b, hh, p + 1, t])
        let shifted = padded[0..., 0..., 1...].reshaped([b, hh, t, p])
        return shifted[0..., 0..., 0..., ..<(p / 2 + 1)]
    }
    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, posEmb: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let q = linearQ(x).reshaped([b, -1, h, dK])                                   // (B, T, H, D) — the reference's q_t
        let k = linearK(x).reshaped([b, -1, h, dK]).transposed(0, 2, 1, 3)
        let v = linearV(x).reshaped([b, -1, h, dK]).transposed(0, 2, 1, 3)
        let p = linearPos(posEmb).reshaped([posEmb.dim(0), -1, h, dK]).transposed(0, 2, 1, 3)
        let qU = (q + posBiasU).transposed(0, 2, 1, 3), qV = (q + posBiasV).transposed(0, 2, 1, 3)
        let ac = qU.matmul(k.transposed(0, 1, 3, 2))
        let bd = relShift(qV.matmul(p.transposed(0, 1, 3, 2)))
        var scores = (ac + bd) / Float(Double(dK).squareRoot())
        var attn: MLXArray
        if let mask {                                                                 // (B, 1, T) true = valid
            let invalid = .!mask.expandedDimensions(axis: 2)                          // (B, 1, 1, T)
            scores = MLX.where(invalid, MLXArray(Float(-1e9)), scores)
            attn = MLX.where(invalid, MLXArray.zeros(like: scores), softmax(scores, axis: -1))
        } else { attn = softmax(scores, axis: -1) }
        let y = attn.matmul(v).transposed(0, 2, 1, 3).reshaped([b, t, h * dK])
        return linearOut(y)
    }
}

public final class FlowFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    public init(_ idim: Int, _ hidden: Int) { self._linear1.wrappedValue = Linear(idim, hidden, bias: true); self._linear2.wrappedValue = Linear(hidden, idim, bias: true) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class ConformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FlowFeedForward
    @ModuleInfo(key: "norm_ff") var normFF: FlowLayerNorm
    @ModuleInfo(key: "norm_mha") var normMHA: FlowLayerNorm
    public init(size: Int, heads: Int, linearUnits: Int, keyBias: Bool) {
        self._selfAttn.wrappedValue = RelPositionAttention(nHead: heads, nFeat: size, keyBias: keyBias)
        self._feedForward.wrappedValue = FlowFeedForward(size, linearUnits)
        self._normFF.wrappedValue = FlowLayerNorm(size, eps: 1e-12); self._normMHA.wrappedValue = FlowLayerNorm(size, eps: 1e-12)
    }
    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, posEmb: MLXArray) -> MLXArray {
        var y = x + selfAttn(normMHA(x), mask: mask, posEmb: posEmb)
        y = y + feedForward(normFF(y))
        return y
    }
}

public final class Upsample1D: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let stride: Int, scaleFactor: Int
    public init(channels: Int, outChannels: Int, stride: Int, scaleFactor: Double) {
        self.stride = stride; self.scaleFactor = Int(scaleFactor)
        self._conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: stride * 2 + 1, bias: true)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {     // (B, C, T) → (B, C, 2T)
        let rep = repeated(x, count: scaleFactor, axis: 2)
        let pad = padded(rep, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((stride * 2, 0))])
        return applyConv(conv, pad)
    }
}

public final class PreLookaheadLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    let lookahead: Int
    public init(channels: Int, preLookaheadLen: Int) {
        lookahead = preLookaheadLen
        self._conv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: preLookaheadLen + 1, bias: true)
        self._conv2.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, bias: true)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {     // (B, T, C)
        var y = x.transposed(0, 2, 1)
        y = padded(y, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, lookahead))])
        y = leakyReLU(applyConv(conv1, y), 0.01)
        y = padded(y, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((2, 0))])
        y = applyConv(conv2, y).transposed(0, 2, 1)
        return y + x
    }
}

public final class UpsampleConformerEncoder: Module {
    @ModuleInfo(key: "embed") var embed: LinearNoSubsampling
    @ModuleInfo(key: "after_norm") var afterNorm: FlowLayerNorm
    @ModuleInfo(key: "pre_lookahead_layer") var preLookahead: PreLookaheadLayer
    @ModuleInfo(key: "encoders") var encoders: [ConformerLayer]
    @ModuleInfo(key: "up_layer") var upLayer: Upsample1D
    @ModuleInfo(key: "up_embed") var upEmbed: LinearNoSubsampling
    @ModuleInfo(key: "up_encoders") var upEncoders: [ConformerLayer]
    let embedPos: RelPositionalEncoding, upEmbedPos: RelPositionalEncoding

    public init(_ c: FlowModelConfig) {
        self._embed.wrappedValue = LinearNoSubsampling(c.input_size, c.encoder_output_size)
        self._afterNorm.wrappedValue = FlowLayerNorm(c.encoder_output_size, eps: 1e-5)
        self._preLookahead.wrappedValue = PreLookaheadLayer(channels: c.encoder_output_size, preLookaheadLen: c.pre_lookahead_len)
        self._encoders.wrappedValue = (0 ..< c.num_blocks).map { _ in ConformerLayer(size: c.encoder_output_size, heads: c.attention_heads, linearUnits: c.linear_units, keyBias: c.key_bias) }
        self._upLayer.wrappedValue = Upsample1D(channels: c.encoder_output_size, outChannels: c.encoder_output_size, stride: c.up_stride, scaleFactor: c.up_scale_factor)
        self._upEmbed.wrappedValue = LinearNoSubsampling(c.input_size, c.encoder_output_size)
        self._upEncoders.wrappedValue = (0 ..< c.num_up_blocks).map { _ in ConformerLayer(size: c.encoder_output_size, heads: c.attention_heads, linearUnits: c.linear_units, keyBias: c.key_bias) }
        embedPos = RelPositionalEncoding(dModel: c.encoder_output_size); upEmbedPos = RelPositionalEncoding(dModel: c.encoder_output_size)
    }

    /// (B, T, 512) → (B, 2T, 512); one utterance ⇒ every position valid.
    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = embed(xs)
        var pos: MLXArray
        (x, pos) = embedPos(x)
        x = preLookahead(x)
        for layer in encoders { x = layer(x, mask: nil, posEmb: pos) }
        x = upLayer(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        x = upEmbed(x)
        (x, pos) = upEmbedPos(x)
        for layer in upEncoders { x = layer(x, mask: nil, posEmb: pos) }
        return afterNorm(x)
    }
}

// MARK: - DiT estimator

public final class DiTMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    public init(_ inF: Int, _ hidden: Int, _ outF: Int) { self._fc1.wrappedValue = Linear(inF, hidden, bias: true); self._fc2.wrappedValue = Linear(hidden, outF, bias: true) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

public final class DiTAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: FlowLayerNorm
    @ModuleInfo(key: "k_norm") var kNorm: FlowLayerNorm
    @ModuleInfo(key: "proj") var proj: Linear
    let numHeads: Int, headDim: Int, scale: Float
    public init(dim: Int, numHeads: Int, headDim: Int) {
        self.numHeads = numHeads; self.headDim = headDim; scale = Float(pow(Double(headDim), -0.5))
        let inner = numHeads * headDim
        self._toQ.wrappedValue = Linear(dim, inner, bias: true); self._toK.wrappedValue = Linear(dim, inner, bias: true); self._toV.wrappedValue = Linear(dim, inner, bias: true)
        self._qNorm.wrappedValue = FlowLayerNorm(headDim, eps: 1e-5); self._kNorm.wrappedValue = FlowLayerNorm(headDim, eps: 1e-5)
        self._proj.wrappedValue = Linear(inner, dim, bias: true)
    }
    func heads(_ x: MLXArray) -> MLXArray { x.reshaped([x.dim(0), x.dim(1), numHeads, -1]).transposed(0, 2, 1, 3) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = qNorm(heads(toQ(x))), k = kNorm(heads(toK(x))), v = heads(toV(x))
        let attn = softmax(q.matmul(k.transposed(0, 1, 3, 2)) * scale, axis: -1)
        return proj(attn.matmul(v).transposed(0, 2, 1, 3).reshaped([b, t, numHeads * headDim]))
    }
}

public final class TimestepEmbedder: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let freqSize: Int
    public init(hidden: Int, frequencyEmbeddingSize: Int = 256) {
        freqSize = frequencyEmbeddingSize
        self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, hidden, bias: true); self._linear2.wrappedValue = Linear(hidden, hidden, bias: true)
    }
    public func callAsFunction(_ t: MLXArray) -> MLXArray {      // t (B,)
        let half = freqSize / 2
        let freqs = exp(-logf(10000) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = (t * 1000).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let emb = concatenated([cos(args), sin(args)], axis: -1)
        return linear2(silu(linear1(emb)))
    }
}

public final class CausalConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let kernelSize: Int
    public init(_ inC: Int, _ outC: Int, kernelSize: Int) { self.kernelSize = kernelSize; self._conv.wrappedValue = Conv1d(inputChannels: inC, outputChannels: outC, kernelSize: kernelSize, bias: true) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {     // (B, C, T)
        applyConv(conv, padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((kernelSize - 1, 0))]))
    }
}

public final class CausalConvBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: CausalConv1d
    @ModuleInfo(key: "norm") var norm: FlowLayerNorm
    @ModuleInfo(key: "conv2") var conv2: CausalConv1d
    public init(_ inC: Int, _ outC: Int, kernelSize: Int = 3) {
        self._conv1.wrappedValue = CausalConv1d(inC, outC, kernelSize: kernelSize); self._norm.wrappedValue = FlowLayerNorm(outC, eps: 1e-5); self._conv2.wrappedValue = CausalConv1d(outC, outC, kernelSize: kernelSize)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {     // (B, T, C)
        var y = conv1(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        y = mish(norm(y))
        return conv2(y.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

func modulate(_ x: MLXArray, _ shift: MLXArray, _ scale: MLXArray) -> MLXArray { x * (1 + scale) + shift }

public final class DiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: FlowLayerNorm
    @ModuleInfo(key: "attn") var attn: DiTAttention
    @ModuleInfo(key: "norm2") var norm2: FlowLayerNorm
    @ModuleInfo(key: "mlp") var mlp: DiTMLP
    @ModuleInfo(key: "norm3") var norm3: FlowLayerNorm
    @ModuleInfo(key: "conv") var conv: CausalConvBlock
    @ModuleInfo(key: "adaLN_linear") var adaLN: Linear
    public init(hidden: Int, numHeads: Int, headDim: Int, mlpRatio: Double) {
        self._norm1.wrappedValue = FlowLayerNorm(hidden, eps: 1e-6, affine: false)
        self._attn.wrappedValue = DiTAttention(dim: hidden, numHeads: numHeads, headDim: headDim)
        self._norm2.wrappedValue = FlowLayerNorm(hidden, eps: 1e-6, affine: false)
        self._mlp.wrappedValue = DiTMLP(hidden, Int(Double(hidden) * mlpRatio), hidden)
        self._norm3.wrappedValue = FlowLayerNorm(hidden, eps: 1e-6, affine: false)
        self._conv.wrappedValue = CausalConvBlock(hidden, hidden)
        self._adaLN.wrappedValue = Linear(hidden, hidden * 9, bias: true)
    }
    public func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let m = split(adaLN(silu(c)), parts: 9, axis: -1)
        var y = x + m[2] * attn(modulate(norm1(x), m[0], m[1]))
        y = y + m[8] * conv(modulate(norm3(y), m[6], m[7]))
        y = y + m[5] * mlp(modulate(norm2(y), m[3], m[4]))
        return y
    }
}

public final class DiTFinalLayer: Module {
    @ModuleInfo(key: "adaLN_linear") var adaLN: Linear
    @ModuleInfo(key: "norm_final") var normFinal: FlowLayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    public init(hidden: Int, outChannels: Int) {
        self._adaLN.wrappedValue = Linear(hidden, 2 * hidden, bias: true); self._normFinal.wrappedValue = FlowLayerNorm(hidden, eps: 1e-6, affine: false); self._linear.wrappedValue = Linear(hidden, outChannels, bias: true)
    }
    public func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let m = split(adaLN(silu(c)), parts: 2, axis: -1)
        return linear(modulate(normFinal(x), m[0], m[1]))
    }
}

public final class DiT: Module {
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [DiTBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: DiTFinalLayer
    public let outChannels: Int
    public init(_ c: FlowModelConfig) {
        outChannels = c.estimator_out_channels
        self._tEmbedder.wrappedValue = TimestepEmbedder(hidden: c.estimator_hidden_size)
        self._inProj.wrappedValue = Linear(c.estimator_in_channels, c.estimator_hidden_size, bias: true)
        self._blocks.wrappedValue = (0 ..< c.estimator_depth).map { _ in DiTBlock(hidden: c.estimator_hidden_size, numHeads: c.estimator_num_heads, headDim: c.estimator_head_dim, mlpRatio: c.estimator_mlp_ratio) }
        self._finalLayer.wrappedValue = DiTFinalLayer(hidden: c.estimator_hidden_size, outChannels: c.estimator_out_channels)
    }
    /// x, mu, cond (B, 80, T); spks (B, 80); t (B,) → (B, 80, T)
    public func callAsFunction(x: MLXArray, mu: MLXArray, t: MLXArray, spks: MLXArray, cond: MLXArray) -> MLXArray {
        let timestep = tEmbedder(t).reshaped([t.dim(0), 1, -1])
        let s = broadcast(spks.expandedDimensions(axis: 2), to: [spks.dim(0), spks.dim(1), x.dim(2)])
        let merged = concatenated([x, mu, s, cond], axis: 1).transposed(0, 2, 1)   // (B, T, 320)
        var h = inProj(merged)
        for block in blocks { h = block(h, timestep) }
        return finalLayer(h, timestep).transposed(0, 2, 1)
    }
}

public final class ConditionalCFM: Module {
    @ModuleInfo(key: "estimator") var estimator: DiT
    let cfgRate: Float
    /// The fixed noise buffer: the reference draws `randn(1, 80, 30000)` once (seed 0 in mlx-speech, torch's global RNG
    /// upstream); the gate injects the oracle's, production seeds MLXRandom once per load.
    public var randNoise: MLXArray
    public init(_ c: FlowModelConfig) {
        cfgRate = Float(c.inference_cfg_rate); self._estimator.wrappedValue = DiT(c)
        randNoise = MLXRandom.normal([1, c.estimator_out_channels, 50 * 600], key: MLXRandom.key(0))
    }
    public func callAsFunction(mu: MLXArray, spks: MLXArray, cond: MLXArray, nTimesteps: Int = 10, temperature: Float = 1) -> MLXArray {
        let t = mu.dim(2)
        var x = randNoise[0..., 0..., ..<t].asType(mu.dtype) * temperature
        var tSpan = MLXArray((0 ... nTimesteps).map { Float($0) / Float(nTimesteps) })
        tSpan = 1 - cos(tSpan * 0.5 * Float.pi)
        let steps = tSpan.asArray(Float.self)
        let muIn = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
        let spksIn = concatenated([spks, MLXArray.zeros(like: spks)], axis: 0)
        let condIn = concatenated([cond, MLXArray.zeros(like: cond)], axis: 0)
        let b = x.dim(0)
        var tCur = steps[0]
        for step in 1 ..< steps.count {
            let dt = steps[step] - tCur
            let xIn = concatenated([x, x], axis: 0)
            let tIn = MLXArray([tCur, tCur])
            let dphi = estimator(x: xIn, mu: muIn, t: tIn, spks: spksIn, cond: condIn)
            let guided = (1 + cfgRate) * dphi[..<b] - cfgRate * dphi[b...]
            x = x + dt * guided
            tCur = tCur + dt
            eval(x)
        }
        return x
    }
}

/// `CausalMaskedDiffWithXvec` — tokens + prompt mel + speaker → generated mel (B, 80, T_gen).
public final class FlowModel: Module {
    @ModuleInfo(key: "input_embedding") var inputEmbedding: DualCodebookEmbedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkAffine: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "decoder") public var decoder: ConditionalCFM
    public let config: FlowModelConfig

    public init(_ c: FlowModelConfig) {
        config = c
        self._inputEmbedding.wrappedValue = DualCodebookEmbedding(vocabSize: c.vocab_size, inputSize: c.input_size)
        self._spkAffine.wrappedValue = Linear(c.spk_embed_dim, c.output_size, bias: true)
        self._encoder.wrappedValue = UpsampleConformerEncoder(c)
        self._encoderProj.wrappedValue = Linear(c.encoder_output_size, c.output_size, bias: true)
        self._decoder.wrappedValue = ConditionalCFM(c)
    }

    public static func load(bundle: EditXBundle, dtype: DType = .float32) throws -> FlowModel {
        let cfg = try ConfigIO.load(FlowModelConfig.self, from: bundle.file("flow-model-config.json"))
        let m = FlowModel(cfg)
        try WeightIO.apply(try WeightIO.load(bundle.file("flow-model.safetensors"), dtype: dtype), to: m, component: "flow",
                           derived: { $0.hasPrefix("decoder.randNoise") })
        return m
    }

    /// `inference`: (B, 80, T_gen) mel for the generated tokens.
    public func callAsFunction(_ inputs: FlowInputs, nTimesteps: Int = 10) -> MLXArray {
        let tokens = inputs.promptTokenDual + inputs.tokenDual
        let ids = MLXArray(tokens.flatMap { $0 }, [1, tokens.count, 2])
        let embedding = spkAffine(inputs.speakerEmbedding)                                 // (1, 80)
        let tokenEmbed = inputEmbedding(maximum(ids, 0))                                    // (1, T, 512)
        var h = encoder(tokenEmbed)
        h = encoderProj(h)                                                                  // (1, 2T, 80)
        let melLen1 = inputs.promptFeatAligned.dim(1), melLen2 = h.dim(1) - melLen1
        let conds = concatenated([inputs.promptFeatAligned, MLXArray.zeros([1, melLen2, h.dim(2)])], axis: 1).transposed(0, 2, 1)
        let feat = decoder(mu: h.transposed(0, 2, 1), spks: embedding, cond: conds, nTimesteps: nTimesteps)
        return feat[0..., 0..., melLen1...]
    }
}

/// `StepAudioDualCodebookEmbedding`: one table, the two codebooks' embeddings concatenated.
public final class DualCodebookEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    public init(vocabSize: Int, inputSize: Int) { self._embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize / 2) }
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {        // (B, T, 2) → (B, T, inputSize)
        concatenated([embedding(tokens[.ellipsis, 0]), embedding(tokens[.ellipsis, 1])], axis: -1)
    }
}
