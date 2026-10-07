// Step1.swift — StepFun's step1 decoder (the audio LLM): 32 pre-norm blocks of grouped-query attention (48 heads over
// 4 KV groups, head_dim 64, each group's heads contiguous) with a sqrt-ALiBi causal bias — −slope·√(i−j), no RoPE —
// and SwiGLU MLPs, RMSNorm with fp32 accumulation, untied lm_head. Translated 1:1 from mlx-speech's
// `step_audio_editx/model.py` (which reproduces StepFun's `modeling_step1.py` SDPA fallback path); module paths equal
// the bundle keys (`model.layers.N.self_attn.q_proj.weight`, `model.norm.weight`, `lm_head.weight` …).

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// The ALiBi slope table — a constant kept OUT of Module reflection (it is not a parameter).
public final class AlibiSlopes: @unchecked Sendable {
    public let slopes: MLXArray      // (num_heads,) float32
    public init(numHeads: Int) {
        // `_alibi_slopes`: 2^(−8/n) powers for the largest power-of-two head count n, then 2^(−4/n) odd powers for the rest
        let n = 1 << Int(floor(log2(Double(numHeads))))
        let m0 = pow(2.0, -8.0 / Double(n))
        var s = (1 ... n).map { Float(pow(m0, Double($0))) }
        if n < numHeads {
            let m1 = pow(2.0, -4.0 / Double(n))
            s += stride(from: 1, to: 1 + 2 * (numHeads - n), by: 2).map { Float(pow(m1, Double($0))) }
        }
        slopes = MLXArray(s)
    }

    /// `build_sqrt_alibi_bias`: (heads, queryLen, keyLen) fp32, −slope·√(distance) where the key is at or before the
    /// query (query positions start at `offset`), −∞ elsewhere.
    public func bias(queryLen: Int, keyLen: Int, offset: Int) -> MLXArray {
        let q = MLXArray((0 ..< queryLen).map { Float(offset + $0) }).reshaped([queryLen, 1])
        let k = MLXArray((0 ..< keyLen).map { Float($0) }).reshaped([1, keyLen])
        let distance = q - k
        let valid = distance .>= 0
        let safe = MLX.where(valid, distance, MLXArray.zeros(like: distance))
        var b = -sqrt(safe).expandedDimensions(axis: 0) * slopes.reshaped([-1, 1, 1])
        b = MLX.where(valid.expandedDimensions(axis: 0), b, MLXArray(-Float.infinity))
        return b
    }
}

public final class Step1RMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    public init(_ size: Int, eps: Double) { self._weight.wrappedValue = MLXArray.ones([size]); self.eps = Float(eps) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: weight.asType(x.dtype), eps: eps) }
}

public final class Step1MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    public init(_ hidden: Int, _ intermediate: Int) {
        self._gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._up.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// Per-layer KV cache in pre-repeat (grouped) form: (B, groups, T, headDim).
public final class Step1LayerCache {
    var keys: MLXArray? = nil
    var values: MLXArray? = nil
    public var length: Int { keys?.dim(2) ?? 0 }
    public init() {}
    func append(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let ck = keys, let cv = values { keys = concatenated([ck, k], axis: 2); values = concatenated([cv, v], axis: 2) }
        else { keys = k; values = v }
        return (keys!, values!)
    }
}

public final class Step1Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let hidden: Int, numHeads: Int, numGroups: Int, headDim: Int, kvRepeat: Int, scale: Float
    let alibi: AlibiSlopes

    public init(_ c: Step1Config, alibi: AlibiSlopes) {
        hidden = c.hidden_size; numHeads = c.num_attention_heads; numGroups = c.num_attention_groups
        headDim = c.headDim; kvRepeat = numHeads / numGroups; scale = Float(1.0 / Double(headDim).squareRoot()); self.alibi = alibi
        self._qProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._kProj.wrappedValue = Linear(hidden, numGroups * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, numGroups * headDim, bias: false)
        self._oProj.wrappedValue = Linear(hidden, hidden, bias: false)
    }

    public func callAsFunction(_ x: MLXArray, cache: Step1LayerCache?) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)        // (B, H, T, D)
        var k = kProj(x).reshaped([b, t, numGroups, headDim]).transposed(0, 2, 1, 3)       // (B, G, T, D)
        var v = vProj(x).reshaped([b, t, numGroups, headDim]).transposed(0, 2, 1, 3)
        var offset = 0
        if let cache { offset = cache.length; (k, v) = cache.append(k, v) }
        let keyLen = k.dim(2)
        let bias = alibi.bias(queryLen: t, keyLen: keyLen, offset: offset)                   // (H, T, Tk)
        // heads grouped contiguously per KV group: head h ↔ group h / kvRepeat
        let qg = q.reshaped([b, numGroups, kvRepeat, t, headDim])
        let kg = k.expandedDimensions(axis: 2)                                                // (B, G, 1, Tk, D)
        var scores = qg.matmul(kg.transposed(0, 1, 2, 4, 3)).reshaped([b, numHeads, t, keyLen]).asType(.float32)
        scores = scores * scale + bias.expandedDimensions(axis: 0)
        let probs = softmax(scores, axis: -1)
        let pg = probs.asType(v.dtype).reshaped([b, numGroups, kvRepeat, t, keyLen])
        let vg = v.expandedDimensions(axis: 2)                                                // (B, G, 1, Tk, D)
        let h = pg.matmul(vg).reshaped([b, numHeads, t, headDim]).transposed(0, 2, 1, 3).reshaped([b, t, hidden])
        return oProj(h.asType(x.dtype))
    }
}

public final class Step1Block: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: Step1RMSNorm
    @ModuleInfo(key: "self_attn") var attn: Step1Attention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: Step1RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Step1MLP
    public init(_ c: Step1Config, alibi: AlibiSlopes) {
        self._inputNorm.wrappedValue = Step1RMSNorm(c.hidden_size, eps: c.rms_norm_eps)
        self._attn.wrappedValue = Step1Attention(c, alibi: alibi)
        self._postNorm.wrappedValue = Step1RMSNorm(c.hidden_size, eps: c.rms_norm_eps)
        self._mlp.wrappedValue = Step1MLP(c.hidden_size, c.intermediate_size)
    }
    public func callAsFunction(_ x: MLXArray, cache: Step1LayerCache?) -> MLXArray {
        var h = x + attn(inputNorm(x), cache: cache)
        h = h + mlp(postNorm(h))
        return h
    }
}

public final class Step1Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Step1Block]
    @ModuleInfo(key: "norm") var norm: Step1RMSNorm
    public init(_ c: Step1Config, alibi: AlibiSlopes) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        self._layers.wrappedValue = (0 ..< c.num_hidden_layers).map { _ in Step1Block(c, alibi: alibi) }
        self._norm.wrappedValue = Step1RMSNorm(c.hidden_size, eps: c.rms_norm_eps)
    }
    /// Final hidden states (B, T, H); `trace` receives each layer's output (for the per-layer gate).
    public func callAsFunction(_ ids: MLXArray, caches: [Step1LayerCache]?, trace: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = embedTokens(ids)
        trace?(0, h)
        for (i, layer) in layers.enumerated() { h = layer(h, cache: caches?[i]); trace?(i + 1, h) }
        return norm(h)
    }
}

public final class Step1ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Step1Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    public let config: Step1Config
    public let alibi: AlibiSlopes

    public init(_ c: Step1Config) {
        config = c; alibi = AlibiSlopes(numHeads: c.num_attention_heads)
        self._model.wrappedValue = Step1Model(c, alibi: alibi)
        self._lmHead.wrappedValue = Linear(c.hidden_size, c.vocab_size, bias: false)
    }

    /// `quantBits` (4 / 8, group 64) quantises every Linear after the fp load — on the CPU stream, like the load itself;
    /// the quantised forward must then run on the GPU stream (a CPU-pinned quantised matmul grinds for hours).
    public static func load(bundle: EditXBundle, dtype: DType = .bfloat16, quantBits: Int? = nil, groupSize: Int = 64) throws -> Step1ForCausalLM {
        let cfg = try ConfigIO.load(Step1Config.self, from: bundle.file("config.json"))
        let lm = Step1ForCausalLM(cfg)
        try WeightIO.apply(try WeightIO.load(bundle.file("model.safetensors"), dtype: dtype), to: lm, component: "step1")
        if let bits = quantBits {
            try Device.withDefaultDevice(.cpu) {
                quantize(model: lm, groupSize: groupSize, bits: bits)
                eval(lm.parameters())
            }
        }
        return lm
    }

    public func makeCaches() -> [Step1LayerCache] { (0 ..< config.num_hidden_layers).map { _ in Step1LayerCache() } }

    /// Logits (B, T, V) for `ids`, updating `caches` when given.
    public func callAsFunction(_ ids: MLXArray, caches: [Step1LayerCache]? = nil, trace: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        lmHead(model(ids, caches: caches, trace: trace))
    }

    /// `_generate_audio_tokens`: prefill, then sample at `temperature` until EOS or `maxNewTokens`; returns every
    /// generated id (audio ids are ≥ `audioTokenBase`; the caller strips the rest). `checkpoint` runs per token.
    public func generate(prompt: [Int32], maxNewTokens: Int, temperature: Float, eosTokenId: Int32,
                         checkpoint: (() throws -> Void)? = nil) rethrows -> [Int32] {
        let caches = makeCaches()
        var logits = self(MLXArray(prompt).reshaped([1, -1]), caches: caches)[0..., -1, 0...]
        eval(logits)
        var out = [Int32]()
        for _ in 0 ..< maxNewTokens {
            try checkpoint?()
            let next: Int32
            if temperature <= 0 { next = argMax(logits, axis: -1).item(Int32.self) }
            else { next = MLXRandom.categorical(logits.asType(.float32) / temperature, axis: -1).item(Int32.self) }
            if next == eosTokenId { break }
            out.append(next)
            logits = self(MLXArray([next]).reshaped([1, 1]), caches: caches)[0..., -1, 0...]
            eval(logits)
        }
        return out
    }
}
