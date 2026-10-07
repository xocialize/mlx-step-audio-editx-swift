// FlowConditioner.swift — what the flow is fed: the LM's mixed audio tokens regrouped into the dual-codebook layout
// CosyVoice's `_reshape` produces ([02, 02, 06, 06, 06] → rows of (vq02, vq06) with a pad for the third row), the
// prompt's 24 kHz mel aligned to 2 frames per token (nearest-neighbour), the CAM++ embedding L2-normalised.
// Translated from mlx-speech's `step_audio_editx/flow.py`.

import Foundation
import MLX

public struct FlowInputs {
    public let tokenDual: [[Int32]]          // (T, 2) generated
    public let promptTokenDual: [[Int32]]    // (Tp, 2)
    public let promptFeatAligned: MLXArray   // (1, 2·Tp, 80)
    public let speakerEmbedding: MLXArray    // (1, 192) normalised
}

public enum FlowConditioning {
    /// `reshape_mixed_audio_tokens`: mixed ids (vq02 < 1024, vq06 ≥ 1024) → (T, 2) with vq02 padded by 1024 every
    /// third row and vq06 shifted into the vocoder vocabulary (−1024 + 1025).
    public static func reshapeMixed(_ mixed: [Int32], vq02Pad: Int32, vq06PromptOffset: Int32, vq06VocoderBase: Int32) -> [[Int32]] {
        var m = mixed
        let rem = m.count % 5
        if rem != 0 { m += Array([0, 0, 0, 1024, 1024, 1024].suffix(5 - rem)) }
        var vq02 = [Int32](), vq06 = [Int32]()
        for g in 0 ..< (m.count / 5) {
            let s = g * 5
            vq02 += [m[s], m[s + 1], vq02Pad]
            vq06 += [m[s + 2], m[s + 3], m[s + 4]]
        }
        return zip(vq02, vq06).map { [$0, $1 - vq06PromptOffset + vq06VocoderBase] }
    }

    /// `interpolate_prompt_features`: nearest-neighbour time resampling of (1, T, C) to `target` frames.
    public static func interpolate(_ feat: MLXArray, target: Int) -> MLXArray {
        let n = feat.dim(1)
        if n == target { return feat }
        let idx = (0 ..< target).map { Int32(min(max(Int(floor(Float($0) * Float(n) / Float(target))), 0), n - 1)) }
        return feat[0..., MLXArray(idx), 0...]
    }

    public static func normalize(_ embedding: MLXArray) -> MLXArray {
        let e = embedding.asType(.float32).reshaped([1, -1])
        let norm = maximum(sqrt((e * e).sum(axis: 1, keepDims: true)), MLXArray(Float(1e-12)))
        return e / norm
    }

    /// `prepare_nonstream_inputs`.
    public static func prepare(config c: FlowConditionerConfig, tokens: [Int32], promptTokens: [Int32],
                               promptFeat: MLXArray, speakerEmbedding: MLXArray) -> FlowInputs {
        let t = reshapeMixed(tokens, vq02Pad: Int32(c.vq02_pad_token), vq06PromptOffset: Int32(c.vq06_prompt_offset), vq06VocoderBase: Int32(c.vq06_vocoder_base))
        let p = reshapeMixed(promptTokens, vq02Pad: Int32(c.vq02_pad_token), vq06PromptOffset: Int32(c.vq06_prompt_offset), vq06VocoderBase: Int32(c.vq06_vocoder_base))
        let feat = promptFeat.ndim == 2 ? promptFeat.expandedDimensions(axis: 0) : promptFeat
        return FlowInputs(tokenDual: t, promptTokenDual: p,
                          promptFeatAligned: interpolate(feat, target: p.count * c.prompt_mel_upsample),
                          speakerEmbedding: normalize(speakerEmbedding))
    }
}
