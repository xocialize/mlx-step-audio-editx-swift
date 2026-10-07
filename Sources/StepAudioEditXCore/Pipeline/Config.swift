// Config.swift — the bundle's `*-config.json` files, decoded as written by mlx-speech's converter (the resolved
// values are the oracle truths; nothing here is defaulted).

import Foundation

/// `step-audio-tokenizer-config.json` — preprocessing + both tokenizers' front ends.
public struct TokenizerConfig: Codable, Sendable {
    public let model_type: String
    public let vq02_sample_rate: Int
    public let vq06_sample_rate: Int
    public let vq02_codebook_size: Int
    public let vq06_token_rate_hz: Double
    public let vq06_n_fft: Int
    public let vq06_hop_length: Int
    public let vq06_num_mels: Int
    public let vq06_max_chunk_seconds: Double
    public let vq06_min_chunk_samples: Int
    public let trim_top_db: Double
    public let trim_frame_length: Int
    public let trim_hop_length: Int
    public let trim_keep_left_seconds: Double
    public let trim_keep_right_seconds: Double
    public let trim_output_hop_samples: Int
    public let vq02_chunk_size: [Int]
    public let encoder_chunk_look_back: Int
}

/// `vq06-config.json` — the S3 v1 semantic tokenizer (Whisper-style encoder + VQ).
public struct VQ06Config: Codable, Sendable {
    public let num_mels: Int
    public let hidden_size: Int
    public let num_heads: Int
    public let num_layers: Int
    public let max_positions: Int
    public let codebook_size: Int
    public let conv1_kernel_size: Int
    public let conv1_stride: Int
    public let conv1_padding: Int
    public let conv2_kernel_size: Int
    public let conv2_stride: Int
    public let conv2_padding: Int
    public let layer_norm_eps: Double
    public let l2_norm_eps: Double
}

/// `vq02-config.json` — the Paraformer (FunASR) streaming encoder and its kaldi front end.
public struct VQ02Config: Codable, Sendable {
    public struct Frontend: Codable, Sendable {
        public let sample_rate: Int
        public let window_type: String
        public let n_mels: Int
        public let frame_length_ms: Double
        public let frame_shift_ms: Double
        public let lfr_m: Int
        public let lfr_n: Int
        public let dither: Double
        public let snip_edges: Bool
        public let remove_dc_offset: Bool
        public let preemphasis_coefficient: Double
        public let round_to_power_of_two: Bool
        public let low_freq: Double
        public let high_freq: Double
        public let use_power: Bool
        public let use_log_fbank: Bool
        public let use_energy: Bool
    }
    public struct Encoder: Codable, Sendable {
        public let input_size: Int
        public let output_size: Int
        public let attention_heads: Int
        public let linear_units: Int
        public let num_blocks: Int
        public let normalize_before: Bool
        public let kernel_size: Int
        public let sanm_shift: Int
        public let input_layer: String
        public let tp_blocks: Int?
        public let dropout_rate: Double?
        public let positional_dropout_rate: Double?
        public let attention_dropout_rate: Double?
        public let concat_after: Bool?
        public let pos_enc_class: String?
    }
    public let model_name: String
    public let frontend: Frontend
    public let encoder: Encoder
}

/// `config.json` — the step1 LM.
public struct Step1Config: Codable, Sendable {
    public let hidden_size: Int
    public let intermediate_size: Int
    public let num_attention_heads: Int
    public let num_attention_groups: Int
    public let num_hidden_layers: Int
    public let vocab_size: Int
    public let rms_norm_eps: Double
    public let bos_token_id: Int
    public let pad_token_id: Int
    public let eos_token_id: Int
    public let max_seq_len: Int
    /// Present in a pre-quantised bundle (the published int8 tier): the LM's Linear + Embedding layers are stored as
    /// MLX affine-quantised weights (`weight` / `scales` / `biases`) — `Step1ForCausalLM.load` quantises the module
    /// structure to match before the key contract.
    public let quantization: Quantization?
    public struct Quantization: Codable, Sendable { public let bits: Int; public let group_size: Int; public let mode: String? }
    public var headDim: Int { hidden_size / num_attention_heads }
}

/// `flow-conditioner-config.json`.
public struct FlowConditionerConfig: Codable, Sendable {
    public let vocab_size: Int
    public let input_size: Int
    public let output_size: Int
    public let spk_embed_dim: Int
    public let vq02_pad_token: Int
    public let vq06_prompt_offset: Int
    public let vq06_vocoder_base: Int
    public let prompt_mel_upsample: Int
}

/// `flow-model-config.json` — upsample conformer encoder + DiT CFM.
public struct FlowModelConfig: Codable, Sendable {
    public let input_size: Int
    public let output_size: Int
    public let spk_embed_dim: Int
    public let vocab_size: Int
    public let encoder_output_size: Int
    public let pre_lookahead_len: Int
    public let num_blocks: Int
    public let num_up_blocks: Int
    public let up_stride: Int
    public let up_scale_factor: Double
    public let attention_heads: Int
    public let linear_units: Int
    public let key_bias: Bool
    public let estimator_in_channels: Int
    public let estimator_out_channels: Int
    public let estimator_hidden_size: Int
    public let estimator_depth: Int
    public let estimator_num_heads: Int
    public let estimator_head_dim: Int
    public let estimator_mlp_ratio: Double
    public let inference_cfg_rate: Double
}

/// `frontend-config.json` — the CosyVoice 24 kHz mel the flow is conditioned on.
public struct MelFrontendConfig: Codable, Sendable {
    public let num_mels: Int
    public let n_fft: Int
    public let hop_size: Int
    public let win_size: Int
    public let sampling_rate: Int
    public let fmin: Double
    public let fmax: Double
}

/// `hift-config.json`.
public struct HiFTConfig: Codable, Sendable {
    public struct F0Predictor: Codable, Sendable {
        public let num_class: Int
        public let in_channels: Int
        public let cond_channels: Int
    }
    public let in_channels: Int
    public let base_channels: Int
    public let nb_harmonics: Int
    public let sampling_rate: Int
    public let nsf_alpha: Double
    public let nsf_sigma: Double
    public let nsf_voiced_threshold: Double
    public let upsample_rates: [Int]
    public let upsample_kernel_sizes: [Int]
    public let istft_n_fft: Int
    public let istft_hop_len: Int
    public let resblock_kernel_sizes: [Int]
    public let resblock_dilation_sizes: [[Int]]
    public let source_resblock_kernel_sizes: [Int]
    public let source_resblock_dilation_sizes: [[Int]]
    public let lrelu_slope: Double
    public let audio_limit: Double
    public let f0_predictor: F0Predictor
}

/// The bundle directory and its files.
public struct EditXBundle: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }
    public func file(_ name: String) -> URL { root.appendingPathComponent(name) }
    public static let requiredFiles = [
        "campplus-config.json", "config.json", "flow-conditioner-config.json", "flow-conditioner.safetensors",
        "flow-model-config.json", "flow-model.safetensors", "frontend-config.json", "hift-config.json", "hift.safetensors",
        "model.safetensors", "step-audio-tokenizer-assets.safetensors", "step-audio-tokenizer-config.json",
        "tokenizer.json", "tokenizer_config.json", "vq02-config.json", "vq02.safetensors", "vq06-config.json", "vq06.safetensors",
    ]
    public func validate() throws {
        for f in Self.requiredFiles where !FileManager.default.fileExists(atPath: file(f).path) {
            throw StepAudioEditXError.missingFile(file(f).path)
        }
    }
}
