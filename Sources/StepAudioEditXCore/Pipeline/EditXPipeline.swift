// EditXPipeline.swift — the whole edit, as upstream `tts.py: edit()` runs it: the take (peak-capped to 0.6) → 16 kHz
// preprocessing → vq02 + vq06 codes → the chat-templated edit prompt → step1 samples a new dual-codebook token
// sequence (temperature 0.7) → flow (prompt tokens + the take's mel + CAM++ embedding condition the DiT CFM) → HiFT.
// `clone()` is the same chain with upstream's zero-shot TTS prompt. Per-token `checkpoint` gives the engine its
// cancellation seam.

import Foundation
import MLX
import MLXRandom

public struct EditResult: Sendable {
    public let waveform: [Float]
    public let sampleRate: Int
    public let generatedTokens: [Int32]        // every sampled id
    public let audioTokens: [Int32]            // the audio ids, rebased to 0…5119
    public let promptLength: Int
    public let stoppedAtEOS: Bool
    public let timings: [String: Double]
}

public struct EditXDTypes: Sendable {
    public var lm: DType = .bfloat16
    public var flow: DType = .float32
    public var hift: DType = .float32
    public var tokenizers: DType = .float32
    /// Quantise the LM's linears at load (4 / 8 bits, group 64); nil = the `lm` dtype as is.
    public var lmQuantBits: Int? = nil
    public init() {}
    public static let parity: EditXDTypes = { var d = EditXDTypes(); d.lm = .float32; return d }()
}

public final class EditXPipeline {
    public let bundle: EditXBundle
    public let tokenizer: EditXTokenizer
    public let vq02: VQ02Tokenizer
    public let vq06: VQ06Tokenizer
    public let lm: Step1ForCausalLM
    public let mel: MelFrontend
    public let speaker: SpeakerEncoder
    public let flow: FlowModel
    public let hift: HiFTGenerator
    public let conditionerConfig: FlowConditionerConfig
    public static let totalSequenceLimit = 8192

    init(bundle: EditXBundle, tokenizer: EditXTokenizer, vq02: VQ02Tokenizer, vq06: VQ06Tokenizer, lm: Step1ForCausalLM, mel: MelFrontend,
         speaker: SpeakerEncoder, flow: FlowModel, hift: HiFTGenerator, conditionerConfig: FlowConditionerConfig) {
        self.bundle = bundle; self.tokenizer = tokenizer; self.vq02 = vq02; self.vq06 = vq06; self.lm = lm; self.mel = mel
        self.speaker = speaker; self.flow = flow; self.hift = hift; self.conditionerConfig = conditionerConfig
    }

    /// Weights load on the CPU stream (WeightIO); the forwards run wherever the caller's default device points.
    public static func load(bundle: EditXBundle, dtypes: EditXDTypes = EditXDTypes()) throws -> EditXPipeline {
        try bundle.validate()
        let tokenizer = try EditXTokenizer.load(bundle: bundle)
        let vq02 = try VQ02Tokenizer.load(bundle: bundle, dtype: dtypes.tokenizers)
        let vq06 = try VQ06Tokenizer.load(bundle: bundle, dtype: dtypes.tokenizers)
        let lm = try Step1ForCausalLM.load(bundle: bundle, dtype: dtypes.lm, quantBits: dtypes.lmQuantBits)
        let mel = try MelFrontend.load(bundle: bundle)
        let speaker = try SpeakerEncoder.load(dtype: .float32)
        let flow = try FlowModel.load(bundle: bundle, dtype: dtypes.flow)
        let hift = try HiFTGenerator.load(bundle: bundle, dtype: dtypes.hift)
        let ccfg = try ConfigIO.load(FlowConditionerConfig.self, from: bundle.file("flow-conditioner-config.json"))
        return EditXPipeline(bundle: bundle, tokenizer: tokenizer, vq02: vq02, vq06: vq06, lm: lm, mel: mel, speaker: speaker, flow: flow, hift: hift, conditionerConfig: ccfg)
    }

    /// `_prepare_prompt_audio` / upstream `preprocess_prompt_wav`: mono, peak capped to 0.6.
    public static func capPeak(_ wav: [Float], target: Float = 0.6) -> [Float] {
        guard let peak = wav.map({ abs($0) }).max(), peak > target else { return wav }
        let s = target / peak
        return wav.map { $0 * s }
    }

    /// The take's dual-codebook codes and the prompt-token packing the LM and the flow both consume.
    public func tokenize(_ capped: [Float], sampleRate: Int) -> (vq02: [Int32], vq06: [Int32], promptTokens: [Int32]) {
        let pre = Preprocess.run(capped, sampleRate: sampleRate, config: vq02.tokenizerConfig)
        let c02 = vq02.encodePreprocessed(pre), c06 = vq06.encodePreprocessed(pre)
        return (c02, c06, EditXTokenizer.packPromptTokens(vq02: c02, vq06: c06, vq06Offset: tokenizer.vq06Offset))
    }

    func synthesize(promptIds: [Int32], promptTokens: [Int32], capped: [Float], sampleRate: Int, seed: UInt64?, temperature: Float,
                    maxNewTokens: Int?, flowSteps: Int, timings: inout [String: Double], checkpoint: (() throws -> Void)?,
                    onToken: ((Int, Int) -> Void)? = nil) throws -> EditResult {
        if let seed { MLXRandom.seed(seed) }
        let limit = min(maxNewTokens ?? Int.max, max(Self.totalSequenceLimit - promptIds.count, 1), lm.config.max_seq_len - promptIds.count)
        var t0 = Date()
        let generated = try lm.generate(prompt: promptIds, maxNewTokens: limit, temperature: temperature, eosTokenId: Int32(lm.config.eos_token_id),
                                        checkpoint: checkpoint, onToken: onToken.map { f in { f($0, limit) } })
        timings["lm"] = Date().timeIntervalSince(t0)
        let audio = generated.filter { $0 >= tokenizer.audioTokenBase }.map { $0 - tokenizer.audioTokenBase }
        guard !audio.isEmpty else { throw StepAudioEditXError.invalidInput("the LM produced no audio tokens (\(generated.count) ids)") }
        try checkpoint?()
        t0 = Date()
        let feat = mel.features(capped, sampleRate: sampleRate)
        let emb = try speaker.embed(capped, sampleRate: sampleRate)
        let inputs = FlowConditioning.prepare(config: conditionerConfig, tokens: audio, promptTokens: promptTokens, promptFeat: feat, speakerEmbedding: emb)
        let melOut = flow(inputs, nTimesteps: flowSteps); eval(melOut)
        timings["flow"] = Date().timeIntervalSince(t0)
        try checkpoint?()
        t0 = Date()
        let (wav, _) = hift(melOut); eval(wav)
        timings["hift"] = Date().timeIntervalSince(t0)
        return EditResult(waveform: wav[0].asArray(Float.self), sampleRate: hift.config.sampling_rate, generatedTokens: generated, audioTokens: audio,
                          promptLength: promptIds.count, stoppedAtEOS: generated.count < limit, timings: timings)
    }

    /// `edit`: re-deliver `wav` (its transcript `text`) per `edit`.
    public func edit(_ wav: [Float], sampleRate: Int, text: String, edit: SpeechEdit, seed: UInt64? = 42, temperature: Float = 0.7,
                     maxNewTokens: Int? = nil, flowSteps: Int = 10, checkpoint: (() throws -> Void)? = nil,
                     onToken: ((Int, Int) -> Void)? = nil) throws -> EditResult {
        try checkpoint?()
        var timings = [String: Double](); var t0 = Date()
        let capped = Self.capPeak(wav)
        let (_, _, promptTokens) = tokenize(capped, sampleRate: sampleRate)
        timings["tokenize"] = Date().timeIntervalSince(t0); t0 = Date()
        let promptIds = tokenizer.editPromptIds(instruction: try EditPrompts.instruction(promptText: text, edit: edit),
                                                audioTokenString: EditXTokenizer.audioTokenString(promptTokens))
        timings["prompt"] = Date().timeIntervalSince(t0)
        return try synthesize(promptIds: promptIds, promptTokens: promptTokens, capped: capped, sampleRate: sampleRate, seed: seed, temperature: temperature,
                              maxNewTokens: maxNewTokens, flowSteps: flowSteps, timings: &timings, checkpoint: checkpoint, onToken: onToken)
    }

    /// `clone`: zero-shot TTS of `targetText` in the voice of `wav` (its transcript `promptText`).
    public func clone(_ wav: [Float], sampleRate: Int, promptText: String, targetText: String, speaker: String = "debug", seed: UInt64? = 42,
                      temperature: Float = 0.7, maxNewTokens: Int? = nil, flowSteps: Int = 10, checkpoint: (() throws -> Void)? = nil) throws -> EditResult {
        try checkpoint?()
        var timings = [String: Double](); let t0 = Date()
        let capped = Self.capPeak(wav)
        let (_, _, promptTokens) = tokenize(capped, sampleRate: sampleRate)
        let promptIds = tokenizer.clonePromptIds(speaker: speaker, promptText: promptText, promptWavTokens: EditXTokenizer.audioTokenString(promptTokens), targetText: targetText)
        timings["tokenize+prompt"] = Date().timeIntervalSince(t0)
        return try synthesize(promptIds: promptIds, promptTokens: promptTokens, capped: capped, sampleRate: sampleRate, seed: seed, temperature: temperature,
                              maxNewTokens: maxNewTokens, flowSteps: flowSteps, timings: &timings, checkpoint: checkpoint)
    }
}
