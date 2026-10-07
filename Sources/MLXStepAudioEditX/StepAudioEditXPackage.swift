import Foundation
import MLX
import MLXNN
import MLXToolKit
import StepAudioEditXCore

/// Step-Audio-EditX (StepFun, Apache-2.0) on the canonical `speechEdit` surface (contract 1.50.0, AB-A-0137): an
/// existing take + what it says → the same words in the same voice, re-delivered — an emotion, a speaking style,
/// inserted paralinguistics, denoised, or with its silences trimmed. 24 kHz mono `.wav` out.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `StepAudioEditXConfiguration`, materializes the declared
/// bundle into its store, pages weights in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// What the surface declares (`SpeechEditControls`): the five operations, upstream's 15 emotion labels (incl. `remove`),
/// 33 style labels, 10 paralinguistic tags, and a 90 s input ceiling (the LM's 8 192-token window must hold the prompt
/// — ≈ 42 audio tokens per second — and a regeneration of the same length). The engine refuses an undeclared label or
/// operation before admission; this package re-checks for a direct caller and judges the paralinguistic tags, which the
/// engine does not parse.
///
/// The output is REGENERATED (AR sampling, temperature 0.7 by default): nothing guarantees the words survive, so a
/// consumer that must keep them compares an ASR pass of the result against `SpeechEditResponse.transcript`
/// (AB-L-0119), and every edit changes the take's length, so a cue is re-fitted afterwards. Speed and content
/// replacement are deliberately absent (E19 V0).
///
/// `metaData` keys (package-specific, C5):
/// - `temperature` (double, 0.7): the LM's sampling temperature.
/// - `flowSteps` (int, 10): Euler steps of the mel flow.
/// `seed` rides the request; absent, a random one is drawn. Sampling is reproducible within Swift (MLXRandom), not
/// across to the PyTorch reference's stream.
@InferenceActor
public final class StepAudioEditXPackage: ModelPackage {
    public typealias Configuration = StepAudioEditXConfiguration

    /// Split footprints, MEASURED as phys_footprint THROUGH THE ENGINE (`editx-gates --validate`: MLXServeEngine register →
    /// prepare (incl. the one-second warm-up edit, cache cleared after) → three edits → evict, M5 Max, 2026-10-07,
    /// PORTING-SPEC S7, v0.2.1), pool-inclusive as the fleet declares (AB-L-0113 / AB-L-0155): bf16 phys +10.19 GB after
    /// prepare, 11.39 GB at the highest reading over 5–8.5 s edits → 1.19 GB activation; int8 +6.94 GB, 8.37 GB → 1.43 GB.
    /// The warm-up leaves ≈ 0.8 GB resident that 0.2.0 counted as activation; the peaks did not move. Declared with headroom.
    nonisolated static let bf16ResidentBytes: UInt64 = 10_300_000_000
    nonisolated static let int8ResidentBytes: UInt64 = 7_000_000_000
    nonisolated static let peakActivationBytes: UInt64 = 2_000_000_000

    /// Upstream's vocabularies (`config/edit_config.py`; the paralinguistic tags from the reference UI).
    public nonisolated static let emotionLabels = [
        "happy", "angry", "sad", "humour", "confusion", "disgusted", "empathy", "embarrass", "fear", "surprised",
        "excited", "depressed", "coldness", "admiration", "remove",
    ]
    public nonisolated static let styleLabels = [
        "serious", "arrogant", "child", "older", "girl", "pure", "sister", "sweet", "ethereal", "whisper", "gentle",
        "recite", "generous", "act_coy", "warm", "shy", "comfort", "authority", "chat", "radio", "soulful", "story",
        "vivid", "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly", "remove", "exaggerated",
    ]
    public nonisolated static let paralinguisticTags = [
        "[Breathing]", "[Laughter]", "[Surprise-oh]", "[Confirmation-en]", "[Uhm]", "[Surprise-ah]", "[Surprise-wa]",
        "[Sigh]", "[Question-ei]", "[Dissatisfaction-hnn]",
    ]
    /// The LM window (8 192 tokens) holds ≈ 130 prompt tokens + 41.7 audio tokens per second of take, and the
    /// regeneration is about as long as the take: 90 s leaves headroom.
    public nonisolated static let maxInputSeconds: Double = 90

    public nonisolated static var controls: SpeechEditControls {
        SpeechEditControls(operations: [.emotion, .style, .paralinguistic, .denoise, .trimSilence],
                           emotionLabels: emotionLabels, styleLabels: styleLabels, paralinguisticTags: paralinguisticTags,
                           maxInputSeconds: maxInputSeconds)
    }

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7 (AB-R-0409): the LM, flow, HiFT and S3 tokenizer are StepFun's (Apache-2.0); CAM++ is Alibaba's (Apache-2.0);
            // the Paraformer vq02 encoder is FunASR's model (`funasrModel`, allowlisted). C8: port code MIT; the lifted
            // CAM++ port MIT (xocialize); the translation reference mlx-speech MIT.
            license: LicenseDeclaration(weightLicense: .apache2, additionalWeightLicenses: [.funasrModel], portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "stepfun-ai/Step-Audio-EditX", revision: "5fe2f8a05c2353301ad47d3c1747b262115da138", tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes, peakActivationBytes: peakActivationBytes),
                    QuantFootprint(quant: .int8, residentBytes: int8ResidentBytes, peakActivationBytes: peakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                SpeechEditContract.descriptor(
                    name: "step-audio-editx",
                    summary: "Step-Audio-EditX (StepFun) speech editor (.wav, 24 kHz mono; en / zh, some ja / ko / dialects): "
                        + "an existing take + its transcript comes back saying the same words in the same voice, re-delivered — "
                        + "emotion (15 labels incl. remove), speaking style (33), inserted non-verbal sounds ([Laughter], [Sigh], "
                        + "[Breathing] … inline in a target transcript), denoised, or silence-trimmed. The output is regenerated "
                        + "(check the words with ASR) and changes length (re-fit afterwards). ≤ 90 s per take. metaData: "
                        + "temperature (0.7) / flowSteps (10); seed on the request.",
                    controls: Self.controls
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: EditXPipeline?

    /// C14/INF seam: the module graphs this package holds — CAM++ carries BatchNorms, which `WeightIO.apply` switches to
    /// inference at load (the single choke point); the rest are norm-only transformers and convs.
    var inferenceModeGraphs: [String: MLXNN.Module?] {
        ["lm": pipeline?.lm, "flow": pipeline?.flow, "hift": pipeline?.hift, "campplus": pipeline?.speaker.model,
         "vq02": pipeline?.vq02.model, "vq06": pipeline?.vq06.model]
    }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard pipeline == nil else { return }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw StepAudioEditXPackageError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let dir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw StepAudioEditXPackageError.missingWeights("unresolved bundle directory (no store root)")
        }
        pipeline = try EditXPipeline.load(bundle: EditXBundle(root: dir), dtypes: configuration.dtypes)
        // A one-second edit compiles every Metal kernel the real edits use (≈ 10 s, once per process — AB-R-0420),
        // so the first request runs at the steady RTF. Its outcome is discarded; only cancellation propagates.
        if configuration.warmUp {
            do {
                try pipeline?.warmUp(checkpoint: { try Task.checkCancellation() })
                MLX.Memory.clearCache()   // the compiled kernels stay; the warm-up's pool must not read as resident weights
            } catch is CancellationError {
                pipeline = nil
                throw CancellationError()
            } catch {
                // A failed warm-up is not a failed load: the first real edit simply pays the compile.
            }
        }
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every generated token and every stage
        // boundary (the Core checks the closure); the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .speechEdit, let req = request as? SpeechEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req)
        let (samples, rate) = try EditXAudioIO.decode(req.audio)
        let seconds = Double(samples.count) / Double(max(rate, 1))
        guard seconds >= 0.5 else { throw PackageError.unsupportedRequestFeature("take shorter than 0.5 s") }
        guard seconds <= Self.maxInputSeconds else {
            throw PackageError.unsupportedRequestFeature(
                String(format: "take is %.1f s — longer than the %.0f s this surface declares (the LM window); split it", seconds, Self.maxInputSeconds))
        }
        let meta = req.metaData
        let temperature = Float(meta.doubleValue("temperature") ?? 0.7)
        let flowSteps = meta.intValue("flowSteps") ?? 10
        guard temperature > 0, flowSteps > 0 else { throw PackageError.unsupportedRequestFeature("temperature and flowSteps must be positive") }
        let seed = req.seed ?? UInt64.random(in: 0 ... UInt64.max)
        try Task.checkCancellation()
        let result = try pipeline.edit(samples, sampleRate: rate, text: plan.transcript, edit: plan.edit, seed: seed, temperature: temperature,
                                       flowSteps: flowSteps, checkpoint: { try Task.checkCancellation() },
                                       onToken: { RunProgress.report(.generate, step: $0, totalSteps: $1) })
        try Task.checkCancellation()
        guard result.stoppedAtEOS else {
            throw PackageError.unsupportedRequestFeature(
                "the take did not finish within the LM window (\(result.audioTokens.count) audio tokens) — a shorter take, or split it")
        }
        let wav = EditXAudioIO.encodeWAV16(samples: result.waveform, sampleRate: result.sampleRate)
        return SpeechEditResponse(audio: Audio(format: .wav, data: wav, sampleRate: result.sampleRate, channels: 1),
                                  transcript: plan.outputTranscript)
    }

    /// What the request asks for, judged against the declaration (the engine admits only declared edits; a direct
    /// caller meets the same rules here). `.trimSilence` is the core's `.vad`.
    public struct Plan: Equatable {
        public let edit: SpeechEdit
        public let transcript: String
        /// The text the output should carry: the request's transcript, or a `.paralinguistic` target with its tags.
        public let outputTranscript: String
    }

    public nonisolated static func plan(_ req: SpeechEditRequest) throws -> Plan {
        let transcript = req.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else { throw PackageError.unsupportedRequestFeature("empty transcript — the model conditions on the take's words") }
        switch req.edit {
        case .emotion(let label):
            guard emotionLabels.contains(label) else {
                throw PackageError.unsupportedRequestFeature("emotion '\(label)' — declared labels: \(emotionLabels.joined(separator: ", "))")
            }
            return Plan(edit: .emotion(label), transcript: transcript, outputTranscript: transcript)
        case .style(let label):
            guard styleLabels.contains(label) else {
                throw PackageError.unsupportedRequestFeature("style '\(label)' — declared labels: \(styleLabels.joined(separator: ", "))")
            }
            return Plan(edit: .style(label), transcript: transcript, outputTranscript: transcript)
        case .paralinguistic(let target):
            let t = target.trimmingCharacters(in: .whitespacesAndNewlines)
            let tags = Self.tags(in: t)
            guard !tags.isEmpty else {
                throw PackageError.unsupportedRequestFeature("paralinguistic target carries no tag — write one of \(paralinguisticTags.joined(separator: " ")) inline")
            }
            if let bad = tags.first(where: { !paralinguisticTags.contains($0) }) {
                throw PackageError.unsupportedRequestFeature("paralinguistic tag \(bad) — declared tags: \(paralinguisticTags.joined(separator: " "))")
            }
            return Plan(edit: .paralinguistic(targetText: t), transcript: transcript, outputTranscript: t)
        case .denoise:
            return Plan(edit: .denoise, transcript: transcript, outputTranscript: transcript)
        case .trimSilence:
            return Plan(edit: .vad, transcript: transcript, outputTranscript: transcript)
        @unknown default:
            throw PackageError.unsupportedRequestFeature("speech edit kind not known to this package")
        }
    }

    /// Every `[…]` in a target transcript.
    nonisolated static func tags(in text: String) -> [String] {
        var out = [String](); var i = text.startIndex
        while let open = text[i...].firstIndex(of: "["), let close = text[open...].firstIndex(of: "]") {
            out.append(String(text[open ... close])); i = text.index(after: close)
        }
        return out
    }
}

extension StepAudioEditXPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(StepAudioEditXPackage.self) }
}

/// Wrapper-level errors (weight resolution). Runtime request errors use `PackageError`.
public enum StepAudioEditXPackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    public var description: String {
        switch self {
        case .missingWeights(let why): return "Step-Audio-EditX weights unavailable: \(why)"
        }
    }
}

extension MetaData {
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}
