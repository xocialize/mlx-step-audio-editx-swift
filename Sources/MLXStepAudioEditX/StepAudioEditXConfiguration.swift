// StepAudioEditXConfiguration.swift — init-time configuration for `StepAudioEditXPackage` (C9). Per-request take /
// transcript / edit / seed ride the canonical `SpeechEditRequest`.
//
// ONE weight source backs the model: the published bundle for the configured tier — `mlx-community/Step-Audio-EditX-bf16`
// (8.0 GB, every component bf16) or `mlx-community/Step-Audio-EditX-8bit` (4.4 GB, the LM 8-bit group 64, written by this
// package from the bf16 bundle) — holding the LM, both tokenizers, the flow, HiFT, the text tokenizer and the configs
// (`EditXBundle.requiredFiles`). CAM++ ships inside the package's own resources and is never downloaded. `.fp32` is the
// parity-gate precision: it upcasts the bf16 bundle at load and is not a published tier.

import Foundation
import MLXToolKit
import StepAudioEditXCore

public struct StepAudioEditXConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The bf16 bundle repo; the int8 tier swaps the `-bf16` suffix for `-8bit`.
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// `.bf16` (the shipped precision), `.int8` (the LM quantised — RTF 0.50, 6.2 GB resident), `.fp32` (parity work).
    public var quant: Quant
    /// Explicit bundle directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Run a one-second edit at load (≈ 10 s, once per process) so the first real edit runs at the steady RTF
    /// instead of paying the Metal kernel compile (AB-R-0420). Off for parity work that times the first edit.
    public var warmUp: Bool

    public init(repo: String = "mlx-community/Step-Audio-EditX-bf16", revision: String? = nil, quant: Quant = .bf16,
                modelDirectory: URL? = nil, modelsRootDirectory: URL? = nil, warmUp: Bool = true) {
        self.repo = repo; self.revision = revision; self.quant = quant
        self.modelDirectory = modelDirectory; self.modelsRootDirectory = modelsRootDirectory
        self.warmUp = warmUp
    }

    /// The repo backing the configured tier. fp32 upcasts the bf16 bundle, so it materializes the same files.
    public var tierRepo: String {
        quant == .int8 ? repo.replacingOccurrences(of: "-bf16", with: "-8bit") : repo
    }

    /// The Core dtypes for this tier.
    var dtypes: EditXDTypes {
        var d = EditXDTypes()
        switch quant {
        case .fp32: d.lm = .float32
        case .int8: d.lm = .bfloat16          // the bundle's config.json carries the quantisation; the loader reads it
        default: d.lm = .bfloat16
        }
        return d
    }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case repo, revision, quant, warmUp }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
        warmUp = try c.decodeIfPresent(Bool.self, forKey: .warmUp) ?? true
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension StepAudioEditXConfiguration: WeightSourcing {
    /// Everything `EditXPipeline.load` opens — the bundle contract.
    static let files = EditXBundle.requiredFiles
    /// Representative file for the missing-probe (the largest; a partial download is most likely to lack it).
    static let probeFile = "model.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "bundle", repo: tierRepo, revision: revision, matching: Self.files)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        if let modelDirectory, fm.fileExists(atPath: modelDirectory.appending(path: Self.probeFile).path) { return [] }
        if let dir = ModelStore(root: storeRoot).directory(for: tierRepo),
           fm.fileExists(atPath: dir.appending(path: Self.probeFile).path) { return [] }
        return weightSources
    }

    /// The configuration with a nil directory resolved to the store layout — what `load()` uses AFTER
    /// materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> StepAudioEditXConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil { cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: tierRepo) }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension StepAudioEditXConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return ["model.safetensors", "vq02.safetensors", "vq06.safetensors", "flow-model.safetensors"].map { dir.appending(path: $0) }
    }
}
