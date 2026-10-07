// swift-tools-version: 6.2
// mlx-step-audio-editx-swift — StepFun's Step-Audio-EditX (Apache-2.0) ported to Swift-MLX for MLXEngine: the fleet's
// speech EDITOR. An existing take (audio + its transcript) comes back re-delivered — emotion, speaking style,
// inserted paralinguistics (laughter, sighs, breaths), denoised, or silence-trimmed — in the same voice.
// Evaluation and the port decision: mlxengine-audio/Docs/ENHANCEMENTS.md E19 (AB-D-0113); port task AB-T-0202.
//
// Pipeline (all in StepAudioEditXCore, translated 1:1 from the parity-locked pure-MLX reference appautomaton/mlx-speech
// — MIT — which was itself gated against StepFun's torch code; mlxengine-audio/WIP/speech-edit-eval/out/v1/parity.md):
//   • Tokenizer  — dual codebook: vq02 = Paraformer SAN-M encoder (16.7 Hz) + k-means (1 024); vq06 = S3 v1
//                  (Whisper-style conformer, 25 Hz) + VQ (4 096); interleaved 2:3 into the LM vocabulary.
//   • LM         — step1 (StepFun's own family): 32 × 3072, 48 heads over 4 KV groups, sqrt-ALiBi, no RoPE.
//   • Flow       — CosyVoice-lineage conditioner + upsample conformer + DiT CFM (10 steps, cfg 0.7) → 50 Hz mel.
//   • Vocoder    — HiFT (NSF harmonic source + iSTFT) → 24 kHz.
//   • Speaker    — CAM++ (lifted from mlx-indextts2-swift — the same checkpoint, already parity-locked).
//   • MLXStepAudioEditX — the engine-facing package (awaits `Capability.speechEdit`, ask filed from AB-T-0202 V3).
//   • editx-gates — parity gates against goldens from the torch fp32 oracle (Tools/oracle-capture), plus a render lane.
import PackageDescription

let package = Package(
    name: "mlx-step-audio-editx-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "StepAudioEditXCore", targets: ["StepAudioEditXCore"]),
        .library(name: "MLXStepAudioEditX", targets: ["MLXStepAudioEditX"]),
        .executable(name: "editx-gates", targets: ["editx-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        .package(url: "https://github.com/xocialize/mlx-audio-dsp.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.64.0"),
    ],
    targets: [
        .target(
            name: "StepAudioEditXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXAudioDSP", package: "mlx-audio-dsp"),
            ],
            path: "Sources/StepAudioEditXCore",
            resources: [.copy("Resources")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXStepAudioEditX",
            dependencies: [
                "StepAudioEditXCore",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXStepAudioEditX",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXStepAudioEditXTests",
            dependencies: [
                "StepAudioEditXCore",
                "MLXStepAudioEditX",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXStepAudioEditXTests",
            exclude: [],
            resources: []
        ),
        .executableTarget(
            name: "editx-gates",
            dependencies: [
                "StepAudioEditXCore", "MLXStepAudioEditX",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/editx-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
