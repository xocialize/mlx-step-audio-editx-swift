# mlx-step-audio-editx-swift

[Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX) (StepFun, Apache-2.0) on Swift-MLX: a speech
**editor**. Give it a take and its transcript, and it comes back saying the same words in the same voice, re-delivered —
an emotion (`angry`, `happy`, `sad`, …), a speaking style (`whisper`, `shout`, `gentle`, `older`, …), inserted
paralinguistics (`[Laughter]`, `[Sigh]`, `[Breathing]` at a place in the transcript), denoised, or with its silences
trimmed. The fleet's measured reason to port it (E19, `mlxengine-audio/Docs/ENHANCEMENTS.md`): against the Studio's own
delivery levers it wins every paralinguistic row, zh shout and zh angry at the codec floor (AB-R-0412), and it is the only
delivery lever for engines that have none.

```swift
import StepAudioEditXCore

let pipe = try EditXPipeline.load(bundle: EditXBundle(root: bundleDir))        // bf16 LM, fp32 flow / HiFT
let take = try pipe.edit(samples, sampleRate: 24000, text: "We don't have much time.", edit: .emotion("angry"), seed: 42)
// take.waveform (24 kHz), take.audioTokens, take.timings["lm"/"flow"/"hift"]
let laugh = try pipe.edit(samples, sampleRate: 24000, text: line, edit: .paralinguistic(targetText: "Great[Laughter], the weather is lovely."))
let voice = try pipe.clone(samples, sampleRate: 24000, promptText: line, targetText: "A new sentence in that voice.")
```

Speed edits exist upstream (`.speed("faster")`) but lose to a DSP stretch in the band they would replace (E19 V0) and
replace words outright in 5 % of runs — they are not in the Dub ladder. Every delivery edit changes the take's length;
re-fit afterwards. The LM regenerates rather than edits in place, so gate the output with an ASR transcript (AB-L-0119).

## What is inside

| stage | module | note |
|---|---|---|
| preprocessing | `Preprocess`, `Resample.sinc` | 16 kHz (torchaudio's sinc kernel), peak-normalised, silence-trimmed — exactly the reference's |
| dual tokenizer | `VQ02Tokenizer` (Paraformer SAN-M, 16.7 Hz, k-means 1 024), `VQ06Tokenizer` (S3 v1, 25 Hz, VQ 4 096) | streaming Paraformer run offline as upstream does; **dither off** — deterministic (upstream's default is a noise draw, 95.8 % self-agreement) |
| prompt | `UnigramTokenizer`, `EditXTokenizer`, `EditPrompts` | the sentencepiece Unigram model read from `tokenizer.json`, the chat template by hand; id-exact to the reference |
| LM | `Step1ForCausalLM` | 32 × 3072, GQA 48/4, sqrt-ALiBi, SwiGLU; logits mean\|Δ\| 8e-6 vs torch fp32 |
| flow | `FlowModel` (+ `FlowConditioning`, `MelFrontend`) | upsample conformer + DiT CFM, 10 steps, cfg 0.7; mel cos 0.999999 |
| vocoder | `HiFTGenerator` | NSF source + iSTFT, 24 kHz; decoder 44 dB on a shared source |
| speaker | `SpeakerEncoder` (CAM++) | lifted from mlx-indextts2-swift, the same checkpoint; cos 1.000000 |

Weights: [`mlx-community/Step-Audio-EditX-bf16`](https://huggingface.co/mlx-community/Step-Audio-EditX-bf16) — the
[mlx-speech](https://github.com/appautomaton/mlx-speech) bundle layout (`model.safetensors`, `vq02/vq06`, `flow-*`, `hift`,
`step-audio-tokenizer-assets` + `*-config.json`, `tokenizer.json`), converted from the stock weights without quantisation;
[`mlx-community/Step-Audio-EditX-8bit`](https://huggingface.co/mlx-community/Step-Audio-EditX-8bit) — the int8 tier (LM 8-bit, group 64;
RTF 0.50, 6.2 GB resident), written by this package from the bf16 bundle and read back by `config.json`'s `quantization`
(or quantise at load: `EditXDTypes.lmQuantBits = 8`). CAM++ ships in the package resources.

## Gates

`swift run -c release editx-gates --keys --g-vq06 --g-vq02 --g-lm --g-frontend --g-flow --g-hift --prompt --bundle DIR` (CPU
stream, against the torch fp32 oracle goldens in `Tests/Goldens` — not in the repo; regenerate them with
`Tools/oracle-capture/v1_oracle_dump.py` in the upstream reference's environment, dither off) and
`--e2e [--quant 8] --cues DIR` (GPU, the shipped precision). The record is `PORTING-SPEC.md`.

## Engine

`MLXStepAudioEditX` is the engine-facing package on `Capability.speechEdit` (mlx-engine-swift ≥ 0.65.0, contract 1.50.0):

```swift
import MLXServeCore
import MLXStepAudioEditX

let id = try await engine.register(StepAudioEditXPackage.registration,
                                   configuration: StepAudioEditXConfiguration(quant: .int8))   // .bf16 default
try await engine.prepare(.speechEdit, package: id)   // materializes mlx-community/Step-Audio-EditX-8bit into the store
let r = try await engine.run(SpeechEditRequest(audio: take, transcript: line, edit: .style("whisper"), seed: 42), package: id)
let out = r as! SpeechEditResponse       // out.audio (24 kHz .wav), out.transcript (what the take should now say)
```

The surface declares everything the engine admits: the five operations, upstream's 15 emotion labels (incl. `remove`),
33 style labels, 10 paralinguistic tags, and a 90 s input ceiling (`StepAudioEditXPackage.controls`). `.trimSilence` is
upstream's `vad`. `metaData`: `temperature` (0.7), `flowSteps` (10). Cancellation is honoured per generated token and
per stage; `RunProgress` reports the generate phase per token. A take that does not finish within the LM window is
refused, never returned short. Licence declaration: Apache-2.0 + `funasrModel` (the Paraformer encoder), port code MIT.

Two tiers are declared and measured through the engine (phys_footprint, M5 Max): bf16 9.4 GB resident, int8 6.1 GB;
≈ 2 GB activation either way (declared 2.5 GB) with the engine's pool cap. `swift test` runs the offline gate (manifest, MAT, CAN,
request plane); `editx-gates --validate [--quant 8] --bundle DIR` drives the real engine path and prints the footprint.

## Licence

Port code MIT (xocialize). Step-Audio-EditX weights and code Apache-2.0 (StepFun); the Paraformer encoder under the
FunASR model licence; CAM++ Apache-2.0; mlx-speech (the translation reference) MIT. See `THIRD_PARTY_NOTICES.md`.
