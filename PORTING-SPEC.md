# mlx-step-audio-editx-swift — porting spec

Step-Audio-EditX (StepFun, Apache-2.0) → Swift-MLX. Evaluation, licence provenance and the port decision live in
`mlxengine-audio/Docs/ENHANCEMENTS.md` **E19** (AB-D-0113, receipts AB-R-0409 / AB-R-0412 / AB-R-0413); this file is
the port's contract and its gate record. Task AB-T-0202.

## Published

- Code: [xocialize/mlx-step-audio-editx-swift](https://github.com/xocialize/mlx-step-audio-editx-swift) — `v0.1.0`
  Core + gates (2026-10-07); `v0.1.1` pre-quantised bundles; `v0.2.0` the engine-facing `MLXStepAudioEditX` on
  `Capability.speechEdit` (engine 0.65.0, contract 1.50.0); `v0.2.1` the load-time warm-up edit (the first real edit
  no longer pays the per-process Metal compile — AB-R-0420).
- Weights: [mlx-community/Step-Audio-EditX-bf16](https://huggingface.co/mlx-community/Step-Audio-EditX-bf16) — the bf16
  bundle (8.0 GB) the gate record below was measured on; [mlx-community/Step-Audio-EditX-8bit](https://huggingface.co/mlx-community/Step-Audio-EditX-8bit)
  — the int8 tier (LM 8-bit group 64, 4.4 GB), written by this package (`editx-gates --write-int8-bundle`) so loading
  it reproduces quantise-at-load exactly (`v0.1.1`: the loader reads `config.json`'s `quantization`).

## Upstream

| piece | source | licence | pinned |
|---|---|---|---|
| algorithm + weights | `stepfun-ai/Step-Audio-EditX` (GitHub `a652e87`; HF main, 2026-02-14) — `tts.py`, `tokenizer.py`, `modeling_step1.py`, `stepvocoder/cosyvoice2/*`, `config/prompts.py` | Apache-2.0 (LM, flow, HiFT and the S3 tokenizer are StepFun-trained; `campplus.onnx` is Alibaba's, Apache-2.0 card — AB-R-0409) | same |
| linguistic tokenizer | FunASR Paraformer-large zh-cantonese-en online encoder (`dengcunqin/…`) + StepFun's k-means codebook (`stepfun-ai/Step-Audio-Tokenizer`) | FunASR model licence (allowlisted `funasrModel`) / Apache-2.0 | same |
| translation reference | `appautomaton/mlx-speech` `4577ad2` (pure-MLX pipeline; at parity with the torch reference after our resampler fix — `WIP/speech-edit-eval/out/v1/parity.md`) | MIT | local fork in `WIP/speech-edit-eval/ref/mlx-speech` |
| CAM++ | lifted from `mlx-indextts2-swift` (`Models/CampPlus.swift`, `Resources/campplus_cn_common.safetensors`, `kaldi_mel_banks.npy`) — the same 3D-Speaker checkpoint (433/715 tensors byte-identical to EditX's ONNX, the rest BN-folded) | Apache-2.0 / MIT | — |

Weights: this package consumes the mlx-speech bundle layout (`model.safetensors`, `vq02.safetensors`, `vq06.safetensors`,
`flow-conditioner.safetensors`, `flow-model.safetensors`, `hift.safetensors`, `step-audio-tokenizer-assets.safetensors`
+ `*-config.json`, `tokenizer.json`), converted from the stock weights by `scripts/convert/step_audio_editx.py --no-quantize`
(our local patch); CAM++ from the package resources.

## Goldens

`Tests/Goldens/<cue>_<edit>/` — the torch fp32 oracle (`Tools/oracle-capture/v1_oracle_dump.py`, CPU, **Paraformer dither
off** — the reference's default dithers with an unseeded torch RNG and only 95.8 % of its own codes agree run to run) for
`en01` / `zh01` × `emotion:angry`: preprocessed audio, codes, prompt ids, logits at 32 positions + all 33 hidden
states there, a seed-42 generation (a fixed decoder input), speech_feat, CAM++ fbank/embedding, the flow's
`rand_noise`, mel, HiFT f0 / source / noise draws, waveform. Plus the Python-MLX reference's per-chunk vq02 encoder
inputs (`dump_vq02_stages.py`, CPU, dither off). Gates run on the CPU stream (`editx-gates`, `--gpu` to measure the Metal gap).

## Phases

| phase | gate | result |
|---|---|---|
| S0 key contracts | `editx-gates --keys`: weight-free modules' flattened paths + shapes == safetensors headers | vq06 96 keys, vq02 652 keys — PASSED 2026-10-07 |
| S1 vq06 (S3 tokenizer) | `--g-vq06`: preprocess vs oracle audio; codes on the oracle's 16 kHz audio exact | preprocess max\|Δ\| 3.0e-7 / 4.2e-7; codes **142/142, 153/153 exact** — PASSED 2026-10-07 (0.06 s per take, CPU) |
| S1 vq02 (Paraformer tokenizer) | `--g-vq02`: per-chunk encoder input vs the Python-MLX reference; encoder output vs the dither-free torch oracle (max ≤ 5e-2, cos ≥ 0.9999); codes ≥ 98 % with flips localised | front end max\|Δ\| 2.4e-5 / 1.7e-5 (24 / 26 chunks); encoder vs oracle max\|Δ\| 8.9e-3 / 2.4e-3, cos 0.99997 / 0.99999, vs mlx-speech 2.8e-7 / 4.0e-7; codes **100 % / 100 %** vs the dither-free oracle (94.7 / 99.0 % vs one dithered draw — the reference's own noise band) — PASSED 2026-10-07 (0.6 s per take, CPU) |
| S1 step1 LM | `--g-lm`: logits at 32 oracle positions (argmax 100 %, mean\|Δ\| ≤ 1e-2 on \|logits\| ≈ 35), per-layer hidden trace vs the 33 dumped states, last position top-5 | 291 keys 7.06 GB; logits **mean\|Δ\| 8.3e-6 / 9.9e-6, max 1.8e-4 / 2.0e-4**, argmax 100 %, last position cos 1.000000 top-5 5/5; every layer's hidden ≤ 1e-3 of torch (scale 33) — PASSED 2026-10-07 (fp32 prefill of 416 tokens 1.9 s, CPU) |
| S1 frontend + CAM++ | `--g-frontend`: 24 kHz mel vs oracle (≤ 1e-3); CAM++ on the oracle's fbank (cos ≥ 0.9999 — the lifted port's own lock) | mel (280, 80) / (300, 80) max\|Δ\| 7.4e-5 / 1.1e-4, cos 1.000000; CAM++ on the oracle fbank **cos 1.000000** (max 8e-6); our fbank from the 24 kHz prompt (sinc → kaldi) max\|Δ\| 4.3e-4 / 6.4e-4 — PASSED 2026-10-07 |
| S1 flow | `--g-flow`: mel from the oracle's tokens / feats / rand_noise (cos ≥ 0.9999, max ≤ 0.1) | 605 keys 307 MB; mel (1, 80, 324) / (1, 80, 258) **cos 0.999999**, max\|Δ\| 7.0e-2 / 4.3e-2 (the Python-MLX reference: 9.4e-2 / 5.1e-2) — PASSED 2026-10-07 (10 steps 4.2 s, CPU) |
| S1 HiFT | `--g-hift`: f0 (u/v 100 %, mean ≤ 0.5 Hz), decoder on the oracle source ≥ 40 dB, end to end RMS within 5 % with the oracle's noise replayed | 246 keys 42 MB; f0 u/v 100 %, mean\|Δ\| 0.21 / 0.09 Hz (max 8.7 / 4.1 at voicing onsets — the reference's own residual); decoder on the oracle source **SNR 44.6 / 43.5 dB** (max 8e-3); end to end RMS 0.1167 vs 0.1166 / 0.0882 vs 0.0883 (sample SNR −3.8 dB: NSF phase from fp32 cumsum, as in the Python-MLX reference) — PASSED 2026-10-07 (decode 25 s / 20 s on the CPU stream; the GPU lane is the production path) |
| S2 prompt | `--prompt`: `<audio_N>` string and chat-templated edit prompt ids on the dither-free codes == the Python-MLX reference's | token strings byte-equal (2 720 / 2 939 chars); prompt ids **416/416, 430/430 exact** — PASSED 2026-10-07 (in-package Unigram tokenizer; swift-transformers' path was dropped: its `LlamaTokenizer` label selects BPE and its Unigram has no byte fallback, so every `\n` became `<unk>`) |
| S2 e2e | `--e2e`: wav → edit → wav on the GPU, seed 42; content / voice / quality scored by the E19 scorer | bf16 LM + fp32 flow / HiFT: en01 angry 260 tokens → 6.24 s in 4.7 s (**RTF 0.75**), zh01 angry 0.75, en01 whisper 0.72, zh05 [Laughter] 0.72 (first call +15 s Metal compile); loaded in 0.6 s; E19 scorer: content err **0.000 / 0.000 / 0.000 / 0.000**, SIM 0.963 / 0.883 / 0.774 (whisper, by design) / 0.908, DNSMOS OVRL 3.40 / 3.56 / 3.17 / 3.48 — the V0b bars (torch path 24.5 s per edit; Python-MLX rung RTF 0.77–1.05). Sampled tokens differ from the Python-MLX rung at the first near-tie (bf16 logits differ; AB-L-0090 — not a gate) — PASSED 2026-10-07 |
| S6 tiers | `--e2e --quant 8`: int8 (group 64) LM on the GPU, same four edits | int8: RTF **0.49–0.58** (LM 1.5× faster), content err 0.000 ×4, SIM 0.955 / 0.889 / 0.799 / 0.864, OVRL 3.15–3.58 — the bars hold; load 4.7 s (quantised on the CPU stream). **Footprint** (M5 Max, phys_footprint): bf16 resident 9.40 GB (MLX active 8.99 GB) → after an edit 13.3–14.7 GB with the pool, 11.6–12.6 GB with the pool cleared; MLX peak 10.5–10.9 GB ⇒ activation ≈ 1.9 GB per 6–8.5 s take; int8 resident 5.69 GB MLX active, peak 9.0 GB. Without clearing, the pool ratchets +4–5 GB per edit (26 GB after four) — the engine's `cacheLimit` is the bound; `Memory.clearCache()` on unload — PASSED 2026-10-07 **int8 as FILES** (`--write-int8-bundle`, the package's own quantisation on the CPU stream, 743 tensors / 3.75 GB): key contract PASSED; `--e2e --bundle <int8>` sampled tokens **IDENTICAL** to the quantise-at-load run on all four edits; loaded in 0.5 s, resident 6.16 GB phys (MLX active 5.69), RTF 0.50–0.51, MLX peak 7.3–7.6 GB; scorer err 0.000 ×4, SIM 0.83–0.97, OVRL 3.45–3.49 — PASSED 2026-10-07 |
| S7 engine wrap | `MLXStepAudioEditX` on `Capability.speechEdit` (contract 1.50.0, engine 0.65.0 — AB-A-0137): `swift test` (17: manifest C1/C7/C8/C-memory/provenance, MAT per tier, CAN pre-cancelled + cadence, request plane); `editx-gates --validate [--quant 8]` through `MLXServeEngine` | 17/17 offline; engine lane bf16: 0 licence advisories, prepared 1.0 s, angry / [Laughter] / trimSilence at RTF 0.77 / 0.75 / 0.77, an undeclared label refused before admission; **phys resident 9.44 GB, peak 11.40 → activation 1.95 GB**; int8: prepared 0.8 s, RTF 0.53 / 0.52 / 0.55, **resident 6.14 GB, activation 1.94 GB**. Manifest declared 9.5 / 6.3 GB resident, 2.5 GB activation — PASSED 2026-10-07 (v0.2.0). **v0.2.1 (the load-time warm-up, cache cleared after it):** bf16 prepared 3.3 s, resident 10.19 GB, peak 11.39 → activation 1.19 GB, RTF 0.74 / 0.73 / 0.76; int8 prepared 1.6 s, resident 6.94 GB, peak 8.37 → 1.43 GB — the peaks did not move, ≈ 0.8 GB of the warm-up stays resident; manifest re-declared **10.3 / 7.0 GB resident, 2.0 GB activation** — PASSED 2026-10-07 |

## Key contracts

- Module paths equal the bundle keys; `WeightIO.apply` refuses partial loads (0 missing / 0 unused, shapes equal) and
  switches BatchNorms to inference at that single choke point.
- Linear weights in `vq06.safetensors` are stored **(in, out)** (`x @ W`) — `VQ06Linear` keeps that layout; every other
  linear is MLXNN `Linear` (out, in). Conv weights are MLX `(O, K, I)` as the bundle ships them.
- vq02 runs the STREAMING path offline, as upstream does: 240 ms chunks (`chunk_size [0, 4, 5]`, stride 4 × 960
  samples), kaldi fbank with a carried sample tail, LFR (7, 6) with a 3-frame splice cache, CMVN after LFR, 5 frames of
  encoder-input overlap, per-layer K/V caches of 4 × 4 frames with a 5-frame right context dropped, position
  encoding counting from the utterance start, the last T frames of each chunk kept. **Dither is off** (deterministic;
  the reference's default is a noise draw).
- The resampler is torchaudio's windowed sinc (`Resample.sinc`, width 6, rolloff 0.99) — a linear resampler here
  perturbs every 16 kHz consumer (AB-L-0209).
- step1 attention: heads grouped contiguously (head h ↔ KV group h / 12); bias = −slope·√(i−j) in fp32 with the 32 + 16
  slope table; softmax in fp32.
- Text tokenizer: `tokenizer.json` is a sentencepiece UNIGRAM model (74 752 pieces, `unk_id` 0, byte fallback,
  Metaspace `prepend_scheme: first`, `split: false`, 9 284 added tokens) — `UnigramTokenizer` reads it directly:
  added-token split (longest `<…>` match), spaces → ▁ with a leading ▁ on the first text section only, Viterbi over
  the pieces, uncovered characters → their UTF-8 `<0xNN>` pieces (`\n` = 78). The chat template is rendered by hand
  (`<s>` before a system message; `<|BOT|> role\n` + content + `<|EOT|>`; `user` spelled `human`; `<|BOT|> assistant\n`).
