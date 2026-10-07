# Third-party notices

This port translates code from, and its weights derive from, the projects below. Their licences travel with it.

## Step-Audio-EditX — stepfun-ai/Step-Audio-EditX (code and weights)

Apache License 2.0 — Copyright StepFun. Full text: `LICENSES/Apache-2.0-stepfun-step-audio-editx.txt`. The LM, the
flow (DiT), HiFT and the S3 semantic tokenizer are StepFun-trained (AB-R-0409); the k-means linguistic codebook and the
tokenizer assets ship in `stepfun-ai/Step-Audio-Tokenizer` under the same licence. Translated upstream files:
`tts.py`, `tokenizer.py`, `config/prompts.py`, `modeling_step1.py`, `stepvocoder/cosyvoice2/{flow,hifigan,transformer,cli}`.

## FunASR Paraformer — the vq02 encoder weights (`dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online`)

FunASR model licence (the fleet's allowlisted `funasrModel` class). The SAN-M encoder and its kaldi front end are
translated from FunASR's `models/scama/encoder.py`, `frontends/wav_frontend.py`.

## CosyVoice — FunAudioLLM/CosyVoice (architecture) and FunAudioLLM/CosyVoice-300M (`campplus.onnx`)

Apache License 2.0 — Copyright Alibaba. The flow / HiFT architectures are CosyVoice's; only `campplus.onnx` among
EditX's vendored decoder files is Alibaba's checkpoint.

## CAM++ — 3D-Speaker (`campplus_cn_common`)

Apache License 2.0. `Sources/StepAudioEditXCore/Speaker/CampPlus.swift` and the resource weights are lifted from
`mlx-indextts2-swift` (xocialize, MIT), which ported them from `index-tts`' vendored `DTDNN.py` / `layers.py`.

## mlx-speech — appautomaton/mlx-speech (the pure-MLX translation reference)

MIT License — Copyright (c) 2026 AppAutomaton swarm of agents. Full text: `LICENSES/MIT-appautomaton-mlx-speech.txt`.
The Swift sources are 1:1 translations of its `models/step_audio_editx/*.py`, `models/step_audio_tokenizer/*.py`,
`generation/step_audio_editx.py`; the weight bundle layout is its converter's.

## mlx-indextts2-swift — xocialize (MIT)

`Support/NPY.swift`, `Speaker/CampPlus.swift`, the kaldi fbank core in `Speaker/CampPlusFrontend.swift`,
`Resources/{campplus_cn_common.safetensors, kaldi_mel_banks.npy}`.
