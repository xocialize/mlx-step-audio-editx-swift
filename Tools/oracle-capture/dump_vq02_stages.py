"""vq02 ladder goldens from the pure-MLX reference (mlx-speech, CPU stream, dither OFF): for each cue, the streaming
front end's encoder INPUT per chunk (LFR+CMVN, (T_chunk, 560)) concatenated, the chunk boundaries, and the encoder
OUTPUT (T, 512) — so the Swift front end gates exactly against this and the encoder against the torch oracle.
usage (.venv-mlxspeech, from WIP/speech-edit-eval): dump_vq02_stages.py <goldens dir>"""
import sys, pathlib, numpy as np, mlx.core as mx
mx.set_default_device(mx.cpu)
G = pathlib.Path(sys.argv[1]); ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
from mlx_speech.generation.step_audio_editx import StepAudioEditXModel
import mlx_speech.models.step_audio_tokenizer.vq02 as V
m = StepAudioEditXModel.from_dir("weights/editx-mlx-bf16")
for o in [m.vq02.model]: o.set_dtype(mx.float32)
rt = m.vq02.runtime; rt.frontend.frontend = rt.frontend.frontend.__class__(**{**rt.frontend.frontend.__dict__, "dither": 0.0}) if hasattr(rt.frontend.frontend, "__dict__") else rt.frontend.frontend
try: object.__setattr__(rt.frontend, "frontend", type(rt.frontend.frontend)(**{**vars(rt.frontend.frontend), "dither": 0.0}))
except Exception as e: print("dither patch:", e)
print("frontend dither now", rt.frontend.frontend.dither)
for tag in sorted(p.name for p in G.iterdir() if p.is_dir()):
    w16 = np.load(G / tag / "wav16_pre.npy").astype(np.float32)
    ins, outs, bounds = [], [], []
    orig_fc = rt.model.encoder.forward_chunk
    def hooked(xs_pad, ilens, *, cache):
        ins.append(np.asarray(xs_pad, dtype=np.float32)[0]); y, l = orig_fc(xs_pad, ilens, cache=cache); return y, l
    rt.model.encoder.forward_chunk = hooked
    feats = rt.extract_encoder_features(w16, is_final=True)
    rt.model.encoder.forward_chunk = orig_fc
    enc_in = np.concatenate(ins, axis=0); lens = np.array([x.shape[0] for x in ins], dtype=np.int32)
    np.save(G / tag / "vq02_encoder_in.npy", np.ascontiguousarray(enc_in)); np.save(G / tag / "vq02_chunk_lens.npy", lens)
    np.save(G / tag / "vq02_feats_mlx.npy", np.ascontiguousarray(np.asarray(feats, dtype=np.float32)))
    codes = np.array(rt.processor.cluster_linguistic_features(feats), dtype=np.int32); np.save(G / tag / "vq02_mlx.npy", codes)
    oracle = np.load(G / tag / "vq02.npy"); fo = np.load(G / tag / "vq02_feats.npy").reshape(-1, 512)
    print(f"{tag}: {len(ins)} chunks, encoder input {enc_in.shape}, output {np.asarray(feats).shape}; vs torch oracle: feats max|Δ| {np.abs(np.asarray(feats)-fo).max():.3e}, codes exact {np.mean(codes == oracle)*100:.1f} %")
