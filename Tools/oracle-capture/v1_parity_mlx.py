"""V1 parity: mlx-speech's pure-MLX Step-Audio-EditX pipeline (our bf16 bundle, upcast to fp32 for the numerics)
against the torch oracle dumps from v1_oracle_dump.py — stage by stage, each stage fed the ORACLE's inputs so a
miss is localised:
  T  dual tokenizer      vq02 / vq06 codes on the oracle's prompt wav          → exact-match fraction
  P  prompt builder      chat-templated edit prompt from the oracle's codes     → id-exact
  L  LM                  fp32 logits over the oracle prompt ids                 → max|Δ|, argmax, top-5 at the last position
  F  frontend            speech_feat (mel) + CAM++ embedding                    → max|Δ| / cosine
  M  flow                mel from the oracle's tokens + feats, same rand_noise  → max|Δ|, cosine
  H  HiFT                waveform from the oracle mel, same noise draws         → SNR dB
  E  end to end          mlx's own edit() on the cue (bf16, seed 42)            → runs, duration, token count (no parity claim)
usage (.venv-mlxspeech): v1_parity_mlx.py [--bundle weights/editx-mlx-bf16] [--fp32] [--skip E]  -> out/v1/parity.md"""
import sys, glob, time, argparse, pathlib
import numpy as np, mlx.core as mx, mlx.nn as nn, soundfile as sf
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUTD = ROOT / "out/v1"
from mlx_speech.generation.step_audio_editx import StepAudioEditXModel
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from sinc_resample import sinc_resample

def patch_resampler():
    """mlx-speech resamples with np.interp (no anti-aliasing); swap every module-level binding for the torchaudio-faithful sinc."""
    def faithful(samples, orig_sample_rate=None, target_sample_rate=None, *a, **k):
        o = orig_sample_rate if orig_sample_rate is not None else a[0]; t = target_sample_rate if target_sample_rate is not None else a[1]
        return mx.array(sinc_resample(np.asarray(samples, dtype=np.float32).reshape(-1), int(o), int(t)), dtype=mx.float32)
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_speech") and hasattr(mod, "resample_audio"):
            setattr(mod, "resample_audio", faithful); n += 1
    return n
from mlx_speech.models.step_audio_tokenizer import format_audio_token_string, pack_raw_codes_to_prompt_tokens

L = []
def log(s): print(s, flush=True); L.append(s)
def cos(a, b): a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def snr(ref, x):
    n = min(len(ref), len(x)); ref, x = ref[:n].astype(np.float64), x[:n].astype(np.float64)
    return float(10 * np.log10((ref ** 2).sum() / (((ref - x) ** 2).sum() + 1e-20)))
def walk(obj, depth=0, seen=None):
    """Every object reachable through attributes, dict values (mlx nn.Module is a dict) and sequences."""
    seen = set() if seen is None else seen
    if obj is None or id(obj) in seen or depth > 6 or isinstance(obj, (str, bytes, int, float, bool, np.ndarray, mx.array)): return
    seen.add(id(obj)); yield obj
    kids = []
    if isinstance(obj, dict): kids += list(obj.values())
    if hasattr(obj, "__dict__"): kids += list(vars(obj).values())
    if isinstance(obj, (list, tuple)): kids += list(obj)
    for k in kids: yield from walk(k, depth + 1, seen)
def find_attr(root, name): return next((o for o in walk(root) if hasattr(o, name)), None)
def upcast_all(root):
    n = 0
    for o in walk(root):
        if isinstance(o, nn.Module): o.set_dtype(mx.float32); n += 1
    return n

class Replay:
    """Stands in for the HiFT source module's numpy Generator and hands back the oracle's draws in call order."""
    def __init__(self, rand, randn): self.rand, self.randn = list(rand), list(randn)
    def random(self, shape, dtype=np.float32):
        x = self.rand.pop(0); assert tuple(x.shape) == tuple(shape), (x.shape, shape); return x.astype(dtype)
    def standard_normal(self, shape, dtype=np.float32):
        x = self.randn.pop(0); assert tuple(x.shape) == tuple(shape), (x.shape, shape); return x.astype(dtype)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default=str(ROOT / "weights/editx-mlx-bf16"))
    ap.add_argument("--fp32", action="store_true", help="upcast every module to fp32 (the numerics gate)"); ap.add_argument("--skip", default="")
    ap.add_argument("--sinc", action="store_true", help="replace mlx-speech's np.interp resampler with the torchaudio-faithful sinc")
    ap.add_argument("--cpu", action="store_true", help="run on MLX's CPU stream (numerics discriminator: GPU accumulation vs port)")
    a = ap.parse_args(); skip = set(a.skip.split(",")) if a.skip else set()
    if a.cpu: mx.set_default_device(mx.cpu); log("device: CPU stream")
    t0 = time.time(); m = StepAudioEditXModel.from_dir(a.bundle); log(f"loaded {a.bundle} in {time.time()-t0:.1f}s")
    if a.sinc: log(f"resampler: sinc (torchaudio-faithful) patched into {patch_resampler()} mlx_speech modules")
    if a.fp32:
        log(f"upcast {upcast_all(m)} modules (LM, flow, conditioner, HiFT, vq02, vq06, CAM++) to fp32")
    cfm = find_attr(m.flow.model, "_rand_noise"); src = find_attr(m.hift.model, "_rng")
    log(f"flow noise holder: {type(cfm).__name__}; hift rng holder: {type(src).__name__}")
    for npz in sorted(glob.glob(str(OUTD / "*.npz"))):
        tag = pathlib.Path(npz).stem; d = np.load(npz, allow_pickle=True); log(f"\n## {tag}")
        wav, sr = d["wav"].astype(np.float32), int(d["sr"])
        o02, o06, o0206 = d["vq02"].tolist(), d["vq06"].tolist(), d["vq0206"].tolist()
        # T — tokenizer
        if "T" not in skip:
            try:
                t = time.time(); v02 = list(m.vq02.runtime.encode(wav, sr)); v06 = list(m.vq06.runtime.encode(wav, sr))
                e02 = np.mean([x == y for x, y in zip(v02, o02)]) if len(v02) == len(o02) else float("nan")
                e06 = np.mean([x == y for x, y in zip(v06, o06)]) if len(v06) == len(o06) else float("nan")
                log(f"T tokenizer: vq02 {len(v02)} vs {len(o02)} codes, exact {e02*100:.1f} %; vq06 {len(v06)} vs {len(o06)}, exact {e06*100:.1f} %  ({time.time()-t:.1f}s)")
                mixed = list(pack_raw_codes_to_prompt_tokens(o02, o06)); ref_mixed = [x - 65536 for x in o0206]
                log(f"T packing (oracle codes → mixed ids): {'EXACT' if mixed == ref_mixed else 'DIFFERS'} ({len(mixed)} vs {len(ref_mixed)})")
            except Exception as e: log(f"T tokenizer: FAILED {e!r}")
        # P — prompt
        if "P" not in skip:
            try:
                ids = list(m.tokenizer.build_edit_prompt_ids(instruct_prefix=str(d["instruct"]), audio_token_str=format_audio_token_string(o02, o06)))
                ref = d["prompt_ids"].tolist(); same = ids == ref
                first = next((i for i, (x, y) in enumerate(zip(ids, ref)) if x != y), None)
                log(f"P prompt ids: {'EXACT' if same else 'DIFFERS'} ({len(ids)} vs {len(ref)}; first diff at {first})")
            except Exception as e: log(f"P prompt: FAILED {e!r}")
        # L — LM logits on the oracle prompt
        if "L" not in skip:
            try:
                t = time.time(); out = m.step1.model(input_ids=mx.array([d["prompt_ids"].tolist()], dtype=mx.int32), cache=None)
                lg = np.asarray(out.logits.astype(mx.float32))[0]; mx.eval(out.logits)
                last, ref_last = lg[-1], d["logits_last"]
                top5 = set(np.argsort(-last)[:5]) & set(np.argsort(-ref_last)[:5])
                ours = np.concatenate([lg[:16], lg[-16:]]); ref = np.concatenate([d["logits_head"], d["logits_tail"]]); dd = np.abs(ours - ref)
                per_pos = dd.max(axis=1); am = (ours.argmax(1) == ref.argmax(1)).mean()
                log(f"L LM logits (fp32 oracle vs mlx {'fp32' if a.fp32 else 'bf16'}): last max|Δ| {np.abs(last-ref_last).max():.3e} mean|Δ| {np.abs(last-ref_last).mean():.3e} cos {cos(last, ref_last):.6f} argmax {'=' if last.argmax()==ref_last.argmax() else '≠'} top5∩ {len(top5)}/5; "
                    f"over 32 dumped positions: max|Δ| {dd.max():.3e} mean|Δ| {dd.mean():.3e} p99 {np.percentile(dd, 99):.3e}, per-position max {np.array2string(per_pos[:8], precision=2)}…, argmax agreement {am*100:.0f} %, |logits| scale {np.abs(ref).max():.1f}  ({time.time()-t:.1f}s)")
            except Exception as e: log(f"L LM: FAILED {e!r}")
        # F — frontend
        feat_o = d["speech_feat"].astype(np.float32); emb_o = d["speech_emb"].astype(np.float32).reshape(-1)
        try:
            feat_m, _ = m.frontend.extract_speech_feat(wav, sr); feat_m = np.asarray(feat_m, dtype=np.float32)
            emb_m = np.asarray(m.frontend.extract_spk_embedding(wav, sr), dtype=np.float32).reshape(-1)
            fo = feat_o.reshape(-1, 80); fm = feat_m.reshape(-1, 80); n = min(len(fo), len(fm))
            log(f"F frontend: mel {fm.shape} vs {fo.shape} max|Δ| {np.abs(fm[:n]-fo[:n]).max():.3e} cos {cos(fm[:n], fo[:n]):.6f}; CAM++ emb max|Δ| {np.abs(emb_m-emb_o).max():.3e} cos {cos(emb_m, emb_o):.6f}")
            feat_shape = feat_m.shape
        except Exception as e: log(f"F frontend: FAILED {e!r}"); feat_shape = feat_o.shape
        # M — flow on the oracle's tokens / feats / noise
        mel_o = d["mel"].astype(np.float32)
        if "M" not in skip and mel_o.size:
            try:
                t = time.time(); cfm._rand_noise = d["rand_noise"].astype(np.float32)
                tok = [int(x) - 65536 for x in d["audio_ids"].tolist()]; ptok = [x - 65536 for x in o0206]
                prepared = m.conditioner.model.prepare_nonstream_inputs(token=tok, prompt_token=ptok, prompt_feat=feat_o.reshape(feat_shape), speaker_embedding=emb_o)
                mel_m = np.asarray(m.flow.model.inference(prepared, n_timesteps=10), dtype=np.float32)
                n = min(mel_m.shape[-1], mel_o.shape[-1])
                log(f"M flow: mel {mel_m.shape} vs {mel_o.shape} max|Δ| {np.abs(mel_m[..., :n]-mel_o[..., :n]).max():.3e} cos {cos(mel_m[..., :n], mel_o[..., :n]):.6f}  ({time.time()-t:.1f}s)")
            except Exception as e: log(f"M flow: FAILED {e!r}")
        # H — HiFT on the oracle mel with the oracle's noise draws
        if "H" not in skip and mel_o.size:
            try:
                draws = lambda pre: [d[k] for k in sorted((k for k in d.files if k.startswith(pre)), key=lambda k: int(k.rsplit('_', 1)[1]))]
                t = time.time(); src._rng = Replay(draws("hift_rand_"), draws("hift_randn_"))
                w, _ = m.hift.model.inference(mel_o); w = np.asarray(w, dtype=np.float32).reshape(-1); ref = d["wav_out"].astype(np.float32)
                log(f"H HiFT: wav {len(w)} vs {len(ref)} samples, SNR {snr(ref, w):.1f} dB, max|Δ| {np.abs(w[:len(ref)]-ref[:len(w)]).max():.3e}  ({time.time()-t:.1f}s)")
                sf.write(OUTD / f"{tag}_mlx_hift.wav", w, 24000)
                src._rng = np.random.default_rng(seed=0)
            except Exception as e: log(f"H HiFT: FAILED {e!r}")
        # T2 — tokenizer isolation: preprocessing vs the oracle's 16 kHz audio, then the encoders on the ORACLE's audio
        if "T2" not in skip and "wav16_pre" in d.files:
            try:
                w16o = d["wav16_pre"].astype(np.float32)
                w16m = np.asarray(m.vq02.runtime.processor.preprocess_wav(wav, sr), dtype=np.float32).reshape(-1)
                n = min(len(w16o), len(w16m))
                log(f"T2 preprocess (resample+energy+trim): {len(w16m)} vs {len(w16o)} samples, max|Δ| {np.abs(w16m[:n]-w16o[:n]).max():.3e}, rms Δ {np.sqrt(((w16m[:n]-w16o[:n])**2).mean()):.3e}")
                f = np.asarray(m.vq02.runtime.extract_encoder_features(w16o, is_final=True), dtype=np.float32); fo = d["vq02_feats"].astype(np.float32)
                f2, fo2 = f.reshape(-1, f.shape[-1]), fo.reshape(-1, fo.shape[-1]); k = min(len(f2), len(fo2))
                codes = list(m.vq02.runtime.processor.cluster_linguistic_features(f)); e = np.mean([x == y for x, y in zip(codes, o02)]) if len(codes) == len(o02) else float("nan")
                log(f"T2 vq02 on the oracle's audio: features {f2.shape} vs {fo2.shape} max|Δ| {np.abs(f2[:k]-fo2[:k]).max():.3e} cos {cos(f2[:k], fo2[:k]):.6f}; codes exact {e*100:.1f} % ({len(codes)} vs {len(o02)})")
                try: v06 = list(m.vq06.runtime.encode(w16o, 16000, enable_trim=False, energy_norm=False))
                except TypeError: v06 = list(m.vq06.runtime.encode(w16o, 16000))
                e6 = np.mean([x == y for x, y in zip(v06, o06)]) if len(v06) == len(o06) else float("nan")
                log(f"T2 vq06 on the oracle's audio: codes exact {e6*100:.1f} % ({len(v06)} vs {len(o06)})")
            except Exception as e: log(f"T2: FAILED {e!r}")
        # F2 — CAM++ isolation: the model on the oracle's fbank (bypassing resample + fbank)
        if "F2" not in skip and "cam_fbank" in d.files:
            try:
                rt = m.frontend._ensure_campplus_model().runtime if hasattr(m.frontend, "_ensure_campplus_model") else None
                fb = d["cam_fbank"].astype(np.float32)
                cand = [n for n in dir(rt) if "fbank" in n.lower() or "feature" in n.lower()]
                emb = None
                for name in ("embed_features", "extract_embedding_from_features", "embed_fbank", "forward_features"):
                    if hasattr(rt, name): emb = np.asarray(getattr(rt, name)(fb), dtype=np.float32).reshape(-1); break
                if emb is None and hasattr(rt, "model"):
                    emb = np.asarray(rt.model(mx.array(fb if fb.ndim == 3 else fb[None], dtype=mx.float32)), dtype=np.float32).reshape(-1)
                log(f"F2 CAM++ on the oracle's fbank {fb.shape}: " + (f"emb max|Δ| {np.abs(emb-emb_o).max():.3e} cos {cos(emb, emb_o):.6f}" if emb is not None else f"no entry point found (runtime attrs: {cand})"))
            except Exception as e: log(f"F2 CAM++: FAILED {e!r}")
        # H2 — HiFT isolation: f0 predictor, and the decoder on the oracle's source signal
        if "H2" not in skip and mel_o.size and "hift_source" in d.files:
            try:
                f0m = np.asarray(m.hift.model.f0_predictor(mx.array(mel_o, dtype=mx.float32)), dtype=np.float32).reshape(-1); f0o = d["hift_f0"].astype(np.float32).reshape(-1)
                k = min(len(f0m), len(f0o)); dd = np.abs(f0m[:k]-f0o[:k]); log(f"H2 f0 predictor: {len(f0m)} vs {len(f0o)} frames, max|Δ| {dd.max():.3e} Hz mean {dd.mean():.3e} p99 {np.percentile(dd, 99):.3e}, frames > 1 Hz {(dd > 1).sum()}, voiced agreement {((f0m[:k]>0)==(f0o[:k]>0)).mean()*100:.1f} %, f0 range {f0o.min():.0f}–{f0o.max():.0f}")
                src_o = d["hift_source"].astype(np.float32)                                      # torch (B, 1, T)
                w = np.asarray(m.hift.model.decode(mx.array(mel_o, dtype=mx.float32), mx.array(src_o, dtype=mx.float32)), dtype=np.float32).reshape(-1); ref = d["wav_out"].astype(np.float32)
                log(f"H2 decode on the oracle's source: wav {len(w)} vs {len(ref)}, SNR {snr(ref, w):.1f} dB, max|Δ| {np.abs(w[:len(ref)]-ref[:len(w)]).max():.3e}")
            except Exception as e: log(f"H2 HiFT: FAILED {e!r}")
        # L2 — LM per-layer trace at the 32 dumped positions
        if "L2" not in skip and "hidden" in d.files:
            try:
                lm = m.step1.model.model; ids = mx.array([d["prompt_ids"].tolist()], dtype=mx.int32)
                h = lm.embed_tokens(ids); H = d["hidden"].astype(np.float32); rows = []
                def take(x): x = np.asarray(x.astype(mx.float32))[0]; return np.concatenate([x[:16], x[-16:]])
                rows.append(np.abs(take(h) - H[0]).max())
                for i, layer in enumerate(lm.layers):
                    h, _ = layer(h, cache=None); mx.eval(h)
                    if i + 1 < len(H): rows.append(np.abs(take(h) - H[i + 1]).max())
                hn = lm.norm(h); post = np.abs(take(hn) - H[-1]).max()
                worst = int(np.argmax(rows)); log(f"L2 per-layer max|Δ| (embed, L1..L{len(rows)-1}): {np.array2string(np.array(rows), precision=3, max_line_width=400)}; after final norm vs last dump {post:.3e}; first layer with Δ > 1e-2: {next((i for i, r in enumerate(rows) if r > 1e-2), None)}; worst L{worst} = {rows[worst]:.3e}; hidden scale {np.abs(H[-2]).max():.1f}")
                p5 = int(d["prompt_ids"][5]); log(f"L2 position 5 token id {p5} = {m.tokenizer.decode([p5]) if hasattr(m.tokenizer, 'decode') else '?'!r}")
            except Exception as e: log(f"L2 LM trace: FAILED {e!r}")
        # E — the whole mlx path on its own
        if "E" not in skip:
            try:
                etype, step = tag.split("_", 1)[1].split("_", 1)
                import csv; text = next(c["text"] for c in csv.DictReader(open(ROOT / "cues/cues.tsv"), delimiter="\t") if c["id"] == tag.split("_")[0])
                t = time.time(); res = m.edit(wav, sr, text, etype, edit_info=step, seed=42)
                sf.write(OUTD / f"{tag}_mlx_e2e.wav", res.waveform, res.sample_rate)
                log(f"E end-to-end (mlx own sampling, seed 42): {len(res.generated_token_ids)} audio tokens, {len(res.waveform)/res.sample_rate:.2f}s, stop {res.stop_reason}, {time.time()-t:.1f}s")
            except Exception as e: log(f"E end-to-end: FAILED {e!r}")
    (OUTD / "parity.md").write_text("\n".join(L) + "\n"); print("wrote", OUTD / "parity.md")

if __name__ == "__main__":
    main()
