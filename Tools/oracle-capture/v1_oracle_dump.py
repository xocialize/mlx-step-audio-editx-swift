"""V1 oracle dump (torch reference, fp32 on CPU): everything the MLX side needs to prove parity stage by stage,
for a few cues × one delivery edit each.  Per cue writes out/v1/<cue>_<etype>_<step>.npz with
  wav (the volume-normalised 24 kHz prompt the reference tokenizes), vq02 / vq06 / vq0206 codes, the audio token
  string, prompt_ids (chat-templated edit prompt), fp32 LM logits at the last prompt position and at the first /
  last 16 positions, the seed-42 sampled generation (a FIXED decoder input, not a sampling-parity claim — AB-L-0090),
  speech_feat + speech_embedding (CosyVoice frontend), the flow's rand_noise buffer, the flow mel, every HiFT noise
  draw in call order (rand_ini, randn) and the HiFT waveform.
usage (.venv-editx): v1_oracle_dump.py [--cues en01,zh01] [--edits emotion:angry] [--max-new 320]"""
import os, sys, json, time, argparse, pathlib
import numpy as np, torch, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ROOT, CUES, OUT, read_cues
import run_editx                                     # the V0 runner: reference tts.py path, no vLLM (its patches apply on import)
from run_editx import Runner, AUDIO_LO, AUDIO_HI

OUTD = OUT / "v1"; OUTD.mkdir(parents=True, exist_ok=True)


class Recorder:
    """Replace torch.rand / torch.randn_like for the HiFT call and keep every draw so MLX can replay them."""
    def __init__(self): self.rand, self.randn = [], []
    def __enter__(self):
        self._rand, self._randn = torch.rand, torch.randn_like
        def rand(*a, **k):
            x = self._rand(*a, **k); self.rand.append(x.detach().cpu().float().numpy()); return x
        def randn_like(t, *a, **k):
            x = self._randn(t, *a, **k); self.randn.append(x.detach().cpu().float().numpy()); return x
        torch.rand, torch.randn_like = rand, randn_like; return self
    def __exit__(self, *e): torch.rand, torch.randn_like = self._rand, self._randn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cues", default="en01,zh01"); ap.add_argument("--edits", default="emotion:angry")
    ap.add_argument("--max-new", type=int, default=320); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    torch.manual_seed(a.seed)
    r = Runner(lm_device="cpu", cosy_device="cpu", cosy_dtype=torch.float32)
    r.lm = r.lm.float().eval()                         # fp32 oracle (the V0 runs were bf16 on MPS)
    # the Paraformer front end dithers with an unseeded torch RNG by default (95.8 % of its own codes agree run to run);
    # the oracle is the deterministic path
    r.audio_tokenizer.funasr_model.kwargs["frontend"].dither = 0.0
    cosy = r.cosy_model.cosy_impl
    cues = {c["id"]: c for c in read_cues()}
    for cid in a.cues.split(","):
        c = cues[cid]; wav_path = str(CUES / f"{cid}.wav")
        for edit in a.edits.split(","):
            etype, step = edit.split(":", 1); tag = f"{cid}_{etype}_{step}"
            t0 = time.time()
            # --- prompt audio, exactly as the reference prepares it
            feats = {}
            _dump_label = r.audio_tokenizer.dump_label
            r.audio_tokenizer.dump_label = lambda samples, mean: (feats.__setitem__("vq02_feats", np.asarray(samples[0], dtype=np.float32)), _dump_label(samples, mean))[1]
            _sess_run = r.cosy_model.frontend.campplus_session.run
            r.cosy_model.frontend.campplus_session.run = lambda outs, inp: (feats.__setitem__("cam_fbank", next(iter(inp.values())).astype(np.float32)), _sess_run(outs, inp))[1]
            vq0206, vq02, vq06, speech_feat, _, speech_emb = r.preprocess_prompt_wav(wav_path)
            r.audio_tokenizer.dump_label = _dump_label; r.cosy_model.frontend.campplus_session.run = _sess_run
            prompt_wav, sr = __import__("torchaudio").load(wav_path); prompt_wav = prompt_wav.mean(0, keepdim=True)
            norm = prompt_wav.abs().max()
            if norm > 0.6: prompt_wav = prompt_wav / norm * 0.6
            audio_tokens = r.audio_tokenizer.merge_vq0206_to_token_str(vq02, vq06)
            feats["wav16_pre"] = r.audio_tokenizer.preprocess_wav(prompt_wav, sr)[0].numpy().astype(np.float32)   # what both tokenizers consume
            instruct = r._build_audio_edit_instruction(c["text"], etype, step, None)
            prompt_ids = r._encode_audio_edit_prompt(r.edit_sys_prompt, instruct, audio_tokens)
            # --- LM: fp32 logits over the prompt, then a seeded sampled rollout (fixed decoder input)
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            with torch.no_grad():
                out = r.lm(ids, use_cache=False, output_hidden_states=True); logits = out.logits[0].float()
            hidden = np.stack([np.concatenate([h[0, :16].float().numpy(), h[0, -16:].float().numpy()]) for h in out.hidden_states])   # (layers+1, 32, 3072)
            logits_last = logits[-1].numpy(); logits_head = logits[:16].numpy(); logits_tail = logits[-16:].numpy()
            torch.manual_seed(a.seed)
            with torch.no_grad():
                gen = r.lm.generate(ids, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=a.max_new,
                                    eos_token_id=[2, 3], pad_token_id=0, use_cache=True)[0, ids.shape[1]:].tolist()
            while gen and gen[-1] in (2, 3): gen = gen[:-1]
            audio_ids = [t for t in gen if AUDIO_LO <= t < AUDIO_HI]
            # --- decoder: flow (fixed rand_noise buffer) + HiFT (recorded noise), fp32
            tok = torch.tensor([audio_ids], dtype=torch.long) - 65536
            prompt_tok = torch.tensor([vq0206], dtype=torch.long) - 65536
            rand_noise = cosy.flow.decoder.rand_noise.detach().cpu().float().numpy()
            # the reference's token2wav_nonstream runs flow then hift; capture the mel by hooking hift.inference
            captured = {}
            hift_inf = cosy.hift.inference
            f0_fwd = cosy.hift.f0_predictor.forward
            def f0_hook(x):
                y = f0_fwd(x); captured["hift_f0"] = y.detach().cpu().float().numpy(); return y
            cosy.hift.f0_predictor.forward = f0_hook
            def hift_hook(speech_feat=None, **kw):
                mel = speech_feat if speech_feat is not None else kw.get("mel")
                captured["mel"] = mel.detach().cpu().float().numpy()
                res = hift_inf(speech_feat=speech_feat, **kw) if speech_feat is not None else hift_inf(**kw)
                captured["hift_source"] = res[1].detach().cpu().float().numpy(); return res
            cosy.hift.inference = hift_hook
            with torch.no_grad(), Recorder() as rec:
                wav_out = r.cosy_model.token2wav_nonstream(tok, prompt_tok, speech_feat.float(), speech_emb.float())
            cosy.hift.inference = hift_inf; cosy.hift.f0_predictor.forward = f0_fwd
            wav_out = wav_out.detach().cpu().float().numpy().reshape(-1)
            np.savez(OUTD / f"{tag}.npz", wav=prompt_wav[0].numpy(), sr=sr, vq02=np.array(vq02), vq06=np.array(vq06), vq0206=np.array(vq0206),
                     audio_token_str=np.array(audio_tokens), instruct=np.array(instruct), prompt_ids=np.array(prompt_ids),
                     logits_last=logits_last, logits_head=logits_head, logits_tail=logits_tail, gen_ids=np.array(gen), audio_ids=np.array(audio_ids),
                     speech_feat=speech_feat.float().numpy(), speech_emb=speech_emb.float().numpy(), rand_noise=rand_noise,
                     mel=captured.get("mel", np.zeros(0)), wav_out=wav_out, hidden=hidden, hift_f0=captured.get("hift_f0", np.zeros(0)), hift_source=captured.get("hift_source", np.zeros(0)), **feats,
                     **{f"hift_rand_{i}": x for i, x in enumerate(rec.rand)}, **{f"hift_randn_{i}": x for i, x in enumerate(rec.randn)})   # one key per draw: shapes differ
            sf.write(OUTD / f"{tag}_oracle.wav", wav_out, 24000)
            print(f"{tag}: vq02 {len(vq02)} vq06 {len(vq06)} prompt {len(prompt_ids)} gen {len(gen)} (audio {len(audio_ids)}) mel {captured.get('mel', np.zeros(0)).shape} "
                  f"hift draws rand {len(rec.rand)} randn {len(rec.randn)} wav {len(wav_out)/24000:.2f}s in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
