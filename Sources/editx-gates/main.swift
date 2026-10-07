// editx-gates — parity gates against goldens from the torch fp32 oracle (Tools/oracle-capture/v1_oracle_dump.py, run in
// mlxengine-audio/WIP/speech-edit-eval; exported per key as .npy into Tests/Goldens/<cue>_<edit>/), plus render lanes.
//   swift run -c release editx-gates --keys | --g-vq06 | --g-vq02 | --g-lm | --g-frontend | --g-flow | --g-hift | --prompt | --all [--bundle DIR] [--goldens DIR] [--gpu]
//   editx-gates --e2e [--quant 8] [--cues DIR] [--bundle DIR]   (GPU; the shipped precision, four Studio cues, seed 42)
// Gates run on the CPU stream by default (the oracle is torch CPU fp32; stream-match both sides); --gpu runs them
// on the default device to measure the Metal fp32 gap.

import Foundation
import MLX
import MLXNN
import StepAudioEditXCore

setvbuf(stdout, nil, _IONBF, 0)   // progress survives a trap
var args = Array(CommandLine.arguments.dropFirst())
func flag(_ n: String) -> Bool { if let i = args.firstIndex(of: n) { args.remove(at: i); return true }; return false }
func option(_ n: String) -> String? {
    guard let i = args.firstIndex(of: n), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i ... i + 1); return v
}
let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let bundleDir = URL(fileURLWithPath: option("--bundle")
    ?? "/Volumes/Satechi/Development/mlxengine-audio/WIP/speech-edit-eval/weights/editx-mlx-bf16")
let goldensDir = URL(fileURLWithPath: option("--goldens") ?? packageRoot.appendingPathComponent("Tests/Goldens").path)
let useGPU = flag("--gpu")
let quantBits = option("--quant").flatMap(Int.init)
let cuesDir = option("--cues")
let writeInt8 = option("--write-int8-bundle")
let bundle = EditXBundle(root: bundleDir)
let goldenTags = ((try? FileManager.default.contentsOfDirectory(atPath: goldensDir.path)) ?? []).filter { !$0.hasPrefix(".") && $0 != "index.json" }.sorted()

var failures = [String]()
func check(_ ok: Bool, _ what: String) { print((ok ? "  PASS " : "  FAIL ") + what); if !ok { failures.append(what) } }
func npy(_ tag: String, _ name: String) throws -> MLXArray { try NPY.load(goldensDir.appendingPathComponent(tag).appendingPathComponent("\(name).npy")) }
func floats(_ tag: String, _ name: String) throws -> [Float] { let a = try npy(tag, name).asType(.float32); eval(a); return a.asArray(Float.self) }
func ints(_ tag: String, _ name: String) throws -> [Int32] { let a = try npy(tag, name).asType(.int32); eval(a); return a.asArray(Int32.self) }
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) }
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).reshaped([-1]), y = b.asType(.float32).reshaped([-1])
    return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()) + 1e-12)).item(Float.self)
}
func snr(_ a: MLXArray, _ ref: MLXArray) -> Float {
    let n = sqrt(((a.asType(.float32) - ref.asType(.float32)) ** 2).mean()).item(Float.self), s = sqrt((ref.asType(.float32) ** 2).mean()).item(Float.self)
    return 20 * log10(s / max(n, 1e-12))
}
func exactFraction(_ a: [Int32], _ b: [Int32]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return .nan }
    return Double(zip(a, b).filter { $0 == $1 }.count) / Double(a.count)
}
/// The process's phys_footprint (what the fleet's [VAL] numbers read), in GB.
func physFootprintGB() -> Double {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : .nan
}
func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
    let t0 = Date(); let r = try body(); print(String(format: "  (%@ %.2fs)", label, Date().timeIntervalSince(t0))); return r
}

// MARK: - S0: key contracts — every module's flattened keys + shapes vs the bundle's safetensors headers (no Metal)

func gateKeys() throws {
    print("S0 key contracts vs \(bundleDir.lastPathComponent)")
    try bundle.validate()
    let vq06 = VQ06Model(try ConfigIO.load(VQ06Config.self, from: bundle.file("vq06-config.json")))
    let r = try WeightIO.contract(vq06, against: bundle.file("vq06.safetensors"), component: "vq06")
    check(true, "vq06: \(r.keys) keys, \(r.bytes / 1_000_000) MB — module paths == safetensors keys, shapes equal")
    let vq02 = VQ02Model(try ConfigIO.load(VQ02Config.self, from: bundle.file("vq02-config.json")))
    let r2 = try WeightIO.contract(vq02, against: bundle.file("vq02.safetensors"), component: "vq02")
    check(true, "vq02: \(r2.keys) keys, \(r2.bytes / 1_000_000) MB — module paths == safetensors keys, shapes equal")
    let lmCfg = try ConfigIO.load(Step1Config.self, from: bundle.file("config.json"))
    let lm = Step1ForCausalLM(lmCfg)
    if let q = lmCfg.quantization { quantize(model: lm, groupSize: q.group_size, bits: q.bits) }
    let r3 = try WeightIO.contract(lm, against: bundle.file("model.safetensors"), component: "step1")
    check(true, "step1\(lmCfg.quantization.map { " int\($0.bits) g\($0.group_size)" } ?? ""): \(r3.keys) keys, \(r3.bytes / 1_000_000) MB — module paths == safetensors keys, shapes equal")
    let flow = FlowModel(try ConfigIO.load(FlowModelConfig.self, from: bundle.file("flow-model-config.json")))
    let r4 = try WeightIO.contract(flow, against: bundle.file("flow-model.safetensors"), component: "flow", derived: { $0.hasPrefix("decoder.randNoise") })
    check(true, "flow: \(r4.keys) keys, \(r4.bytes / 1_000_000) MB — module paths == safetensors keys, shapes equal")
    let hift = HiFTGenerator(try ConfigIO.load(HiFTConfig.self, from: bundle.file("hift-config.json")))
    let r5 = try WeightIO.contract(hift, against: bundle.file("hift.safetensors"), component: "hift")
    check(true, "hift: \(r5.keys) keys, \(r5.bytes / 1_000_000) MB — module paths == safetensors keys, shapes equal")
    let spk = try SpeakerEncoder.load()
    check(true, "campplus: \(spk.model.parameters().flattened().count) parameters loaded from the package resources (sanitized keys == module paths)")
}

// MARK: - S1 frontend: the 24 kHz mel vs oracle speech_feat; CAM++ on the oracle's fbank vs its embedding

func gateFrontend() throws {
    print("g-frontend mel + CAM++ vs oracle (\(goldenTags.joined(separator: ", ")))")
    let mel = try MelFrontend.load(bundle: bundle); let spk = try SpeakerEncoder.load()
    for tag in goldenTags {
        let wav = try floats(tag, "wav"), sr = Int(try ints(tag, "sr")[0])
        let feat = mel.features(wav, sampleRate: sr); let ref = try npy(tag, "speech_feat").reshaped([-1, 80])
        let d = feat.shape == ref.shape ? maxAbs(feat, ref) : .nan
        check(feat.shape == ref.shape && d < 1e-3, String(format: "%@ mel: %@ vs %@, max|Δ| %.2e cos %.6f", tag, "\(feat.shape)", "\(ref.shape)", d, feat.shape == ref.shape ? cosine(feat, ref) : .nan))
        let fb = try npy(tag, "cam_fbank").reshaped([-1, 80]); let embRef = try npy(tag, "speech_emb").reshaped([-1])
        let emb = spk.embed(fbankCMN: fb)
        check(cosine(emb, embRef) > 0.9999, String(format: "%@ CAM++ on the oracle's fbank: cos %.6f max|Δ| %.2e", tag, cosine(emb, embRef), maxAbs(emb, embRef)))
        // and our own fbank from the oracle's 24 kHz prompt (sinc → 16 k → kaldi fbank − mean) vs the oracle's fbank
        let w16 = Resample.sinc(wav, from: sr, to: 16000)
        if let ours = CampPlusFbank.fbankCMN(MLXArray(w16)) {
            let n = min(ours.dim(0), fb.dim(0))
            check(ours.dim(0) == fb.dim(0) && maxAbs(ours[..<n], fb[..<n]) < 1e-3, String(format: "%@ CAM++ fbank: %d vs %d frames, max|Δ| %.2e", tag, ours.dim(0), fb.dim(0), maxAbs(ours[..<n], fb[..<n])))
        }
    }
}

// MARK: - S1 flow: mel from the oracle's tokens / feats / embedding with the oracle's rand_noise

func gateFlow() throws {
    print("g-flow DiT CFM flow vs oracle (\(goldenTags.joined(separator: ", ")))")
    let flow = try timed("load flow fp32") { try FlowModel.load(bundle: bundle, dtype: .float32) }
    let ccfg = try ConfigIO.load(FlowConditionerConfig.self, from: bundle.file("flow-conditioner-config.json"))
    for tag in goldenTags {
        let tokens = try ints(tag, "audio_ids").map { $0 - 65536 }, prompt = try ints(tag, "vq0206").map { $0 - 65536 }
        let feat = try npy(tag, "speech_feat"), emb = try npy(tag, "speech_emb")
        flow.decoder.randNoise = try npy(tag, "rand_noise")
        let inputs = FlowConditioning.prepare(config: ccfg, tokens: tokens, promptTokens: prompt, promptFeat: feat, speakerEmbedding: emb)
        let out = timed("flow") { () -> MLXArray in let m = flow(inputs); eval(m); return m }
        let ref = try npy(tag, "mel")
        let d = out.shape == ref.shape ? maxAbs(out, ref) : .nan, c = out.shape == ref.shape ? cosine(out, ref) : .nan
        check(out.shape == ref.shape && c > 0.9999 && d < 0.1, String(format: "%@ flow mel: %@ vs %@, max|Δ| %.2e cos %.6f", tag, "\(out.shape)", "\(ref.shape)", d, c))
    }
}

// MARK: - S1 HiFT: f0 on the oracle mel, the decoder on the oracle source, end to end with the oracle's noise draws

func gateHiFT() throws {
    print("g-hift HiFT vocoder vs oracle (\(goldenTags.joined(separator: ", ")))")
    let hift = try timed("load hift fp32") { try HiFTGenerator.load(bundle: bundle, dtype: .float32) }
    for tag in goldenTags {
        let mel = try npy(tag, "mel"), refWav = try npy(tag, "wav_out"), refF0 = try npy(tag, "hift_f0").reshaped([-1]), src = try npy(tag, "hift_source")
        let f0 = hift.f0Predictor(mel)[0]; eval(f0)
        let n = min(f0.dim(0), refF0.dim(0)); let df0 = abs(f0[..<n] - refF0[..<n])
        let uv = ((f0[..<n] .> 0) .== (refF0[..<n] .> 0)).asType(.float32).mean().item(Float.self)
        check(f0.dim(0) == refF0.dim(0) && df0.mean().item(Float.self) < 0.5 && uv == 1, String(format: "%@ f0: %d vs %d frames, max|Δ| %.2f Hz mean %.3f, u/v agreement %.1f %%", tag, f0.dim(0), refF0.dim(0), df0.max().item(Float.self), df0.mean().item(Float.self), uv * 100))
        let y = timed("decode") { () -> MLXArray in let w = hift.decode(mel, source: src)[0]; eval(w); return w }
        let m = min(y.dim(0), refWav.dim(0))
        check(y.dim(0) == refWav.dim(0) && snr(y[..<m], refWav[..<m]) > 40, String(format: "%@ decoder on the oracle's source: %d vs %d samples, SNR %.1f dB, max|Δ| %.2e", tag, y.dim(0), refWav.dim(0), snr(y[..<m], refWav[..<m]), maxAbs(y[..<m], refWav[..<m])))
        // end to end with the oracle's noise draws replayed (hift_rand_0 = rand_ini (1, 9), hift_randn_0 = the sine noise (1, T, 9))
        var uniforms = [MLXArray](), normals = [MLXArray]()
        var i = 0; while let a = try? npy(tag, "hift_rand_\(i)") { uniforms.append(a); i += 1 }
        i = 0; while let a = try? npy(tag, "hift_randn_\(i)") { normals.append(a); i += 1 }
        hift.mSource.noise = ReplaySourceNoise(uniforms: uniforms, normals: normals)
        let (full, _) = hift(mel); eval(full)
        let yy = full[0]; let k = min(yy.dim(0), refWav.dim(0))
        let rmsOurs = sqrt((yy * yy).mean()).item(Float.self), rmsRef = sqrt((refWav * refWav).mean()).item(Float.self)
        check(yy.dim(0) == refWav.dim(0) && abs(rmsOurs - rmsRef) / rmsRef < 0.05, String(format: "%@ end to end (noise replayed): %d samples, RMS %.4f vs %.4f, sample SNR %.1f dB (phase from fp32 cumsum — spectral parity is the claim)", tag, yy.dim(0), rmsOurs, rmsRef, snr(yy[..<k], refWav[..<k])))
        hift.mSource.noise = RandomSourceNoise()
    }
}

// MARK: - S1 step1 LM: logits at the oracle's 32 positions, per-layer hidden trace, last-position top-5

func gateLM() throws {
    print("g-lm step1 LM vs torch fp32 oracle (\(goldenTags.joined(separator: ", ")))")
    let lm = try timed("load step1 fp32") { try Step1ForCausalLM.load(bundle: bundle, dtype: .float32) }
    for tag in goldenTags {
        let ids = try ints(tag, "prompt_ids"); let t = ids.count
        let hidden = try npy(tag, "hidden")                                   // (33, 32, 3072) at positions [:16] + [-16:]
        var perLayer = [Float](repeating: .nan, count: hidden.dim(0))
        func take(_ x: MLXArray) -> MLXArray { concatenated([x[0, ..<16], x[0, (t - 16)...]], axis: 0) }
        let logits = timed("prefill \(t)") { () -> MLXArray in
            let l = lm(MLXArray(ids).reshaped([1, -1]), caches: nil) { layer, h in
                if layer < perLayer.count - 1 { perLayer[layer] = maxAbs(take(h), hidden[layer]) }   // the dump's last entry is post-norm
            }
            eval(l); return l[0]
        }
        let ours = take(logits.expandedDimensions(axis: 0)), ref = concatenated([try npy(tag, "logits_head"), try npy(tag, "logits_tail")], axis: 0)
        let d = abs(ours - ref); let perPos = d.max(axis: 1); eval(perPos)
        let agree = (ours.argMax(axis: 1) .== ref.argMax(axis: 1)).asType(.float32).mean().item(Float.self)
        let last = logits[t - 1], refLast = try npy(tag, "logits_last")
        let top5 = Set(argSort(-last)[..<5].asType(.int32).asArray(Int32.self)).intersection(Set(argSort(-refLast)[..<5].asType(.int32).asArray(Int32.self))).count
        let scale = abs(ref).max().item(Float.self)
        check(agree == 1 && d.mean().item(Float.self) < 1e-2 && last.argMax().item(Int32.self) == refLast.argMax().item(Int32.self),
              String(format: "%@ logits (32 positions, |logits| %.1f): argmax agreement %.0f %%, mean|Δ| %.3e, max|Δ| %.3e; last: max|Δ| %.3e cos %.6f top5∩ %d/5",
                     tag, scale, agree * 100, d.mean().item(Float.self), d.max().item(Float.self), maxAbs(last, refLast), cosine(last, refLast), top5))
        let trace = perLayer.dropLast().enumerated().map { String(format: "L%d %.3f", $0.offset, $0.element) }.joined(separator: " ")
        check(perLayer[1] < 0.1, "\(tag) per-layer max|Δ| (embed, L1..L32; hidden scale \(String(format: "%.0f", abs(hidden[1]).max().item(Float.self)))): \(trace)")
    }
}

// MARK: - S1 vq02: front end vs the Python-MLX reference (exact), encoder vs the torch oracle, codes with flips localised

func gateVQ02() throws {
    print("g-vq02 Paraformer linguistic tokenizer vs oracle (\(goldenTags.joined(separator: ", ")))")
    let tok = try timed("load vq02 fp32") { try VQ02Tokenizer.load(bundle: bundle, dtype: .float32) }
    for tag in goldenTags {
        let wav16 = try floats(tag, "wav16_pre")
        let (feats, inputs) = timed("encode") { tok.encoderFeatures(wav16, collectInputs: true) }
        // front end: the concatenated per-chunk encoder inputs vs mlx-speech's (dither off, CPU)
        let refIn = try npy(tag, "vq02_encoder_in"), refLens = try ints(tag, "vq02_chunk_lens")
        let ours = concatenated(inputs.map { $0[0] }, axis: 0)
        let lensOK = inputs.map { Int32($0.dim(1)) } == refLens
        let dIn = ours.shape == refIn.shape ? maxAbs(ours, refIn) : .nan
        check(lensOK && dIn < 1e-3, String(format: "%@ front end: %d chunks %@, encoder input %@ vs %@, max|Δ| %.2e", tag, inputs.count, lensOK ? "lengths equal" : "LENGTHS DIFFER", "\(ours.shape)", "\(refIn.shape)", dIn))
        // encoder vs torch oracle (fp32 cross-implementation: ≤ 5e-2 max, cos ≥ 0.9999) and vs the mlx reference
        // the oracle is the reference's DITHER-FREE path (its default dithers with an unseeded torch RNG: 95.8 % of its own
        // codes agree run to run); the dithered dump stays as information
        let refOut = try npy(tag, "vq02_feats_nodither").reshaped([-1, 512]), mlxOut = try npy(tag, "vq02_feats_mlx")
        let dithered = try ints(tag, "vq02")
        let d = feats.shape == refOut.shape ? maxAbs(feats, refOut) : .nan, c = feats.shape == refOut.shape ? cosine(feats, refOut) : .nan
        let dm = feats.shape == mlxOut.shape ? maxAbs(feats, mlxOut) : .nan
        check(feats.shape == refOut.shape && d < 5e-2 && c > 0.9999, String(format: "%@ encoder: %@ vs oracle max|Δ| %.2e cos %.6f; vs mlx-speech max|Δ| %.2e", tag, "\(feats.shape)", d, c, dm))
        // codes: flips localised against the oracle's own centroid margins
        let codes = tok.cluster(feats), oracle = try ints(tag, "vq02_nodither"), mlxCodes = try ints(tag, "vq02_mlx")
        let exact = exactFraction(codes, oracle), exactMLX = exactFraction(codes, mlxCodes)
        let flips = zip(codes, oracle).enumerated().filter { $0.element.0 != $0.element.1 }.map(\.offset)
        check(codes.count == oracle.count && exact >= 0.98, String(format: "%@ codes: exact %.1f %% vs the dither-free oracle (%d flips at %@), %.1f %% vs mlx-speech, %.1f %% vs one dithered draw", tag, exact * 100, flips.count, "\(flips)", exactMLX * 100, exactFraction(codes, dithered) * 100))
    }
}

// MARK: - S1 vq06: codes on the oracle's preprocessed 16 kHz audio must be exact; preprocessing vs the oracle's audio

func gateVQ06() throws {
    print("g-vq06 S3 semantic tokenizer vs oracle (\(goldenTags.joined(separator: ", ")))")
    let tok = try timed("load vq06 fp32") { try VQ06Tokenizer.load(bundle: bundle, dtype: .float32) }
    for tag in goldenTags {
        let wav16 = try floats(tag, "wav16_pre"), oracle = try ints(tag, "vq06")
        // preprocessing on the oracle's 24 kHz prompt (`wav`) vs its `wav16_pre`
        let wav = try floats(tag, "wav"), sr = Int(try ints(tag, "sr")[0])
        let pre = Preprocess.run(wav, sampleRate: sr, config: tok.config)
        let n = min(pre.count, wav16.count)
        let dPre = zip(pre.prefix(n), wav16.prefix(n)).map { abs($0 - $1) }.max() ?? .nan
        check(pre.count == wav16.count && dPre < 1e-5, String(format: "%@ preprocess: %d vs %d samples, max|Δ| %.2e", tag, pre.count, wav16.count, dPre))
        let codes = timed("encode") { tok.encodePreprocessed(wav16) }
        let exact = exactFraction(codes, oracle)
        check(codes.count == oracle.count && exact == 1.0, String(format: "%@ vq06 codes: %d vs %d, exact %.1f %%", tag, codes.count, oracle.count, exact * 100))
    }
}

// MARK: - S2 prompt: the Swift text side on the dither-free codes must reproduce the reference's prompt ids exactly

func gatePrompt() async throws {
    print("prompt tokenizer + chat template vs the Python-MLX reference (\(goldenTags.joined(separator: ", ")))")
    let tok = try EditXTokenizer.load(bundle: bundle)
    check(tok.audioTokenBase == 65536, "audio token base id \(tok.audioTokenBase) == 65536")
    for tag in goldenTags {
        let vq02 = try ints(tag, "vq02_nodither"), vq06 = try ints(tag, "vq06")
        let packed = EditXTokenizer.packPromptTokens(vq02: vq02, vq06: vq06)
        let str = EditXTokenizer.audioTokenString(packed)
        let refStr = try String(contentsOf: goldensDir.appendingPathComponent(tag).appendingPathComponent("audio_token_str_nodither.txt"), encoding: .utf8)
        check(str == refStr, "\(tag) audio token string: \(str.count) chars \(str == refStr ? "==" : "≠") reference")
        let instruct = try String(contentsOf: goldensDir.appendingPathComponent(tag).appendingPathComponent("instruct.txt"), encoding: .utf8)
        let ids = tok.editPromptIds(instruction: instruct, audioTokenString: str), ref = try ints(tag, "prompt_ids_nodither")
        let first = zip(ids, ref).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        check(ids == ref, "\(tag) edit prompt ids: \(ids.count) vs \(ref.count)\(ids == ref ? ", exact" : ", first diff at \(String(describing: first))")")
        if ids != ref {
            let show = { (xs: [Int32]) in xs.map { "\($0)=\(tok.inner.piece($0))" }.joined(separator: " ") }
            print("    ours first 16: " + show(Array(ids.prefix(16)))); print("    ref  first 16: " + show(Array(ref.prefix(16))))
            // where do the sequences re-align after the first difference?
            if let f = first {
                for off in -3 ... 3 where f + off >= 0 && f + off + 8 <= ids.count && f + 8 <= ref.count {
                    if Array(ids[(f + off) ..< (f + off + 8)]) == Array(ref[f ..< (f + 8)]) { print("    re-aligns with offset \(off) after position \(f)") }
                }
            }
            print("    ours last 12: " + show(Array(ids.suffix(12)))); print("    ref  last 12: " + show(Array(ref.suffix(12))))
        }
    }
}

// MARK: - S2 end to end on the GPU: the shipped precision (bf16 LM), seed 42, the four V1 edits → wavs + timings

func gateE2E() async throws {
    print("e2e (GPU, \(quantBits.map { "int\($0)" } ?? "bf16") LM) — en01/zh01 angry, en01 whisper, zh05 [Laughter], seed 42")
    let t0 = Date()
    var dt = EditXDTypes(); dt.lmQuantBits = quantBits
    let pipe = try EditXPipeline.load(bundle: bundle, dtypes: dt)
    print(String(format: "  loaded in %.1fs; resident phys_footprint %.2f GB (MLX active %.2f GB)", Date().timeIntervalSince(t0), physFootprintGB(), Double(Memory.activeMemory) / 1e9))
    let tier = quantBits.map { "int\($0)" } ?? (pipe.lm.isQuantized ? "int\(pipe.lm.config.quantization?.bits ?? 8)-bundle" : "bf16")
    let cues = URL(fileURLWithPath: cuesDir ?? "/Volumes/Satechi/Development/mlxengine-audio/WIP/speech-edit-eval/cues")   // --cues DIR: <id>.wav + cues.tsv (id, lang, speaker, text)
    let text = try String(contentsOf: cues.appendingPathComponent("cues.tsv"), encoding: .utf8).split(separator: "\n").dropFirst()
        .reduce(into: [String: String]()) { d, line in let f = line.split(separator: "\t", omittingEmptySubsequences: false); if f.count >= 4 { d[String(f[0])] = String(f[3]) } }
    let outDir = goldensDir.appendingPathComponent("_out"); try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let jobs: [(String, SpeechEdit, String)] = [("en01", .emotion("angry"), "emotion_angry"), ("zh01", .emotion("angry"), "emotion_angry"), ("en01", .style("whisper"), "style_whisper"),
                                               ("zh05", .paralinguistic(targetText: text["zh05"]!.replacingOccurrences(of: "？", with: "？[Laughter]", options: [], range: text["zh05"]!.range(of: "？"))), "paralinguistic_Laughter")]
    for (cue, edit, name) in jobs {
        let (wav, sr) = try WAV.read(cues.appendingPathComponent("\(cue).wav"))
        let t = Date()
        let r = try pipe.edit(wav, sampleRate: sr, text: text[cue]!, edit: edit, seed: 42)
        let wall = Date().timeIntervalSince(t)
        try WAV.write(r.waveform, sampleRate: r.sampleRate, to: outDir.appendingPathComponent("\(cue)_\(name)_swift_\(tier).wav"))
        try r.generatedTokens.map(String.init).joined(separator: " ").write(to: outDir.appendingPathComponent("\(cue)_\(name)_swift_\(tier)_tokens.txt"), atomically: true, encoding: .utf8)
        if tier.hasSuffix("-bundle"), let ref = try? String(contentsOf: outDir.appendingPathComponent("\(cue)_\(name)_swift_int8_tokens.txt"), encoding: .utf8) {
            let refIds = ref.split(separator: " ").compactMap { Int32($0) }
            print("    tokens vs the quantise-at-load int8 run (seed 42): \(refIds == r.generatedTokens ? "IDENTICAL (\(refIds.count) ids)" : "differ — \(refIds.count) vs \(r.generatedTokens.count)")")
        }
        if let ref = try? String(contentsOf: outDir.appendingPathComponent("\(cue)_\(name)_mlx_tokens.txt"), encoding: .utf8) {
            let refIds = ref.split(separator: " ").compactMap { Int32($0) }
            let same = refIds == r.generatedTokens
            let firstDiff = zip(refIds, r.generatedTokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            print("    tokens vs the Python-MLX rung (seed 42): \(same ? "IDENTICAL (\(refIds.count) ids)" : "differ — \(refIds.count) vs \(r.generatedTokens.count), first diff at \(String(describing: firstDiff))")")
        }
        let dur = Double(r.waveform.count) / Double(r.sampleRate)
        let footBefore = physFootprintGB(), pool = Double(Memory.cacheMemory) / 1e9, active = Double(Memory.activeMemory) / 1e9
        Memory.clearCache()
        check(r.stoppedAtEOS && dur > 1, String(format: "%@ %@: %d audio tokens → %.2fs audio in %.1fs (RTF %.2f; lm %.1fs flow %.1fs hift %.1fs tokenize %.1fs), prompt %d, eos %@, MLX peak %.2f GB; after the edit: active %.2f GB + pool %.2f GB, phys_footprint %.2f GB → %.2f GB once the pool is cleared",
              cue, name, r.audioTokens.count, dur, wall, wall / dur, r.timings["lm"] ?? 0, r.timings["flow"] ?? 0, r.timings["hift"] ?? 0, r.timings["tokenize"] ?? 0, r.promptLength, r.stoppedAtEOS ? "yes" : "NO", Double(Memory.peakMemory) / 1e9, active, pool, footBefore, physFootprintGB()))
    }
}

// MARK: - tier: write the int8 bundle the package itself produces (bf16 → quantise on the CPU stream → save), so the
// published files reproduce the quantise-at-load path exactly; every other file is copied from the bf16 bundle

func writeInt8Bundle(to dir: URL, bits: Int = 8, groupSize: Int = 64) throws {
    print("tier: writing the int\(bits) (group \(groupSize)) bundle from \(bundleDir.lastPathComponent) to \(dir.path)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let lm = try timed("load bf16 + quantise (CPU stream)") { try Step1ForCausalLM.load(bundle: bundle, dtype: .bfloat16, quantBits: bits, groupSize: groupSize) }
    var arrays = [String: MLXArray]()
    for (k, v) in lm.parameters().flattened() { arrays[k] = v }
    try timed("save model.safetensors") { try save(arrays: arrays, metadata: ["format": "mlx"], url: dir.appendingPathComponent("model.safetensors")) }
    var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: bundle.file("config.json"))) as? [String: Any] ?? [:]
    cfg["quantization"] = ["bits": bits, "group_size": groupSize, "mode": "affine"]
    try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys]).write(to: dir.appendingPathComponent("config.json"))
    for f in EditXBundle.requiredFiles where f != "model.safetensors" && f != "config.json" {
        let dst = dir.appendingPathComponent(f); try? FileManager.default.removeItem(at: dst)
        try FileManager.default.copyItem(at: bundle.file(f), to: dst)
    }
    let bytes = (try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent("model.safetensors").path)[.size] as? Int) ?? 0
    check(bytes > 0, "int\(bits) bundle written: \(arrays.count) LM tensors, \(bytes / 1_000_000) MB model.safetensors, \(EditXBundle.requiredFiles.count) files")
}

// MARK: - Entry

let all = flag("--all")
let modes: [(String, () throws -> Void)] = [("--keys", gateKeys), ("--g-vq06", gateVQ06), ("--g-vq02", gateVQ02), ("--g-lm", gateLM), ("--g-frontend", gateFrontend), ("--g-flow", gateFlow), ("--g-hift", gateHiFT)]
let asyncModes: [(String, () async throws -> Void, Bool)] = [("--prompt", gatePrompt, false), ("--e2e", gateE2E, true)]   // (name, gate, wants the GPU)
let selected = modes.filter { all || flag($0.0) }
let selectedAsync = asyncModes.filter { all || flag($0.0) }
if let dir = writeInt8 {
    do { try writeInt8Bundle(to: URL(fileURLWithPath: dir)) } catch { print("  ERROR \(error)"); failures.append("\(error)") }
    print(failures.isEmpty ? "\nALL GATES PASSED" : "\n\(failures.count) FAILURE(S)"); exit(failures.isEmpty ? 0 : 1)
}
if selected.isEmpty && selectedAsync.isEmpty { print("usage: editx-gates --keys | --g-vq06 | --g-vq02 | --g-lm | --g-frontend | --g-flow | --g-hift | --prompt | --e2e | --all [--bundle DIR] [--goldens DIR] [--gpu]"); exit(2) }
do {
    try Device.withDefaultDevice(useGPU ? Device.gpu : Device.cpu) {
        for (name, gate) in selected { print("\n== \(name)"); try gate() }
    }
    for (name, gate, gpu) in selectedAsync {
        print("\n== \(name)")
        Device.setDefault(device: (gpu || useGPU) ? Device.gpu : Device.cpu)
        try await gate()
    }
} catch {
    print("  ERROR \(error)"); failures.append("\(error)")
}
print(failures.isEmpty ? "\nALL GATES PASSED" : "\n\(failures.count) FAILURE(S):\n  " + failures.joined(separator: "\n  "))
exit(failures.isEmpty ? 0 : 1)
