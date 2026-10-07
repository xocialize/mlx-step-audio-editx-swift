// WeightIO.swift — safetensors in, key contract enforced. The idiom is mlx-dia2-tts-swift's (`Dia2Core/MimiLoader.swift`):
// weights load on the CPU stream (the Metal-watchdog rule — a multi-GB lazy `Load` bound to the GPU stream holds one
// command buffer open past the ~10 s watchdog), are cast to the requested dtype and eval'd there; `apply` compares the
// module's own flattened parameter paths with the on-disk keys (0 missing / 0 unused, or throw — a partial load is the
// silent-failure class) and then `update(verify:)`s with `.noUnusedKeys` + `.shapeMismatch`, and switches BatchNorms to
// inference mode at this single choke point (the C14 INF lesson).
//
// The weight layout this package consumes is the mlx-speech bundle (`model.safetensors`, `vq02.safetensors`,
// `vq06.safetensors`, `flow-conditioner.safetensors`, `flow-model.safetensors`, `hift.safetensors`,
// `step-audio-tokenizer-assets.safetensors`, plus their `*-config.json`): conv weights already in MLX `(O, K, I)` layout,
// HiFT weight-norms already materialised. CAM++ comes from this package's own resources instead (see CampPlusLoader).

import Foundation
import MLX
import MLXNN

public enum StepAudioEditXError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        }
    }
}

public enum WeightIO {
    /// Reads only the safetensors HEADER (8-byte little-endian length + JSON): key names, dtypes and shapes. Pure
    /// Foundation — the S0 key contract runs without touching MLX or Metal.
    public static func header(_ url: URL) throws -> [String: (dtype: String, shape: [Int])] {
        guard let h = FileHandle(forReadingAtPath: url.path) else { throw StepAudioEditXError.missingFile(url.path) }
        defer { try? h.close() }
        let lenData = h.readData(ofLength: 8)
        guard lenData.count == 8 else { throw StepAudioEditXError.badConfig("safetensors header too short: \(url.lastPathComponent)") }
        let len = lenData.withUnsafeBytes { Int($0.load(as: UInt64.self).littleEndian) }
        let json = h.readData(ofLength: len)
        guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw StepAudioEditXError.badConfig("safetensors header is not a JSON object: \(url.lastPathComponent)")
        }
        var out = [String: (dtype: String, shape: [Int])]()
        for (k, v) in obj where k != "__metadata__" {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int] else { continue }
            out[k] = (dt, sh)
        }
        return out
    }

    /// Loads a safetensors file on the CPU stream, casting floating tensors to `dtype` (nil keeps the on-disk dtype).
    public static func load(_ url: URL, dtype: DType?) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw StepAudioEditXError.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                out[k] = (dtype != nil && floating && v.dtype != dtype!) ? v.asType(dtype!) : v
            }
            eval(Array(out.values))
            return out
        }
    }

    /// The key contract, then the update. `derived`: reflected arrays computed at load (never on disk); `buffers`:
    /// on-disk tensors consumed through `update` without being listed as parameters.
    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String,
                             derived: (String) -> Bool = { _ in false }, buffers: (String) -> Bool = { _ in false }) throws {
        let expected = Set(module.parameters().flattened().map(\.0).filter { !derived($0) })
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).filter { !buffers($0) }.sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw StepAudioEditXError.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
    }

    /// The structural half of S0: a weight-free module's flattened keys against a safetensors header.
    public static func contract(_ module: Module, against url: URL, component: String,
                                derived: (String) -> Bool = { _ in false }) throws -> (keys: Int, bytes: Int) {
        let hdr = try header(url)
        let expected = Set(module.parameters().flattened().map(\.0).filter { !derived($0) })
        let onDisk = Set(hdr.keys)
        let missing = expected.subtracting(onDisk).sorted(), unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw StepAudioEditXError.keyContract(component: component, missing: missing, unused: unused)
        }
        // shapes: the module's parameter shapes must equal the header's
        var mismatched = [String]()
        for (k, v) in module.parameters().flattened() where !derived(k) {
            if let h = hdr[k], h.shape != v.shape { mismatched.append("\(k) \(v.shape) vs disk \(h.shape)") }
        }
        guard mismatched.isEmpty else {
            throw StepAudioEditXError.keyContract(component: component, missing: [], unused: mismatched)
        }
        return (hdr.count, (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0)
    }
}

/// JSON configs from the bundle.
public enum ConfigIO {
    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else { throw StepAudioEditXError.missingFile(url.path) }
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
}
