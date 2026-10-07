// UnigramTokenizer.swift — the bundle's `tokenizer.json` read directly: a sentencepiece UNIGRAM model (74 752 pieces with
// log-probabilities, byte fallback) behind HF's Metaspace pre-tokenizer (`▁` for space, `prepend_scheme: first`,
// `split: false`) and 9 284 added tokens (`<audio_N>`, `<|BOT|>`, `<|EOT|>`, `<s>` …). Encoding = split out the added
// tokens, then per text section: spaces → ▁ (a leading ▁ on the very first section), Viterbi over the pieces, and any
// character no piece covers becomes its UTF-8 bytes' `<0xNN>` pieces. This is what `tokenizers` does for
// `LlamaTokenizerFast(legacy=False)`; the `--prompt` gate holds it id-exact to the reference.

import Foundation

public final class UnigramTokenizer: @unchecked Sendable {
    public let pieces: [String]                     // id → piece
    public let scores: [Float]
    public let pieceToId: [String: Int32]
    public let addedTokens: [String: Int32]         // content → id (specials included)
    public let unkId: Int32
    public let bytePieceIds: [Int32]                // <0x00> … <0xFF>
    let maxPieceScalars: Int
    static let space: Character = "▁"

    public init(tokenizerJSON url: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let model = root["model"] as? [String: Any], (model["type"] as? String) == "Unigram",
              let vocab = model["vocab"] as? [[Any]] else { throw StepAudioEditXError.badConfig("tokenizer.json is not a Unigram model") }
        var ps = [String](), ss = [Float](), map = [String: Int32]()
        ps.reserveCapacity(vocab.count); ss.reserveCapacity(vocab.count)
        for (i, entry) in vocab.enumerated() {
            guard entry.count == 2, let p = entry[0] as? String, let s = entry[1] as? NSNumber else { throw StepAudioEditXError.badConfig("malformed vocab entry \(i)") }
            ps.append(p); ss.append(s.floatValue); map[p] = Int32(i)
        }
        var added = [String: Int32]()
        for a in root["added_tokens"] as? [[String: Any]] ?? [] {
            if let c = a["content"] as? String, let id = a["id"] as? NSNumber { added[c] = id.int32Value }
        }
        pieces = ps; scores = ss; pieceToId = map; addedTokens = added
        unkId = Int32((model["unk_id"] as? NSNumber)?.intValue ?? 0)
        bytePieceIds = (0 ..< 256).map { map[String(format: "<0x%02X>", $0)] ?? Int32((model["unk_id"] as? NSNumber)?.intValue ?? 0) }
        maxPieceScalars = ps.reduce(0) { max($0, $1.unicodeScalars.count) }
    }

    public func id(_ token: String) -> Int32? { addedTokens[token] ?? pieceToId[token] }
    public func piece(_ id: Int32) -> String { Int(id) < pieces.count ? pieces[Int(id)] : addedTokens.first { $0.value == id }?.key ?? "<unk>" }

    /// Added-token split: the longest added token at each `<`.
    func split(_ text: String) -> [(String, Bool)] {
        var out = [(String, Bool)](); var buf = ""; var i = text.startIndex
        while i < text.endIndex {
            if text[i] == "<", let close = text[i...].firstIndex(of: ">") {
                let cand = String(text[i ... close])
                if addedTokens[cand] != nil {
                    if !buf.isEmpty { out.append((buf, false)); buf = "" }
                    out.append((cand, true)); i = text.index(after: close); continue
                }
            }
            buf.append(text[i]); i = text.index(after: i)
        }
        if !buf.isEmpty { out.append((buf, false)) }
        return out
    }

    /// Unigram Viterbi over Unicode scalars; characters no piece covers get the unk id, replaced by byte pieces after.
    func viterbi(_ section: String) -> [Int32] {
        let scalars = Array(section.unicodeScalars); let n = scalars.count
        if n == 0 { return [] }
        let unkScore = (scores.min() ?? 0) - 10
        var best = [Float](repeating: -.infinity, count: n + 1), back = [(start: Int, id: Int32)](repeating: (0, unkId), count: n + 1)
        best[0] = 0
        for end in 1 ... n {
            let lo = max(0, end - maxPieceScalars)
            for start in stride(from: end - 1, through: lo, by: -1) {
                var s = String.UnicodeScalarView(); s.append(contentsOf: scalars[start ..< end])
                if let id = pieceToId[String(s)] {
                    let cand = best[start] + scores[Int(id)]
                    if cand > best[end] { best[end] = cand; back[end] = (start, id) }
                }
            }
            if best[end] == -.infinity {                    // nothing covers this character: unk at single-scalar step
                let cand = best[end - 1] + unkScore
                best[end] = cand; back[end] = (end - 1, unkId)
            }
        }
        var ids = [Int32](); var pos = n
        while pos > 0 {
            let (start, id) = back[pos]
            if id == unkId {                                  // byte fallback for the uncovered scalar(s)
                var bytes = [UInt8]()
                for sc in scalars[start ..< pos] { bytes += Array(String(sc).utf8) }
                ids.append(contentsOf: bytes.reversed().map { bytePieceIds[Int($0)] })
            } else { ids.append(id) }
            pos = start
        }
        return ids.reversed()
    }

    public func encode(_ text: String) -> [Int32] {
        var ids = [Int32](); var firstSection = true
        for (chunk, isAdded) in split(text) {
            if isAdded { ids.append(addedTokens[chunk]!); continue }
            var s = chunk.replacingOccurrences(of: " ", with: String(Self.space))
            if firstSection, s.first != Self.space { s = String(Self.space) + s }   // prepend_scheme "first"
            firstSection = false
            ids += viterbi(s)
        }
        return ids
    }

    public func decode(_ ids: [Int32]) -> String {
        var out = ""; var pendingBytes = [UInt8]()
        func flush() { if !pendingBytes.isEmpty { out += String(decoding: pendingBytes, as: UTF8.self); pendingBytes = [] } }
        for id in ids {
            let p = piece(id)
            if p.count == 6, p.hasPrefix("<0x"), p.hasSuffix(">"), let b = UInt8(p.dropFirst(3).dropLast(), radix: 16) { pendingBytes.append(b); continue }
            flush(); out += p.replacingOccurrences(of: String(Self.space), with: " ")
        }
        flush()
        return out.hasPrefix(" ") ? String(out.dropFirst()) : out
    }
}
