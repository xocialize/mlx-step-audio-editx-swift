// WAV.swift — 16-bit / 32-bit float PCM WAV in and out for the gate lanes (mono; multi-channel averaged).
import Foundation

enum WAV {
    static func read(_ url: URL) throws -> ([Float], Int) {
        let d = try Data(contentsOf: url)
        func u32(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 | Int(d[o + 2]) << 16 | Int(d[o + 3]) << 24 }
        func u16(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
        var o = 12; var fmt = 1, channels = 1, rate = 0, bits = 16; var samples = [Float]()
        while o + 8 <= d.count {
            let id = String(bytes: d[o ..< o + 4], encoding: .ascii) ?? "", size = u32(o + 4); let body = o + 8
            if id == "fmt " { fmt = u16(body); channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14) }
            if id == "data" {
                let n = min(size, d.count - body) / (bits / 8)
                samples.reserveCapacity(n / channels)
                var acc: Float = 0
                for i in 0 ..< n {
                    let p = body + i * (bits / 8); var v: Float
                    if fmt == 3 && bits == 32 { v = d[p ..< p + 4].withUnsafeBytes { $0.loadUnaligned(as: Float.self) } }
                    else if bits == 16 { v = Float(Int16(bitPattern: UInt16(u16(p)))) / 32768 }
                    else if bits == 24 { let x = Int32(d[p]) | (Int32(d[p + 1]) << 8) | (Int32(d[p + 2]) << 16); v = Float((x << 8) >> 8) / 8388608 }
                    else { v = Float(Int32(bitPattern: UInt32(u32(p)))) / 2147483648 }
                    acc += v
                    if (i + 1) % channels == 0 { samples.append(acc / Float(channels)); acc = 0 }
                }
                break
            }
            o = body + size + (size & 1)
        }
        return (samples, rate)
    }

    static func write(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let pcm = samples.map { Int16(max(-32768, min(32767, ($0 * 32767).rounded()))) }
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + pcm.count * 2)); data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count * 2))
        pcm.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try data.write(to: url)
    }
}
