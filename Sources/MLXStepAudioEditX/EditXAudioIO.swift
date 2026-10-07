// EditXAudioIO.swift — canonical `Audio` (.wav) in and out. Takes are decoded with AVFoundation (any valid WAV layout)
// and mixed to mono at their native rate; the Core resamples with the same windowed sinc the reference uses, so no
// AVAudioConverter sits in the path. Lifted from mlx-dia2-tts-swift's `Dia2AudioIO` (decode / encodeWAV16).

import AVFoundation
import Foundation
import MLXToolKit

enum EditXAudioIO {
    /// Mono float samples at the file's own rate.
    static func decode(_ audio: Audio, what: String = "take") throws -> (samples: [Float], sampleRate: Int) {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("editx-\(what)-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: tmp) } catch {
            throw PackageError.unsupportedRequestFeature("\(what) is not a readable .wav (\(error.localizedDescription))")
        }
        let src = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames) else {
            throw PackageError.unsupportedRequestFeature("\(what) is empty")
        }
        try file.read(into: buf)
        return (mixToMono(buf), Int(src.sampleRate))
    }

    static func mixToMono(_ buf: AVAudioPCMBuffer) -> [Float] {
        let ch = Int(buf.format.channelCount), n = Int(buf.frameLength)
        guard let data = buf.floatChannelData, ch > 0 else { return [] }
        var mono = [Float](repeating: 0, count: n)
        for c in 0 ..< ch { for i in 0 ..< n { mono[i] += data[c][i] } }
        if ch > 1 { let inv = 1 / Float(ch); for i in 0 ..< n { mono[i] *= inv } }
        return mono
    }

    /// Mono float [-1, 1] → 16-bit PCM WAV (upstream's write_wav scaling: × 32767).
    static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + bytes)); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(bytes))
        d.reserveCapacity(d.count + bytes)
        for s in samples {
            let v = Int16(max(-1, min(1, s.isFinite ? s : 0)) * 32767)
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        return d
    }
}
