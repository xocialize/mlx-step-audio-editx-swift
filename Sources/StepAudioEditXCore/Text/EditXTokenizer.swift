// EditXTokenizer.swift — the text side of the prompt: StepFun's Llama-style sentencepiece vocabulary (65 536 text ids,
// 1 024 + 4 096 audio ids `<audio_N>`, chat markers `<|BOT|>` / `<|EOT|>`) through swift-transformers' `tokenizer.json`
// reader, the chat template rendered by hand (upstream's Jinja: `<s>` when the first message is a system message,
// `<|BOT|> role\n content <|EOT|>` per message with `user` spelled `human`, then `<|BOT|> assistant\n`), the audio
// token string packer (vq02 raw, vq06 + 1024, interleaved 2:3) and the edit / clone prompts of upstream `tts.py` and
// `config/prompts.py`. Translated from mlx-speech's `step_audio_editx/tokenizer.py` + `step_audio_tokenizer/packing.py`.

import Foundation

public enum EditPrompts {
    public static let editSystemPrompt = """
    As a highly skilled audio editing and tuning specialist, you excel in interpreting user instructions and applying precise adjustments to meet their needs. Your expertise spans a wide range of enhancement capabilities, including but not limited to:
    # Emotional Enhancement
    # Speaking Style Transfer
    # Non-linguistic Adjustments
    # Audio Tuning & Editing
    Note: You will receive instructions in natural language and are expected to accurately interpret and execute the most suitable audio edits and enhancements.

    """

    public static func cloneSystemPrompt(speaker: String, promptText: String, promptWavTokens: String) -> String {
        """
        Generate audio with the following timbre, prosody and speaking style

        [speaker_start]
        speaker name: \(speaker)
        speaker prompt text:
        \(promptText)
        speaker audio tokens:
        \(promptWavTokens)
        [speaker_end]

        """
    }

    /// `_build_audio_edit_instruction`.
    public static func instruction(promptText: String, edit: SpeechEdit) throws -> String {
        let text = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch edit {
        case .emotion(let info), .speed(let info):
            if info == "remove" { return "Remove any emotion in the following audio and the reference text is: \(text)\n" }
            return "Make the following audio more \(info). The text corresponding to the audio is: \(text)\n"
        case .style(let info):
            if info == "remove" { return "Remove any speaking styles in the following audio and the reference text is: \(text)\n" }
            return "Make the following audio more \(info) style. The text corresponding to the audio is: \(text)\n"
        case .denoise:
            return "Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        case .vad:
            return "Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        case .paralinguistic(let targetText):
            return "Add some non-verbal sounds to make the audio more natural, the new text is : \(targetText)\n  The text corresponding to the audio is: \(text)\n"
        }
    }
}

/// The edit operations upstream exposes. Speed is categorical (`faster` / `slower` / `more faster` / `more slower`) and
/// measurably loses to a DSP stretch (E19 V0) — it is here for completeness, not for the Dub ladder.
public enum SpeechEdit: Sendable, Equatable {
    case emotion(String)          // happy, sad, angry, surprised, fearful, disgusted, … or "remove"
    case style(String)            // whisper, shout, gentle, older, child, … or "remove"
    case paralinguistic(targetText: String)   // the transcript with inline tags: "Great[Laughter], the weather…"
    case speed(String)
    case denoise
    case vad
}

public struct ChatMessage { public let role: String; public let content: String; public init(role: String, content: String) { self.role = role; self.content = content } }

public final class EditXTokenizer: @unchecked Sendable {
    public let inner: UnigramTokenizer
    public let audioTokenBase: Int32       // id of <audio_0> = 65536
    public let vq06Offset: Int32 = 1024

    public init(_ inner: UnigramTokenizer) throws {
        self.inner = inner
        guard let base = inner.id("<audio_0>") else { throw StepAudioEditXError.badConfig("tokenizer has no <audio_0>") }
        audioTokenBase = base
    }

    public static func load(bundle: EditXBundle) throws -> EditXTokenizer {
        try EditXTokenizer(try UnigramTokenizer(tokenizerJSON: bundle.file("tokenizer.json")))
    }

    public func encode(_ text: String) -> [Int32] { inner.encode(text) }
    public func decode(_ ids: [Int32]) -> String { inner.decode(ids) }

    /// `render_messages`
    public static func render(_ messages: [ChatMessage], addGenerationPrompt: Bool) -> String {
        var parts = [String]()
        if messages.first?.role == "system" { parts.append("<s>") }
        for (i, m) in messages.enumerated() {
            let role = m.role == "user" ? "human" : m.role
            parts.append("<|BOT|> \(role)\n\(m.content)")
            if i != messages.count - 1 || m.role != "assistant" { parts.append("<|EOT|>") }
        }
        if addGenerationPrompt { parts.append("<|BOT|> assistant\n") }
        return parts.joined()
    }

    /// `interleave_step_audio_tokens` on prompt tokens (vq02 raw, vq06 + 1024): 2 vq02 then 3 vq06 per group.
    public static func packPromptTokens(vq02: [Int32], vq06: [Int32], vq06Offset: Int32 = 1024) -> [Int32] {
        let groups = min(vq02.count / 2, vq06.count / 3)
        var mixed = [Int32](); mixed.reserveCapacity(groups * 5)
        for g in 0 ..< groups {
            mixed += vq02[(g * 2) ..< (g * 2 + 2)]
            mixed += vq06[(g * 3) ..< (g * 3 + 3)].map { $0 + vq06Offset }
        }
        return mixed
    }

    public static func audioTokenString(_ promptTokens: [Int32]) -> String { promptTokens.map { "<audio_\($0)>" }.joined() }

    public func editPromptIds(instruction: String, audioTokenString: String) -> [Int32] {
        encode(Self.render([ChatMessage(role: "system", content: EditPrompts.editSystemPrompt),
                            ChatMessage(role: "user", content: "\(instruction)\n\(audioTokenString)\n")], addGenerationPrompt: true))
    }

    public func clonePromptIds(speaker: String, promptText: String, promptWavTokens: String, targetText: String) -> [Int32] {
        encode(Self.render([ChatMessage(role: "system", content: EditPrompts.cloneSystemPrompt(speaker: speaker, promptText: promptText, promptWavTokens: promptWavTokens)),
                            ChatMessage(role: "user", content: targetText)], addGenerationPrompt: true))
    }
}
