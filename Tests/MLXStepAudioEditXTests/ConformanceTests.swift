// ConformanceTests.swift — the offline conformance gate for MLXStepAudioEditX: manifest (C1 / C7 / C8 / C-memory /
// provenance), the MAT gate per tier (auto-materialization declarations), the CAN gate (pre-cancelled run propagation,
// checkpoint cadence), the unloaded refusal, and the request plane (operations, label vocabularies, paralinguistic
// tags, trimSilence → vad). Nothing here evals a kernel: the live run is `editx-gates --validate` (GPU).

import Foundation
import MLX
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXStepAudioEditX
@testable import StepAudioEditXCore

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — Apache-2.0 weights with the Paraformer encoder under the FunASR model licence (AB-R-0409), MIT port code;
    /// the default policy judges both and admits.
    func testLicenseDeclaresBothWeightLicencesAndAdmits() {
        let license = StepAudioEditXPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.additionalWeightLicenses, [.funasrModel])
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
    }

    /// C-memory — split footprints for the two published tiers, at least what `--validate` measured through the engine.
    func testFootprintsAreTheMeasuredTiers() {
        let footprints = StepAudioEditXPackage.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.bf16, .int8])
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 10_190_000_000)       // bf16 phys after load (warm-up incl.)
        XCTAssertGreaterThanOrEqual(footprints[1].residentBytes, 6_940_000_000)        // int8 phys after load (warm-up incl.)
        for f in footprints { XCTAssertGreaterThanOrEqual(f.peakActivationBytes, 1_430_000_000) }   // measured through the engine (--validate, 0.2.1)
    }

    /// C1 — the package serves `speechEdit` and nothing else, and its surface is born declared (1.50.0).
    func testCapabilityIsSpeechEditWithAFullDeclaration() {
        XCTAssertEqual(Set(StepAudioEditXPackage.manifest.capabilities), [.speechEdit])
        let surface = StepAudioEditXPackage.manifest.surfaces.first { $0.capability == .speechEdit }
        let controls = surface?.speechEditControls
        XCTAssertEqual(controls?.operations, [.emotion, .style, .paralinguistic, .denoise, .trimSilence])
        XCTAssertEqual(controls?.emotionLabels.count, 15)
        XCTAssertEqual(controls?.styleLabels.count, 33)
        XCTAssertEqual(controls?.paralinguisticTags.count, 10)
        XCTAssertEqual(controls?.maxInputSeconds, 90)
        XCTAssertTrue(controls?.emotionLabels.contains("remove") ?? false)
    }

    func testProvenancePinsUpstream() {
        XCTAssertEqual(StepAudioEditXPackage.manifest.provenance.sourceRepo, "stepfun-ai/Step-Audio-EditX")
        XCTAssertEqual(StepAudioEditXPackage.manifest.provenance.revision.count, 40)
    }
}

final class MaterializationConformanceTests: XCTestCase {

    /// MAT-1..5: a fresh (dir-less) configuration reports its one source missing; an explicit path satisfies — for both
    /// published tiers and for the fp32 parity precision (which upcasts the bf16 bundle).
    func testMaterializationGatePerTier() throws {
        for quant in [Quant.bf16, .int8, .fp32] {
            let satisfied = try satisfiedConfiguration(quant: quant)
            defer { try? FileManager.default.removeItem(at: satisfied.modelDirectory!) }
            let report = MaterializationConformance.check(
                freshConfiguration: StepAudioEditXConfiguration(quant: quant, modelsRootDirectory: emptyStoreRoot()),
                satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(quant): \(report.summary)")
        }
    }

    /// The tiers materialize the mlx-community repos named by fleet convention (<upstream name>-<tier>).
    func testPublishedRepos() {
        XCTAssertEqual(StepAudioEditXConfiguration().weightSources.map(\.repo), ["mlx-community/Step-Audio-EditX-bf16"])
        XCTAssertEqual(StepAudioEditXConfiguration(quant: .int8).weightSources.map(\.repo), ["mlx-community/Step-Audio-EditX-8bit"])
        XCTAssertEqual(StepAudioEditXConfiguration(quant: .fp32).weightSources.map(\.repo), ["mlx-community/Step-Audio-EditX-bf16"])
        XCTAssertEqual(StepAudioEditXConfiguration().quant, .bf16)
    }

    /// The declared file list is the bundle contract `EditXPipeline.load` validates.
    func testDeclaredFilesAreTheBundleContract() {
        XCTAssertEqual(Set(StepAudioEditXConfiguration.files), Set(EditXBundle.requiredFiles))
        XCTAssertTrue(StepAudioEditXConfiguration.files.contains("model.safetensors"))
        XCTAssertTrue(StepAudioEditXConfiguration.files.contains("tokenizer.json"))
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = StepAudioEditXConfiguration(quant: .int8, modelDirectory: URL(fileURLWithPath: "/tmp/x"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"))
        let decoded = try JSONDecoder().decode(StepAudioEditXConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.modelDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.repo, c.repo)
        XCTAssertEqual(decoded.quant, .int8)
    }

    /// The load-time warm-up is on by default, survives Codable, and a 0.2.0 configuration without the key decodes on.
    func testWarmUpDefaultsOnAndDecodesFromOlderConfigurations() throws {
        XCTAssertTrue(StepAudioEditXConfiguration().warmUp)
        let off = try JSONDecoder().decode(StepAudioEditXConfiguration.self,
                                           from: JSONEncoder().encode(StepAudioEditXConfiguration(warmUp: false)))
        XCTAssertFalse(off.warmUp)
        let older = Data(#"{"repo":"mlx-community/Step-Audio-EditX-bf16","quant":"bf16"}"#.utf8)
        XCTAssertTrue(try JSONDecoder().decode(StepAudioEditXConfiguration.self, from: older).warmUp)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("editx-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(quant: Quant) throws -> StepAudioEditXConfiguration {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("editx-explicit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data().write(to: dir.appending(path: StepAudioEditXConfiguration.probeFile))
        return StepAudioEditXConfiguration(quant: quant, modelDirectory: dir)
    }
}

final class CancellationConformanceTests: XCTestCase {

    private func probe() -> SpeechEditRequest {
        SpeechEditRequest(audio: Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24_000, channels: 1),
                          transcript: "cancellation probe", edit: .emotion("happy"))
    }

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = StepAudioEditXPackage(configuration: StepAudioEditXConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: probe())
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every generated token (RunProgress on the same seam) and at every stage boundary.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: StepAudioEditXPackage.manifest,
            posture: .cadence([.init(phase: .generate, unit: .token, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {

    private func request(_ edit: SpeechEditOperation, transcript: String = "We don't have much time.") -> SpeechEditRequest {
        SpeechEditRequest(audio: Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24_000, channels: 1), transcript: transcript, edit: edit)
    }

    func testUnloadedPackageRejectsLegibly() async {
        let package = StepAudioEditXPackage(configuration: StepAudioEditXConfiguration())
        do {
            _ = try await package.run(request(.emotion("happy")))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    /// Every declared operation maps 1:1 onto the Core's edit; `trimSilence` is upstream's `vad`; the output transcript
    /// is the request's, or the paralinguistic target with its tags.
    func testOperationsMapOntoTheCore() throws {
        XCTAssertEqual(try StepAudioEditXPackage.plan(request(.emotion("angry"))).edit, .emotion("angry"))
        XCTAssertEqual(try StepAudioEditXPackage.plan(request(.style("whisper"))).edit, .style("whisper"))
        XCTAssertEqual(try StepAudioEditXPackage.plan(request(.denoise)).edit, .denoise)
        XCTAssertEqual(try StepAudioEditXPackage.plan(request(.trimSilence)).edit, .vad)
        let p = try StepAudioEditXPackage.plan(request(.paralinguistic(targetTranscript: "We don't[Laughter] have much time.")))
        XCTAssertEqual(p.edit, .paralinguistic(targetText: "We don't[Laughter] have much time."))
        XCTAssertEqual(p.outputTranscript, "We don't[Laughter] have much time.")
        XCTAssertEqual(try StepAudioEditXPackage.plan(request(.emotion("remove"))).outputTranscript, "We don't have much time.")
    }

    /// An undeclared label or tag is refused legibly (the engine refuses labels before admission; tags are the package's).
    func testUndeclaredLabelsAndTagsAreRefused() {
        XCTAssertThrowsError(try StepAudioEditXPackage.plan(request(.emotion("furious"))))
        XCTAssertThrowsError(try StepAudioEditXPackage.plan(request(.style("pirate"))))
        XCTAssertThrowsError(try StepAudioEditXPackage.plan(request(.paralinguistic(targetTranscript: "no tags here"))))
        XCTAssertThrowsError(try StepAudioEditXPackage.plan(request(.paralinguistic(targetTranscript: "a [Giggle] here"))))
        XCTAssertThrowsError(try StepAudioEditXPackage.plan(request(.emotion("happy"), transcript: "   ")))
    }

    func testTagScan() {
        XCTAssertEqual(StepAudioEditXPackage.tags(in: "Great[Laughter], the weather is so nice today[Surprise-ah]."), ["[Laughter]", "[Surprise-ah]"])
        XCTAssertEqual(StepAudioEditXPackage.tags(in: "no tags"), [])
    }
}

final class CoreStructuralTests: XCTestCase {

    func testResampleIdentity() { XCTAssertEqual(Resample.sinc([1, 2, 3], from: 16000, to: 16000), [1, 2, 3]) }

    /// The warm-up signal: one second at 24 kHz, finite, peak 0.3, faded at both ends — a take the package admits.
    func testWarmUpSignalIsOneBoundedSecond() {
        let s = EditXPipeline.warmUpSignal()
        XCTAssertEqual(s.count, 24_000)
        XCTAssertTrue(s.allSatisfy(\.isFinite))
        XCTAssertEqual(s.map(abs).max() ?? 0, 0.3, accuracy: 1e-5)
        XCTAssertEqual(s[0], 0, accuracy: 1e-6)
        XCTAssertEqual(s[s.count - 1], 0, accuracy: 1e-6)
        XCTAssertGreaterThan(s[12_000 ... 12_240].map(abs).max() ?? 0, 0.05)   // voiced in the middle, not a fade
    }

    /// The 48-head sqrt-ALiBi slope table: 32 powers of 2^(−8/32) then 16 odd powers of 2^(−4/32) (modeling_step1.py).
    func testAlibiSlopes() {
        let s = AlibiSlopes(numHeads: 48).slopes.asArray(Float.self)
        XCTAssertEqual(s.count, 48)
        XCTAssertEqual(s[0], Float(pow(2.0, -8.0 / 32)), accuracy: 1e-7)
        XCTAssertEqual(s[31], Float(pow(2.0, -8.0)), accuracy: 1e-7)
        XCTAssertEqual(s[32], Float(pow(2.0, -4.0 / 32)), accuracy: 1e-7)
    }

    /// The prompt-token packing: 2 vq02 then 3 vq06 (+1024) per group, the tail dropped.
    func testPromptTokenPacking() {
        XCTAssertEqual(EditXTokenizer.packPromptTokens(vq02: [1, 2, 3, 4, 5], vq06: [10, 11, 12, 13, 14, 15, 16]), [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039])
    }
}
