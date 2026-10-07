import XCTest
@testable import StepAudioEditXCore
final class StructuralTests: XCTestCase {
    func testResampleIdentity() { XCTAssertEqual(Resample.sinc([1, 2, 3], from: 16000, to: 16000), [1, 2, 3]) }
}
