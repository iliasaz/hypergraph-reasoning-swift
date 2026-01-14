import Foundation
import CryptoKit

/// A text chunk with a unique identifier based on its content.
///
/// The chunk ID is an MD5 hash of the text content, ensuring that identical
/// text always produces the same ID. This matches the behavior of the Python
/// implementation.
public struct TextChunk: Hashable, Sendable, Codable {

    /// The text content of this chunk.
    public let text: String

    /// A unique identifier for this chunk, derived from MD5 hash of the text.
    public let chunkID: String

    /// Creates a new text chunk with an auto-generated ID.
    ///
    /// - Parameter text: The text content of the chunk.
    public init(text: String) {
        self.text = text
        self.chunkID = text.md5Hex()
    }

    /// Creates a new text chunk with a custom ID.
    ///
    /// - Parameters:
    ///   - text: The text content of the chunk.
    ///   - chunkID: A custom identifier for this chunk.
    public init(text: String, chunkID: String) {
        self.text = text
        self.chunkID = chunkID
    }

    /// The length of the text in characters.
    public var length: Int {
        text.count
    }

    /// Whether this chunk is empty.
    public var isEmpty: Bool {
        text.isEmpty
    }
}

// MARK: - CustomStringConvertible

extension TextChunk: CustomStringConvertible {
    public var description: String {
        let preview = text.prefix(50)
        let suffix = text.count > 50 ? "..." : ""
        return "TextChunk(\(chunkID.prefix(8))..., \"\(preview)\(suffix)\")"
    }
}

// MARK: - String MD5 Extension

extension String {
    /// Returns the MD5 hash of this string as a hexadecimal string.
    ///
    /// This matches the Python behavior: `md5(text.encode()).hexdigest()`
    public func md5Hex() -> String {
        let data = Data(self.utf8)
        let hash = Insecure.MD5.hash(data: data)
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}
