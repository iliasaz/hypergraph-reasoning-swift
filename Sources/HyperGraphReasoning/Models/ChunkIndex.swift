import Foundation

/// An index for O(1) lookup of text chunks by their ID.
///
/// This is the Swift equivalent of Python's `chunk_to_df` dictionary,
/// enabling fast retrieval of original text for citations and provenance.
///
/// Example:
/// ```swift
/// let index = ChunkIndex(chunks: chunks)
/// if let text = index["7a8b9c3d"] {
///     print("Source: \(text)")
/// }
/// ```
public struct ChunkIndex: Sendable, Codable {

    /// The underlying storage mapping chunk IDs to text content.
    private var storage: [String: String]

    /// Creates an empty chunk index.
    public init() {
        self.storage = [:]
    }

    /// Creates a chunk index from an array of TextChunks.
    ///
    /// If multiple chunks have the same ID (identical content), the first one is kept.
    ///
    /// - Parameter chunks: The chunks to index.
    public init(chunks: [TextChunk]) {
        self.storage = Dictionary(
            chunks.map { ($0.chunkID, $0.text) },
            uniquingKeysWith: { first, _ in first }
        )
    }

    /// Creates a chunk index from a dictionary.
    ///
    /// - Parameter dictionary: Mapping of chunk IDs to text content.
    public init(dictionary: [String: String]) {
        self.storage = dictionary
    }

    // MARK: - Lookup

    /// Retrieves the text content for a chunk ID.
    ///
    /// - Parameter chunkID: The chunk identifier.
    /// - Returns: The text content, or nil if not found.
    /// - Complexity: O(1)
    public subscript(chunkID: String) -> String? {
        storage[chunkID]
    }

    /// Retrieves the text content for a chunk ID.
    ///
    /// - Parameter chunkID: The chunk identifier.
    /// - Returns: The text content, or nil if not found.
    /// - Complexity: O(1)
    public func text(for chunkID: String) -> String? {
        storage[chunkID]
    }

    /// Retrieves a preview of the text content (first N characters).
    ///
    /// - Parameters:
    ///   - chunkID: The chunk identifier.
    ///   - maxLength: Maximum preview length. Default: 200.
    /// - Returns: The text preview, or nil if not found.
    public func preview(for chunkID: String, maxLength: Int = 200) -> String? {
        guard let text = storage[chunkID] else { return nil }
        if text.count <= maxLength {
            return text
        }
        return String(text.prefix(maxLength)) + "..."
    }

    // MARK: - Mutation

    /// Adds a chunk to the index.
    ///
    /// - Parameter chunk: The chunk to add.
    public mutating func add(_ chunk: TextChunk) {
        storage[chunk.chunkID] = chunk.text
    }

    /// Adds multiple chunks to the index.
    ///
    /// - Parameter chunks: The chunks to add.
    public mutating func add(contentsOf chunks: [TextChunk]) {
        for chunk in chunks {
            storage[chunk.chunkID] = chunk.text
        }
    }

    /// Removes a chunk from the index.
    ///
    /// - Parameter chunkID: The chunk ID to remove.
    /// - Returns: The removed text, or nil if not found.
    @discardableResult
    public mutating func remove(_ chunkID: String) -> String? {
        storage.removeValue(forKey: chunkID)
    }

    // MARK: - Properties

    /// The number of chunks in the index.
    public var count: Int {
        storage.count
    }

    /// Whether the index is empty.
    public var isEmpty: Bool {
        storage.isEmpty
    }

    /// All chunk IDs in the index.
    public var chunkIDs: [String] {
        Array(storage.keys)
    }

    // MARK: - Merge

    /// Merges another chunk index into this one.
    ///
    /// - Parameter other: The other index to merge.
    public mutating func merge(_ other: ChunkIndex) {
        storage.merge(other.storage) { _, new in new }
    }

    /// Returns a new index by merging this index with another.
    ///
    /// - Parameter other: The other index to merge.
    /// - Returns: A new merged index.
    public func merging(_ other: ChunkIndex) -> ChunkIndex {
        var result = self
        result.merge(other)
        return result
    }
}

// MARK: - File Operations

extension ChunkIndex {

    /// Saves the chunk index to a JSON file.
    ///
    /// - Parameter url: The file URL to save to.
    /// - Throws: Encoding or file system errors.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(storage)
        try data.write(to: url)
    }

    /// Loads a chunk index from a JSON file.
    ///
    /// - Parameter url: The file URL to load from.
    /// - Returns: The loaded chunk index.
    /// - Throws: Decoding or file system errors.
    public static func load(from url: URL) throws -> ChunkIndex {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let storage = try decoder.decode([String: String].self, from: data)
        return ChunkIndex(dictionary: storage)
    }
}

// MARK: - CustomStringConvertible

extension ChunkIndex: CustomStringConvertible {
    public var description: String {
        "ChunkIndex(\(count) chunks)"
    }
}

// MARK: - ExpressibleByDictionaryLiteral

extension ChunkIndex: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, String)...) {
        self.storage = Dictionary(uniqueKeysWithValues: elements)
    }
}
