import Foundation

/// Metadata for a hyperedge, tracking its origin and structure.
///
/// This corresponds to the rows in the Python `chunk_df` DataFrame that stores:
/// - edge: The hyperedge identifier
/// - nodes: All nodes in the edge
/// - source: Original source entities from the event
/// - target: Original target entities from the event
/// - chunk: The chunk ID this edge was extracted from
public struct ChunkMetadata: Codable, Sendable, Hashable {

    /// The hyperedge identifier.
    public let edge: String

    /// All nodes contained in this edge.
    public let nodes: Set<String>

    /// The source entities from the original event.
    public let source: [String]

    /// The target entities from the original event.
    public let target: [String]

    /// The ID of the text chunk this edge was extracted from.
    public let chunkID: String

    /// Creates new chunk metadata.
    ///
    /// - Parameters:
    ///   - edge: The hyperedge identifier.
    ///   - nodes: All nodes in the edge.
    ///   - source: Source entities.
    ///   - target: Target entities.
    ///   - chunkID: The source chunk ID.
    public init(
        edge: String,
        nodes: Set<String>,
        source: [String],
        target: [String],
        chunkID: String
    ) {
        self.edge = edge
        self.nodes = nodes
        self.source = source
        self.target = target
        self.chunkID = chunkID
    }

    /// Creates metadata from an event.
    ///
    /// - Parameters:
    ///   - event: The source event.
    ///   - edgeID: The assigned edge ID.
    ///   - chunkID: The source chunk ID.
    public init(event: Event, edgeID: String, chunkID: String) {
        self.edge = edgeID
        self.nodes = event.allNodes
        self.source = event.source
        self.target = event.target
        self.chunkID = chunkID
    }
}

// MARK: - CustomStringConvertible

extension ChunkMetadata: CustomStringConvertible {
    public var description: String {
        "ChunkMetadata(edge: \(edge), nodes: \(nodes.count), chunk: \(chunkID.prefix(8))...)"
    }
}

// MARK: - Collection of ChunkMetadata

extension Array where Element == ChunkMetadata {

    /// All unique edges in this collection.
    public var allEdges: Set<String> {
        Set(map(\.edge))
    }

    /// All unique nodes across all metadata entries.
    public var allNodes: Set<String> {
        reduce(into: Set<String>()) { result, meta in
            result.formUnion(meta.nodes)
        }
    }

    /// All unique chunk IDs.
    public var allChunkIDs: Set<String> {
        Set(map(\.chunkID))
    }

    /// Filters metadata by chunk ID.
    ///
    /// - Parameter chunkID: The chunk ID to filter by.
    /// - Returns: Metadata entries from the specified chunk.
    public func filter(byChunkID chunkID: String) -> [ChunkMetadata] {
        filter { $0.chunkID == chunkID }
    }

    /// Groups metadata by chunk ID.
    ///
    /// - Returns: Dictionary mapping chunk IDs to their metadata entries.
    public func groupedByChunk() -> [String: [ChunkMetadata]] {
        Dictionary(grouping: self, by: \.chunkID)
    }
}
