import Foundation

/// Container for node embeddings (vector representations of nodes).
///
/// Stores mappings from node identifiers to their embedding vectors,
/// supporting serialization and common operations like similarity calculation.
public struct NodeEmbeddings: Codable, Sendable {

    /// The underlying storage mapping node IDs to embedding vectors.
    public var embeddings: [String: [Float]]

    /// The dimensionality of the embeddings (assumes all have same dimension).
    public var dimension: Int? {
        embeddings.values.first?.count
    }

    /// Number of nodes with embeddings.
    public var count: Int {
        embeddings.count
    }

    /// Whether the embeddings are empty.
    public var isEmpty: Bool {
        embeddings.isEmpty
    }

    /// All node IDs that have embeddings.
    public var nodeIDs: Set<String> {
        Set(embeddings.keys)
    }

    /// Creates empty node embeddings.
    public init() {
        self.embeddings = [:]
    }

    /// Creates node embeddings from a dictionary.
    ///
    /// - Parameter embeddings: Dictionary mapping node IDs to vectors.
    public init(embeddings: [String: [Float]]) {
        self.embeddings = embeddings
    }

    /// Gets the embedding for a node.
    ///
    /// - Parameter nodeID: The node identifier.
    /// - Returns: The embedding vector, or nil if not found.
    public subscript(nodeID: String) -> [Float]? {
        get { embeddings[nodeID] }
        set { embeddings[nodeID] = newValue }
    }

    /// Adds or updates an embedding.
    ///
    /// - Parameters:
    ///   - nodeID: The node identifier.
    ///   - embedding: The embedding vector.
    public mutating func set(_ nodeID: String, embedding: [Float]) {
        embeddings[nodeID] = embedding
    }

    /// Removes an embedding.
    ///
    /// - Parameter nodeID: The node identifier.
    /// - Returns: The removed embedding, or nil if not found.
    @discardableResult
    public mutating func remove(_ nodeID: String) -> [Float]? {
        embeddings.removeValue(forKey: nodeID)
    }

    /// Merges another set of embeddings into this one.
    ///
    /// - Parameter other: The embeddings to merge.
    /// - Note: Existing embeddings are overwritten if there are conflicts.
    public mutating func merge(_ other: NodeEmbeddings) {
        for (nodeID, embedding) in other.embeddings {
            embeddings[nodeID] = embedding
        }
    }

    /// Returns embeddings only for nodes in the given set.
    ///
    /// - Parameter nodeIDs: The set of node IDs to keep.
    /// - Returns: A new NodeEmbeddings containing only the specified nodes.
    public func filtered(to nodeIDs: Set<String>) -> NodeEmbeddings {
        var filtered = [String: [Float]]()
        for nodeID in nodeIDs {
            if let embedding = embeddings[nodeID] {
                filtered[nodeID] = embedding
            }
        }
        return NodeEmbeddings(embeddings: filtered)
    }

    /// Returns nodes that are missing embeddings.
    ///
    /// - Parameter nodeIDs: The set of node IDs that should have embeddings.
    /// - Returns: Node IDs that don't have embeddings.
    public func missing(from nodeIDs: Set<String>) -> Set<String> {
        nodeIDs.subtracting(self.nodeIDs)
    }

    /// Removes embeddings for nodes not in the given set.
    ///
    /// - Parameter nodeIDs: The set of node IDs to keep.
    public mutating func prune(to nodeIDs: Set<String>) {
        let toRemove = self.nodeIDs.subtracting(nodeIDs)
        for nodeID in toRemove {
            embeddings.removeValue(forKey: nodeID)
        }
    }
}

// MARK: - Similarity Operations

extension NodeEmbeddings {

    /// Calculates cosine similarity between two embeddings.
    ///
    /// - Parameters:
    ///   - a: First embedding vector.
    ///   - b: Second embedding vector.
    /// - Returns: Cosine similarity in range [-1, 1], or nil if vectors have different dimensions.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float? {
        guard a.count == b.count, !a.isEmpty else { return nil }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denominator = sqrt(normA) * sqrt(normB)
        guard denominator > 0 else { return nil }

        return dotProduct / denominator
    }

    /// Finds the most similar nodes to a given embedding.
    ///
    /// - Parameters:
    ///   - embedding: The query embedding.
    ///   - topK: Maximum number of results.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of (nodeID, similarity) pairs, sorted by similarity descending.
    public func findSimilar(
        to embedding: [Float],
        topK: Int = 10,
        threshold: Float = 0.0
    ) -> [(nodeID: String, similarity: Float)] {
        var results = [(String, Float)]()

        for (nodeID, nodeEmbedding) in embeddings {
            if let similarity = Self.cosineSimilarity(embedding, nodeEmbedding),
               similarity >= threshold {
                results.append((nodeID, similarity))
            }
        }

        return results
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { ($0.0, $0.1) }
    }

    /// Finds nodes similar to a given node.
    ///
    /// - Parameters:
    ///   - nodeID: The query node ID.
    ///   - topK: Maximum number of results.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of (nodeID, similarity) pairs, excluding the query node.
    public func findSimilar(
        to nodeID: String,
        topK: Int = 10,
        threshold: Float = 0.0
    ) -> [(nodeID: String, similarity: Float)] {
        guard let embedding = embeddings[nodeID] else { return [] }
        return findSimilar(to: embedding, topK: topK + 1, threshold: threshold)
            .filter { $0.nodeID != nodeID }
            .prefix(topK)
            .map { $0 }
    }
}

// MARK: - CustomStringConvertible

extension NodeEmbeddings: CustomStringConvertible {
    public var description: String {
        let dimStr = dimension.map { "dim=\($0)" } ?? "empty"
        return "NodeEmbeddings(count: \(count), \(dimStr))"
    }
}
