import Foundation

/// Matches keywords to hypergraph nodes using embedding similarity.
///
/// Uses the pre-computed node embeddings to find the most semantically
/// similar nodes for each extracted keyword.
public struct NodeMatcher: Sendable {

    /// The node embeddings for similarity search.
    private let embeddings: NodeEmbeddings

    /// The embedding service for generating query embeddings.
    private let embeddingService: EmbeddingService

    /// Creates a node matcher.
    ///
    /// - Parameters:
    ///   - embeddings: Pre-computed node embeddings.
    ///   - embeddingService: Service for generating keyword embeddings.
    public init(
        embeddings: NodeEmbeddings,
        embeddingService: EmbeddingService
    ) {
        self.embeddings = embeddings
        self.embeddingService = embeddingService
    }

    // MARK: - Node Matching

    /// Finds matching nodes for a list of keywords.
    ///
    /// For each keyword, finds the top-K most similar nodes based on
    /// embedding cosine similarity.
    ///
    /// - Parameters:
    ///   - keywords: The keywords to match.
    ///   - topK: Maximum number of matches per keyword.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of node matches with similarity scores.
    public func findMatchingNodes(
        for keywords: [String],
        topK: Int = 5,
        threshold: Float = 0.5
    ) async throws -> [NodeMatch] {
        guard !keywords.isEmpty else { return [] }

        var allMatches: [NodeMatch] = []

        for keyword in keywords {
            let matches = try await findMatches(
                for: keyword,
                topK: topK,
                threshold: threshold
            )
            allMatches.append(contentsOf: matches)
        }

        // Deduplicate and sort by similarity
        return deduplicateMatches(allMatches)
    }

    /// Finds matching nodes for a single keyword.
    ///
    /// - Parameters:
    ///   - keyword: The keyword to match.
    ///   - topK: Maximum number of matches.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of node matches.
    public func findMatches(
        for keyword: String,
        topK: Int = 5,
        threshold: Float = 0.5
    ) async throws -> [NodeMatch] {
        // First, check for exact match (case-insensitive)
        let exactMatches = findExactMatches(for: keyword)
        if !exactMatches.isEmpty {
            return exactMatches
        }

        // Use embedding similarity
        let similarNodes = try await embeddingService.findSimilarNodes(
            to: keyword,
            in: embeddings,
            topK: topK,
            threshold: threshold
        )

        return similarNodes.map { nodeID, similarity in
            NodeMatch(
                node: nodeID,
                keyword: keyword,
                similarity: similarity,
                matchType: .embedding
            )
        }
    }

    /// Finds exact or substring matches in node names.
    ///
    /// - Parameter keyword: The keyword to search for.
    /// - Returns: Array of exact matches.
    public func findExactMatches(for keyword: String) -> [NodeMatch] {
        let lowercaseKeyword = keyword.lowercased()
        var matches: [NodeMatch] = []

        for nodeID in embeddings.nodeIDs {
            let lowercaseNode = nodeID.lowercased()

            if lowercaseNode == lowercaseKeyword {
                // Perfect match
                matches.append(NodeMatch(
                    node: nodeID,
                    keyword: keyword,
                    similarity: 1.0,
                    matchType: .exact
                ))
            } else if lowercaseNode.contains(lowercaseKeyword) ||
                      lowercaseKeyword.contains(lowercaseNode) {
                // Substring match
                let similarity = Float(min(lowercaseKeyword.count, lowercaseNode.count)) /
                                 Float(max(lowercaseKeyword.count, lowercaseNode.count))
                matches.append(NodeMatch(
                    node: nodeID,
                    keyword: keyword,
                    similarity: max(0.8, similarity),  // Boost substring matches
                    matchType: .substring
                ))
            }
        }

        return matches.sorted { $0.similarity > $1.similarity }
    }

    // MARK: - Batch Operations

    /// Finds the best matching node for each keyword.
    ///
    /// - Parameters:
    ///   - keywords: The keywords to match.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Dictionary mapping keywords to their best matching nodes.
    public func findBestMatches(
        for keywords: [String],
        threshold: Float = 0.5
    ) async throws -> [String: String] {
        var bestMatches: [String: String] = [:]

        for keyword in keywords {
            let matches = try await findMatches(
                for: keyword,
                topK: 1,
                threshold: threshold
            )
            if let best = matches.first {
                bestMatches[keyword] = best.node
            }
        }

        return bestMatches
    }

    /// Gets the unique matched nodes from a list of matches.
    ///
    /// - Parameter matches: The matches to extract nodes from.
    /// - Returns: Array of unique node IDs.
    public func uniqueNodes(from matches: [NodeMatch]) -> [String] {
        var seen = Set<String>()
        var nodes: [String] = []

        for match in matches.sorted(by: { $0.similarity > $1.similarity }) {
            if !seen.contains(match.node) {
                seen.insert(match.node)
                nodes.append(match.node)
            }
        }

        return nodes
    }

    // MARK: - Private Methods

    /// Deduplicates matches, keeping the highest similarity for each node.
    private func deduplicateMatches(_ matches: [NodeMatch]) -> [NodeMatch] {
        var bestByNode: [String: NodeMatch] = [:]

        for match in matches {
            if let existing = bestByNode[match.node] {
                if match.similarity > existing.similarity {
                    bestByNode[match.node] = match
                }
            } else {
                bestByNode[match.node] = match
            }
        }

        return bestByNode.values.sorted { $0.similarity > $1.similarity }
    }
}

// MARK: - Supporting Types

/// Represents a match between a keyword and a hypergraph node.
public struct NodeMatch: Sendable, Codable, Hashable {
    /// The matched node ID.
    public let node: String

    /// The keyword that matched this node.
    public let keyword: String

    /// The similarity score (0-1).
    public let similarity: Float

    /// How the match was found.
    public let matchType: MatchType

    /// Creates a node match.
    public init(
        node: String,
        keyword: String,
        similarity: Float,
        matchType: MatchType = .embedding
    ) {
        self.node = node
        self.keyword = keyword
        self.similarity = similarity
        self.matchType = matchType
    }

    /// How a match was determined.
    public enum MatchType: String, Sendable, Codable {
        /// Exact string match (case-insensitive).
        case exact
        /// Substring match.
        case substring
        /// Embedding similarity match.
        case embedding
    }
}

// MARK: - CustomStringConvertible

extension NodeMatch: CustomStringConvertible {
    public var description: String {
        "NodeMatch('\(keyword)' -> '\(node)', sim: \(String(format: "%.3f", similarity)), type: \(matchType.rawValue))"
    }
}

// MARK: - Collection Extensions

extension Array where Element == NodeMatch {
    /// Gets all unique node IDs from the matches.
    public var uniqueNodeIDs: [String] {
        var seen = Set<String>()
        var nodes: [String] = []

        for match in self.sorted(by: { $0.similarity > $1.similarity }) {
            if !seen.contains(match.node) {
                seen.insert(match.node)
                nodes.append(match.node)
            }
        }

        return nodes
    }

    /// Gets the total number of unique nodes.
    public var uniqueNodeCount: Int {
        Set(map(\.node)).count
    }

    /// Filters matches above a similarity threshold.
    public func filtered(threshold: Float) -> [NodeMatch] {
        filter { $0.similarity >= threshold }
    }
}
