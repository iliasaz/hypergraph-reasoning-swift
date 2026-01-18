import Foundation

/// Result of a hypergraph simplification operation.
public struct SimplificationResult: Sendable {
    /// The simplified hypergraph.
    public let hypergraph: StringHypergraph

    /// Updated embeddings with merged nodes removed.
    public let embeddings: NodeEmbeddings

    /// Number of merge operations performed.
    public let mergeCount: Int

    /// Number of nodes removed.
    public let nodesRemoved: Int

    /// Number of edges removed (due to becoming singletons or empty).
    public let edgesRemoved: Int

    /// Number of embeddings recomputed for keeper nodes.
    public let embeddingsRecomputed: Int

    /// Details of each merge operation.
    public let mergeHistory: [MergeRecord]

    public init(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        mergeCount: Int,
        nodesRemoved: Int,
        edgesRemoved: Int,
        embeddingsRecomputed: Int = 0,
        mergeHistory: [MergeRecord]
    ) {
        self.hypergraph = hypergraph
        self.embeddings = embeddings
        self.mergeCount = mergeCount
        self.nodesRemoved = nodesRemoved
        self.edgesRemoved = edgesRemoved
        self.embeddingsRecomputed = embeddingsRecomputed
        self.mergeHistory = mergeHistory
    }
}

/// Record of a single merge operation.
public struct MergeRecord: Sendable, Codable {
    /// The node that was kept.
    public let keptNode: String

    /// The node that was removed.
    public let removedNode: String

    /// Cosine similarity between the two nodes.
    public let similarity: Float

    /// Degree of the kept node (before merge).
    public let keptNodeDegree: Int

    /// Degree of the removed node (before merge).
    public let removedNodeDegree: Int

    public init(
        keptNode: String,
        removedNode: String,
        similarity: Float,
        keptNodeDegree: Int,
        removedNodeDegree: Int
    ) {
        self.keptNode = keptNode
        self.removedNode = removedNode
        self.similarity = similarity
        self.keptNodeDegree = keptNodeDegree
        self.removedNodeDegree = removedNodeDegree
    }
}

/// Simplifies hypergraphs by merging similar nodes based on embedding similarity.
///
/// Based on the algorithm from the HyperGraphReasoning Python project:
/// 1. Build a cosine similarity matrix for all node embeddings
/// 2. Find pairs of nodes with similarity above threshold
/// 3. For each pair, keep the node with higher degree
/// 4. Rebuild the hypergraph with merged nodes collapsed
/// 5. Remove edges that become singletons or empty
/// 6. Update embeddings by removing merged nodes
public struct HypergraphSimplifier: Sendable {

    /// Default similarity threshold for merging (0.9 = 90% similar).
    public static let defaultSimilarityThreshold: Float = 0.9

    /// Minimum edge size after merging (edges with fewer nodes are removed).
    public static let minimumEdgeSize: Int = 2

    /// Creates a new hypergraph simplifier.
    public init() {}

    /// Simplify a hypergraph by merging similar nodes.
    ///
    /// - Parameters:
    ///   - hypergraph: The hypergraph to simplify.
    ///   - embeddings: Node embeddings for similarity computation.
    ///   - similarityThreshold: Minimum cosine similarity to merge nodes (default 0.9).
    ///   - excludeSuffixes: Node suffixes to exclude from merging (e.g., [".png"] for images).
    /// - Returns: SimplificationResult containing the simplified hypergraph and updated embeddings.
    public func simplify(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        similarityThreshold: Float = defaultSimilarityThreshold,
        excludeSuffixes: [String] = []
    ) -> SimplificationResult {
        // 1. Get nodes with embeddings, excluding special nodes
        let allNodes = Array(hypergraph.nodes)
        let eligibleNodes = allNodes.filter { node in
            // Exclude nodes with specified suffixes
            let isExcluded = excludeSuffixes.contains { suffix in
                node.hasSuffix(suffix)
            }
            // Must have an embedding
            let hasEmbedding = embeddings[node] != nil
            return !isExcluded && hasEmbedding
        }

        guard eligibleNodes.count > 1 else {
            return SimplificationResult(
                hypergraph: hypergraph,
                embeddings: embeddings,
                mergeCount: 0,
                nodesRemoved: 0,
                edgesRemoved: 0,
                mergeHistory: []
            )
        }

        // 2. Build embeddings matrix
        let embeddingVectors = eligibleNodes.compactMap { embeddings[$0] }
        guard embeddingVectors.count == eligibleNodes.count else {
            // Some nodes are missing embeddings
            return SimplificationResult(
                hypergraph: hypergraph,
                embeddings: embeddings,
                mergeCount: 0,
                nodesRemoved: 0,
                edgesRemoved: 0,
                mergeHistory: []
            )
        }

        // 3. Find similar pairs using Accelerate
        let similarPairs = AccelerateVectorOps.findSimilarPairs(
            embeddings: embeddingVectors,
            threshold: similarityThreshold
        )

        guard !similarPairs.isEmpty else {
            return SimplificationResult(
                hypergraph: hypergraph,
                embeddings: embeddings,
                mergeCount: 0,
                nodesRemoved: 0,
                edgesRemoved: 0,
                mergeHistory: []
            )
        }

        // 4. Build merge plan: for each pair, keep node with higher degree
        var nodeMapping: [String: String] = [:]  // removed -> kept
        var mergedNodes = Set<String>()
        var mergeHistory: [MergeRecord] = []

        for pair in similarPairs {
            let nodeI = eligibleNodes[pair.i]
            let nodeJ = eligibleNodes[pair.j]

            // Skip if either node is already being merged
            if mergedNodes.contains(nodeI) || mergedNodes.contains(nodeJ) {
                continue
            }

            let degreeI = hypergraph.degree(of: nodeI)
            let degreeJ = hypergraph.degree(of: nodeJ)

            // Keep the node with higher degree
            let (keepNode, removeNode, keepDegree, removeDegree): (String, String, Int, Int)
            if degreeI >= degreeJ {
                (keepNode, removeNode, keepDegree, removeDegree) = (nodeI, nodeJ, degreeI, degreeJ)
            } else {
                (keepNode, removeNode, keepDegree, removeDegree) = (nodeJ, nodeI, degreeJ, degreeI)
            }

            nodeMapping[removeNode] = keepNode
            mergedNodes.insert(removeNode)

            mergeHistory.append(MergeRecord(
                keptNode: keepNode,
                removedNode: removeNode,
                similarity: pair.similarity,
                keptNodeDegree: keepDegree,
                removedNodeDegree: removeDegree
            ))
        }

        // 5. Rebuild incidence dict with merged nodes collapsed
        var newIncidenceDict: [String: Set<String>] = [:]
        var edgesRemoved = 0

        for (edgeID, members) in hypergraph.incidenceDict {
            // Map each member to its kept node (or itself if not merged)
            let newMembers = Set(members.map { node in
                nodeMapping[node] ?? node
            })

            // Only keep edges with at least minimumEdgeSize members
            if newMembers.count >= Self.minimumEdgeSize {
                newIncidenceDict[edgeID] = newMembers
            } else {
                edgesRemoved += 1
            }
        }

        let newHypergraph = StringHypergraph(incidenceDict: newIncidenceDict)

        // 6. Update embeddings: remove merged nodes
        var newEmbeddings = embeddings
        for removedNode in mergedNodes {
            newEmbeddings.remove(removedNode)
        }

        // Prune embeddings to only include nodes in the new hypergraph
        newEmbeddings.prune(to: newHypergraph.nodes)

        return SimplificationResult(
            hypergraph: newHypergraph,
            embeddings: newEmbeddings,
            mergeCount: mergeHistory.count,
            nodesRemoved: mergedNodes.count,
            edgesRemoved: edgesRemoved,
            embeddingsRecomputed: 0,
            mergeHistory: mergeHistory
        )
    }

    /// Simplify a hypergraph by merging similar nodes, with optional embedding recomputation.
    ///
    /// This async version supports recomputing embeddings for keeper nodes that absorbed
    /// other nodes during merging. This can improve downstream tasks by ensuring the
    /// embedding reflects the merged concept.
    ///
    /// - Parameters:
    ///   - hypergraph: The hypergraph to simplify.
    ///   - embeddings: Node embeddings for similarity computation.
    ///   - similarityThreshold: Minimum cosine similarity to merge nodes (default 0.9).
    ///   - excludeSuffixes: Node suffixes to exclude from merging (e.g., [".png"] for images).
    ///   - recomputeEmbeddings: Whether to recompute embeddings for keeper nodes.
    ///   - embeddingService: The embedding service to use for recomputation (required if recomputeEmbeddings is true).
    /// - Returns: SimplificationResult containing the simplified hypergraph and updated embeddings.
    public func simplify(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        similarityThreshold: Float = defaultSimilarityThreshold,
        excludeSuffixes: [String] = [],
        recomputeEmbeddings: Bool,
        embeddingService: EmbeddingService?
    ) async throws -> SimplificationResult {
        // First, run the synchronous simplification
        let initialResult = simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: similarityThreshold,
            excludeSuffixes: excludeSuffixes
        )

        // If no recomputation needed or no merges happened, return as-is
        guard recomputeEmbeddings,
              let service = embeddingService,
              !initialResult.mergeHistory.isEmpty else {
            return initialResult
        }

        // Find keeper nodes that received merges (need embedding recomputation)
        let keeperNodes = Set(initialResult.mergeHistory.map { $0.keptNode })

        // Filter to only nodes still in the graph (some might have been merged into others)
        let nodesToRecompute = keeperNodes.intersection(initialResult.hypergraph.nodes)

        guard !nodesToRecompute.isEmpty else {
            return initialResult
        }

        // Recompute embeddings for keeper nodes
        let recomputedEmbeddings = try await service.generateEmbeddings(for: Array(nodesToRecompute))

        // Update embeddings with recomputed values
        var updatedEmbeddings = initialResult.embeddings
        for (node, embedding) in recomputedEmbeddings {
            updatedEmbeddings[node] = embedding
        }

        return SimplificationResult(
            hypergraph: initialResult.hypergraph,
            embeddings: updatedEmbeddings,
            mergeCount: initialResult.mergeCount,
            nodesRemoved: initialResult.nodesRemoved,
            edgesRemoved: initialResult.edgesRemoved,
            embeddingsRecomputed: recomputedEmbeddings.count,
            mergeHistory: initialResult.mergeHistory
        )
    }

    /// Find potential merge candidates without actually merging.
    ///
    /// - Parameters:
    ///   - hypergraph: The hypergraph to analyze.
    ///   - embeddings: Node embeddings for similarity computation.
    ///   - similarityThreshold: Minimum cosine similarity to consider as candidates.
    /// - Returns: Array of potential merge pairs with their similarities.
    public func findMergeCandidates(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        similarityThreshold: Float = defaultSimilarityThreshold
    ) -> [(nodeA: String, nodeB: String, similarity: Float)] {
        let nodes = Array(hypergraph.nodes).filter { embeddings[$0] != nil }
        guard nodes.count > 1 else { return [] }

        let embeddingVectors = nodes.compactMap { embeddings[$0] }
        guard embeddingVectors.count == nodes.count else { return [] }

        let pairs = AccelerateVectorOps.findSimilarPairs(
            embeddings: embeddingVectors,
            threshold: similarityThreshold
        )

        return pairs.map { pair in
            (nodeA: nodes[pair.i], nodeB: nodes[pair.j], similarity: pair.similarity)
        }
    }
}

// MARK: - NodeEmbeddings Extension

extension NodeEmbeddings {
    /// Compute the cosine similarity matrix for all embeddings using Accelerate.
    ///
    /// - Returns: Dictionary mapping node pairs to their similarity.
    public func similarityMatrix() -> [String: [String: Float]] {
        let nodes = Array(embeddings.keys)
        let vectors = nodes.compactMap { embeddings[$0] }
        guard vectors.count == nodes.count else { return [:] }

        let matrix = AccelerateVectorOps.cosineSimilarityMatrix(vectors)

        var result: [String: [String: Float]] = [:]
        for (i, nodeI) in nodes.enumerated() {
            var row: [String: Float] = [:]
            for (j, nodeJ) in nodes.enumerated() {
                row[nodeJ] = matrix[i][j]
            }
            result[nodeI] = row
        }

        return result
    }
}
