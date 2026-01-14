import Foundation

/// Service for generating and managing node embeddings.
///
/// Provides methods to generate embeddings for nodes and update embeddings
/// when the hypergraph structure changes.
public actor EmbeddingService {

    /// The Ollama service for generating embeddings.
    private let ollamaService: OllamaService

    /// The embedding model to use.
    private let model: String

    /// Batch size for embedding generation.
    private let batchSize: Int

    /// Creates an embedding service.
    ///
    /// - Parameters:
    ///   - ollamaService: The Ollama service to use.
    ///   - model: The embedding model name. Defaults to "nomic-embed-text:v1.5".
    ///   - batchSize: Number of texts to embed in each batch. Defaults to 100.
    public init(
        ollamaService: OllamaService,
        model: String = "nomic-embed-text:v1.5",
        batchSize: Int = 100
    ) {
        self.ollamaService = ollamaService
        self.model = model
        self.batchSize = batchSize
    }

    // MARK: - Embedding Generation

    /// Generates embeddings for a set of nodes.
    ///
    /// - Parameter nodes: The node identifiers to generate embeddings for.
    /// - Returns: A dictionary mapping node IDs to their embeddings.
    public func generateEmbeddings(
        for nodes: [String]
    ) async throws -> [String: [Float]] {
        guard !nodes.isEmpty else {
            return [:]
        }

        var embeddings = [String: [Float]]()

        // Process in batches
        for batchStart in stride(from: 0, to: nodes.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, nodes.count)
            let batch = Array(nodes[batchStart..<batchEnd])

            let batchEmbeddings = try await ollamaService.embed(batch, model: model)

            for (index, node) in batch.enumerated() {
                if index < batchEmbeddings.count {
                    embeddings[node] = batchEmbeddings[index]
                }
            }
        }

        return embeddings
    }

    /// Generates embeddings for a set of nodes, returning a NodeEmbeddings object.
    ///
    /// - Parameter nodes: The node identifiers to generate embeddings for.
    /// - Returns: A NodeEmbeddings object with the generated embeddings.
    public func generateNodeEmbeddings(
        for nodes: [String]
    ) async throws -> NodeEmbeddings {
        let embeddingsDict = try await generateEmbeddings(for: nodes)
        return NodeEmbeddings(embeddings: embeddingsDict)
    }

    /// Generates embeddings for all nodes in a hypergraph.
    ///
    /// - Parameter hypergraph: The hypergraph to generate embeddings for.
    /// - Returns: Embeddings for all nodes in the hypergraph.
    public func generateEmbeddings(
        for hypergraph: Hypergraph<String, String>
    ) async throws -> NodeEmbeddings {
        let nodes = Array(hypergraph.nodes)
        return try await generateNodeEmbeddings(for: nodes)
    }

    // MARK: - Embedding Updates

    /// Updates embeddings to match the current hypergraph state.
    ///
    /// - Parameters:
    ///   - existing: Existing embeddings to update.
    ///   - hypergraph: The current hypergraph state.
    ///   - pruneOrphans: Whether to remove embeddings for nodes no longer in the graph.
    /// - Returns: Updated embeddings.
    public func updateEmbeddings(
        existing: NodeEmbeddings,
        for hypergraph: Hypergraph<String, String>,
        pruneOrphans: Bool = true
    ) async throws -> NodeEmbeddings {
        var updated = existing

        let graphNodes = hypergraph.nodes
        let existingNodes = existing.nodeIDs

        // Find nodes that need new embeddings
        let newNodes = graphNodes.subtracting(existingNodes)

        if !newNodes.isEmpty {
            let newEmbeddings = try await generateEmbeddings(for: Array(newNodes))
            for (node, embedding) in newEmbeddings {
                updated[node] = embedding
            }
        }

        // Optionally prune embeddings for nodes no longer in the graph
        if pruneOrphans {
            updated.prune(to: graphNodes)
        }

        return updated
    }

    /// Updates embeddings, generating only for missing nodes.
    ///
    /// - Parameters:
    ///   - existing: Existing embeddings.
    ///   - nodes: Set of nodes that should have embeddings.
    /// - Returns: Updated embeddings with all nodes covered.
    public func ensureEmbeddings(
        existing: NodeEmbeddings,
        for nodes: Set<String>
    ) async throws -> NodeEmbeddings {
        let missing = existing.missing(from: nodes)

        guard !missing.isEmpty else {
            return existing
        }

        var updated = existing
        let newEmbeddings = try await generateEmbeddings(for: Array(missing))
        for (node, embedding) in newEmbeddings {
            updated[node] = embedding
        }

        return updated
    }
}

// MARK: - Similarity Operations

extension EmbeddingService {

    /// Finds similar nodes to a query.
    ///
    /// - Parameters:
    ///   - query: The query text.
    ///   - embeddings: The node embeddings to search.
    ///   - topK: Maximum number of results.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of (nodeID, similarity) pairs.
    public func findSimilarNodes(
        to query: String,
        in embeddings: NodeEmbeddings,
        topK: Int = 10,
        threshold: Float = 0.0
    ) async throws -> [(nodeID: String, similarity: Float)] {
        let queryEmbedding = try await ollamaService.embed(query, model: model)
        return embeddings.findSimilar(
            to: queryEmbedding,
            topK: topK,
            threshold: threshold
        )
    }

    /// Finds pairs of similar nodes above a threshold.
    ///
    /// This is useful for node merging/deduplication.
    ///
    /// - Parameters:
    ///   - embeddings: The node embeddings.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: Array of (node1, node2, similarity) tuples.
    public func findSimilarPairs(
        in embeddings: NodeEmbeddings,
        threshold: Float = 0.9
    ) -> [(node1: String, node2: String, similarity: Float)] {
        var pairs = [(String, String, Float)]()
        let nodes = Array(embeddings.nodeIDs)

        for i in 0..<nodes.count {
            for j in (i + 1)..<nodes.count {
                let node1 = nodes[i]
                let node2 = nodes[j]

                guard let emb1 = embeddings[node1],
                      let emb2 = embeddings[node2],
                      let similarity = NodeEmbeddings.cosineSimilarity(emb1, emb2),
                      similarity >= threshold else {
                    continue
                }

                pairs.append((node1, node2, similarity))
            }
        }

        return pairs.sorted { $0.2 > $1.2 }
    }
}
