import Testing
import Foundation
@testable import HyperGraphReasoning

@Suite("Accelerate Vector Ops Tests")
struct AccelerateVectorOpsTests {

    @Test("Cosine similarity of identical vectors is 1")
    func cosineSimilarityIdentical() {
        let a: [Float] = [1, 2, 3, 4, 5]
        let similarity = AccelerateVectorOps.cosineSimilarity(a, a)
        #expect(abs(similarity - 1.0) < 0.0001)
    }

    @Test("Cosine similarity of orthogonal vectors is 0")
    func cosineSimilarityOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        let similarity = AccelerateVectorOps.cosineSimilarity(a, b)
        #expect(abs(similarity) < 0.0001)
    }

    @Test("Cosine similarity of opposite vectors is -1")
    func cosineSimilarityOpposite() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [-1, -2, -3]
        let similarity = AccelerateVectorOps.cosineSimilarity(a, b)
        #expect(abs(similarity + 1.0) < 0.0001)
    }

    @Test("Similarity matrix is symmetric")
    func similarityMatrixSymmetric() {
        let embeddings: [[Float]] = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ]
        let matrix = AccelerateVectorOps.cosineSimilarityMatrix(embeddings)

        #expect(matrix.count == 3)
        for i in 0..<3 {
            for j in 0..<3 {
                #expect(abs(matrix[i][j] - matrix[j][i]) < 0.0001)
            }
        }
    }

    @Test("Similarity matrix diagonal is 1")
    func similarityMatrixDiagonal() {
        let embeddings: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        let matrix = AccelerateVectorOps.cosineSimilarityMatrix(embeddings)

        for i in 0..<3 {
            #expect(abs(matrix[i][i] - 1.0) < 0.0001)
        }
    }

    @Test("Find similar pairs with high threshold")
    func findSimilarPairsHighThreshold() {
        let embeddings: [[Float]] = [
            [1, 0, 0],       // 0
            [0.99, 0.1, 0],  // 1 - very similar to 0
            [0, 1, 0],       // 2 - orthogonal to 0
            [0, 0, 1]        // 3 - orthogonal to 0
        ]

        let pairs = AccelerateVectorOps.findSimilarPairs(
            embeddings: embeddings,
            threshold: 0.9
        )

        // Should find pair (0, 1) with high similarity
        #expect(pairs.count >= 1)
        let firstPair = pairs.first!
        #expect((firstPair.i == 0 && firstPair.j == 1) || (firstPair.i == 1 && firstPair.j == 0))
    }

    @Test("L2 distance of same vector is 0")
    func l2DistanceSame() {
        let a: [Float] = [1, 2, 3]
        let distance = AccelerateVectorOps.l2Distance(a, a)
        #expect(abs(distance) < 0.0001)
    }

    @Test("Normalize produces unit vector")
    func normalizeUnitVector() {
        let a: [Float] = [3, 4, 0]
        let normalized = AccelerateVectorOps.normalize(a)

        var normSq: Float = 0
        for x in normalized {
            normSq += x * x
        }
        #expect(abs(sqrt(normSq) - 1.0) < 0.0001)
    }
}

@Suite("Hypergraph Simplifier Tests")
struct HypergraphSimplifierTests {

    @Test("No merging when similarity below threshold")
    func noMergingBelowThreshold() {
        // Create hypergraph with distinct nodes
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "B", "C"]))
        hypergraph.addEdge("e2", nodes: Set(["B", "C", "D"]))

        // Create embeddings with low similarity
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0, 0, 0]
        embeddings["B"] = [0, 1, 0, 0]
        embeddings["C"] = [0, 0, 1, 0]
        embeddings["D"] = [0, 0, 0, 1]

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        #expect(result.mergeCount == 0)
        #expect(result.nodesRemoved == 0)
        #expect(result.hypergraph.nodeCount == 4)
    }

    @Test("Merging similar nodes keeps higher degree")
    func mergeSimilarNodesKeepHigherDegree() {
        // Create hypergraph where "A" has degree 2, "B" has degree 1
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "C"]))
        hypergraph.addEdge("e2", nodes: Set(["A", "D"]))
        hypergraph.addEdge("e3", nodes: Set(["B", "E"]))

        // Create embeddings where A and B are very similar
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0, 0]
        embeddings["B"] = [0.99, 0.1, 0]  // Very similar to A
        embeddings["C"] = [0, 1, 0]
        embeddings["D"] = [0, 0, 1]
        embeddings["E"] = [0.5, 0.5, 0]

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        // A should be kept (degree 2), B should be removed (degree 1)
        #expect(result.mergeCount == 1)
        #expect(result.hypergraph.nodes.contains("A"))
        #expect(!result.hypergraph.nodes.contains("B"))
        #expect(result.mergeHistory.first?.keptNode == "A")
        #expect(result.mergeHistory.first?.removedNode == "B")
    }

    @Test("Edges become redirected after merge")
    func edgesRedirectedAfterMerge() {
        // Create hypergraph where A has higher degree than B
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "C"]))
        hypergraph.addEdge("e2", nodes: Set(["B", "D"]))  // B will be merged into A
        hypergraph.addEdge("e3", nodes: Set(["A", "E"]))  // A has degree 2, B has degree 1

        // Create embeddings where A and B are identical
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0, 0]
        embeddings["B"] = [1, 0, 0]  // Identical to A
        embeddings["C"] = [0, 1, 0]
        embeddings["D"] = [0, 0, 1]
        embeddings["E"] = [0.5, 0.5, 0]

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        // After merge, e2 should contain A instead of B
        if let e2Nodes = result.hypergraph.nodes(in: "e2") {
            #expect(e2Nodes.contains("A"))
            #expect(!e2Nodes.contains("B"))
        }
    }

    @Test("Singleton edges are removed")
    func singletonEdgesRemoved() {
        // Create hypergraph where merging creates a singleton edge
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "B"]))  // A and B are similar
        hypergraph.addEdge("e2", nodes: Set(["C", "D"]))

        // A and B are identical
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0]
        embeddings["B"] = [1, 0]
        embeddings["C"] = [0, 1]
        embeddings["D"] = [-1, 0]

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        // e1 should be removed since it becomes {A} after B is merged
        #expect(result.edgesRemoved >= 1)
        #expect(result.hypergraph.nodes(in: "e1") == nil)
    }

    @Test("Embeddings are pruned for removed nodes")
    func embeddingsPrunedForRemovedNodes() {
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "B", "C"]))
        hypergraph.addEdge("e2", nodes: Set(["A", "D"]))  // A has degree 2, B has degree 1

        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0]
        embeddings["B"] = [1, 0]  // Same as A
        embeddings["C"] = [0, 1]
        embeddings["D"] = [0.5, 0.5]

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        // B's embedding should be removed (A kept because higher degree)
        #expect(result.embeddings["A"] != nil)
        #expect(result.embeddings["B"] == nil)
        #expect(result.embeddings["C"] != nil)
        #expect(result.embeddings["D"] != nil)
    }

    @Test("Find merge candidates returns sorted pairs")
    func findMergeCandidatesSorted() {
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "B", "C", "D"]))

        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0, 0]
        embeddings["B"] = [0.95, 0.3, 0]  // Similar to A
        embeddings["C"] = [0.99, 0.1, 0]  // More similar to A
        embeddings["D"] = [0, 1, 0]

        let simplifier = HypergraphSimplifier()
        let candidates = simplifier.findMergeCandidates(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9
        )

        // Should be sorted by similarity descending
        for i in 0..<(candidates.count - 1) {
            #expect(candidates[i].similarity >= candidates[i + 1].similarity)
        }
    }

    @Test("Exclude suffixes are respected")
    func excludeSuffixesRespected() {
        var hypergraph = StringHypergraph()
        hypergraph.addEdge("e1", nodes: Set(["A", "B", "image.png"]))

        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1, 0]
        embeddings["B"] = [1, 0]  // Same as A
        embeddings["image.png"] = [1, 0]  // Same as A, but should be excluded

        let simplifier = HypergraphSimplifier()
        let result = simplifier.simplify(
            hypergraph: hypergraph,
            embeddings: embeddings,
            similarityThreshold: 0.9,
            excludeSuffixes: [".png"]
        )

        // A and B should be merged, but image.png should remain
        #expect(result.hypergraph.nodes.contains("image.png"))
    }
}
