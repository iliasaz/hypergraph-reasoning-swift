import Testing
import Foundation
@testable import HyperGraphReasoning

@Suite("Model Tests")
struct ModelTests {

    // MARK: - Event Tests

    @Test("Event creation")
    func testEventCreation() {
        let event = Event(
            source: ["A", "B"],
            target: ["C"],
            relation: "leads to"
        )

        #expect(event.source == ["A", "B"])
        #expect(event.target == ["C"])
        #expect(event.relation == "leads to")
    }

    @Test("Binary event creation")
    func testBinaryEvent() {
        let event = Event(source: "A", target: "B", relation: "is")

        #expect(event.source == ["A"])
        #expect(event.target == ["B"])
        #expect(event.isBinary)
    }

    @Test("Event all nodes")
    func testEventNodes() {
        let event = Event(
            source: ["A", "B"],
            target: ["C", "D"],
            relation: "relates to"
        )

        #expect(event.allNodes == ["A", "B", "C", "D"])
    }

    @Test("Event JSON encoding")
    func testEventEncoding() throws {
        let event = Event(
            source: ["X"],
            target: ["Y"],
            relation: "is"
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(event)
        let json = String(data: data, encoding: .utf8)!

        #expect(json.contains("\"relation\":\"is\""))
        #expect(json.contains("\"source\":[\"X\"]"))
        #expect(json.contains("\"target\":[\"Y\"]"))
    }

    @Test("Event JSON decoding")
    func testEventDecoding() throws {
        let json = """
        {
            "source": ["chitosan", "hydroxyapatite"],
            "relation": "compose",
            "target": ["nanocomposite"]
        }
        """

        let event = try JSONDecoder().decode(Event.self, from: json.data(using: .utf8)!)

        #expect(event.source == ["chitosan", "hydroxyapatite"])
        #expect(event.relation == "compose")
        #expect(event.target == ["nanocomposite"])
    }

    // MARK: - HypergraphJSON Tests

    @Test("HypergraphJSON empty")
    func testEmptyHypergraphJSON() {
        let hg = HypergraphJSON()

        #expect(hg.isEmpty)
        #expect(hg.count == 0)
    }

    @Test("HypergraphJSON decoding")
    func testHypergraphJSONDecoding() throws {
        let json = """
        {
            "events": [
                {"source": ["A"], "relation": "is", "target": ["B"]},
                {"source": ["B", "C"], "relation": "compose", "target": ["D"]}
            ]
        }
        """

        let hg = try JSONDecoder().decode(HypergraphJSON.self, from: json.data(using: .utf8)!)

        #expect(hg.count == 2)
        #expect(hg.allNodes == ["A", "B", "C", "D"])
        #expect(hg.allRelations == ["is", "compose"])
    }

    @Test("HypergraphJSON to Hypergraph conversion")
    func testToHypergraph() throws {
        let json = """
        {
            "events": [
                {"source": ["A"], "relation": "is", "target": ["B"]},
                {"source": ["C"], "relation": "has", "target": ["D"]}
            ]
        }
        """

        let hgJson = try JSONDecoder().decode(HypergraphJSON.self, from: json.data(using: .utf8)!)
        let (hypergraph, metadata) = hgJson.toHypergraph(chunkID: "test123")

        #expect(hypergraph.edgeCount == 2)
        #expect(hypergraph.nodeCount == 4)
        #expect(metadata.count == 2)
        #expect(metadata.allChunkIDs == ["test123"])
    }

    // MARK: - ChunkMetadata Tests

    @Test("ChunkMetadata creation")
    func testChunkMetadata() {
        let meta = ChunkMetadata(
            edge: "edge1",
            nodes: ["A", "B"],
            source: ["A"],
            target: ["B"],
            chunkID: "chunk123"
        )

        #expect(meta.edge == "edge1")
        #expect(meta.nodes == ["A", "B"])
        #expect(meta.chunkID == "chunk123")
    }

    @Test("ChunkMetadata from event")
    func testChunkMetadataFromEvent() {
        let event = Event(source: ["X"], target: ["Y"], relation: "relates")
        let meta = ChunkMetadata(event: event, edgeID: "e1", chunkID: "c1")

        #expect(meta.edge == "e1")
        #expect(meta.nodes == ["X", "Y"])
        #expect(meta.source == ["X"])
        #expect(meta.target == ["Y"])
        #expect(meta.chunkID == "c1")
    }

    // MARK: - NodeEmbeddings Tests

    @Test("NodeEmbeddings creation")
    func testNodeEmbeddings() {
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1.0, 2.0, 3.0]
        embeddings["B"] = [4.0, 5.0, 6.0]

        #expect(embeddings.count == 2)
        #expect(embeddings.dimension == 3)
        #expect(embeddings.nodeIDs == ["A", "B"])
    }

    @Test("NodeEmbeddings similarity")
    func testCosineSimilarity() {
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [1.0, 0.0, 0.0]
        let c: [Float] = [0.0, 1.0, 0.0]

        let simAB = NodeEmbeddings.cosineSimilarity(a, b)
        let simAC = NodeEmbeddings.cosineSimilarity(a, c)

        #expect(simAB! > 0.99)  // Same vectors
        #expect(simAC! < 0.01)  // Orthogonal vectors
    }

    @Test("NodeEmbeddings find similar")
    func testFindSimilar() {
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1.0, 0.0]
        embeddings["B"] = [0.9, 0.1]  // Similar to A
        embeddings["C"] = [0.0, 1.0]  // Different from A

        let similar = embeddings.findSimilar(to: "A", topK: 2)

        #expect(similar.count == 2)
        #expect(similar[0].nodeID == "B")  // Most similar
    }

    @Test("NodeEmbeddings merge")
    func testMerge() {
        var emb1 = NodeEmbeddings()
        emb1["A"] = [1.0, 2.0]

        var emb2 = NodeEmbeddings()
        emb2["B"] = [3.0, 4.0]

        emb1.merge(emb2)

        #expect(emb1.count == 2)
        #expect(emb1["A"] != nil)
        #expect(emb1["B"] != nil)
    }

    @Test("NodeEmbeddings prune")
    func testPrune() {
        var embeddings = NodeEmbeddings()
        embeddings["A"] = [1.0]
        embeddings["B"] = [2.0]
        embeddings["C"] = [3.0]

        embeddings.prune(to: ["A", "C"])

        #expect(embeddings.count == 2)
        #expect(embeddings["B"] == nil)
    }
}
