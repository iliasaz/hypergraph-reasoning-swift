import Testing
@testable import HyperGraphReasoning

@Suite("Hypergraph Tests")
struct HypergraphTests {

    // MARK: - Initialization Tests

    @Test("Empty hypergraph initialization")
    func testEmptyInit() {
        let h = Hypergraph<String, String>()
        #expect(h.nodeCount == 0)
        #expect(h.edgeCount == 0)
        #expect(h.nodes.isEmpty)
        #expect(h.edges.isEmpty)
    }

    @Test("Initialize with incidence dictionary")
    func testIncidenceInit() {
        let incidence: [String: Set<String>] = [
            "e1": ["A", "B", "C"],
            "e2": ["B", "C", "D"],
            "e3": ["D", "E"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.edgeCount == 3)
        #expect(h.nodeCount == 5)
        #expect(h.nodes == ["A", "B", "C", "D", "E"])
        #expect(h.edges == ["e1", "e2", "e3"])
    }

    @Test("Initialize with array-based incidence")
    func testArrayIncidenceInit() {
        let incidence: [String: [String]] = [
            "e1": ["A", "B"],
            "e2": ["B", "C"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.edgeCount == 2)
        #expect(h.nodeCount == 3)
    }

    // MARK: - Node Operations

    @Test("Node degree calculation")
    func testDegree() {
        let incidence: [String: Set<String>] = [
            "e1": ["A", "B"],
            "e2": ["A", "C"],
            "e3": ["A", "B", "C"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.degree(of: "A") == 3)
        #expect(h.degree(of: "B") == 2)
        #expect(h.degree(of: "C") == 2)
        #expect(h.degree(of: "X") == 0)  // Non-existent node
    }

    @Test("Node neighbors")
    func testNeighbors() {
        let incidence: [String: Set<String>] = [
            "e1": ["A", "B"],
            "e2": ["B", "C"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.neighbors(of: "A") == ["B"])
        #expect(h.neighbors(of: "B") == ["A", "C"])
        #expect(h.neighbors(of: "C") == ["B"])
    }

    // MARK: - Edge Operations

    @Test("Edge nodes query")
    func testEdgeNodes() {
        let incidence: [String: Set<String>] = [
            "e1": ["A", "B", "C"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.nodes(in: "e1") == ["A", "B", "C"])
        #expect(h.nodes(in: "e2") == nil)
    }

    @Test("Edge size")
    func testEdgeSize() {
        let incidence: [String: Set<String>] = [
            "e1": ["A", "B", "C"],
            "e2": ["D"]
        ]
        let h = Hypergraph(incidenceDict: incidence)

        #expect(h.size(of: "e1") == 3)
        #expect(h.size(of: "e2") == 1)
        #expect(h.size(of: "e3") == 0)
    }

    // MARK: - Mutation Tests

    @Test("Add edge")
    func testAddEdge() {
        var h = Hypergraph<String, String>()
        h.addEdge("e1", nodes: Set(["A", "B"]))

        #expect(h.edgeCount == 1)
        #expect(h.nodeCount == 2)
    }

    @Test("Remove edge")
    func testRemoveEdge() {
        var h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"],
            "e2": ["B", "C"]
        ])

        let removed = h.removeEdge("e1")

        #expect(removed == ["A", "B"])
        #expect(h.edgeCount == 1)
        #expect(h.nodes(in: "e1") == nil)
    }

    @Test("Remove node")
    func testRemoveNode() {
        var h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B", "C"],
            "e2": ["B", "C"]
        ])

        h.removeNode("B")

        #expect(h.nodes(in: "e1") == ["A", "C"])
        #expect(h.nodes(in: "e2") == ["C"])
        #expect(!h.nodes.contains("B"))
    }

    // MARK: - Union Tests

    @Test("Hypergraph union")
    func testUnion() {
        let h1 = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"]
        ])
        let h2 = Hypergraph<String, String>(incidenceDict: [
            "e2": ["C", "D"]
        ])

        let combined = h1.union(h2)

        #expect(combined.edgeCount == 2)
        #expect(combined.nodeCount == 4)
    }

    @Test("Union with overlapping edges")
    func testUnionOverlap() {
        let h1 = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"]
        ])
        let h2 = Hypergraph<String, String>(incidenceDict: [
            "e1": ["B", "C"]
        ])

        let combined = h1.union(h2)

        // Same edge ID should merge nodes
        #expect(combined.edgeCount == 1)
        #expect(combined.nodes(in: "e1") == ["A", "B", "C"])
    }

    // MARK: - Connected Components Tests

    @Test("Connected components")
    func testConnectedComponents() {
        let h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"],
            "e2": ["B", "C"],
            "e3": ["X", "Y"]
        ])

        let components = h.connectedComponents()

        #expect(components.count == 2)
        #expect(components[0] == ["A", "B", "C"] || components[0] == ["X", "Y"])
    }

    @Test("Largest component")
    func testLargestComponent() {
        let h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B", "C"],
            "e2": ["X"]
        ])

        let largest = h.largestComponent()

        #expect(largest == ["A", "B", "C"])
    }

    @Test("Is connected")
    func testIsConnected() {
        let connected = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"],
            "e2": ["B", "C"]
        ])
        let disconnected = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"],
            "e2": ["X", "Y"]
        ])

        #expect(connected.isConnected)
        #expect(!disconnected.isConnected)
    }

    // MARK: - Restriction Tests

    @Test("Restrict to nodes")
    func testRestrictToNodes() {
        let h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B", "C"],
            "e2": ["C", "D"]
        ])

        let restricted = h.restrictToNodes(["A", "B", "C"])

        #expect(restricted.nodes == ["A", "B", "C"])
        #expect(restricted.nodes(in: "e1") == ["A", "B", "C"])
        #expect(restricted.nodes(in: "e2") == ["C"])
    }

    @Test("Restrict to edges")
    func testRestrictToEdges() {
        let h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B"],
            "e2": ["C", "D"],
            "e3": ["E", "F"]
        ])

        let restricted = h.restrictToEdges(["e1", "e2"])

        #expect(restricted.edgeCount == 2)
        #expect(restricted.nodes == ["A", "B", "C", "D"])
    }

    @Test("Remove small components")
    func testRemoveSmallComponents() {
        let h = Hypergraph<String, String>(incidenceDict: [
            "e1": ["A", "B", "C"],  // Component of 3
            "e2": ["X"]             // Component of 1
        ])

        let filtered = h.removeSmallComponents(sizeThreshold: 2)

        #expect(filtered.nodeCount == 3)
        #expect(!filtered.nodes.contains("X"))
    }
}
