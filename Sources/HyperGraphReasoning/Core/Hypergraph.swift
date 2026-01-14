import Foundation

/// A hypergraph data structure where edges can connect any number of nodes.
///
/// Unlike traditional graphs where edges connect exactly two nodes, hyperedges
/// can connect any subset of nodes. This is useful for representing higher-order
/// relationships in knowledge graphs.
///
/// - Note: This implementation is inspired by HyperNetX but includes only the
///   features needed for hypergraph-based knowledge representation.
public struct Hypergraph<NodeID: Hashable & Sendable, EdgeID: Hashable & Sendable>: Sendable {

    // MARK: - Storage

    /// Primary storage: maps each edge ID to the set of nodes it connects.
    /// This is the canonical representation of the hypergraph structure.
    public internal(set) var incidenceDict: [EdgeID: Set<NodeID>]

    /// Cached adjacency dictionary: maps each node to the edges it belongs to.
    /// Computed lazily and cached for performance.
    internal var _adjacencyDict: [NodeID: Set<EdgeID>]?

    // MARK: - Computed Properties

    /// All nodes in the hypergraph.
    public var nodes: Set<NodeID> {
        var allNodes = Set<NodeID>()
        for nodeSet in incidenceDict.values {
            allNodes.formUnion(nodeSet)
        }
        return allNodes
    }

    /// All edge IDs in the hypergraph.
    public var edges: Set<EdgeID> {
        Set(incidenceDict.keys)
    }

    /// Number of nodes in the hypergraph.
    public var nodeCount: Int {
        nodes.count
    }

    /// Number of edges in the hypergraph.
    public var edgeCount: Int {
        incidenceDict.count
    }

    /// Returns the adjacency dictionary mapping nodes to their incident edges.
    /// This is computed on first access and cached.
    public var adjacencyDict: [NodeID: Set<EdgeID>] {
        mutating get {
            if let cached = _adjacencyDict {
                return cached
            }
            var adj = [NodeID: Set<EdgeID>]()
            for (edgeID, nodeSet) in incidenceDict {
                for node in nodeSet {
                    adj[node, default: []].insert(edgeID)
                }
            }
            _adjacencyDict = adj
            return adj
        }
    }

    // MARK: - Initialization

    /// Creates an empty hypergraph.
    public init() {
        self.incidenceDict = [:]
        self._adjacencyDict = nil
    }

    /// Creates a hypergraph from an incidence dictionary.
    ///
    /// - Parameter incidenceDict: A dictionary mapping edge IDs to sets of node IDs.
    public init(incidenceDict: [EdgeID: Set<NodeID>]) {
        self.incidenceDict = incidenceDict
        self._adjacencyDict = nil
    }

    /// Creates a hypergraph from an incidence dictionary with array values.
    ///
    /// - Parameter incidenceDict: A dictionary mapping edge IDs to arrays of node IDs.
    public init(incidenceDict: [EdgeID: [NodeID]]) {
        self.incidenceDict = incidenceDict.mapValues { Set($0) }
        self._adjacencyDict = nil
    }

    // MARK: - Node Operations

    /// Returns the degree of a node (number of edges it belongs to).
    ///
    /// - Parameter node: The node to query.
    /// - Returns: The number of edges containing this node, or 0 if the node doesn't exist.
    public func degree(of node: NodeID) -> Int {
        var count = 0
        for nodeSet in incidenceDict.values {
            if nodeSet.contains(node) {
                count += 1
            }
        }
        return count
    }

    /// Returns the set of nodes connected to a given node via shared edges.
    ///
    /// - Parameter node: The node to query.
    /// - Returns: Set of all nodes that share at least one edge with the given node.
    public func neighbors(of node: NodeID) -> Set<NodeID> {
        var neighborSet = Set<NodeID>()
        for nodeSet in incidenceDict.values {
            if nodeSet.contains(node) {
                neighborSet.formUnion(nodeSet)
            }
        }
        neighborSet.remove(node)
        return neighborSet
    }

    // MARK: - Edge Operations

    /// Returns the nodes belonging to a specific edge.
    ///
    /// - Parameter edge: The edge ID to query.
    /// - Returns: The set of nodes in this edge, or nil if the edge doesn't exist.
    public func nodes(in edge: EdgeID) -> Set<NodeID>? {
        incidenceDict[edge]
    }

    /// Returns the size (cardinality) of an edge.
    ///
    /// - Parameter edge: The edge ID to query.
    /// - Returns: The number of nodes in this edge, or 0 if the edge doesn't exist.
    public func size(of edge: EdgeID) -> Int {
        incidenceDict[edge]?.count ?? 0
    }

    // MARK: - Mutation

    /// Adds an edge to the hypergraph.
    ///
    /// - Parameters:
    ///   - edge: The edge ID.
    ///   - nodes: The set of nodes this edge connects.
    public mutating func addEdge(_ edge: EdgeID, nodes: Set<NodeID>) {
        incidenceDict[edge] = nodes
        _adjacencyDict = nil  // Invalidate cache
    }

    /// Adds an edge to the hypergraph.
    ///
    /// - Parameters:
    ///   - edge: The edge ID.
    ///   - nodes: The nodes this edge connects.
    public mutating func addEdge(_ edge: EdgeID, nodes: NodeID...) {
        addEdge(edge, nodes: Set(nodes))
    }

    /// Removes an edge from the hypergraph.
    ///
    /// - Parameter edge: The edge ID to remove.
    /// - Returns: The nodes that were in the removed edge, or nil if the edge didn't exist.
    @discardableResult
    public mutating func removeEdge(_ edge: EdgeID) -> Set<NodeID>? {
        let removed = incidenceDict.removeValue(forKey: edge)
        if removed != nil {
            _adjacencyDict = nil  // Invalidate cache
        }
        return removed
    }

    /// Removes a node from all edges in the hypergraph.
    ///
    /// - Parameter node: The node to remove.
    /// - Note: Edges that become empty after node removal are also removed.
    public mutating func removeNode(_ node: NodeID) {
        var modified = false
        var edgesToRemove = [EdgeID]()

        for (edgeID, var nodeSet) in incidenceDict {
            if nodeSet.remove(node) != nil {
                modified = true
                if nodeSet.isEmpty {
                    edgesToRemove.append(edgeID)
                } else {
                    incidenceDict[edgeID] = nodeSet
                }
            }
        }

        for edgeID in edgesToRemove {
            incidenceDict.removeValue(forKey: edgeID)
        }

        if modified {
            _adjacencyDict = nil  // Invalidate cache
        }
    }
}

// MARK: - Equatable & Hashable

extension Hypergraph: Equatable where NodeID: Equatable, EdgeID: Equatable {
    public static func == (lhs: Hypergraph, rhs: Hypergraph) -> Bool {
        lhs.incidenceDict == rhs.incidenceDict
    }
}

extension Hypergraph: Hashable where NodeID: Hashable, EdgeID: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(incidenceDict.count)
        for key in incidenceDict.keys.sorted(by: { "\($0)" < "\($1)" }) {
            hasher.combine(key)
        }
    }
}

// MARK: - CustomStringConvertible

extension Hypergraph: CustomStringConvertible {
    public var description: String {
        "Hypergraph(nodes: \(nodeCount), edges: \(edgeCount))"
    }
}

// MARK: - Type Aliases

/// A hypergraph with String node and edge IDs (most common use case).
public typealias StringHypergraph = Hypergraph<String, String>
