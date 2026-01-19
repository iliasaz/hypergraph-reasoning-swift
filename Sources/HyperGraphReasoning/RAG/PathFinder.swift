import Foundation

/// Finds paths between nodes in a hypergraph using BFS.
///
/// This is used in the GraphRAG pipeline to find connections between
/// matched nodes, which then provide context for LLM queries.
public struct PathFinder: Sendable {

    /// The hypergraph to search in.
    private let hypergraph: StringHypergraph

    /// Creates a path finder for the given hypergraph.
    ///
    /// - Parameter hypergraph: The hypergraph to search paths in.
    public init(hypergraph: StringHypergraph) {
        self.hypergraph = hypergraph
    }

    // MARK: - Path Finding

    /// Finds shortest paths between all pairs of the given nodes.
    ///
    /// Uses BFS to find the shortest path between each pair of nodes.
    /// Returns paths as sequences of node IDs.
    ///
    /// - Parameters:
    ///   - nodes: The nodes to find paths between.
    ///   - maxLength: Maximum path length (number of nodes). Default is 4.
    /// - Returns: Array of paths, where each path is an array of node IDs.
    public func findShortestPaths(
        between nodes: [String],
        maxLength: Int = 4
    ) -> [[String]] {
        guard nodes.count >= 2 else { return [] }

        var paths: [[String]] = []
        let nodeSet = Set(nodes)

        // Find paths between all pairs
        for i in 0..<nodes.count {
            for j in (i + 1)..<nodes.count {
                if let path = findPath(
                    from: nodes[i],
                    to: nodes[j],
                    maxLength: maxLength
                ) {
                    paths.append(path)
                }
            }
        }

        // Also try to find paths that connect multiple target nodes
        if nodes.count > 2 {
            let multiPaths = findMultiNodePaths(
                through: nodeSet,
                maxLength: maxLength
            )
            paths.append(contentsOf: multiPaths)
        }

        // Remove duplicate paths
        return Array(Set(paths.map { PathWrapper(path: $0) })).map(\.path)
    }

    /// Finds the shortest path between two specific nodes using BFS.
    ///
    /// - Parameters:
    ///   - source: The starting node.
    ///   - target: The destination node.
    ///   - maxLength: Maximum path length (number of nodes).
    /// - Returns: The path as an array of node IDs, or nil if no path exists.
    public func findPath(
        from source: String,
        to target: String,
        maxLength: Int = 4
    ) -> [String]? {
        guard source != target else { return [source] }
        guard hypergraph.nodes.contains(source),
              hypergraph.nodes.contains(target) else {
            return nil
        }

        // BFS state
        var visited = Set<String>()
        var queue: [(node: String, path: [String])] = [(source, [source])]
        visited.insert(source)

        while !queue.isEmpty {
            let (currentNode, currentPath) = queue.removeFirst()

            // Check if we've exceeded max depth
            if currentPath.count >= maxLength {
                continue
            }

            // Get neighbors through hyperedges
            let neighbors = hypergraph.neighbors(of: currentNode)

            for neighbor in neighbors {
                if neighbor == target {
                    // Found the target!
                    return currentPath + [neighbor]
                }

                if !visited.contains(neighbor) {
                    visited.insert(neighbor)
                    queue.append((neighbor, currentPath + [neighbor]))
                }
            }
        }

        return nil
    }

    /// Finds a path from source that passes through the target node.
    ///
    /// - Parameters:
    ///   - source: The starting node.
    ///   - target: The node that must be on the path.
    ///   - maxLength: Maximum path length.
    /// - Returns: A path containing both nodes, or nil if not found.
    public func findPathThrough(
        from source: String,
        through target: String,
        maxLength: Int = 4
    ) -> [String]? {
        findPath(from: source, to: target, maxLength: maxLength)
    }

    // MARK: - Multi-Node Paths

    /// Finds paths that connect multiple target nodes.
    ///
    /// - Parameters:
    ///   - targetNodes: The set of nodes to connect.
    ///   - maxLength: Maximum path length.
    /// - Returns: Paths that pass through multiple target nodes.
    private func findMultiNodePaths(
        through targetNodes: Set<String>,
        maxLength: Int
    ) -> [[String]] {
        var paths: [[String]] = []

        // For each target node, try to find paths that reach other targets
        for startNode in targetNodes {
            let reachable = findReachableTargets(
                from: startNode,
                targets: targetNodes,
                maxLength: maxLength
            )

            for (path, _) in reachable {
                if path.count > 2 {  // Only include non-trivial paths
                    paths.append(path)
                }
            }
        }

        return paths
    }

    /// Finds all reachable target nodes from a source using BFS.
    ///
    /// - Parameters:
    ///   - source: The starting node.
    ///   - targets: The target nodes to find.
    ///   - maxLength: Maximum path length.
    /// - Returns: Array of (path, targets found on path) tuples.
    private func findReachableTargets(
        from source: String,
        targets: Set<String>,
        maxLength: Int
    ) -> [([String], Set<String>)] {
        var results: [([String], Set<String>)] = []
        var visited = Set<String>()
        var queue: [(node: String, path: [String], foundTargets: Set<String>)] = [
            (source, [source], targets.contains(source) ? [source] : [])
        ]
        visited.insert(source)

        while !queue.isEmpty {
            let (currentNode, currentPath, foundTargets) = queue.removeFirst()

            if currentPath.count >= maxLength {
                if foundTargets.count > 1 {
                    results.append((currentPath, foundTargets))
                }
                continue
            }

            let neighbors = hypergraph.neighbors(of: currentNode)

            for neighbor in neighbors where !visited.contains(neighbor) {
                visited.insert(neighbor)

                var newFoundTargets = foundTargets
                if targets.contains(neighbor) {
                    newFoundTargets.insert(neighbor)
                }

                let newPath = currentPath + [neighbor]

                if newFoundTargets.count > 1 {
                    results.append((newPath, newFoundTargets))
                }

                queue.append((neighbor, newPath, newFoundTargets))
            }
        }

        return results
    }

    // MARK: - Subgraph Extraction

    /// Extracts a subgraph containing only the edges along the given paths.
    ///
    /// - Parameter paths: The paths to include in the subgraph.
    /// - Returns: A new hypergraph containing only the relevant edges.
    public func extractSubgraph(paths: [[String]]) -> StringHypergraph {
        // Collect all nodes on the paths
        var pathNodes = Set<String>()
        for path in paths {
            pathNodes.formUnion(path)
        }

        // Find edges that connect consecutive nodes in paths
        var relevantEdges = Set<String>()

        for path in paths {
            for i in 0..<(path.count - 1) {
                let node1 = path[i]
                let node2 = path[i + 1]

                // Find edges containing both nodes
                for (edgeID, nodes) in hypergraph.incidenceDict {
                    if nodes.contains(node1) && nodes.contains(node2) {
                        relevantEdges.insert(edgeID)
                    }
                }
            }
        }

        // Build subgraph
        var subgraph = StringHypergraph()
        for edgeID in relevantEdges {
            if let nodes = hypergraph.incidenceDict[edgeID] {
                // Only include nodes that are on the paths
                let filteredNodes = nodes.intersection(pathNodes)
                if filteredNodes.count >= 2 {
                    subgraph.addEdge(edgeID, nodes: filteredNodes)
                }
            }
        }

        return subgraph
    }

    /// Gets the edges that connect nodes along a specific path.
    ///
    /// - Parameter path: The path to get edges for.
    /// - Returns: Array of edge IDs along the path.
    public func edgesAlongPath(_ path: [String]) -> [String] {
        guard path.count >= 2 else { return [] }

        var edges: [String] = []

        for i in 0..<(path.count - 1) {
            let node1 = path[i]
            let node2 = path[i + 1]

            // Find an edge containing both nodes
            for (edgeID, nodes) in hypergraph.incidenceDict {
                if nodes.contains(node1) && nodes.contains(node2) {
                    edges.append(edgeID)
                    break  // Only need one edge per step
                }
            }
        }

        return edges
    }
}

// MARK: - Helper Types

/// Wrapper for path comparison and hashing.
private struct PathWrapper: Hashable {
    let path: [String]

    func hash(into hasher: inout Hasher) {
        hasher.combine(path)
    }

    static func == (lhs: PathWrapper, rhs: PathWrapper) -> Bool {
        lhs.path == rhs.path
    }
}

// MARK: - Path Representation

/// A path through the hypergraph with associated metadata.
public struct HypergraphPath: Sendable, Codable, Hashable {
    /// The sequence of node IDs in the path.
    public let nodes: [String]

    /// The edge IDs connecting the nodes.
    public let edges: [String]

    /// The source node (first in path).
    public var source: String? { nodes.first }

    /// The target node (last in path).
    public var target: String? { nodes.last }

    /// The length of the path (number of nodes).
    public var length: Int { nodes.count }

    /// Creates a hypergraph path.
    ///
    /// - Parameters:
    ///   - nodes: The sequence of node IDs.
    ///   - edges: The edge IDs connecting the nodes.
    public init(nodes: [String], edges: [String]) {
        self.nodes = nodes
        self.edges = edges
    }
}
