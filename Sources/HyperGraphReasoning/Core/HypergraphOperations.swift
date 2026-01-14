import Foundation

// MARK: - Union Operation

extension Hypergraph {

    /// Creates a new hypergraph by combining this hypergraph with another.
    ///
    /// Edges from both hypergraphs are included. If both hypergraphs have an edge
    /// with the same ID, the edge from the other hypergraph takes precedence.
    ///
    /// - Parameter other: The hypergraph to union with.
    /// - Returns: A new hypergraph containing edges from both hypergraphs.
    public func union(_ other: Hypergraph) -> Hypergraph {
        var combined = incidenceDict
        for (edgeID, nodes) in other.incidenceDict {
            if let existing = combined[edgeID] {
                // Merge nodes if edge exists in both
                combined[edgeID] = existing.union(nodes)
            } else {
                combined[edgeID] = nodes
            }
        }
        return Hypergraph(incidenceDict: combined)
    }

    /// Combines this hypergraph with another in place.
    ///
    /// - Parameter other: The hypergraph to union with.
    public mutating func formUnion(_ other: Hypergraph) {
        for (edgeID, nodes) in other.incidenceDict {
            if let existing = incidenceDict[edgeID] {
                incidenceDict[edgeID] = existing.union(nodes)
            } else {
                incidenceDict[edgeID] = nodes
            }
        }
        _adjacencyDict = nil  // Invalidate cache
    }
}

// MARK: - Connected Components

extension Hypergraph {

    /// Returns the connected components of the hypergraph.
    ///
    /// Two nodes are in the same connected component if there is a path of
    /// hyperedges connecting them (i.e., they can reach each other by traversing
    /// through shared hyperedges).
    ///
    /// - Returns: An array of sets, where each set contains the node IDs of a
    ///   connected component, sorted by size (largest first).
    public func connectedComponents() -> [Set<NodeID>] {
        let allNodes = nodes
        var visited = Set<NodeID>()
        var components = [Set<NodeID>]()

        for startNode in allNodes {
            if visited.contains(startNode) {
                continue
            }

            // BFS to find all nodes in this component
            var component = Set<NodeID>()
            var queue = [startNode]

            while !queue.isEmpty {
                let node = queue.removeFirst()
                if visited.contains(node) {
                    continue
                }
                visited.insert(node)
                component.insert(node)

                // Find all nodes connected via shared edges
                for nodeSet in incidenceDict.values {
                    if nodeSet.contains(node) {
                        for neighbor in nodeSet {
                            if !visited.contains(neighbor) {
                                queue.append(neighbor)
                            }
                        }
                    }
                }
            }

            components.append(component)
        }

        // Sort by size, largest first
        return components.sorted { $0.count > $1.count }
    }

    /// Returns the largest connected component.
    ///
    /// - Returns: The set of nodes in the largest connected component,
    ///   or an empty set if the hypergraph is empty.
    public func largestComponent() -> Set<NodeID> {
        connectedComponents().first ?? []
    }

    /// Returns whether the hypergraph is connected.
    ///
    /// A hypergraph is connected if all nodes belong to the same connected component.
    ///
    /// - Returns: `true` if connected, `false` otherwise.
    public var isConnected: Bool {
        let components = connectedComponents()
        return components.count <= 1
    }
}

// MARK: - Subgraph Operations

extension Hypergraph {

    /// Creates a subhypergraph containing only the specified nodes.
    ///
    /// Edges are retained only if they contain at least one of the specified nodes.
    /// The retained edges are restricted to only include the specified nodes.
    ///
    /// - Parameter nodes: The set of nodes to keep.
    /// - Returns: A new hypergraph restricted to the specified nodes.
    public func restrictToNodes(_ nodes: Set<NodeID>) -> Hypergraph {
        var newIncidence = [EdgeID: Set<NodeID>]()

        for (edgeID, edgeNodes) in incidenceDict {
            let intersection = edgeNodes.intersection(nodes)
            if !intersection.isEmpty {
                newIncidence[edgeID] = intersection
            }
        }

        return Hypergraph(incidenceDict: newIncidence)
    }

    /// Creates a subhypergraph containing only the specified edges.
    ///
    /// - Parameter edges: The set of edge IDs to keep.
    /// - Returns: A new hypergraph containing only the specified edges.
    public func restrictToEdges(_ edges: Set<EdgeID>) -> Hypergraph {
        var newIncidence = [EdgeID: Set<NodeID>]()

        for edgeID in edges {
            if let nodeSet = incidenceDict[edgeID] {
                newIncidence[edgeID] = nodeSet
            }
        }

        return Hypergraph(incidenceDict: newIncidence)
    }

    /// Creates a subhypergraph from a connected component.
    ///
    /// - Parameter component: The set of nodes in the component.
    /// - Returns: A new hypergraph containing only the edges within the component.
    public func subhypergraph(for component: Set<NodeID>) -> Hypergraph {
        restrictToNodes(component)
    }

    /// Removes nodes belonging to small connected components.
    ///
    /// - Parameters:
    ///   - sizeThreshold: Components with fewer nodes than this are removed.
    ///   - keepSingletons: If true, single-node components are kept.
    /// - Returns: A new hypergraph with small components removed.
    public func removeSmallComponents(
        sizeThreshold: Int,
        keepSingletons: Bool = false
    ) -> Hypergraph {
        if sizeThreshold <= 0 {
            return self
        }

        let components = connectedComponents()
        var nodesToKeep = Set<NodeID>()

        for component in components {
            let shouldKeep = component.count >= sizeThreshold ||
                (keepSingletons && component.count == 1)
            if shouldKeep {
                nodesToKeep.formUnion(component)
            }
        }

        return restrictToNodes(nodesToKeep)
    }
}

// MARK: - Edge Filtering

extension Hypergraph {

    /// Returns a hypergraph with edges filtered by a predicate.
    ///
    /// - Parameter isIncluded: A closure that determines whether an edge should be included.
    /// - Returns: A new hypergraph containing only edges that satisfy the predicate.
    public func filterEdges(
        _ isIncluded: (EdgeID, Set<NodeID>) -> Bool
    ) -> Hypergraph {
        var newIncidence = [EdgeID: Set<NodeID>]()
        for (edgeID, nodes) in incidenceDict {
            if isIncluded(edgeID, nodes) {
                newIncidence[edgeID] = nodes
            }
        }
        return Hypergraph(incidenceDict: newIncidence)
    }

    /// Returns a hypergraph with edges of at least a minimum size.
    ///
    /// - Parameter minSize: Minimum number of nodes an edge must have.
    /// - Returns: A new hypergraph containing only edges with at least minSize nodes.
    public func filterEdgesBySize(minSize: Int) -> Hypergraph {
        filterEdges { _, nodes in nodes.count >= minSize }
    }
}
