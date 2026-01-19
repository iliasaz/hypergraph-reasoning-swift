import Foundation

/// The structured response format expected from the LLM for hypergraph extraction.
///
/// This model matches the Pydantic `HypergraphJSON` model in the Python implementation:
/// ```python
/// class HypergraphJSON(BaseModel):
///     events: List[Event]
/// ```
public struct HypergraphJSON: Codable, Sendable {

    /// The list of events (relationships) extracted from text.
    public let events: [Event]

    /// Creates a new hypergraph JSON response.
    ///
    /// - Parameter events: The extracted events.
    public init(events: [Event]) {
        self.events = events
    }

    /// Creates an empty response with no events.
    public init() {
        self.events = []
    }

    /// Whether this response contains no events.
    public var isEmpty: Bool {
        events.isEmpty
    }

    /// The total number of events.
    public var count: Int {
        events.count
    }

    /// All unique nodes mentioned across all events.
    public var allNodes: Set<String> {
        events.reduce(into: Set<String>()) { result, event in
            result.formUnion(event.allNodes)
        }
    }

    /// All unique relations mentioned across all events.
    public var allRelations: Set<String> {
        Set(events.map(\.relation))
    }
}

// MARK: - Conversion to Hypergraph

extension HypergraphJSON {

    /// Converts the events to a hypergraph structure.
    ///
    /// Each event becomes a hyperedge connecting its source and target nodes.
    /// Edge IDs are generated based on the relation and a unique index.
    ///
    /// - Parameter chunkID: Optional chunk ID to include in edge naming.
    /// - Returns: A tuple of (hypergraph, metadata array).
    public func toHypergraph(chunkID: String? = nil) -> (Hypergraph<String, String>, [ChunkMetadata]) {
        var incidenceDict = [String: Set<String>]()
        var metadata = [ChunkMetadata]()

        for (index, event) in events.enumerated() {
            // Normalize relation: convert underscores to spaces for consistency
            // This handles LLMs that might output "is_a" instead of "is a"
            let normalizedRelation = event.relation.replacingOccurrences(of: "_", with: " ")

            // Create edge ID
            let edgeID: String
            if let chunkID = chunkID {
                edgeID = "\(normalizedRelation)_chunk\(chunkID)_\(index)"
            } else {
                edgeID = "\(normalizedRelation)_\(index)"
            }

            // Combine source and target nodes
            let nodes = event.allNodes
            incidenceDict[edgeID] = nodes

            // Create metadata
            let chunkMeta = ChunkMetadata(
                edge: edgeID,
                nodes: nodes,
                source: event.source,
                target: event.target,
                chunkID: chunkID ?? ""
            )
            metadata.append(chunkMeta)
        }

        let hypergraph = Hypergraph<String, String>(incidenceDict: incidenceDict)
        return (hypergraph, metadata)
    }
}

// MARK: - CustomStringConvertible

extension HypergraphJSON: CustomStringConvertible {
    public var description: String {
        "HypergraphJSON(events: \(events.count), nodes: \(allNodes.count))"
    }
}
