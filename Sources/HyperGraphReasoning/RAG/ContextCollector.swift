import Foundation

/// Collects context from hypergraph paths and converts them to natural language.
///
/// This is used in the GraphRAG pipeline to convert hyperedges along paths
/// into sentences that can be injected into LLM prompts.
public struct ContextCollector: Sendable {

    /// The hypergraph to collect context from.
    private let hypergraph: StringHypergraph

    /// Optional chunk metadata for extracting edge labels.
    private let metadata: [ChunkMetadata]?

    /// Creates a context collector.
    ///
    /// - Parameters:
    ///   - hypergraph: The hypergraph to collect context from.
    ///   - metadata: Optional metadata containing edge provenance.
    public init(
        hypergraph: StringHypergraph,
        metadata: [ChunkMetadata]? = nil
    ) {
        self.hypergraph = hypergraph
        self.metadata = metadata
    }

    // MARK: - Context Collection

    /// Collects natural language sentences from the given paths.
    ///
    /// Converts hyperedges along the paths to directional sentences
    /// describing the relationships between nodes.
    ///
    /// - Parameters:
    ///   - paths: The paths to convert to sentences.
    ///   - includeEdgeLabels: Whether to include edge labels in sentences.
    /// - Returns: Array of context sentences.
    public func collectSentences(
        from paths: [[String]],
        includeEdgeLabels: Bool = true
    ) -> [String] {
        var sentences = Set<String>()

        for path in paths {
            let pathSentences = sentencesFromPath(
                path,
                includeEdgeLabels: includeEdgeLabels
            )
            sentences.formUnion(pathSentences)
        }

        return Array(sentences).sorted()
    }

    /// Collects sentences from edges in a subgraph.
    ///
    /// - Parameters:
    ///   - subgraph: The subgraph to convert.
    ///   - includeEdgeLabels: Whether to include edge labels.
    /// - Returns: Array of context sentences.
    public func collectSentences(
        from subgraph: StringHypergraph,
        includeEdgeLabels: Bool = true
    ) -> [String] {
        var sentences: [String] = []

        for (edgeID, nodes) in subgraph.incidenceDict {
            let sentence = edgeToSentence(
                edgeID: edgeID,
                nodes: nodes,
                includeLabel: includeEdgeLabels
            )
            sentences.append(sentence)
        }

        return sentences.sorted()
    }

    /// Formats collected sentences into a context string for LLM injection.
    ///
    /// - Parameters:
    ///   - sentences: The sentences to format.
    ///   - maxTokens: Approximate maximum token count (chars / 4).
    ///   - header: Optional header for the context section.
    /// - Returns: Formatted context string.
    public func formatContext(
        sentences: [String],
        maxTokens: Int = 2000,
        header: String? = "Relevant knowledge from the graph:"
    ) -> String {
        guard !sentences.isEmpty else { return "" }

        var result = ""
        if let header = header {
            result = "\(header)\n\n"
        }

        let maxChars = maxTokens * 4  // Approximate chars per token
        var currentChars = result.count

        for (index, sentence) in sentences.enumerated() {
            let line = "- \(sentence)\n"
            if currentChars + line.count > maxChars {
                if index > 0 {
                    result += "- (... and \(sentences.count - index) more relationships)\n"
                }
                break
            }
            result += line
            currentChars += line.count
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Private Methods

    /// Converts a path to sentences by examining edges between consecutive nodes.
    private func sentencesFromPath(
        _ path: [String],
        includeEdgeLabels: Bool
    ) -> [String] {
        guard path.count >= 2 else { return [] }

        var sentences: [String] = []

        for i in 0..<(path.count - 1) {
            let node1 = path[i]
            let node2 = path[i + 1]

            // Find edges containing both nodes
            for (edgeID, nodes) in hypergraph.incidenceDict {
                if nodes.contains(node1) && nodes.contains(node2) {
                    let sentence = edgeToSentence(
                        edgeID: edgeID,
                        nodes: nodes,
                        sourceNode: node1,
                        targetNode: node2,
                        includeLabel: includeEdgeLabels
                    )
                    sentences.append(sentence)
                    break  // Only take first matching edge
                }
            }
        }

        return sentences
    }

    /// Converts a hyperedge to a natural language sentence.
    ///
    /// Uses the edge ID format to extract the relation label:
    /// - Format: "relation_chunkXXX_N" -> extracts "relation"
    /// - Converts underscores to spaces
    ///
    /// - Parameters:
    ///   - edgeID: The edge identifier.
    ///   - nodes: The nodes in the edge.
    ///   - sourceNode: Optional specific source node.
    ///   - targetNode: Optional specific target node.
    ///   - includeLabel: Whether to include the relation label.
    /// - Returns: Natural language sentence describing the relationship.
    private func edgeToSentence(
        edgeID: String,
        nodes: Set<String>,
        sourceNode: String? = nil,
        targetNode: String? = nil,
        includeLabel: Bool = true
    ) -> String {
        let relation = includeLabel ? extractRelation(from: edgeID) : nil

        // If we have specific source/target from metadata
        if let meta = metadata?.first(where: { $0.edge == edgeID }) {
            return formatDirectionalSentence(
                source: meta.source,
                target: meta.target,
                relation: relation
            )
        }

        // If we have specified source and target nodes
        if let source = sourceNode, let target = targetNode {
            if let relation = relation {
                return "\(source) \(relation) \(target)."
            } else {
                return "\(source) is related to \(target)."
            }
        }

        // General case: format based on node count
        let nodeList = Array(nodes).sorted()

        if nodeList.count == 2 {
            if let relation = relation {
                return "\(nodeList[0]) \(relation) \(nodeList[1])."
            } else {
                return "\(nodeList[0]) is connected to \(nodeList[1])."
            }
        } else if nodeList.count > 2 {
            let allButLast = nodeList.dropLast().joined(separator: ", ")
            let last = nodeList.last!
            if let relation = relation {
                return "\(allButLast), \(last) \(relation)."
            } else {
                return "\(allButLast), \(last) are connected."
            }
        } else {
            return nodeList.joined(separator: ", ") + "."
        }
    }

    /// Formats a directional sentence from source/target arrays.
    ///
    /// Matches Python format: "source1, source2 relation target1, target2."
    private func formatDirectionalSentence(
        source: [String],
        target: [String],
        relation: String?
    ) -> String {
        // Use comma-separated format to match Python behavior
        let sourceStr = source.joined(separator: ", ")
        let targetStr = target.joined(separator: ", ")

        if let relation = relation {
            return "\(sourceStr) \(relation) \(targetStr)."
        } else {
            return "\(sourceStr) is related to \(targetStr)."
        }
    }

    /// Extracts the relation name from an edge ID.
    ///
    /// Edge ID format: "relation_chunkXXX_N"
    /// This extracts "relation" exactly as stored (matching Python behavior).
    ///
    /// Note: Python uses the relation exactly as extracted from the edge ID.
    /// If underscores should be converted to spaces, that should be done
    /// at hypergraph construction time, not here.
    private func extractRelation(from edgeID: String) -> String? {
        // Try to extract relation from edge ID
        // Format: "relation_chunk..." or "relation_chunkXXX_N"
        // Python regex: r"(.+?)_chunk([0-9A-Za-z]+)_(\d+)"

        // First, check if it matches the chunk pattern
        let pattern = "^(.+?)_chunk[0-9A-Za-z]+_\\d+$"
        if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
            let range = NSRange(edgeID.startIndex..<edgeID.endIndex, in: edgeID)
            if let match = regex.firstMatch(in: edgeID, options: [], range: range),
               let relationRange = Range(match.range(at: 1), in: edgeID) {
                let relation = String(edgeID[relationRange])
                // Return relation exactly as stored (matching Python)
                return relation.trimmingCharacters(in: .whitespaces)
            }
        }

        // Fallback: try splitting by "_chunk"
        if let chunkIndex = edgeID.range(of: "_chunk", options: .caseInsensitive) {
            let relation = String(edgeID[..<chunkIndex.lowerBound])
            return relation.trimmingCharacters(in: .whitespaces)
        }

        // Last fallback: use the whole edge ID, cleaned up
        let cleaned = edgeID.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }
}

// MARK: - Convenience Extensions

extension ContextCollector {

    /// Collects context from a path finder result.
    ///
    /// - Parameters:
    ///   - paths: Paths found by PathFinder.
    ///   - maxSentences: Maximum number of sentences to include.
    /// - Returns: Formatted context string.
    public func collectAndFormat(
        from paths: [[String]],
        maxSentences: Int = 50,
        maxTokens: Int = 2000
    ) -> String {
        let sentences = collectSentences(from: paths)
        let limited = Array(sentences.prefix(maxSentences))
        return formatContext(sentences: limited, maxTokens: maxTokens)
    }
}
