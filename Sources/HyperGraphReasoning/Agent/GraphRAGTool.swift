import Foundation
import SwiftAgents

/// A tool that searches a knowledge graph for relevant context.
///
/// This tool integrates with the SwiftAgents framework to provide
/// Graph Retrieval-Augmented Generation (GraphRAG) capabilities.
/// It extracts keywords from a query, matches them to graph nodes,
/// finds paths between nodes, and returns formatted context.
public struct GraphRAGTool: Tool {

    // MARK: - Tool Protocol Properties

    /// The unique name of the tool.
    public let name = "graph_rag_search"

    /// Description of what the tool does.
    public let description = """
        Search the knowledge graph for information related to the query. \
        Returns relevant context from the hypergraph including related concepts \
        and their relationships. Use this to find background information, \
        understand connections between concepts, or verify factual claims.
        """

    /// The parameters this tool accepts.
    public let parameters: [ToolParameter] = [
        ToolParameter(
            name: "query",
            description: "The search query or question to find relevant context for",
            type: .string,
            isRequired: true
        ),
        ToolParameter(
            name: "top_k",
            description: "Maximum number of matching nodes to find per keyword (default: 5)",
            type: .int,
            isRequired: false,
            defaultValue: .int(5)
        ),
        ToolParameter(
            name: "max_path_length",
            description: "Maximum path length for finding connections (default: 4)",
            type: .int,
            isRequired: false,
            defaultValue: .int(4)
        )
    ]

    // MARK: - Properties

    /// The GraphRAG service for context retrieval.
    private let ragService: GraphRAGService

    /// Whether to use simple (non-LLM) keyword extraction.
    private let useSimpleExtraction: Bool

    // MARK: - Initialization

    /// Creates a GraphRAG tool.
    ///
    /// - Parameters:
    ///   - ragService: The GraphRAG service for context retrieval.
    ///   - useSimpleExtraction: Use simple keyword extraction instead of LLM-based.
    public init(ragService: GraphRAGService, useSimpleExtraction: Bool = false) {
        self.ragService = ragService
        self.useSimpleExtraction = useSimpleExtraction
    }

    // MARK: - Tool Protocol Methods

    /// Executes the tool with the given arguments.
    ///
    /// - Parameter arguments: The tool arguments.
    /// - Returns: The formatted context as a SendableValue.
    /// - Throws: AgentError if execution fails.
    public func execute(arguments: [String: SendableValue]) async throws -> SendableValue {
        // Extract required query parameter
        guard let query = arguments["query"]?.stringValue else {
            throw AgentError.invalidToolArguments(
                toolName: name,
                reason: "Missing required parameter: query"
            )
        }

        // Extract optional parameters
        let topK = arguments["top_k"]?.intValue ?? 5
        let maxPathLength = arguments["max_path_length"]?.intValue ?? 4

        // Retrieve context
        let context: RAGContext
        if useSimpleExtraction {
            context = try await ragService.retrieveContextSimple(for: query, topK: topK)
        } else {
            context = try await ragService.retrieveContext(
                for: query,
                topK: topK,
                maxPathLength: maxPathLength
            )
        }

        // Return formatted context or a message if none found
        if context.hasContext {
            // Build a detailed result
            var result = context.formattedContext

            // Add metadata for debugging/transparency
            if !context.keywords.isEmpty {
                result += "\n\n[Keywords extracted: \(context.keywords.joined(separator: ", "))]"
            }
            if !context.matchedNodes.isEmpty {
                let nodeNames = context.matchedNodes.prefix(5).map(\.node)
                result += "\n[Matched nodes: \(nodeNames.joined(separator: ", "))]"
            }

            return .string(result)
        } else {
            return .string("No relevant information found in the knowledge graph for: \(query)")
        }
    }
}

// MARK: - Extended Result Type

/// A detailed result from GraphRAG tool execution.
public struct GraphRAGToolResult: Sendable, Codable {
    /// The formatted context string.
    public let context: String

    /// Keywords extracted from the query.
    public let keywords: [String]

    /// Number of matched nodes.
    public let matchedNodeCount: Int

    /// Number of paths found.
    public let pathCount: Int

    /// Whether any relevant context was found.
    public let hasContext: Bool

    /// Creates a tool result from RAG context.
    public init(from context: RAGContext) {
        self.context = context.formattedContext
        self.keywords = context.keywords
        self.matchedNodeCount = context.matchedNodeCount
        self.pathCount = context.paths.count
        self.hasContext = context.hasContext
    }
}

// MARK: - Query Tool Variant

/// A simpler tool that just queries and returns an answer.
///
/// Unlike GraphRAGTool which returns context, this tool performs
/// the full RAG pipeline including answer generation.
public struct GraphRAGQueryTool: Tool {

    public let name = "graph_rag_query"

    public let description = """
        Query the knowledge graph and get a direct answer. \
        This performs a full RAG query including answer generation, \
        rather than just returning context.
        """

    public let parameters: [ToolParameter] = [
        ToolParameter(
            name: "question",
            description: "The question to answer using the knowledge graph",
            type: .string,
            isRequired: true
        )
    ]

    private let ragService: GraphRAGService

    /// Creates a GraphRAG query tool.
    ///
    /// - Parameter ragService: The GraphRAG service for queries.
    public init(ragService: GraphRAGService) {
        self.ragService = ragService
    }

    public func execute(arguments: [String: SendableValue]) async throws -> SendableValue {
        guard let question = arguments["question"]?.stringValue else {
            throw AgentError.invalidToolArguments(
                toolName: name,
                reason: "Missing required parameter: question"
            )
        }

        let response = try await ragService.query(question)

        // Include context availability in result
        if response.hadContext {
            return .string(response.answer)
        } else {
            return .string("\(response.answer)\n\n[Note: No graph context was available for this query]")
        }
    }
}

// MARK: - Node Search Tool

/// A tool for searching specific nodes in the graph.
public struct GraphNodeSearchTool: Tool {

    public let name = "graph_node_search"

    public let description = """
        Search for specific entities or concepts in the knowledge graph. \
        Returns matching nodes and their connections.
        """

    public let parameters: [ToolParameter] = [
        ToolParameter(
            name: "entity",
            description: "The entity or concept to search for",
            type: .string,
            isRequired: true
        ),
        ToolParameter(
            name: "limit",
            description: "Maximum number of results (default: 10)",
            type: .int,
            isRequired: false,
            defaultValue: .int(10)
        )
    ]

    private let embeddings: NodeEmbeddings
    private let embeddingService: EmbeddingService
    private let hypergraph: StringHypergraph

    /// Creates a node search tool.
    ///
    /// - Parameters:
    ///   - embeddings: The node embeddings.
    ///   - embeddingService: Service for generating query embeddings.
    ///   - hypergraph: The hypergraph to search.
    public init(
        embeddings: NodeEmbeddings,
        embeddingService: EmbeddingService,
        hypergraph: StringHypergraph
    ) {
        self.embeddings = embeddings
        self.embeddingService = embeddingService
        self.hypergraph = hypergraph
    }

    public func execute(arguments: [String: SendableValue]) async throws -> SendableValue {
        guard let entity = arguments["entity"]?.stringValue else {
            throw AgentError.invalidToolArguments(
                toolName: name,
                reason: "Missing required parameter: entity"
            )
        }

        let limit = arguments["limit"]?.intValue ?? 10

        // Find similar nodes
        let matches = try await embeddingService.findSimilarNodes(
            to: entity,
            in: embeddings,
            topK: limit,
            threshold: 0.4
        )

        if matches.isEmpty {
            return .string("No nodes found matching '\(entity)'")
        }

        // Format results
        var result = "Found \(matches.count) matching nodes:\n\n"

        for (nodeID, similarity) in matches {
            let degree = hypergraph.degree(of: nodeID)
            result += "- \(nodeID) (similarity: \(String(format: "%.2f", similarity)), degree: \(degree))\n"

            // Get a few neighbors
            let neighbors = hypergraph.neighbors(of: nodeID).prefix(3)
            if !neighbors.isEmpty {
                result += "  Connected to: \(neighbors.joined(separator: ", "))\n"
            }
        }

        return .string(result)
    }
}
