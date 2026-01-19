import Foundation
import SwiftAgents

/// Builder for creating GraphRAG-enabled agents.
///
/// Provides factory methods for creating ReAct agents configured with
/// GraphRAG tools for knowledge graph-augmented reasoning.
public struct GraphRAGAgentBuilder {

    // MARK: - System Prompts

    /// Default system prompt for GraphRAG agent.
    public static let defaultSystemPrompt = """
        You are a knowledgeable assistant with access to a knowledge graph.
        Use the graph_rag_search tool to find relevant information before answering questions.

        Guidelines:
        - Search the knowledge graph for relevant context before responding
        - Cite specific information from the graph when possible
        - If the graph doesn't contain relevant information, clearly state that
        - Synthesize information from multiple graph queries when helpful
        - Be accurate and don't make claims not supported by the graph context
        """

    /// System prompt emphasizing graph context usage.
    public static let graphFocusedPrompt = """
        You are an expert assistant that leverages a knowledge graph to provide accurate answers.

        ALWAYS use the graph_rag_search tool to retrieve relevant context before answering.
        Your answers should be grounded in the information retrieved from the graph.

        When answering:
        1. First search for relevant concepts using graph_rag_search
        2. Analyze the relationships and connections found
        3. Synthesize a clear answer based on the graph context
        4. Explicitly mention what you found (or didn't find) in the graph
        """

    /// System prompt for research/exploration tasks.
    public static let researchPrompt = """
        You are a research assistant with access to a knowledge graph for finding connections and relationships.

        Use the graph_rag_search and graph_node_search tools to:
        - Find connections between concepts
        - Discover related topics
        - Trace relationships through the graph
        - Identify relevant background information

        Approach questions methodically:
        1. Identify key entities and concepts
        2. Search for each in the graph
        3. Look for connections between them
        4. Synthesize findings into a comprehensive answer
        """

    // MARK: - Builder Methods

    /// Builds a ReAct agent with GraphRAG tools.
    ///
    /// - Parameters:
    ///   - ragService: The GraphRAG service for context retrieval.
    ///   - inferenceProvider: The LLM provider for the agent.
    ///   - instructions: Optional custom system prompt.
    ///   - configuration: Optional agent configuration.
    ///   - additionalTools: Additional tools to include.
    /// - Returns: A configured ReActAgent.
    public static func build(
        ragService: GraphRAGService,
        inferenceProvider: any InferenceProvider,
        instructions: String? = nil,
        configuration: AgentConfiguration = .default,
        additionalTools: [any Tool] = []
    ) -> ReActAgent {
        // Create GraphRAG tools
        let ragTool = GraphRAGTool(ragService: ragService)

        // Combine tools
        var tools: [any Tool] = [ragTool]
        tools.append(contentsOf: additionalTools)

        return ReActAgent(
            tools: tools,
            instructions: instructions ?? defaultSystemPrompt,
            configuration: configuration,
            inferenceProvider: inferenceProvider
        )
    }

    /// Builds a ReAct agent with full GraphRAG toolset.
    ///
    /// Includes both search and query tools, plus node search.
    ///
    /// - Parameters:
    ///   - ragService: The GraphRAG service.
    ///   - embeddings: Node embeddings for search.
    ///   - embeddingService: Embedding service for queries.
    ///   - hypergraph: The hypergraph structure.
    ///   - inferenceProvider: The LLM provider.
    ///   - instructions: Optional custom system prompt.
    ///   - configuration: Optional agent configuration.
    /// - Returns: A configured ReActAgent with full toolset.
    public static func buildWithFullToolset(
        ragService: GraphRAGService,
        embeddings: NodeEmbeddings,
        embeddingService: EmbeddingService,
        hypergraph: StringHypergraph,
        inferenceProvider: any InferenceProvider,
        instructions: String? = nil,
        configuration: AgentConfiguration = .default
    ) -> ReActAgent {
        // Create all GraphRAG tools
        let ragTool = GraphRAGTool(ragService: ragService)
        let queryTool = GraphRAGQueryTool(ragService: ragService)
        let nodeTool = GraphNodeSearchTool(
            embeddings: embeddings,
            embeddingService: embeddingService,
            hypergraph: hypergraph
        )

        let tools: [any Tool] = [ragTool, queryTool, nodeTool]

        return ReActAgent(
            tools: tools,
            instructions: instructions ?? graphFocusedPrompt,
            configuration: configuration,
            inferenceProvider: inferenceProvider
        )
    }

    /// Builds a ReAct agent using the fluent builder API.
    ///
    /// - Parameters:
    ///   - ragService: The GraphRAG service.
    ///   - inferenceProvider: The LLM provider.
    /// - Returns: A ReActAgent.Builder for fluent configuration.
    public static func builder(
        ragService: GraphRAGService,
        inferenceProvider: any InferenceProvider
    ) -> ReActAgent.Builder {
        let ragTool = GraphRAGTool(ragService: ragService)

        return ReActAgent.Builder()
            .tools([ragTool])
            .instructions(defaultSystemPrompt)
            .inferenceProvider(inferenceProvider)
    }

    // MARK: - OpenRouter Integration

    /// Builds a GraphRAG agent using OpenRouter as the inference provider.
    ///
    /// - Parameters:
    ///   - ragService: The GraphRAG service.
    ///   - apiKey: OpenRouter API key.
    ///   - model: Model to use (default: meta-llama/llama-4-maverick).
    ///   - instructions: Optional custom system prompt.
    ///   - configuration: Optional agent configuration.
    /// - Returns: A configured ReActAgent.
    /// - Throws: If OpenRouter configuration fails.
    public static func buildWithOpenRouter(
        ragService: GraphRAGService,
        apiKey: String,
        model: String = "meta-llama/llama-4-maverick",
        instructions: String? = nil,
        configuration: AgentConfiguration = .default
    ) throws -> ReActAgent {
        let openRouterModel = try OpenRouterModel(model)
        let openRouterConfig = try OpenRouterConfiguration(
            apiKey: apiKey,
            model: openRouterModel
        )
        let provider = OpenRouterProvider(configuration: openRouterConfig)

        return build(
            ragService: ragService,
            inferenceProvider: provider,
            instructions: instructions,
            configuration: configuration
        )
    }

    // MARK: - Configuration Helpers

    /// Creates a default agent configuration suitable for GraphRAG.
    ///
    /// - Parameters:
    ///   - maxIterations: Maximum ReAct iterations. Default: 5.
    ///   - timeout: Timeout duration. Default: 120 seconds.
    /// - Returns: An AgentConfiguration.
    public static func defaultConfiguration(
        maxIterations: Int = 5,
        timeout: Duration = .seconds(120)
    ) -> AgentConfiguration {
        AgentConfiguration(
            maxIterations: maxIterations,
            timeout: timeout,
            temperature: 0.7,
            maxTokens: 4096,
            stopOnToolError: false  // Continue even if RAG search fails
        )
    }

    /// Creates a research-focused configuration with more iterations.
    ///
    /// - Returns: An AgentConfiguration optimized for research tasks.
    public static func researchConfiguration() -> AgentConfiguration {
        AgentConfiguration(
            maxIterations: 10,
            timeout: .seconds(300),
            temperature: 0.5,  // More focused
            maxTokens: 4096,
            stopOnToolError: false
        )
    }
}

// MARK: - Convenience Extensions

extension GraphRAGAgentBuilder {

    /// Creates a complete GraphRAG agent setup from loaded data.
    ///
    /// This is a convenience method that sets up the full GraphRAG pipeline.
    ///
    /// - Parameters:
    ///   - hypergraph: The loaded hypergraph.
    ///   - embeddings: The loaded embeddings.
    ///   - llmProvider: LLM provider for keyword extraction.
    ///   - embeddingService: Service for generating embeddings.
    ///   - inferenceProvider: Inference provider for the agent.
    ///   - metadata: Optional chunk metadata.
    ///   - instructions: Optional custom instructions.
    /// - Returns: A configured ReActAgent.
    public static func createFullSetup(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        llmProvider: any LLMProvider,
        embeddingService: EmbeddingService,
        inferenceProvider: any InferenceProvider,
        metadata: [ChunkMetadata]? = nil,
        instructions: String? = nil
    ) async -> ReActAgent {
        // Create the RAG service
        let ragService = GraphRAGService(
            hypergraph: hypergraph,
            embeddings: embeddings,
            llmProvider: llmProvider,
            embeddingService: embeddingService,
            metadata: metadata
        )

        // Build the agent
        return buildWithFullToolset(
            ragService: ragService,
            embeddings: embeddings,
            embeddingService: embeddingService,
            hypergraph: hypergraph,
            inferenceProvider: inferenceProvider,
            instructions: instructions
        )
    }
}
