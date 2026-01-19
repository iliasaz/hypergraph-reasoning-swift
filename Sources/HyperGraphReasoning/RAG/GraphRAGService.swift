import Foundation

/// Main orchestrator for Graph Retrieval-Augmented Generation.
///
/// Coordinates the full RAG pipeline:
/// 1. Extract keywords from user query
/// 2. Match keywords to graph nodes via embeddings
/// 3. Find paths between matched nodes
/// 4. Collect context from paths
/// 5. Augment LLM prompt with context
public actor GraphRAGService {

    // MARK: - Properties

    /// The hypergraph to query.
    private let hypergraph: StringHypergraph

    /// Node embeddings for similarity search.
    private let embeddings: NodeEmbeddings

    /// LLM provider for keyword extraction and response generation.
    private let llmProvider: any LLMProvider

    /// Service for generating query embeddings.
    private let embeddingService: EmbeddingService

    /// Optional chunk metadata for richer context.
    private let metadata: [ChunkMetadata]?

    /// The chat model to use for responses.
    private let chatModel: String?

    // MARK: - Component Instances

    /// Keyword extractor (created lazily).
    private var _keywordExtractor: KeywordExtractor?
    private var keywordExtractor: KeywordExtractor {
        if let extractor = _keywordExtractor {
            return extractor
        }
        let extractor = KeywordExtractor(llmProvider: llmProvider, model: chatModel)
        _keywordExtractor = extractor
        return extractor
    }

    /// Node matcher.
    private var _nodeMatcher: NodeMatcher?
    private var nodeMatcher: NodeMatcher {
        if let matcher = _nodeMatcher {
            return matcher
        }
        let matcher = NodeMatcher(embeddings: embeddings, embeddingService: embeddingService)
        _nodeMatcher = matcher
        return matcher
    }

    /// Path finder.
    private var _pathFinder: PathFinder?
    private var pathFinder: PathFinder {
        if let finder = _pathFinder {
            return finder
        }
        let finder = PathFinder(hypergraph: hypergraph)
        _pathFinder = finder
        return finder
    }

    /// Context collector.
    private var _contextCollector: ContextCollector?
    private var contextCollector: ContextCollector {
        if let collector = _contextCollector {
            return collector
        }
        let collector = ContextCollector(hypergraph: hypergraph, metadata: metadata)
        _contextCollector = collector
        return collector
    }

    // MARK: - Initialization

    /// Creates a GraphRAG service.
    ///
    /// - Parameters:
    ///   - hypergraph: The knowledge graph to query.
    ///   - embeddings: Pre-computed node embeddings.
    ///   - llmProvider: LLM provider for extraction and generation.
    ///   - embeddingService: Service for generating query embeddings.
    ///   - metadata: Optional chunk metadata for context.
    ///   - chatModel: Optional model override for chat.
    public init(
        hypergraph: StringHypergraph,
        embeddings: NodeEmbeddings,
        llmProvider: any LLMProvider,
        embeddingService: EmbeddingService,
        metadata: [ChunkMetadata]? = nil,
        chatModel: String? = nil
    ) {
        self.hypergraph = hypergraph
        self.embeddings = embeddings
        self.llmProvider = llmProvider
        self.embeddingService = embeddingService
        self.metadata = metadata
        self.chatModel = chatModel
    }

    // MARK: - Main RAG Entry Points

    /// Retrieves context from the graph for a query.
    ///
    /// This is the main RAG retrieval method. It:
    /// 1. Extracts keywords from the query
    /// 2. Finds matching nodes in the graph
    /// 3. Finds paths between matched nodes
    /// 4. Collects context sentences from paths
    ///
    /// - Parameters:
    ///   - query: The user's query or question.
    ///   - topK: Number of matching nodes to find per keyword.
    ///   - maxPathLength: Maximum path length for BFS.
    ///   - similarityThreshold: Minimum similarity for node matching.
    /// - Returns: The retrieved context.
    public func retrieveContext(
        for query: String,
        topK: Int = 5,
        maxPathLength: Int = 4,
        similarityThreshold: Float = 0.5
    ) async throws -> RAGContext {
        // Step 1: Extract keywords
        let keywords = try await keywordExtractor.extract(from: query)

        guard !keywords.isEmpty else {
            return RAGContext.empty(query: query)
        }

        // Step 2: Match keywords to nodes
        let matches = try await nodeMatcher.findMatchingNodes(
            for: keywords,
            topK: topK,
            threshold: similarityThreshold
        )

        guard !matches.isEmpty else {
            return RAGContext(
                query: query,
                keywords: keywords,
                matchedNodes: [],
                paths: [],
                contextSentences: [],
                formattedContext: ""
            )
        }

        // Step 3: Find paths between matched nodes
        let matchedNodeIDs = matches.uniqueNodeIDs
        let paths = pathFinder.findShortestPaths(
            between: matchedNodeIDs,
            maxLength: maxPathLength
        )

        // Step 4: Collect context from paths
        let sentences: [String]
        if paths.isEmpty {
            // No paths found, use direct node context
            sentences = collectDirectNodeContext(for: matchedNodeIDs)
        } else {
            sentences = contextCollector.collectSentences(from: paths)
        }

        // Step 5: Format context
        let formattedContext = contextCollector.formatContext(sentences: sentences)

        return RAGContext(
            query: query,
            keywords: keywords,
            matchedNodes: matches,
            paths: paths,
            contextSentences: sentences,
            formattedContext: formattedContext
        )
    }

    /// Queries the graph and generates a response.
    ///
    /// This performs the full RAG pipeline including response generation.
    ///
    /// - Parameters:
    ///   - question: The user's question.
    ///   - topK: Number of matching nodes per keyword.
    ///   - maxPathLength: Maximum path length.
    /// - Returns: The response with context.
    public func query(
        _ question: String,
        topK: Int = 5,
        maxPathLength: Int = 4
    ) async throws -> RAGResponse {
        // Retrieve context
        let context = try await retrieveContext(
            for: question,
            topK: topK,
            maxPathLength: maxPathLength
        )

        // Generate response using LLM
        let prompt = RAGPrompts.contextTemplate(
            context: context.formattedContext,
            question: question
        )

        let answer = try await llmProvider.chat(
            systemPrompt: RAGPrompts.questionAnswering,
            userPrompt: prompt,
            model: chatModel,
            temperature: 0.7
        )

        return RAGResponse(
            answer: answer,
            context: context
        )
    }

    /// Retrieves context using a simple approach (for quick queries).
    ///
    /// Uses simple keyword extraction instead of LLM-based extraction.
    ///
    /// - Parameters:
    ///   - query: The query string.
    ///   - topK: Number of matching nodes.
    /// - Returns: Retrieved context.
    public func retrieveContextSimple(
        for query: String,
        topK: Int = 5
    ) async throws -> RAGContext {
        // Use simple keyword extraction
        let keywords = keywordExtractor.simpleExtract(from: query)

        guard !keywords.isEmpty else {
            return RAGContext.empty(query: query)
        }

        // Match and retrieve
        let matches = try await nodeMatcher.findMatchingNodes(
            for: keywords,
            topK: topK,
            threshold: 0.5
        )

        let matchedNodeIDs = matches.uniqueNodeIDs
        let paths = pathFinder.findShortestPaths(between: matchedNodeIDs)
        let sentences = paths.isEmpty
            ? collectDirectNodeContext(for: matchedNodeIDs)
            : contextCollector.collectSentences(from: paths)

        return RAGContext(
            query: query,
            keywords: keywords,
            matchedNodes: matches,
            paths: paths,
            contextSentences: sentences,
            formattedContext: contextCollector.formatContext(sentences: sentences)
        )
    }

    // MARK: - Helper Methods

    /// Collects context for nodes directly (when no paths found).
    private func collectDirectNodeContext(for nodeIDs: [String]) -> [String] {
        var sentences: [String] = []

        for nodeID in nodeIDs {
            // Get edges containing this node
            let nodeEdges = hypergraph.incidenceDict.filter { _, nodes in
                nodes.contains(nodeID)
            }

            for (edgeID, nodes) in nodeEdges.prefix(5) {  // Limit per node
                let otherNodes = nodes.subtracting([nodeID])
                if !otherNodes.isEmpty {
                    let relation = extractRelationFromEdge(edgeID)
                    let others = otherNodes.sorted().joined(separator: ", ")
                    if let rel = relation {
                        sentences.append("\(nodeID) \(rel) \(others)")
                    } else {
                        sentences.append("\(nodeID) is related to \(others)")
                    }
                }
            }
        }

        return Array(Set(sentences)).sorted()
    }

    /// Extracts relation name from edge ID.
    private func extractRelationFromEdge(_ edgeID: String) -> String? {
        // Try to extract from "relation_chunkXXX_N" format
        if let chunkRange = edgeID.range(of: "_chunk", options: .caseInsensitive) {
            let relation = String(edgeID[..<chunkRange.lowerBound])
            let cleaned = relation.replacingOccurrences(of: "_", with: " ")
            return cleaned.isEmpty ? nil : cleaned
        }
        return nil
    }
}

// MARK: - Result Types

/// Context retrieved from the knowledge graph.
public struct RAGContext: Sendable, Codable {
    /// The original query.
    public let query: String

    /// Extracted keywords from the query.
    public let keywords: [String]

    /// Nodes matched to keywords.
    public let matchedNodes: [NodeMatch]

    /// Paths found between matched nodes.
    public let paths: [[String]]

    /// Context sentences extracted from paths.
    public let contextSentences: [String]

    /// Formatted context string for LLM injection.
    public let formattedContext: String

    /// Whether any relevant context was found.
    public var hasContext: Bool {
        !contextSentences.isEmpty
    }

    /// Number of unique matched nodes.
    public var matchedNodeCount: Int {
        matchedNodes.uniqueNodeCount
    }

    /// Creates an empty context for a query.
    public static func empty(query: String) -> RAGContext {
        RAGContext(
            query: query,
            keywords: [],
            matchedNodes: [],
            paths: [],
            contextSentences: [],
            formattedContext: ""
        )
    }
}

/// Response from a RAG query including the answer and context.
public struct RAGResponse: Sendable, Codable {
    /// The generated answer.
    public let answer: String

    /// The retrieved context used to generate the answer.
    public let context: RAGContext

    /// Whether context was available for the response.
    public var hadContext: Bool {
        context.hasContext
    }
}

// MARK: - CustomStringConvertible

extension RAGContext: CustomStringConvertible {
    public var description: String {
        """
        RAGContext(
          query: "\(query.prefix(50))..."
          keywords: \(keywords)
          matchedNodes: \(matchedNodes.count)
          paths: \(paths.count)
          sentences: \(contextSentences.count)
        )
        """
    }
}
