/// HyperGraphReasoning - Swift library for building hypergraph knowledge representations.
///
/// This library provides tools for extracting knowledge graphs from text using
/// LLM-based inference. It supports:
///
/// - **Hypergraph Data Structure**: A flexible hypergraph implementation where
///   edges can connect any number of nodes.
/// - **Text Processing**: Recursive text splitting for chunking documents.
/// - **LLM Integration**: Multiple providers (Ollama, OpenRouter) for inference.
/// - **Knowledge Extraction**: Automated extraction of Subject-Verb-Object triples
///   from scientific text.
///
/// ## Quick Start
///
/// ```swift
/// import HyperGraphReasoning
///
/// // Using Ollama (local)
/// let ollama = OllamaService()
/// let processor = DocumentProcessor(ollamaService: ollama)
///
/// // Or using OpenRouter (cloud)
/// let openRouter = try OpenRouterService(apiKey: "sk-or-...")
/// let processor = DocumentProcessor(llmProvider: openRouter, ollamaService: ollama)
///
/// // Process a markdown file
/// let result = try await processor.processMarkdownFile(at: fileURL)
///
/// // Access the hypergraph
/// print("Nodes: \(result.hypergraph.nodeCount)")
/// print("Edges: \(result.hypergraph.edgeCount)")
///
/// // Save the result
/// try processor.saveResult(result, to: outputDir)
/// ```
///
/// ## Core Types
///
/// - ``Hypergraph``: The main hypergraph data structure.
/// - ``LLMProvider``: Protocol for LLM providers.
/// - ``OllamaService``: Local LLM inference via Ollama.
/// - ``OpenRouterService``: Cloud LLM inference via OpenRouter.
/// - ``HypergraphExtractor``: Extracts hypergraphs from text.
/// - ``DocumentProcessor``: Full pipeline for processing documents.
/// - ``NodeEmbeddings``: Storage for node embedding vectors.
///
/// ## Python Compatibility
///
/// Hypergraphs can be exported in a format compatible with HyperNetX:
///
/// ```swift
/// try hypergraph.saveForHyperNetX(to: outputURL)
/// ```
///
/// This can then be loaded in Python:
///
/// ```python
/// import hypernetx as hnx
/// import json
///
/// with open('graph.json') as f:
///     data = json.load(f)
/// H = hnx.Hypergraph(data['incidence_dict'])
/// ```

// Re-export all public types for convenience

// Core
@_exported import struct Foundation.URL
@_exported import struct Foundation.Data

// Note: The following would typically be @_exported but Swift 6 doesn't
// allow re-exporting from the same module. The types are public by default.

/// Version of the HyperGraphReasoning library.
public let version = "0.1.0"

/// Default configuration for the library.
public enum Configuration {
    /// Default chunk size for text processing.
    public static let defaultChunkSize = 1000

    /// Default chunk overlap.
    public static let defaultChunkOverlap = 0

    /// Default temperature for LLM generation.
    public static let defaultTemperature = 0.333

    /// Default similarity threshold for node merging.
    public static let defaultSimilarityThreshold: Float = 0.9

    /// Default chat model for Ollama.
    public static let defaultChatModel = "gpt-oss:20b"

    /// Default embedding model.
    public static let defaultEmbeddingModel = "nomic-embed-text:v1.5"

    /// Default chat model for OpenRouter.
    public static let defaultOpenRouterModel = "meta-llama/llama-4-maverick"
}
