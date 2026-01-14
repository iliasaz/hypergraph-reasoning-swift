import Foundation

/// Extracts hypergraph knowledge structures from text using LLM inference.
///
/// This is the main component for converting unstructured text into a hypergraph
/// representation. It uses the OllamaService to call an LLM with specialized
/// prompts that extract Subject-Verb-Object triples.
public actor HypergraphExtractor {

    /// The Ollama service for LLM inference.
    private let ollamaService: OllamaService

    /// The model to use for extraction.
    private let model: String

    /// Text splitter for chunking documents.
    private let textSplitter: RecursiveTextSplitter

    /// Whether to distill text before extraction.
    private let distillByDefault: Bool

    /// Creates a hypergraph extractor.
    ///
    /// - Parameters:
    ///   - ollamaService: The Ollama service to use.
    ///   - model: The LLM model name. Defaults to "gpt-oss:20b".
    ///   - chunkSize: Size of text chunks. Defaults to 10000.
    ///   - chunkOverlap: Overlap between chunks. Defaults to 0.
    ///   - distillByDefault: Whether to distill text by default. Defaults to false.
    public init(
        ollamaService: OllamaService,
        model: String = "gpt-oss:20b",
        chunkSize: Int = 10000,
        chunkOverlap: Int = 0,
        distillByDefault: Bool = false
    ) {
        self.ollamaService = ollamaService
        self.model = model
        self.textSplitter = RecursiveTextSplitter(
            chunkSize: chunkSize,
            chunkOverlap: chunkOverlap
        )
        self.distillByDefault = distillByDefault
    }

    // MARK: - Extraction from Text

    /// Extracts a hypergraph from a single text chunk.
    ///
    /// - Parameters:
    ///   - chunk: The text chunk to process.
    ///   - distill: Whether to distill the text first. Defaults to the extractor's default.
    /// - Returns: A tuple of (hypergraph, metadata array).
    public func extractFromChunk(
        _ chunk: TextChunk,
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata]) {
        let shouldDistill = distill ?? distillByDefault
        var textToProcess = chunk.text

        // Optionally distill the text first
        if shouldDistill {
            textToProcess = try await distillText(chunk.text)
        }

        // Extract events using LLM
        let events = try await extractEvents(from: textToProcess)

        // Convert to hypergraph
        let (hypergraph, metadata) = events.toHypergraph(chunkID: chunk.chunkID)

        return (hypergraph, metadata)
    }

    /// Extracts a hypergraph from a text string.
    ///
    /// - Parameters:
    ///   - text: The text to process.
    ///   - distill: Whether to distill the text first.
    /// - Returns: A tuple of (hypergraph, metadata array).
    public func extractFromText(
        _ text: String,
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata]) {
        let chunk = TextChunk(text: text)
        return try await extractFromChunk(chunk, distill: distill)
    }

    /// Extracts a hypergraph from a document by splitting it into chunks.
    ///
    /// - Parameters:
    ///   - text: The full document text.
    ///   - distill: Whether to distill each chunk.
    /// - Returns: A tuple of (combined hypergraph, all metadata).
    public func extractFromDocument(
        _ text: String,
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata]) {
        let chunks = textSplitter.split(text)

        guard !chunks.isEmpty else {
            return (Hypergraph(), [])
        }

        var combinedGraph = Hypergraph<String, String>()
        var allMetadata = [ChunkMetadata]()

        for chunk in chunks {
            do {
                let (chunkGraph, chunkMetadata) = try await extractFromChunk(chunk, distill: distill)
                combinedGraph.formUnion(chunkGraph)
                allMetadata.append(contentsOf: chunkMetadata)
            } catch {
                // Log error but continue with other chunks
                print("Warning: Failed to extract from chunk \(chunk.chunkID): \(error)")
            }
        }

        return (combinedGraph, allMetadata)
    }

    // MARK: - Helper Methods

    /// Distills text using the LLM.
    private func distillText(_ text: String) async throws -> String {
        try await ollamaService.chat(
            systemPrompt: SystemPrompts.distillation,
            userPrompt: SystemPrompts.distillationUserPrompt(text: text),
            model: model
        )
    }

    /// Extracts events from text using the LLM.
    private func extractEvents(from text: String) async throws -> HypergraphJSON {
        try await ollamaService.generate(
            systemPrompt: SystemPrompts.hypergraphExtraction,
            userPrompt: SystemPrompts.extractionUserPrompt(text: text),
            responseType: HypergraphJSON.self,
            model: model
        )
    }
}

// MARK: - Batch Processing

extension HypergraphExtractor {

    /// Processes multiple documents and returns individual results.
    ///
    /// - Parameters:
    ///   - documents: Array of (identifier, text) tuples.
    ///   - distill: Whether to distill each chunk.
    /// - Returns: Dictionary mapping identifiers to (hypergraph, metadata) results.
    public func processDocuments(
        _ documents: [(id: String, text: String)],
        distill: Bool? = nil
    ) async throws -> [String: (Hypergraph<String, String>, [ChunkMetadata])] {
        var results = [String: (Hypergraph<String, String>, [ChunkMetadata])]()

        for (id, text) in documents {
            let result = try await extractFromDocument(text, distill: distill)
            results[id] = result
        }

        return results
    }

    /// Processes multiple documents and merges results into a single hypergraph.
    ///
    /// - Parameters:
    ///   - documents: Array of (identifier, text) tuples.
    ///   - distill: Whether to distill each chunk.
    /// - Returns: A tuple of (merged hypergraph, all metadata).
    public func processAndMergeDocuments(
        _ documents: [(id: String, text: String)],
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata]) {
        var combinedGraph = Hypergraph<String, String>()
        var allMetadata = [ChunkMetadata]()

        for (_, text) in documents {
            let (docGraph, docMetadata) = try await extractFromDocument(text, distill: distill)
            combinedGraph.formUnion(docGraph)
            allMetadata.append(contentsOf: docMetadata)
        }

        return (combinedGraph, allMetadata)
    }
}

// MARK: - Result Type

/// Result of hypergraph extraction from a document.
public struct ExtractionResult: Sendable {
    /// The extracted hypergraph.
    public let hypergraph: Hypergraph<String, String>

    /// Metadata for each extracted edge.
    public let metadata: [ChunkMetadata]

    /// The document identifier (if provided).
    public let documentID: String?

    /// Number of chunks processed.
    public let chunkCount: Int

    /// Whether any errors occurred during processing.
    public let hasErrors: Bool

    /// Creates an extraction result.
    public init(
        hypergraph: Hypergraph<String, String>,
        metadata: [ChunkMetadata],
        documentID: String? = nil,
        chunkCount: Int = 1,
        hasErrors: Bool = false
    ) {
        self.hypergraph = hypergraph
        self.metadata = metadata
        self.documentID = documentID
        self.chunkCount = chunkCount
        self.hasErrors = hasErrors
    }
}
