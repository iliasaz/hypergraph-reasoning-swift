import Foundation

/// Extracts hypergraph knowledge structures from text using LLM inference.
///
/// This is the main component for converting unstructured text into a hypergraph
/// representation. It uses an LLMProvider to call an LLM with specialized
/// prompts that extract Subject-Verb-Object triples.
public actor HypergraphExtractor {

    /// The LLM provider for inference.
    private let llmProvider: any LLMProvider

    /// The model to use for extraction.
    private let model: String

    /// Text splitter for chunking documents.
    private let textSplitter: RecursiveTextSplitter

    /// Whether to distill text before extraction.
    private let distillByDefault: Bool

    /// Creates a hypergraph extractor with any LLM provider.
    ///
    /// - Parameters:
    ///   - llmProvider: The LLM provider to use.
    ///   - model: The LLM model name to use for extraction.
    ///   - chunkSize: Size of text chunks. Defaults to 1000.
    ///   - chunkOverlap: Overlap between chunks. Defaults to 0.
    ///   - distillByDefault: Whether to distill text by default. Defaults to false.
    public init(
        llmProvider: any LLMProvider,
        model: String,
        chunkSize: Int = 1000,
        chunkOverlap: Int = 0,
        distillByDefault: Bool = false
    ) {
        self.llmProvider = llmProvider
        self.model = model
        self.textSplitter = RecursiveTextSplitter(
            chunkSize: chunkSize,
            chunkOverlap: chunkOverlap
        )
        self.distillByDefault = distillByDefault
    }

    /// Creates a hypergraph extractor with an Ollama service (convenience initializer).
    ///
    /// - Parameters:
    ///   - ollamaService: The Ollama service to use.
    ///   - model: The LLM model name. Defaults to "gpt-oss:20b".
    ///   - chunkSize: Size of text chunks. Defaults to 1000.
    ///   - chunkOverlap: Overlap between chunks. Defaults to 0.
    ///   - distillByDefault: Whether to distill text by default. Defaults to false.
    @MainActor
    public init(
        ollamaService: OllamaService,
        model: String? = nil,
        chunkSize: Int = 1000,
        chunkOverlap: Int = 0,
        distillByDefault: Bool = false
    ) {
        self.llmProvider = ollamaService
        self.model = model ?? ollamaService.defaultModel
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
    /// - Returns: A tuple of (combined hypergraph, all metadata, chunk index).
    public func extractFromDocument(
        _ text: String,
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata], ChunkIndex) {
        let chunks = textSplitter.split(text)

        guard !chunks.isEmpty else {
            return (Hypergraph(), [], ChunkIndex())
        }

        // Build chunk index for O(1) lookup (provenance and citations)
        let chunkIndex = ChunkIndex(chunks: chunks)

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

        return (combinedGraph, allMetadata, chunkIndex)
    }

    // MARK: - Helper Methods

    /// Distills text using the LLM.
    private func distillText(_ text: String) async throws -> String {
        try await llmProvider.chat(
            systemPrompt: SystemPrompts.distillation,
            userPrompt: SystemPrompts.distillationUserPrompt(text: text),
            model: model,
            temperature: nil
        )
    }

    /// Extracts events from text using the LLM.
    private func extractEvents(from text: String) async throws -> HypergraphJSON {
        let rawResult = try await llmProvider.generate(
            systemPrompt: SystemPrompts.hypergraphExtraction,
            userPrompt: SystemPrompts.extractionUserPrompt(text: text),
            responseType: HypergraphJSON.self,
            model: model,
            temperature: nil
        )

        // Filter out vague/pronominal nodes that slipped through
        return NodeFilter.filter(rawResult)
    }
}

// MARK: - Batch Processing

extension HypergraphExtractor {

    /// Processes multiple documents and returns individual results.
    ///
    /// - Parameters:
    ///   - documents: Array of (identifier, text) tuples.
    ///   - distill: Whether to distill each chunk.
    /// - Returns: Dictionary mapping identifiers to (hypergraph, metadata, chunkIndex) results.
    public func processDocuments(
        _ documents: [(id: String, text: String)],
        distill: Bool? = nil
    ) async throws -> [String: (Hypergraph<String, String>, [ChunkMetadata], ChunkIndex)] {
        var results = [String: (Hypergraph<String, String>, [ChunkMetadata], ChunkIndex)]()

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
    /// - Returns: A tuple of (merged hypergraph, all metadata, merged chunkIndex).
    public func processAndMergeDocuments(
        _ documents: [(id: String, text: String)],
        distill: Bool? = nil
    ) async throws -> (Hypergraph<String, String>, [ChunkMetadata], ChunkIndex) {
        var combinedGraph = Hypergraph<String, String>()
        var allMetadata = [ChunkMetadata]()
        var combinedChunks = ChunkIndex()

        for (_, text) in documents {
            let (docGraph, docMetadata, docChunks) = try await extractFromDocument(text, distill: distill)
            combinedGraph.formUnion(docGraph)
            allMetadata.append(contentsOf: docMetadata)
            combinedChunks.merge(docChunks)
        }

        return (combinedGraph, allMetadata, combinedChunks)
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
