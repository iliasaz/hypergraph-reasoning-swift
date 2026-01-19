import Foundation

/// Processes markdown documents into hypergraphs with embeddings.
///
/// This is the main pipeline for the hypergraph building process,
/// combining text extraction, hypergraph generation, and embedding creation.
public actor DocumentProcessor {

    /// The hypergraph extractor.
    private let extractor: HypergraphExtractor

    /// The embedding service.
    private let embeddingService: EmbeddingService

    /// Default chunk size for processing.
    private let defaultChunkSize: Int

    /// Creates a document processor with Ollama service.
    ///
    /// - Parameters:
    ///   - ollamaService: The Ollama service for LLM inference and embeddings.
    ///   - chatModel: Model for chat/extraction. Defaults to "gpt-oss:20b".
    ///   - embeddingModel: Model for embeddings. Defaults to "nomic-embed-text:v1.5".
    ///   - chunkSize: Default chunk size. Defaults to 1000.
    @MainActor
    public init(
        ollamaService: OllamaService,
        chatModel: String = "gpt-oss:20b",
        embeddingModel: String = "nomic-embed-text:v1.5",
        chunkSize: Int = 1000
    ) {
        self.extractor = HypergraphExtractor(
            ollamaService: ollamaService,
            model: chatModel,
            chunkSize: chunkSize
        )
        self.embeddingService = EmbeddingService(
            ollamaService: ollamaService,
            model: embeddingModel
        )
        self.defaultChunkSize = chunkSize
    }

    /// Creates a document processor with any LLM provider.
    ///
    /// - Parameters:
    ///   - llmProvider: The LLM provider for chat/extraction.
    ///   - ollamaService: The Ollama service for embeddings (required for embedding generation).
    ///   - chatModel: Model for chat/extraction.
    ///   - embeddingModel: Model for embeddings. Defaults to "nomic-embed-text:v1.5".
    ///   - chunkSize: Default chunk size. Defaults to 1000.
    public init(
        llmProvider: any LLMProvider,
        ollamaService: OllamaService,
        chatModel: String,
        embeddingModel: String = "nomic-embed-text:v1.5",
        chunkSize: Int = 1000
    ) {
        self.extractor = HypergraphExtractor(
            llmProvider: llmProvider,
            model: chatModel,
            chunkSize: chunkSize
        )
        self.embeddingService = EmbeddingService(
            ollamaService: ollamaService,
            model: embeddingModel
        )
        self.defaultChunkSize = chunkSize
    }

    /// Creates a document processor with existing services.
    ///
    /// - Parameters:
    ///   - extractor: The hypergraph extractor to use.
    ///   - embeddingService: The embedding service to use.
    ///   - chunkSize: Default chunk size.
    public init(
        extractor: HypergraphExtractor,
        embeddingService: EmbeddingService,
        chunkSize: Int = 1000
    ) {
        self.extractor = extractor
        self.embeddingService = embeddingService
        self.defaultChunkSize = chunkSize
    }

    // MARK: - Single Document Processing

    /// Processes a markdown file.
    ///
    /// - Parameters:
    ///   - path: Path to the markdown file.
    ///   - generateEmbeddings: Whether to generate embeddings. Defaults to true.
    /// - Returns: The processing result.
    public func processMarkdownFile(
        at path: URL,
        generateEmbeddings: Bool = true
    ) async throws -> ProcessingResult {
        let text = try String(contentsOf: path, encoding: .utf8)
        let documentID = path.deletingPathExtension().lastPathComponent

        return try await processText(
            text,
            documentID: documentID,
            generateEmbeddings: generateEmbeddings
        )
    }

    /// Processes text content.
    ///
    /// - Parameters:
    ///   - text: The text to process.
    ///   - documentID: Optional document identifier.
    ///   - generateEmbeddings: Whether to generate embeddings.
    /// - Returns: The processing result.
    public func processText(
        _ text: String,
        documentID: String? = nil,
        generateEmbeddings: Bool = true
    ) async throws -> ProcessingResult {
        // Extract hypergraph and chunks
        let (hypergraph, metadata, chunkIndex) = try await extractor.extractFromDocument(text)

        // Optionally generate embeddings
        var embeddings = NodeEmbeddings()
        if generateEmbeddings && !hypergraph.nodes.isEmpty {
            embeddings = try await embeddingService.generateEmbeddings(for: hypergraph)
        }

        return ProcessingResult(
            hypergraph: hypergraph,
            metadata: metadata,
            embeddings: embeddings,
            chunkIndex: chunkIndex,
            documentID: documentID
        )
    }

    // MARK: - Batch Processing

    /// Processes a directory of markdown files in parallel.
    ///
    /// Files are processed concurrently for faster execution. Each file's
    /// document ID is preserved in the result for provenance tracking.
    ///
    /// - Parameters:
    ///   - directoryPath: Path to the directory.
    ///   - recursive: Whether to search subdirectories. Defaults to true.
    ///   - maxConcurrency: Maximum concurrent tasks. Defaults to 4.
    ///   - generateEmbeddings: Whether to generate embeddings.
    ///   - verbose: Whether to print progress. Defaults to false.
    /// - Returns: Array of processing results (one per file).
    public func processMarkdownDirectory(
        at directoryPath: URL,
        recursive: Bool = true,
        maxConcurrency: Int = 4,
        generateEmbeddings: Bool = true,
        verbose: Bool = false
    ) async throws -> [ProcessingResult] {
        // Find all markdown files
        let markdownFiles = try findMarkdownFiles(in: directoryPath, recursive: recursive)

        guard !markdownFiles.isEmpty else {
            return []
        }

        let total = markdownFiles.count
        var completed = 0
        var results = [ProcessingResult]()

        if verbose {
            print("Found \(total) markdown files to process")
        }

        // Process in batches to limit concurrency
        for batch in markdownFiles.chunked(into: maxConcurrency) {
            let batchResults = await withTaskGroup(of: (URL, ProcessingResult?).self) { group in
                for file in batch {
                    group.addTask {
                        do {
                            let result = try await self.processMarkdownFile(
                                at: file,
                                generateEmbeddings: generateEmbeddings
                            )
                            return (file, result)
                        } catch {
                            print("Warning: Failed to process \(file.lastPathComponent): \(error)")
                            return (file, nil)
                        }
                    }
                }

                var batchResults = [(URL, ProcessingResult?)]()
                for await result in group {
                    batchResults.append(result)
                }
                return batchResults
            }

            // Collect successful results and report progress
            for (file, result) in batchResults {
                completed += 1
                if verbose {
                    print("[\(completed)/\(total)] Processed: \(file.lastPathComponent)")
                }
                if let result = result {
                    results.append(result)
                }
            }
        }

        return results
    }

    /// Finds all markdown files in a directory.
    ///
    /// - Parameters:
    ///   - directory: The directory to search.
    ///   - recursive: Whether to search subdirectories.
    /// - Returns: Array of file URLs.
    private func findMarkdownFiles(in directory: URL, recursive: Bool) throws -> [URL] {
        let fileManager = FileManager.default

        if recursive {
            var files = [URL]()
            if let enumerator = fileManager.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) {
                for case let fileURL as URL in enumerator {
                    if fileURL.pathExtension.lowercased() == "md" {
                        files.append(fileURL)
                    }
                }
            }
            return files.sorted { $0.path < $1.path }
        } else {
            let contents = try fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil
            )
            return contents
                .filter { $0.pathExtension.lowercased() == "md" }
                .sorted { $0.path < $1.path }
        }
    }

    /// Processes multiple markdown files and merges them into a single result.
    ///
    /// This is the recommended approach for multi-file processing:
    /// 1. Process each file individually (preserving provenance)
    /// 2. Merge hypergraphs
    /// 3. Optionally update embeddings for the merged graph
    ///
    /// - Parameters:
    ///   - directoryPath: Path to the directory.
    ///   - recursive: Whether to search subdirectories. Defaults to true.
    ///   - maxConcurrency: Maximum concurrent tasks. Defaults to 4.
    ///   - generateEmbeddings: Whether to generate embeddings.
    ///   - verbose: Whether to print progress. Defaults to false.
    /// - Returns: A single merged processing result.
    public func processAndMergeDirectory(
        at directoryPath: URL,
        recursive: Bool = true,
        maxConcurrency: Int = 4,
        generateEmbeddings: Bool = true,
        verbose: Bool = false
    ) async throws -> ProcessingResult {
        let results = try await processMarkdownDirectory(
            at: directoryPath,
            recursive: recursive,
            maxConcurrency: maxConcurrency,
            generateEmbeddings: generateEmbeddings,
            verbose: verbose
        )

        guard !results.isEmpty else {
            return ProcessingResult(
                hypergraph: Hypergraph(),
                metadata: [],
                embeddings: NodeEmbeddings(),
                chunkIndex: ChunkIndex()
            )
        }

        // Merge results (embeddings will be combined)
        return try await mergeResults(results, updateEmbeddings: generateEmbeddings)
    }

    // MARK: - Merging

    /// Merges multiple processing results into a single result.
    ///
    /// - Parameters:
    ///   - results: The results to merge.
    ///   - updateEmbeddings: Whether to update embeddings for the merged graph.
    /// - Returns: A merged processing result.
    public func mergeResults(
        _ results: [ProcessingResult],
        updateEmbeddings: Bool = true
    ) async throws -> ProcessingResult {
        var combinedGraph = Hypergraph<String, String>()
        var allMetadata = [ChunkMetadata]()
        var combinedEmbeddings = NodeEmbeddings()
        var combinedChunks = ChunkIndex()

        for result in results {
            combinedGraph.formUnion(result.hypergraph)
            allMetadata.append(contentsOf: result.metadata)
            combinedEmbeddings.merge(result.embeddings)
            combinedChunks.merge(result.chunkIndex)
        }

        // Update embeddings if needed
        if updateEmbeddings {
            combinedEmbeddings = try await embeddingService.updateEmbeddings(
                existing: combinedEmbeddings,
                for: combinedGraph
            )
        }

        return ProcessingResult(
            hypergraph: combinedGraph,
            metadata: allMetadata,
            embeddings: combinedEmbeddings,
            chunkIndex: combinedChunks
        )
    }

    // MARK: - Graph Simplification

    /// Simplifies a hypergraph by merging similar nodes.
    ///
    /// - Parameters:
    ///   - result: The processing result to simplify.
    ///   - similarityThreshold: Minimum similarity for merging. Defaults to 0.9.
    /// - Returns: A simplified processing result.
    public func simplifyGraph(
        _ result: ProcessingResult,
        similarityThreshold: Float = 0.9
    ) async throws -> ProcessingResult {
        // Find similar node pairs
        let similarPairs = await embeddingService.findSimilarPairs(
            in: result.embeddings,
            threshold: similarityThreshold
        )

        guard !similarPairs.isEmpty else {
            return result
        }

        // Build merge mapping (merge lower-degree node into higher-degree node)
        var mergeMap = [String: String]()
        var merged = Set<String>()

        for (node1, node2, _) in similarPairs {
            if merged.contains(node1) || merged.contains(node2) {
                continue
            }

            let degree1 = result.hypergraph.degree(of: node1)
            let degree2 = result.hypergraph.degree(of: node2)

            let (keep, remove) = degree1 >= degree2 ? (node1, node2) : (node2, node1)
            mergeMap[remove] = keep
            merged.insert(remove)
        }

        // Apply merges to create new incidence dict
        var newIncidence = [String: Set<String>]()
        for (edgeID, nodes) in result.hypergraph.incidenceDict {
            let newNodes = Set(nodes.map { mergeMap[$0] ?? $0 })
            if newNodes.count > 1 {  // Keep only edges with 2+ nodes after merge
                newIncidence[edgeID] = newNodes
            }
        }

        let simplifiedGraph = Hypergraph<String, String>(incidenceDict: newIncidence)

        // Update embeddings (remove merged nodes)
        var simplifiedEmbeddings = result.embeddings
        simplifiedEmbeddings.prune(to: simplifiedGraph.nodes)

        // Update metadata (apply node renames)
        let updatedMetadata = result.metadata.map { meta in
            ChunkMetadata(
                edge: meta.edge,
                nodes: Set(meta.nodes.map { mergeMap[$0] ?? $0 }),
                source: meta.source.map { mergeMap[$0] ?? $0 },
                target: meta.target.map { mergeMap[$0] ?? $0 },
                chunkID: meta.chunkID
            )
        }

        return ProcessingResult(
            hypergraph: simplifiedGraph,
            metadata: updatedMetadata,
            embeddings: simplifiedEmbeddings,
            chunkIndex: result.chunkIndex,
            documentID: result.documentID
        )
    }

    // MARK: - Saving Results

    /// Saves a processing result to disk.
    ///
    /// - Parameters:
    ///   - result: The result to save.
    ///   - directory: The output directory.
    public func saveResult(_ result: ProcessingResult, to directory: URL) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        let baseName = result.documentID ?? "hypergraph"
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        // Save hypergraph
        let graphPath = directory.appendingPathComponent("\(baseName)_graph.json")
        let graphData = try encoder.encode(result.hypergraph)
        try graphData.write(to: graphPath)

        // Save metadata
        let metadataPath = directory.appendingPathComponent("\(baseName)_metadata.json")
        let metadataData = try encoder.encode(result.metadata)
        try metadataData.write(to: metadataPath)

        // Save embeddings
        let embeddingsPath = directory.appendingPathComponent("\(baseName)_embeddings.json")
        let embeddingsData = try encoder.encode(result.embeddings)
        try embeddingsData.write(to: embeddingsPath)

        // Save chunks (for provenance and citations)
        let chunksPath = directory.appendingPathComponent("\(baseName)_chunks.json")
        try result.chunkIndex.save(to: chunksPath)
    }
}

// MARK: - Processing Result

/// The result of document processing.
public struct ProcessingResult: Sendable, Codable {
    /// The extracted hypergraph.
    public let hypergraph: Hypergraph<String, String>

    /// Metadata for each edge.
    public let metadata: [ChunkMetadata]

    /// Node embeddings.
    public let embeddings: NodeEmbeddings

    /// Index of original text chunks for provenance and citations.
    public let chunkIndex: ChunkIndex

    /// Document identifier (if any).
    public let documentID: String?

    /// Creates a processing result.
    public init(
        hypergraph: Hypergraph<String, String>,
        metadata: [ChunkMetadata],
        embeddings: NodeEmbeddings,
        chunkIndex: ChunkIndex = ChunkIndex(),
        documentID: String? = nil
    ) {
        self.hypergraph = hypergraph
        self.metadata = metadata
        self.embeddings = embeddings
        self.chunkIndex = chunkIndex
        self.documentID = documentID
    }

    /// Number of nodes in the hypergraph.
    public var nodeCount: Int {
        hypergraph.nodeCount
    }

    /// Number of edges in the hypergraph.
    public var edgeCount: Int {
        hypergraph.edgeCount
    }
}

extension ProcessingResult: CustomStringConvertible {
    public var description: String {
        "ProcessingResult(nodes: \(nodeCount), edges: \(edgeCount), embeddings: \(embeddings.count))"
    }
}

// MARK: - Array Chunking Extension

extension Array {
    /// Splits the array into chunks of the specified size.
    ///
    /// - Parameter size: The maximum size of each chunk.
    /// - Returns: An array of arrays, each containing at most `size` elements.
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [self] }
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
