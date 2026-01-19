import ArgumentParser
import Foundation
import HyperGraphReasoning

/// LLM provider options for the CLI.
enum LLMProviderOption: String, ExpressibleByArgument, CaseIterable {
    case ollama
    case openrouter

    static var defaultValueDescription: String { "ollama" }
}

@main
struct HypergraphCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hypergraph-cli",
        abstract: "CLI tool for building hypergraph knowledge representations",
        version: "0.1.0",
        subcommands: [Process.self, Extract.self, Embed.self, Simplify.self, Info.self, Query.self]
    )
}

// MARK: - Process Command

extension HypergraphCLI {
    struct Process: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Process a markdown file or directory into a hypergraph"
        )

        @Argument(help: "Input markdown file or directory")
        var input: String

        @Option(name: .shortAndLong, help: "Output directory for results")
        var output: String = "./output"

        @Option(name: .long, help: "LLM provider (ollama or openrouter)")
        var provider: LLMProviderOption = .ollama

        @Option(name: .long, help: "OpenRouter API key (required if provider is openrouter)")
        var apiKey: String?

        @Option(name: .long, help: "Chat model for extraction")
        var chatModel: String?

        @Option(name: .long, help: "Embedding model (Ollama only)")
        var embeddingModel: String = "nomic-embed-text:v1.5"

        @Option(name: .long, help: "Chunk size for text splitting")
        var chunkSize: Int = 10000

        @Flag(name: .long, help: "Skip embedding generation")
        var skipEmbeddings: Bool = false

        @Flag(name: .shortAndLong, help: "Enable verbose output")
        var verbose: Bool = false

        func run() async throws {
            let inputURL = URL(fileURLWithPath: input)
            let outputURL = URL(fileURLWithPath: output)

            // Determine the chat model
            let effectiveChatModel: String
            switch provider {
            case .ollama:
                effectiveChatModel = chatModel ?? "gpt-oss:20b"
            case .openrouter:
                effectiveChatModel = chatModel ?? "meta-llama/llama-4-maverick"
            }

            if verbose {
                print("Input: \(inputURL.path)")
                print("Output: \(outputURL.path)")
                print("Provider: \(provider.rawValue)")
                print("Chat Model: \(effectiveChatModel)")
                print("Embedding Model: \(embeddingModel)")
                print("Chunk Size: \(chunkSize)")
            }

            // Create Ollama service (always needed for embeddings)
            let ollama = await MainActor.run {
                OllamaService(
                    chatModel: effectiveChatModel,
                    embeddingModel: embeddingModel
                )
            }

            // Create the appropriate LLM provider
            let llmProvider: any LLMProvider
            switch provider {
            case .ollama:
                llmProvider = ollama
            case .openrouter:
                guard let key = apiKey else {
                    throw ValidationError("--api-key is required when using openrouter provider")
                }
                llmProvider = try OpenRouterService(
                    apiKey: key,
                    model: effectiveChatModel
                )
            }

            let processor = DocumentProcessor(
                llmProvider: llmProvider,
                ollamaService: ollama,
                chatModel: effectiveChatModel,
                embeddingModel: embeddingModel,
                chunkSize: chunkSize
            )

            // Create output directory
            try FileManager.default.createDirectory(
                at: outputURL,
                withIntermediateDirectories: true
            )

            // Check if input is file or directory
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory) else {
                throw ValidationError("Input path does not exist: \(input)")
            }

            if isDirectory.boolValue {
                // Process directory
                print("Processing directory: \(inputURL.path)")
                let results = try await processor.processMarkdownDirectory(
                    at: inputURL,
                    outputDir: outputURL,
                    generateEmbeddings: !skipEmbeddings
                )

                print("Processed \(results.count) files")

                // Merge all results
                if !results.isEmpty {
                    print("Merging results...")
                    let merged = try await processor.mergeResults(results)
                    try await processor.saveResult(merged, to: outputURL)
                    print("Final hypergraph: \(merged.nodeCount) nodes, \(merged.edgeCount) edges")
                }
            } else {
                // Process single file
                print("Processing file: \(inputURL.path)")
                let result = try await processor.processMarkdownFile(
                    at: inputURL,
                    generateEmbeddings: !skipEmbeddings
                )

                try await processor.saveResult(result, to: outputURL)
                print("Hypergraph: \(result.nodeCount) nodes, \(result.edgeCount) edges")
                print("Results saved to: \(outputURL.path)")
            }
        }
    }
}

// MARK: - Extract Command

extension HypergraphCLI {
    struct Extract: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Extract hypergraph from text (without embeddings)"
        )

        @Argument(help: "Input text or file path")
        var input: String

        @Option(name: .shortAndLong, help: "Output JSON file")
        var output: String = "hypergraph.json"

        @Option(name: .long, help: "LLM provider (ollama or openrouter)")
        var provider: LLMProviderOption = .ollama

        @Option(name: .long, help: "OpenRouter API key (required if provider is openrouter)")
        var apiKey: String?

        @Option(name: .long, help: "Chat model for extraction")
        var chatModel: String?

        @Flag(name: .long, help: "Input is a file path")
        var file: Bool = false

        func run() async throws {
            let text: String
            if file {
                let url = URL(fileURLWithPath: input)
                text = try String(contentsOf: url, encoding: .utf8)
            } else {
                text = input
            }

            // Determine the chat model
            let effectiveChatModel: String
            switch provider {
            case .ollama:
                effectiveChatModel = chatModel ?? "gpt-oss:20b"
            case .openrouter:
                effectiveChatModel = chatModel ?? "meta-llama/llama-4-maverick"
            }

            print("Extracting hypergraph using \(provider.rawValue)...")

            // Create the appropriate LLM provider
            let llmProvider: any LLMProvider
            switch provider {
            case .ollama:
                llmProvider = await MainActor.run {
                    OllamaService(chatModel: effectiveChatModel)
                }
            case .openrouter:
                guard let key = apiKey else {
                    throw ValidationError("--api-key is required when using openrouter provider")
                }
                llmProvider = try OpenRouterService(
                    apiKey: key,
                    model: effectiveChatModel
                )
            }

            let extractor = HypergraphExtractor(
                llmProvider: llmProvider,
                model: effectiveChatModel
            )

            let (hypergraph, metadata) = try await extractor.extractFromDocument(text)

            print("Extracted \(hypergraph.nodeCount) nodes, \(hypergraph.edgeCount) edges")

            // Save hypergraph
            let outputURL = URL(fileURLWithPath: output)
            try hypergraph.save(to: outputURL)
            print("Saved to: \(outputURL.path)")

            // Save metadata
            let metadataURL = outputURL.deletingPathExtension()
                .appendingPathExtension("metadata.json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let metadataData = try encoder.encode(metadata)
            try metadataData.write(to: metadataURL)
            print("Metadata saved to: \(metadataURL.path)")
        }
    }
}

// MARK: - Embed Command

extension HypergraphCLI {
    struct Embed: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Generate embeddings for nodes in a hypergraph"
        )

        @Argument(help: "Input hypergraph JSON file")
        var input: String

        @Option(name: .shortAndLong, help: "Output embeddings JSON file")
        var output: String = "embeddings.json"

        @Option(name: .long, help: "Embedding model")
        var model: String = "nomic-embed-text:v1.5"

        func run() async throws {
            let inputURL = URL(fileURLWithPath: input)
            let outputURL = URL(fileURLWithPath: output)

            print("Loading hypergraph from: \(inputURL.path)")
            let hypergraph = try Hypergraph<String, String>.load(from: inputURL)
            print("Loaded \(hypergraph.nodeCount) nodes")

            print("Generating embeddings with model: \(model)")
            let ollama = await MainActor.run {
                OllamaService(embeddingModel: model)
            }
            let embeddingService = EmbeddingService(
                ollamaService: ollama,
                model: model
            )

            let embeddings = try await embeddingService.generateEmbeddings(for: hypergraph)
            print("Generated \(embeddings.count) embeddings")

            // Save embeddings
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted]
            let data = try encoder.encode(embeddings)
            try data.write(to: outputURL)
            print("Saved to: \(outputURL.path)")
        }
    }
}

// MARK: - Simplify Command

extension HypergraphCLI {
    struct Simplify: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Simplify a hypergraph by merging similar nodes"
        )

        @Argument(help: "Input hypergraph JSON file")
        var input: String

        @Option(name: .shortAndLong, help: "Embeddings JSON file (required)")
        var embeddings: String

        @Option(name: .shortAndLong, help: "Output directory for results")
        var output: String = "./simplified"

        @Option(name: .long, help: "Similarity threshold for merging (0.0-1.0)")
        var threshold: Float = 0.9

        @Option(name: .long, help: "Node suffixes to exclude from merging (comma-separated)")
        var excludeSuffixes: String = ""

        @Flag(name: .long, help: "Recompute embeddings for keeper nodes after merging")
        var recomputeEmbeddings: Bool = false

        @Option(name: .long, help: "Embedding model for recomputation (default: nomic-embed-text:v1.5)")
        var embeddingModel: String = "nomic-embed-text:v1.5"

        @Flag(name: .shortAndLong, help: "Enable verbose output")
        var verbose: Bool = false

        func run() async throws {
            let inputURL = URL(fileURLWithPath: input)
            let embeddingsURL = URL(fileURLWithPath: embeddings)
            let outputURL = URL(fileURLWithPath: output)

            // Validate threshold
            guard threshold >= 0 && threshold <= 1 else {
                throw ValidationError("Threshold must be between 0.0 and 1.0")
            }

            // Load hypergraph
            if verbose {
                print("Loading hypergraph from: \(inputURL.path)")
            }
            let hypergraph = try StringHypergraph.load(from: inputURL)
            if verbose {
                print("Loaded \(hypergraph.nodeCount) nodes, \(hypergraph.edgeCount) edges")
            }

            // Load embeddings
            if verbose {
                print("Loading embeddings from: \(embeddingsURL.path)")
            }
            let embeddingsData = try Data(contentsOf: embeddingsURL)
            let nodeEmbeddings = try JSONDecoder().decode(NodeEmbeddings.self, from: embeddingsData)
            if verbose {
                print("Loaded \(nodeEmbeddings.count) embeddings")
            }

            // Parse exclude suffixes
            let suffixes = excludeSuffixes
                .split(separator: ",")
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }

            // Run simplification
            print("Simplifying hypergraph with threshold \(threshold)...")
            if recomputeEmbeddings {
                print("Will recompute embeddings for merged nodes using model: \(embeddingModel)")
            }

            let simplifier = HypergraphSimplifier()
            let result: SimplificationResult

            if recomputeEmbeddings {
                // Create embedding service for recomputation
                let ollama = await MainActor.run {
                    OllamaService(embeddingModel: embeddingModel)
                }
                let embeddingService = EmbeddingService(
                    ollamaService: ollama,
                    model: embeddingModel
                )

                result = try await simplifier.simplify(
                    hypergraph: hypergraph,
                    embeddings: nodeEmbeddings,
                    similarityThreshold: threshold,
                    excludeSuffixes: suffixes,
                    recomputeEmbeddings: true,
                    embeddingService: embeddingService
                )
            } else {
                result = simplifier.simplify(
                    hypergraph: hypergraph,
                    embeddings: nodeEmbeddings,
                    similarityThreshold: threshold,
                    excludeSuffixes: suffixes
                )
            }

            print("Merged \(result.mergeCount) pairs")
            print("Removed \(result.nodesRemoved) nodes, \(result.edgesRemoved) edges")
            if result.embeddingsRecomputed > 0 {
                print("Recomputed \(result.embeddingsRecomputed) embeddings")
            }
            print("Simplified: \(result.hypergraph.nodeCount) nodes, \(result.hypergraph.edgeCount) edges")

            // Create output directory
            try FileManager.default.createDirectory(
                at: outputURL,
                withIntermediateDirectories: true
            )

            // Save simplified hypergraph
            let graphURL = outputURL.appendingPathComponent("simplified_graph.json")
            try result.hypergraph.save(to: graphURL)
            print("Saved hypergraph to: \(graphURL.path)")

            // Save updated embeddings
            let embURL = outputURL.appendingPathComponent("simplified_embeddings.json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted]
            let embData = try encoder.encode(result.embeddings)
            try embData.write(to: embURL)
            print("Saved embeddings to: \(embURL.path)")

            // Save merge history
            let historyURL = outputURL.appendingPathComponent("merge_history.json")
            let historyData = try encoder.encode(result.mergeHistory)
            try historyData.write(to: historyURL)
            print("Saved merge history to: \(historyURL.path)")

            // Print merge details if verbose
            if verbose && !result.mergeHistory.isEmpty {
                print("\nMerge Details:")
                for (i, record) in result.mergeHistory.prefix(20).enumerated() {
                    let kept = record.keptNode.count > 30
                        ? String(record.keptNode.prefix(27)) + "..."
                        : record.keptNode
                    let removed = record.removedNode.count > 30
                        ? String(record.removedNode.prefix(27)) + "..."
                        : record.removedNode
                    print("  \(i + 1). '\(removed)' → '\(kept)' (sim: \(String(format: "%.3f", record.similarity)))")
                }
                if result.mergeHistory.count > 20 {
                    print("  ... and \(result.mergeHistory.count - 20) more merges")
                }
            }
        }
    }
}

// MARK: - Info Command

extension HypergraphCLI {
    struct Info: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Display information about a hypergraph"
        )

        @Argument(help: "Hypergraph JSON file")
        var input: String

        func run() throws {
            let inputURL = URL(fileURLWithPath: input)

            let hypergraph = try Hypergraph<String, String>.load(from: inputURL)

            print("Hypergraph Information")
            print("======================")
            print("Nodes: \(hypergraph.nodeCount)")
            print("Edges: \(hypergraph.edgeCount)")

            // Compute some statistics
            let components = hypergraph.connectedComponents()
            print("Connected Components: \(components.count)")

            if let largest = components.first {
                print("Largest Component: \(largest.count) nodes")
            }

            // Edge size distribution
            var sizeDistribution = [Int: Int]()
            for edge in hypergraph.edges {
                let size = hypergraph.size(of: edge)
                sizeDistribution[size, default: 0] += 1
            }

            print("\nEdge Size Distribution:")
            for size in sizeDistribution.keys.sorted() {
                let count = sizeDistribution[size]!
                print("  Size \(size): \(count) edges")
            }

            // Node degree distribution (top 10)
            var degrees = [(String, Int)]()
            for node in hypergraph.nodes {
                degrees.append((node, hypergraph.degree(of: node)))
            }
            degrees.sort { $0.1 > $1.1 }

            print("\nTop 10 Nodes by Degree:")
            for (node, degree) in degrees.prefix(10) {
                let displayNode = node.count > 40 ? String(node.prefix(37)) + "..." : node
                print("  \(displayNode): \(degree)")
            }
        }
    }
}

// MARK: - Query Command

extension HypergraphCLI {
    struct Query: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Query the hypergraph using GraphRAG"
        )

        @Argument(help: "The question to ask")
        var question: String

        @Option(name: .shortAndLong, help: "Hypergraph JSON file")
        var graph: String

        @Option(name: .shortAndLong, help: "Embeddings JSON file")
        var embeddings: String

        @Option(name: .shortAndLong, help: "Metadata JSON file (provides source/target for directional sentences)")
        var metadata: String?

        @Option(name: .long, help: "LLM provider for chat (ollama or openrouter)")
        var provider: LLMProviderOption = .ollama

        @Option(name: .long, help: "OpenRouter API key")
        var apiKey: String?

        @Option(name: .long, help: "Chat model")
        var chatModel: String?

        @Option(name: .long, help: "Embedding model")
        var embeddingModel: String = "nomic-embed-text:v1.5"

        @Option(name: .long, help: "Number of matching nodes per keyword")
        var topK: Int = 5

        @Option(name: .long, help: "Maximum path length for BFS")
        var maxPathLength: Int = 4

        @Option(name: .long, help: "Similarity threshold for node matching (0.0-1.0)")
        var threshold: Float = 0.5

        @Flag(name: .long, help: "Show retrieved context")
        var showContext: Bool = false

        @Flag(name: .long, help: "Use simple keyword extraction (no LLM)")
        var simpleKeywords: Bool = false

        @Flag(name: .long, help: "Only retrieve context (don't generate answer)")
        var contextOnly: Bool = false

        @Flag(name: .shortAndLong, help: "Enable verbose output")
        var verbose: Bool = false

        func run() async throws {
            // Load hypergraph
            let graphURL = URL(fileURLWithPath: graph)
            if verbose {
                print("Loading hypergraph from: \(graphURL.path)")
            }
            let hypergraph = try StringHypergraph.load(from: graphURL)
            if verbose {
                print("Loaded \(hypergraph.nodeCount) nodes, \(hypergraph.edgeCount) edges")
            }

            // Load embeddings
            let embeddingsURL = URL(fileURLWithPath: embeddings)
            if verbose {
                print("Loading embeddings from: \(embeddingsURL.path)")
            }
            let embeddingsData = try Data(contentsOf: embeddingsURL)
            let nodeEmbeddings = try JSONDecoder().decode(NodeEmbeddings.self, from: embeddingsData)
            if verbose {
                print("Loaded \(nodeEmbeddings.count) embeddings")
            }

            // Load metadata (optional)
            var chunkMetadata: [ChunkMetadata]? = nil
            if let metadataPath = metadata {
                let metadataURL = URL(fileURLWithPath: metadataPath)
                if verbose {
                    print("Loading metadata from: \(metadataURL.path)")
                }
                chunkMetadata = try [ChunkMetadata].load(from: metadataURL)
                if verbose {
                    print("Loaded \(chunkMetadata?.count ?? 0) metadata entries")
                }
            }

            // Determine the chat model
            let effectiveChatModel: String
            switch provider {
            case .ollama:
                effectiveChatModel = chatModel ?? "gpt-oss:20b"
            case .openrouter:
                effectiveChatModel = chatModel ?? "meta-llama/llama-4-maverick"
            }

            // Create Ollama service for embeddings
            let ollama = await MainActor.run {
                OllamaService(
                    chatModel: effectiveChatModel,
                    embeddingModel: embeddingModel
                )
            }

            // Create LLM provider
            let llmProvider: any LLMProvider
            switch provider {
            case .ollama:
                llmProvider = ollama
            case .openrouter:
                guard let key = apiKey else {
                    throw ValidationError("--api-key is required when using openrouter provider")
                }
                llmProvider = try OpenRouterService(
                    apiKey: key,
                    model: effectiveChatModel
                )
            }

            // Create embedding service
            let embeddingService = EmbeddingService(
                ollamaService: ollama,
                model: embeddingModel
            )

            // Create GraphRAG service
            let ragService = GraphRAGService(
                hypergraph: hypergraph,
                embeddings: nodeEmbeddings,
                llmProvider: llmProvider,
                embeddingService: embeddingService,
                metadata: chunkMetadata,
                chatModel: effectiveChatModel
            )

            if verbose {
                print("Query: \(question)")
                print("Provider: \(provider.rawValue)")
                print("Model: \(effectiveChatModel)")
                print("Top-K: \(topK)")
                print("Max Path Length: \(maxPathLength)")
                print("")
            }

            if contextOnly {
                // Only retrieve context
                let context: RAGContext
                if simpleKeywords {
                    context = try await ragService.retrieveContextSimple(
                        for: question,
                        topK: topK
                    )
                } else {
                    context = try await ragService.retrieveContext(
                        for: question,
                        topK: topK,
                        maxPathLength: maxPathLength,
                        similarityThreshold: threshold
                    )
                }

                printContext(context, verbose: verbose || showContext)
            } else {
                // Full RAG query with answer generation
                print("Searching knowledge graph...")

                let response = try await ragService.query(
                    question,
                    topK: topK,
                    maxPathLength: maxPathLength
                )

                if showContext || verbose {
                    printContext(response.context, verbose: true)
                    print("\n" + String(repeating: "=", count: 60) + "\n")
                }

                print("Answer:")
                print(response.answer)

                if !response.hadContext {
                    print("\n[Note: No relevant graph context was found for this query]")
                }
            }
        }

        private func printContext(_ context: RAGContext, verbose: Bool) {
            if verbose {
                print("Keywords extracted: \(context.keywords.joined(separator: ", "))")
                print("Matched nodes: \(context.matchedNodeCount)")
                print("Paths found: \(context.paths.count)")
                print("")
            }

            if context.hasContext {
                print("Retrieved Context:")
                print(context.formattedContext)
            } else {
                print("No relevant context found in the knowledge graph.")
            }

            if verbose && !context.matchedNodes.isEmpty {
                print("\nMatched Nodes:")
                for match in context.matchedNodes.prefix(10) {
                    print("  - \(match.node) (keyword: \(match.keyword), sim: \(String(format: "%.3f", match.similarity)))")
                }
                if context.matchedNodes.count > 10 {
                    print("  ... and \(context.matchedNodes.count - 10) more")
                }
            }

            if verbose && !context.paths.isEmpty {
                print("\nPaths:")
                for (i, path) in context.paths.prefix(5).enumerated() {
                    print("  \(i + 1). \(path.joined(separator: " -> "))")
                }
                if context.paths.count > 5 {
                    print("  ... and \(context.paths.count - 5) more")
                }
            }
        }
    }
}
