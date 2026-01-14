import ArgumentParser
import Foundation
import HyperGraphReasoning

@main
struct HypergraphCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hypergraph-cli",
        abstract: "CLI tool for building hypergraph knowledge representations",
        version: "0.1.0",
        subcommands: [Process.self, Extract.self, Embed.self, Info.self]
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

        @Option(name: .long, help: "Chat model for extraction")
        var chatModel: String = "gpt-oss:20b"

        @Option(name: .long, help: "Embedding model")
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

            if verbose {
                print("Input: \(inputURL.path)")
                print("Output: \(outputURL.path)")
                print("Chat Model: \(chatModel)")
                print("Embedding Model: \(embeddingModel)")
                print("Chunk Size: \(chunkSize)")
            }

            // Create services
            let ollama = await MainActor.run {
                OllamaService(
                    chatModel: chatModel,
                    embeddingModel: embeddingModel
                )
            }

            let processor = DocumentProcessor(
                ollamaService: ollama,
                chatModel: chatModel,
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

        @Option(name: .long, help: "Chat model for extraction")
        var chatModel: String = "gpt-oss:20b"

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

            print("Extracting hypergraph...")

            let ollama = await MainActor.run {
                OllamaService(chatModel: chatModel)
            }
            let extractor = HypergraphExtractor(
                ollamaService: ollama,
                model: chatModel
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
