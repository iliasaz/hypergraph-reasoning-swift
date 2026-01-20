# HyperGraphReasoning Swift

[![Swift](https://github.com/iliasaz/hypergraph-reasoning-swift/actions/workflows/swift.yml/badge.svg)](https://github.com/iliasaz/hypergraph-reasoning-swift/actions/workflows/swift.yml)
[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2014%20|%20iOS%2017-blue.svg)](https://developer.apple.com)
[![Release](https://img.shields.io/github/v/release/iliasaz/hypergraph-reasoning-swift)](https://github.com/iliasaz/hypergraph-reasoning-swift/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Swift implementation of hypergraph-based knowledge representation extraction, ported from the [HyperGraphReasoning](https://github.com/lamm-mit/HyperGraphReasoning) Python project.

## Overview

This package provides tools for building hypergraph knowledge representations from scientific text using Large Language Models. Unlike traditional graphs where edges connect exactly two nodes, hypergraphs allow edges (hyperedges) to connect any number of nodes, enabling richer representation of complex relationships.

The implementation supports multiple LLM providers:
- **[Ollama](https://ollama.ai)** for local LLM inference and embedding generation
- **[OpenRouter](https://openrouter.ai)** for cloud-based access to various models (OpenAI, Anthropic, Meta, etc.)

## Features

- **Hypergraph Data Structure**: Core hypergraph implementation with support for:
  - Union and intersection operations
  - Connected component analysis
  - Node/edge restriction (subgraph extraction)
  - Degree calculations and neighbor queries

- **Text Processing Pipeline**:
  - Recursive text splitting with configurable chunk sizes
  - MD5-based deterministic chunk identification
  - Markdown document processing

- **LLM Integration**:
  - Multiple provider support via `LLMProvider` protocol
  - **Ollama**: Local inference with any Ollama model
  - **OpenRouter**: Cloud access to OpenAI, Anthropic, Meta Llama, and more
  - Batch embedding generation using nomic-embed-text (Ollama)
  - Structured JSON output parsing

- **Knowledge Extraction**:
  - Subject-Verb-Object triple extraction
  - Hyperedge construction from semantic relationships
  - Graph simplification via embedding similarity

## Requirements

- macOS 14.0+ / iOS 17.0+
- Swift 6.2+
- For **Ollama** provider:
  - [Ollama](https://ollama.ai) running locally with required models
  - Chat model (default: `gpt-oss:20b`)
  - Embedding model (default: `nomic-embed-text:v1.5`)
- For **OpenRouter** provider:
  - OpenRouter API key from [openrouter.ai](https://openrouter.ai)
  - Default model: `meta-llama/llama-4-maverick`

## Installation

### Swift Package Manager

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(path: "../hypergraph-reasoning-swift")
]
```

Or add it as a local package in Xcode.

## Usage

### Library Usage

#### Using Ollama (Local)

```swift
import HyperGraphReasoning

// Create Ollama service
let ollama = OllamaService(
    chatModel: "gpt-oss:20b",
    embeddingModel: "nomic-embed-text:v1.5"
)

// Create document processor
let processor = DocumentProcessor(
    ollamaService: ollama,
    chunkSize: 1000
)

// Process a markdown file
let result = try await processor.processMarkdownFile(
    at: URL(fileURLWithPath: "document.md"),
    generateEmbeddings: true
)

print("Extracted \(result.nodeCount) nodes, \(result.edgeCount) edges")

// Save results
try processor.saveResult(result, to: URL(fileURLWithPath: "./output"))
```

#### Using OpenRouter (Cloud)

```swift
import HyperGraphReasoning

// Create OpenRouter service for LLM
let openRouter = try OpenRouterService(
    apiKey: "sk-or-v1-...",
    model: "meta-llama/llama-4-maverick"
)

// Ollama still needed for embeddings
let ollama = OllamaService(embeddingModel: "nomic-embed-text:v1.5")

// Create document processor with OpenRouter for extraction
let processor = DocumentProcessor(
    llmProvider: openRouter,
    ollamaService: ollama,
    chatModel: "meta-llama/llama-4-maverick"
)

let result = try await processor.processMarkdownFile(at: fileURL)
```

**Supported OpenRouter Models:**
- `meta-llama/llama-4-maverick` (default)
- `openai/gpt-4.1-nano`
- `openai/gpt-4.1-mini`
- `openai/gpt-4.1`
- `openai/gpt-5-nano`
- `openai/gpt-5-mini`
- `openai/gpt-5.2`

### Working with Hypergraphs

```swift
// Create a hypergraph
var graph = Hypergraph<String, String>()

// Add edges (hyperedges can connect multiple nodes)
graph.addEdge("e1", nodes: "A", "B", "C")
graph.addEdge("e2", nodes: "B", "D")
graph.addEdge("e3", nodes: "D", "E", "F")

// Query the graph
let degree = graph.degree(of: "B")        // 2
let neighbors = graph.neighbors(of: "B")  // {"A", "C", "D"}

// Find connected components
let components = graph.connectedComponents()

// Merge two hypergraphs
let merged = graph.union(otherGraph)

// Extract subgraph
let subgraph = graph.restrictToNodes(["A", "B", "C"])
```

### CLI Tool

The package includes a command-line interface for processing documents:

```bash
# Build the CLI
swift build

# Process with Ollama (default)
swift run hypergraph-cli process ./documents --output ./output --verbose

# Process with OpenRouter
swift run hypergraph-cli process ./documents --provider openrouter --api-key "sk-or-..." --output ./output

# Extract hypergraph from text
swift run hypergraph-cli extract "Your text here" --output graph.json

# Extract using OpenRouter with a specific model
swift run hypergraph-cli extract --file document.md --provider openrouter --api-key "sk-or-..." --chat-model "anthropic/claude-3.5-sonnet"

# Generate embeddings for an existing hypergraph
swift run hypergraph-cli embed graph.json --output embeddings.json

# Display hypergraph information
swift run hypergraph-cli info graph.json
```

#### CLI Options

**process** - Full pipeline processing
```
USAGE: hypergraph-cli process <input> [--output <output>] [--provider <provider>]
                              [--api-key <key>] [--chat-model <model>]
                              [--embedding-model <model>] [--chunk-size <size>]
                              [--skip-embeddings] [--verbose]

ARGUMENTS:
  <input>                 Input markdown file or directory

OPTIONS:
  -o, --output <output>   Output directory (default: ./output)
  --provider <provider>   LLM provider: ollama or openrouter (default: ollama)
  --api-key <key>         OpenRouter API key (required if provider is openrouter)
  --chat-model <model>    Chat model for extraction
                          (default: gpt-oss:20b for ollama, meta-llama/llama-4-maverick for openrouter)
  --embedding-model       Embedding model (default: nomic-embed-text:v1.5)
  --chunk-size <size>     Chunk size for text splitting (default: 1000)
  --skip-embeddings       Skip embedding generation
  -v, --verbose           Enable verbose output
```

**extract** - Extract hypergraph without embeddings
```
USAGE: hypergraph-cli extract <input> [--output <output>] [--provider <provider>]
                              [--api-key <key>] [--chat-model <model>] [--file]

ARGUMENTS:
  <input>                 Input text or file path

OPTIONS:
  -o, --output <output>   Output JSON file (default: hypergraph.json)
  --provider <provider>   LLM provider: ollama or openrouter (default: ollama)
  --api-key <key>         OpenRouter API key (required if provider is openrouter)
  --chat-model <model>    Chat model for extraction
  --file                  Treat input as file path
```

**embed** - Generate embeddings
```
USAGE: hypergraph-cli embed <input> [--output <output>] [--model <model>]

ARGUMENTS:
  <input>                 Input hypergraph JSON file

OPTIONS:
  -o, --output <output>   Output embeddings file (default: embeddings.json)
  --model <model>         Embedding model (default: nomic-embed-text:v1.5)
```

**info** - Display hypergraph statistics
```
USAGE: hypergraph-cli info <input>

ARGUMENTS:
  <input>                 Hypergraph JSON file
```

## Package Structure

```
hypergraph-reasoning-swift/
├── Package.swift
├── Sources/
│   ├── HyperGraphReasoning/
│   │   ├── Core/
│   │   │   ├── Hypergraph.swift           # Core data structure
│   │   │   └── HypergraphOperations.swift # Union, components, restrict
│   │   ├── TextProcessing/
│   │   │   ├── RecursiveTextSplitter.swift
│   │   │   ├── TextChunk.swift
│   │   │   └── DocumentProcessor.swift
│   │   ├── LLM/
│   │   │   ├── OllamaService.swift
│   │   │   ├── EmbeddingService.swift
│   │   │   ├── HypergraphExtractor.swift
│   │   │   └── SystemPrompts.swift
│   │   ├── Models/
│   │   │   ├── Event.swift
│   │   │   ├── HypergraphJSON.swift
│   │   │   ├── ChunkMetadata.swift
│   │   │   └── NodeEmbeddings.swift
│   │   └── Serialization/
│   │       └── HypergraphCodable.swift
│   └── hypergraph-cli/
│       └── main.swift
└── Tests/
    └── HyperGraphReasoningTests/
```

## Output Format

The processor outputs three JSON files:

- `*_graph.json` - The hypergraph structure (incidence dictionary)
- `*_metadata.json` - Edge metadata with source/target/relation info
- `*_embeddings.json` - Node embeddings for similarity calculations

## Dependencies

- [ollama-swift](https://github.com/mattt/ollama-swift) - Ollama client for Swift
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - CLI argument parsing

## References

This is a Swift port of the hypergraph building pipeline from:

**HyperGraphReasoning: Higher-Order Knowledge Representations for Agentic Scientific Reasoning**

- Original Repository: [github.com/lamm-mit/HyperGraphReasoning](https://github.com/lamm-mit/HyperGraphReasoning)
- Paper: [arXiv:2505.00091](https://arxiv.org/abs/2505.00091)
- Dataset: [HuggingFace](https://huggingface.co/datasets/lamm-mit/HyperGraphReasoning)

```bibtex
@article{buehler2025hypergraphreasoning,
    title={Higher-Order Knowledge Representations for Agentic Scientific Reasoning},
    author={Buehler, Markus J.},
    journal={arXiv preprint arXiv:2505.00091},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This is a Swift port inspired by the original [HyperGraphReasoning](https://github.com/lamm-mit/HyperGraphReasoning) Python project by Markus J. Buehler (MIT). Please refer to the original repository for its licensing terms.
