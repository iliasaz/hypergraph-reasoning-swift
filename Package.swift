// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "hypergraph-reasoning-swift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "HyperGraphReasoning",
            targets: ["HyperGraphReasoning"]
        ),
        .executable(
            name: "hypergraph-cli",
            targets: ["hypergraph-cli"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/mattt/ollama-swift.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/iliasaz/SwiftAgents.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "HyperGraphReasoning",
            dependencies: [
                .product(name: "Ollama", package: "ollama-swift"),
                .product(name: "SwiftAgents", package: "SwiftAgents"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "hypergraph-cli",
            dependencies: [
                "HyperGraphReasoning",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "HyperGraphReasoningTests",
            dependencies: ["HyperGraphReasoning"]
        ),
    ]
)
