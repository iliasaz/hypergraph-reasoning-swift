import Testing
import Foundation
@testable import HyperGraphReasoning

/// Integration tests for OpenRouterService with different models.
///
/// These tests require the OPENROUTER_API_KEY environment variable to be set.
///
/// **Note:** These tests hit the real OpenRouter API and may be affected by rate limiting
/// when run in parallel. The `.serialized` trait ensures tests run sequentially.
@Suite("OpenRouterService Integration Tests", .serialized)
struct OpenRouterServiceTests {

    // MARK: - Test Configuration

    /// Simple test text for extraction
    static let testText = """
    Spider silk is a remarkable biomaterial with exceptional mechanical properties.
    The silk fibers exhibit high tensile strength and elasticity.
    Researchers have found that spider silk proteins can be used for tissue engineering.
    """

    /// Get API key from environment
    static var apiKey: String? {
        ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"]
    }

    /// Get chat model from environment, defaulting to Llama 4 Maverick
    static var chatModel: String {
        ProcessInfo.processInfo.environment["CHAT_MODEL"] ?? "meta-llama/llama-4-maverick"
    }

    /// Check if API key is available
    static var hasAPIKey: Bool {
        apiKey != nil && !apiKey!.isEmpty
    }

    // MARK: - Chat Model Tests (configurable via CHAT_MODEL env var)

    @Test("Basic chat completion")
    func testBasicChat() async throws {
        try skipIfNoAPIKey()

        let service = try OpenRouterService(
            apiKey: Self.apiKey!,
            model: Self.chatModel
        )

        let response = try await service.chat(
            systemPrompt: "You are a helpful assistant. Reply concisely.",
            userPrompt: "What is 2 + 2?",
            model: nil,
            temperature: 0.1
        )

        print("\(Self.chatModel) chat response: \(response)")
        #expect(!response.isEmpty, "Response should not be empty")
        #expect(response.contains("4"), "Response should contain '4'")
    }

    @Test("JSON extraction")
    func testJSONExtraction() async throws {
        try skipIfNoAPIKey()

        let service = try OpenRouterService(
            apiKey: Self.apiKey!,
            model: Self.chatModel
        )

        // Test structured JSON extraction directly
        // Note: Use nil temperature to match HypergraphExtractor's default behavior
        do {
            let result: HypergraphJSON = try await service.generate(
                systemPrompt: SystemPrompts.hypergraphExtraction,
                userPrompt: SystemPrompts.extractionUserPrompt(text: Self.testText),
                responseType: HypergraphJSON.self,
                model: nil,
                temperature: nil
            )

            print("\(Self.chatModel) extraction result:")
            print("  Events count: \(result.events.count)")
            for event in result.events {
                print("  - \(event.source) --[\(event.relation)]--> \(event.target)")
            }

            #expect(!result.events.isEmpty, "Should extract at least one event")
        } catch {
            print("\(Self.chatModel) JSON extraction error: \(error)")
            throw error
        }
    }

    @Test("Full extraction pipeline")
    func testFullPipeline() async throws {
        try skipIfNoAPIKey()

        let service = try OpenRouterService(
            apiKey: Self.apiKey!,
            model: Self.chatModel
        )

        let extractor = HypergraphExtractor(
            llmProvider: service,
            model: Self.chatModel,
            chunkSize: 1000
        )

        let (hypergraph, metadata, _) = try await extractor.extractFromDocument(Self.testText)

        print("\(Self.chatModel) pipeline result:")
        print("  Nodes: \(hypergraph.nodeCount)")
        print("  Edges: \(hypergraph.edgeCount)")
        print("  Metadata count: \(metadata.count)")

        #expect(hypergraph.nodeCount > 0, "Should have nodes")
        #expect(hypergraph.edgeCount > 0, "Should have edges")
    }

    // MARK: - Model Validation Tests

    @Test("Verify CHAT_MODEL identifier is valid on OpenRouter")
    func testVerifyChatModelIdentifier() async throws {
        try skipIfNoAPIKey()

        // Test if we can get a response from OpenRouter with the configured model
        let url = URL(string: "https://openrouter.ai/api/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(Self.apiKey!)", forHTTPHeaderField: "Authorization")

        let body: [String: Any] = [
            "model": Self.chatModel,
            "messages": [
                ["role": "user", "content": "Say hello"]
            ],
            "max_tokens": 50
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        let httpResponse = response as! HTTPURLResponse

        print("=== Direct OpenRouter API Test for \(Self.chatModel) ===")
        print("HTTP Status: \(httpResponse.statusCode)")
        print("Response body: \(String(data: data, encoding: .utf8) ?? "N/A")")

        // Check if we get a 404 (model not found) or other error
        if httpResponse.statusCode == 404 {
            print("ERROR: Model '\(Self.chatModel)' does not exist on OpenRouter!")
            print("Check https://openrouter.ai/models for valid model names.")
        }

        #expect(httpResponse.statusCode == 200, "Model '\(Self.chatModel)' should exist and return 200")
    }

    // MARK: - Helpers

    private func skipIfNoAPIKey() throws {
        guard Self.hasAPIKey else {
            throw TestSkipError(message: "OPENROUTER_API_KEY not set")
        }
    }
}

/// Error for skipping tests when API key is not available
struct TestSkipError: Error {
    let message: String
}
