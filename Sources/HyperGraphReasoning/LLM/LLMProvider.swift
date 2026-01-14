import Foundation

/// Protocol for LLM providers that can generate text and structured responses.
///
/// This protocol abstracts the underlying LLM backend, allowing the hypergraph
/// extraction pipeline to work with different providers (Ollama, OpenRouter, etc.).
public protocol LLMProvider: Sendable {
    /// The default model identifier for this provider.
    var defaultModel: String { get }

    /// Generates a text response from the LLM.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt setting the context.
    ///   - userPrompt: The user's prompt/question.
    ///   - model: Optional model override.
    ///   - temperature: Optional temperature override.
    /// - Returns: The generated text response.
    func chat(
        systemPrompt: String,
        userPrompt: String,
        model: String?,
        temperature: Double?
    ) async throws -> String

    /// Generates a structured JSON response from the LLM.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt (should request JSON output).
    ///   - userPrompt: The user's prompt.
    ///   - responseType: The Decodable type to decode the response into.
    ///   - model: Optional model override.
    ///   - temperature: Optional temperature override.
    /// - Returns: The decoded response object.
    func generate<T: Decodable & Sendable>(
        systemPrompt: String,
        userPrompt: String,
        responseType: T.Type,
        model: String?,
        temperature: Double?
    ) async throws -> T
}

/// Extension providing convenience methods for LLMProvider.
extension LLMProvider {
    /// Extracts hypergraph events from text using the standard extraction prompt.
    ///
    /// - Parameters:
    ///   - text: The text to extract events from.
    ///   - model: Optional model override.
    /// - Returns: The extracted hypergraph events.
    public func extractHypergraphEvents(
        from text: String,
        model: String? = nil
    ) async throws -> HypergraphJSON {
        try await generate(
            systemPrompt: SystemPrompts.hypergraphExtraction,
            userPrompt: "Context: ```\(text)``` \n Extract the hypergraph knowledge graph in structured JSON format: ",
            responseType: HypergraphJSON.self,
            model: model,
            temperature: nil
        )
    }
}

/// Error types for LLM provider operations.
public enum LLMProviderError: Error, LocalizedError {
    case connectionFailed(Error)
    case invalidResponse(String)
    case decodingFailed(Error)
    case configurationError(String)
    case modelNotAvailable(String)

    public var errorDescription: String? {
        switch self {
        case .connectionFailed(let error):
            return "Failed to connect to LLM provider: \(error.localizedDescription)"
        case .invalidResponse(let message):
            return "Invalid response from LLM: \(message)"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .modelNotAvailable(let model):
            return "Model not available: \(model)"
        }
    }
}
