import Foundation
import SwiftAgents

/// Service for interacting with OpenRouter for LLM inference.
///
/// This service uses SwiftAgents' OpenRouterProvider to access various LLM models
/// through the OpenRouter API.
public actor OpenRouterService: LLMProvider {

    /// Default model for OpenRouter.
    public static let defaultOpenRouterModel = "meta-llama/llama-4-maverick"

    /// Default timeout for requests (5 minutes).
    public static let defaultTimeout: Duration = .seconds(300)

    /// Default temperature for generation.
    public static let defaultTemperature: Double = 0.333

    /// The OpenRouter provider from SwiftAgents.
    private let provider: OpenRouterProvider

    /// The default model identifier.
    public let defaultModel: String

    /// Default temperature for generation.
    public let temperature: Double

    /// Creates an OpenRouterService with the specified configuration.
    ///
    /// - Parameters:
    ///   - apiKey: The OpenRouter API key.
    ///   - model: The model to use. Defaults to "meta-llama/llama-4-maverick".
    ///   - temperature: Default temperature. Defaults to 0.333.
    ///   - timeout: Request timeout. Defaults to 5 minutes.
    ///   - systemPrompt: Optional default system prompt.
    /// - Throws: `LLMProviderError.configurationError` if initialization fails.
    public init(
        apiKey: String,
        model: String = defaultOpenRouterModel,
        temperature: Double = defaultTemperature,
        timeout: Duration = defaultTimeout,
        systemPrompt: String? = nil
    ) throws {
        do {
            let openRouterModel = try OpenRouterModel(model)
            let config = try OpenRouterConfiguration(
                apiKey: apiKey,
                model: openRouterModel,
                timeout: timeout,
                systemPrompt: systemPrompt,
                temperature: temperature
            )
            self.provider = OpenRouterProvider(configuration: config)
            self.defaultModel = model
            self.temperature = temperature
        } catch {
            throw LLMProviderError.configurationError("Failed to initialize OpenRouter: \(error)")
        }
    }

    // MARK: - LLMProvider Conformance

    /// Generates a text response from the LLM.
    public func chat(
        systemPrompt: String,
        userPrompt: String,
        model: String?,
        temperature: Double?
    ) async throws -> String {
        let fullPrompt = """
        System: \(systemPrompt)

        User: \(userPrompt)
        """

        do {
            let response = try await provider.generate(
                prompt: fullPrompt,
                options: .default
            )
            return response
        } catch {
            throw LLMProviderError.connectionFailed(error)
        }
    }

    /// Generates a structured JSON response from the LLM.
    public func generate<T: Decodable & Sendable>(
        systemPrompt: String,
        userPrompt: String,
        responseType: T.Type,
        model: String?,
        temperature: Double?
    ) async throws -> T {
        // Add JSON instruction to the prompt
        let jsonSystemPrompt = """
        \(systemPrompt)

        IMPORTANT: You must respond with valid JSON only. No additional text or explanation.
        """

        let fullPrompt = """
        System: \(jsonSystemPrompt)

        User: \(userPrompt)
        """

        do {
            let response = try await provider.generate(
                prompt: fullPrompt,
                options: .default
            )

            // Extract JSON from response (handle potential markdown code blocks)
            let jsonString = extractJSON(from: response)

            guard let data = jsonString.data(using: .utf8) else {
                throw LLMProviderError.invalidResponse("Response is not valid UTF-8")
            }

            do {
                let decoded = try JSONDecoder().decode(T.self, from: data)
                return decoded
            } catch {
                throw LLMProviderError.decodingFailed(error)
            }
        } catch let error as LLMProviderError {
            throw error
        } catch {
            throw LLMProviderError.connectionFailed(error)
        }
    }

    /// Extracts JSON from a response that might contain markdown code blocks.
    private func extractJSON(from response: String) -> String {
        var content = response.trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove markdown code block if present
        if content.hasPrefix("```json") {
            content = String(content.dropFirst(7))
        } else if content.hasPrefix("```") {
            content = String(content.dropFirst(3))
        }

        if content.hasSuffix("```") {
            content = String(content.dropLast(3))
        }

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
