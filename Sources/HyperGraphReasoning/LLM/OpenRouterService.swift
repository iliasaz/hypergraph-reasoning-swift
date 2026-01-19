import Foundation
import SwiftAgents

/// Service for interacting with OpenRouter for LLM inference.
///
/// This service uses SwiftAgents' OpenRouterProvider to access various LLM models
/// through the OpenRouter API.
public actor OpenRouterService: LLMProvider {

    /// Default model for OpenRouter.
    public static let defaultOpenRouterModel = "meta-llama/llama-4-maverick"

    /// Supported OpenRouter chat models.
    ///
    /// These models are known to work well with the hypergraph extraction pipeline.
    public static let supportedModels: [String] = [
        // Meta Llama
        "meta-llama/llama-4-maverick",
        // OpenAI via OpenRouter
        "openai/gpt-4.1-nano",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1",
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/gpt-5.2",
    ]

    /// Default timeout for requests (5 minutes).
    public static let defaultTimeout: Duration = .seconds(300)

    /// Default temperature for generation.
    public static let defaultTemperature: Double = 0.333

    /// Default max tokens for generation (8192 for longer extraction responses).
    public static let defaultMaxTokens: Int = 8192

    /// The OpenRouter provider from SwiftAgents.
    private let provider: OpenRouterProvider

    /// The default model identifier.
    public let defaultModel: String

    /// Default temperature for generation.
    public let temperature: Double

    /// Maximum tokens for generation.
    public let maxTokens: Int

    /// Creates an OpenRouterService with the specified configuration.
    ///
    /// - Parameters:
    ///   - apiKey: The OpenRouter API key.
    ///   - model: The model to use. Defaults to "meta-llama/llama-4-maverick".
    ///   - temperature: Default temperature. Defaults to 0.333.
    ///   - maxTokens: Maximum tokens for generation. Defaults to 8192.
    ///   - timeout: Request timeout. Defaults to 5 minutes.
    ///   - systemPrompt: Optional default system prompt.
    /// - Throws: `LLMProviderError.configurationError` if initialization fails.
    public init(
        apiKey: String,
        model: String = defaultOpenRouterModel,
        temperature: Double = defaultTemperature,
        maxTokens: Int = defaultMaxTokens,
        timeout: Duration = defaultTimeout,
        systemPrompt: String? = nil
    ) throws {
        do {
            let openRouterModel = try OpenRouterModel(model)
            let config = try OpenRouterConfiguration(
                apiKey: apiKey,
                model: openRouterModel,
                timeout: timeout,
                maxTokens: maxTokens,
                systemPrompt: systemPrompt,
                temperature: temperature
            )
            self.provider = OpenRouterProvider(configuration: config)
            self.defaultModel = model
            self.temperature = temperature
            self.maxTokens = maxTokens
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

        // Use explicit options with maxTokens to prevent truncation
        let options = InferenceOptions.default
            .maxTokens(maxTokens)
            .temperature(temperature ?? self.temperature)

        do {
            let response = try await provider.generate(
                prompt: fullPrompt,
                options: options
            )

            // Check for empty response
            if response.isEmpty {
                throw LLMProviderError.invalidResponse(
                    "Empty response from model '\(defaultModel)'. Raw response: (empty string)"
                )
            }

            return response
        } catch let error as LLMProviderError {
            throw error
        } catch {
            throw mapProviderError(error)
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

        // Use explicit options with maxTokens to prevent truncation
        let options = InferenceOptions.default
            .maxTokens(maxTokens)
            .temperature(temperature ?? self.temperature)

        do {
            let response = try await provider.generate(
                prompt: fullPrompt,
                options: options
            )

            // Check for empty response
            if response.isEmpty {
                throw LLMProviderError.invalidResponse(
                    "Empty response from model '\(defaultModel)'. Raw response: (empty string)"
                )
            }

            // Extract JSON from response (handle potential markdown code blocks)
            let jsonString = extractJSON(from: response)

            // Check if we extracted any JSON content
            if jsonString.isEmpty {
                throw LLMProviderError.invalidResponse(
                    "No JSON content found in response. Response length: \(response.count) characters."
                )
            }

            guard let data = jsonString.data(using: .utf8) else {
                throw LLMProviderError.invalidResponse("Response is not valid UTF-8")
            }

            do {
                let decoded = try JSONDecoder().decode(T.self, from: data)
                return decoded
            } catch let decodeError {
                // Provide context about the JSON that failed to parse
                let preview = String(jsonString.prefix(200))
                let suffix = jsonString.count > 200 ? "..." : ""
                throw LLMProviderError.decodingFailed(
                    DecodingErrorWithContext(
                        underlying: decodeError,
                        jsonPreview: preview + suffix,
                        jsonLength: jsonString.count
                    )
                )
            }
        } catch let error as LLMProviderError {
            throw error
        } catch {
            throw mapProviderError(error)
        }
    }

    // MARK: - Private Helpers

    /// Maps SwiftAgents errors to LLMProviderError.
    private func mapProviderError(_ error: Error) -> LLMProviderError {
        let errorDescription = String(describing: error)

        // Check for HTTP status code errors
        if errorDescription.contains("HTTP 4") || errorDescription.contains("HTTP 5") {
            // Extract the error message if possible
            if let messageStart = errorDescription.range(of: "\"message\":\""),
               let messageEnd = errorDescription[messageStart.upperBound...].range(of: "\"") {
                let message = String(errorDescription[messageStart.upperBound..<messageEnd.lowerBound])
                return .invalidResponse("OpenRouter error: \(message)")
            }

            // Check for common error patterns
            if errorDescription.contains("does not have access to model") {
                return .modelNotAvailable(
                    "Your account does not have access to model '\(defaultModel)'. " +
                    "Check your OpenRouter/OpenAI account permissions."
                )
            }

            if errorDescription.contains("401") {
                return .configurationError("Invalid API key")
            }

            if errorDescription.contains("429") {
                return .connectionFailed(error)  // Rate limit - let caller handle retry
            }

            return .invalidResponse("OpenRouter API error: \(errorDescription)")
        }

        return .connectionFailed(error)
    }

    /// Extracts JSON from a response that might contain markdown code blocks or extra text.
    ///
    /// GPT models sometimes include reasoning or explanatory text before/after the JSON.
    /// This function attempts to extract just the JSON object.
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

        content = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // If the response doesn't start with {, try to find the JSON object
        // GPT models sometimes include explanatory text before the JSON
        if !content.hasPrefix("{") && !content.hasPrefix("[") {
            // Try to find the start of a JSON object
            if let jsonStart = content.firstIndex(of: "{") {
                content = String(content[jsonStart...])
            } else if let jsonStart = content.firstIndex(of: "[") {
                content = String(content[jsonStart...])
            }
        }

        // If the response doesn't end with } or ], try to find the end
        // GPT models sometimes include text after the JSON
        if !content.hasSuffix("}") && !content.hasSuffix("]") {
            // Find the last closing brace/bracket
            if let jsonEnd = content.lastIndex(of: "}") {
                content = String(content[...jsonEnd])
            } else if let jsonEnd = content.lastIndex(of: "]") {
                content = String(content[...jsonEnd])
            }
        }

        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Error Context

/// Wrapper to provide context for JSON decoding errors.
struct DecodingErrorWithContext: Error, LocalizedError {
    let underlying: Error
    let jsonPreview: String
    let jsonLength: Int

    var errorDescription: String? {
        "JSON decoding failed (\(jsonLength) chars). Preview: \(jsonPreview). Error: \(underlying.localizedDescription)"
    }
}
