import Foundation
import Ollama

/// Error types for Ollama operations.
public enum OllamaError: Error, LocalizedError {
    case connectionFailed(Error)
    case invalidResponse(String)
    case decodingFailed(Error)
    case embeddingFailed(String)
    case modelNotAvailable(String)

    public var errorDescription: String? {
        switch self {
        case .connectionFailed(let error):
            return "Failed to connect to Ollama: \(error.localizedDescription)"
        case .invalidResponse(let message):
            return "Invalid response from Ollama: \(message)"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .embeddingFailed(let message):
            return "Embedding generation failed: \(message)"
        case .modelNotAvailable(let model):
            return "Model not available: \(model)"
        }
    }
}

/// Service for interacting with Ollama for LLM inference and embeddings.
///
/// This service wraps the ollama-swift Client for use in the hypergraph extraction pipeline.
@MainActor
public final class OllamaService: @preconcurrency LLMProvider {

    /// Default timeout for requests (5 minutes).
    public static let defaultTimeout: TimeInterval = 300

    /// The Ollama client.
    private let client: Client

    /// Default model for chat completions.
    public let defaultChatModel: String

    /// The default model identifier (conforms to LLMProvider).
    public var defaultModel: String { defaultChatModel }

    /// Default model for embeddings.
    public let defaultEmbeddingModel: String

    /// Default temperature for generation.
    public let defaultTemperature: Double

    /// Creates a URLSession with extended timeout for LLM operations.
    private static func createSession(timeout: TimeInterval = defaultTimeout) -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout
        return URLSession(configuration: config)
    }

    /// Creates an OllamaService with the default localhost connection.
    ///
    /// - Parameters:
    ///   - chatModel: Default model for chat. Defaults to "gpt-oss:20b".
    ///   - embeddingModel: Default model for embeddings. Defaults to "nomic-embed-text:v1.5".
    ///   - temperature: Default temperature. Defaults to 0.333.
    ///   - timeout: Request timeout in seconds. Defaults to 300 (5 minutes).
    public init(
        chatModel: String = "gpt-oss:20b",
        embeddingModel: String = "nomic-embed-text:v1.5",
        temperature: Double = 0.333,
        timeout: TimeInterval = defaultTimeout
    ) {
        self.client = Client(
            session: Self.createSession(timeout: timeout),
            host: Client.defaultHost
        )
        self.defaultChatModel = chatModel
        self.defaultEmbeddingModel = embeddingModel
        self.defaultTemperature = temperature
    }

    /// Creates an OllamaService with a custom host.
    ///
    /// - Parameters:
    ///   - host: The Ollama server URL.
    ///   - chatModel: Default model for chat.
    ///   - embeddingModel: Default model for embeddings.
    ///   - temperature: Default temperature.
    ///   - timeout: Request timeout in seconds. Defaults to 300 (5 minutes).
    public init(
        host: URL,
        chatModel: String = "gpt-oss:20b",
        embeddingModel: String = "nomic-embed-text:v1.5",
        temperature: Double = 0.333,
        timeout: TimeInterval = defaultTimeout
    ) {
        self.client = Client(
            session: Self.createSession(timeout: timeout),
            host: host
        )
        self.defaultChatModel = chatModel
        self.defaultEmbeddingModel = embeddingModel
        self.defaultTemperature = temperature
    }

    // MARK: - Chat Completion

    /// Generates a response from the LLM.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt setting the context.
    ///   - userPrompt: The user's prompt/question.
    ///   - model: The model to use. Defaults to the service's default chat model.
    ///   - temperature: Sampling temperature. Defaults to the service's default.
    /// - Returns: The generated text response.
    public func chat(
        systemPrompt: String,
        userPrompt: String,
        model: String? = nil,
        temperature: Double? = nil
    ) async throws -> String {
        let modelToUse = model ?? defaultChatModel
        let tempToUse = temperature ?? defaultTemperature

        do {
            let modelID: Model.ID = Model.ID(stringLiteral: modelToUse)
            let response = try await client.chat(
                model: modelID,
                messages: [
                    .system(systemPrompt),
                    .user(userPrompt)
                ],
                options: ["temperature": .double(tempToUse)]
            )
            return response.message.content
        } catch {
            throw OllamaError.connectionFailed(error)
        }
    }

    /// Generates a structured response from the LLM.
    ///
    /// The LLM is instructed to return JSON, which is then decoded to the specified type.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt (should request JSON output).
    ///   - userPrompt: The user's prompt.
    ///   - responseType: The Decodable type to decode the response into.
    ///   - model: The model to use.
    ///   - temperature: Sampling temperature.
    /// - Returns: The decoded response object.
    public func generate<T: Decodable & Sendable>(
        systemPrompt: String,
        userPrompt: String,
        responseType: T.Type,
        model: String? = nil,
        temperature: Double? = nil
    ) async throws -> T {
        let modelToUse = model ?? defaultChatModel
        let tempToUse = temperature ?? defaultTemperature

        do {
            let modelID: Model.ID = Model.ID(stringLiteral: modelToUse)
            let response = try await client.chat(
                model: modelID,
                messages: [
                    .system(systemPrompt),
                    .user(userPrompt)
                ],
                options: ["temperature": .double(tempToUse)],
                format: .string("json")
            )

            let content = response.message.content

            // Parse JSON response
            guard let data = content.data(using: String.Encoding.utf8) else {
                throw OllamaError.invalidResponse("Response is not valid UTF-8")
            }

            do {
                let decoded = try JSONDecoder().decode(T.self, from: data)
                return decoded
            } catch {
                throw OllamaError.decodingFailed(error)
            }
        } catch let error as OllamaError {
            throw error
        } catch {
            throw OllamaError.connectionFailed(error)
        }
    }

    // MARK: - Streaming Chat

    /// Generates a streaming response from the LLM, calling `onToken` for each chunk.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt setting the context.
    ///   - userPrompt: The user's prompt/question.
    ///   - model: The model to use. Defaults to the service's default chat model.
    ///   - temperature: Sampling temperature. Defaults to the service's default.
    ///   - onToken: Closure called with each new token as it arrives.
    /// - Returns: The full accumulated text response.
    public func chatStream(
        systemPrompt: String,
        userPrompt: String,
        model: String? = nil,
        temperature: Double? = nil,
        onToken: @escaping (String) -> Void
    ) async throws -> String {
        let modelToUse = model ?? defaultChatModel
        let tempToUse = temperature ?? defaultTemperature

        do {
            let modelID: Model.ID = Model.ID(stringLiteral: modelToUse)
            let stream = try client.chatStream(
                model: modelID,
                messages: [
                    .system(systemPrompt),
                    .user(userPrompt)
                ],
                options: ["temperature": .double(tempToUse)]
            )

            var accumulated = ""
            for try await chunk in stream {
                let token = chunk.message.content
                if !token.isEmpty {
                    accumulated += token
                    onToken(token)
                }
            }
            return accumulated
        } catch {
            throw OllamaError.connectionFailed(error)
        }
    }

    // MARK: - Embeddings

    /// Generates an embedding for a single text.
    ///
    /// - Parameters:
    ///   - text: The text to embed.
    ///   - model: The embedding model to use.
    /// - Returns: The embedding vector.
    public func embed(
        _ text: String,
        model: String? = nil
    ) async throws -> [Float] {
        let embeddings = try await embed([text], model: model)
        guard let first = embeddings.first else {
            throw OllamaError.embeddingFailed("No embedding returned")
        }
        return first
    }

    /// Generates embeddings for multiple texts.
    ///
    /// - Parameters:
    ///   - texts: The texts to embed.
    ///   - model: The embedding model to use.
    /// - Returns: Array of embedding vectors, one per input text.
    public func embed(
        _ texts: [String],
        model: String? = nil
    ) async throws -> [[Float]] {
        let modelToUse = model ?? defaultEmbeddingModel

        guard !texts.isEmpty else {
            return []
        }

        do {
            let modelID: Model.ID = Model.ID(stringLiteral: modelToUse)
            let response = try await client.embed(
                model: modelID,
                inputs: texts
            )

            // Convert Double arrays to Float arrays
            return response.embeddings.rawValue.map { embedding in
                embedding.map { Float($0) }
            }
        } catch {
            throw OllamaError.connectionFailed(error)
        }
    }

    // MARK: - Model Management

    /// Lists available models.
    ///
    /// - Returns: Array of model names.
    public func listModels() async throws -> [String] {
        do {
            let response = try await client.listModels()
            return response.models.map { $0.name }
        } catch {
            throw OllamaError.connectionFailed(error)
        }
    }

    /// Checks if a model is available.
    ///
    /// - Parameter model: The model name to check.
    /// - Returns: `true` if the model is available.
    public func isModelAvailable(_ model: String) async throws -> Bool {
        let models = try await listModels()
        return models.contains { $0.hasPrefix(model) }
    }
}

