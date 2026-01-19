import Foundation

/// Extracts keywords from user queries using an LLM.
///
/// Keywords are used to find matching nodes in the hypergraph
/// for the GraphRAG retrieval pipeline.
public struct KeywordExtractor: Sendable {

    /// The LLM provider for keyword extraction.
    private let llmProvider: any LLMProvider

    /// The model to use for extraction.
    private let model: String?

    /// Creates a keyword extractor.
    ///
    /// - Parameters:
    ///   - llmProvider: The LLM provider to use.
    ///   - model: Optional model override.
    public init(llmProvider: any LLMProvider, model: String? = nil) {
        self.llmProvider = llmProvider
        self.model = model
    }

    // MARK: - Keyword Extraction

    /// Extracts keywords from a user query.
    ///
    /// Uses the LLM to identify key concepts, entities, and technical
    /// terms that can be matched against the knowledge graph.
    ///
    /// - Parameter query: The user's query or question.
    /// - Returns: Array of extracted keywords.
    public func extract(from query: String) async throws -> [String] {
        let response: KeywordsResponse = try await llmProvider.generate(
            systemPrompt: RAGPrompts.keywordExtraction,
            userPrompt: "Question: \(query)",
            responseType: KeywordsResponse.self,
            model: model,
            temperature: 0.1  // Low temperature for consistent extraction
        )

        // Clean and deduplicate keywords
        return cleanKeywords(response.keywords)
    }

    /// Extracts keywords with fallback if LLM extraction fails.
    ///
    /// Falls back to simple tokenization if the LLM call fails.
    ///
    /// - Parameter query: The user's query.
    /// - Returns: Array of keywords (from LLM or fallback).
    public func extractWithFallback(from query: String) async -> [String] {
        do {
            return try await extract(from: query)
        } catch {
            // Fallback to simple keyword extraction
            return simpleExtract(from: query)
        }
    }

    // MARK: - Simple Extraction (Fallback)

    /// Simple keyword extraction using basic NLP rules.
    ///
    /// - Parameter query: The query to extract keywords from.
    /// - Returns: Array of candidate keywords.
    public func simpleExtract(from query: String) -> [String] {
        // Remove common question words and stopwords
        let stopwords: Set<String> = [
            "what", "how", "why", "when", "where", "who", "which", "whom",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "about", "also", "now", "this",
            "that", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "our", "their"
        ]

        // Tokenize and filter
        let words = query
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty && $0.count > 2 }
            .filter { !stopwords.contains($0) }

        // Also try to extract multi-word phrases (simple approach)
        var phrases: [String] = []

        // Find capitalized sequences in original text (likely named entities)
        let pattern = "\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let range = NSRange(query.startIndex..<query.endIndex, in: query)
            let matches = regex.matches(in: query, range: range)
            for match in matches {
                if let matchRange = Range(match.range, in: query) {
                    phrases.append(String(query[matchRange]))
                }
            }
        }

        // Combine and deduplicate
        var keywords = Set(words)
        keywords.formUnion(phrases.map { $0.lowercased() })

        return Array(keywords).sorted()
    }

    // MARK: - Private Methods

    /// Cleans and deduplicates extracted keywords.
    private func cleanKeywords(_ keywords: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []

        for keyword in keywords {
            let cleaned = keyword
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()

            // Skip empty or very short keywords
            guard cleaned.count > 1 else { continue }

            // Skip if already seen
            guard !seen.contains(cleaned) else { continue }

            seen.insert(cleaned)
            result.append(cleaned)
        }

        return result
    }
}

// MARK: - Response Types

/// Response structure for keyword extraction.
struct KeywordsResponse: Codable, Sendable {
    /// The extracted keywords.
    let keywords: [String]
}

// MARK: - Prompts

/// Prompts for RAG operations.
public enum RAGPrompts {

    /// System prompt for keyword extraction.
    public static let keywordExtraction = """
        You are a keyword extraction assistant. Extract key concepts and entities \
        from the user's question that would be useful for searching a knowledge graph.

        Focus on:
        - Nouns and noun phrases
        - Technical terms and jargon
        - Named entities (people, organizations, technologies)
        - Domain-specific concepts

        Return your response as JSON in this exact format:
        {"keywords": ["keyword1", "keyword2", "keyword3"]}

        Guidelines:
        - Extract 3-10 keywords
        - Use lowercase
        - Include both specific terms and broader concepts
        - Do not include stopwords or question words
        - Include acronyms if present
        """

    /// System prompt for RAG-augmented question answering.
    public static let questionAnswering = """
        You are a helpful assistant with access to a knowledge graph. \
        Use the provided context from the graph to answer the user's question.

        Guidelines:
        - Base your answer primarily on the provided graph context
        - If the context doesn't contain relevant information, say so
        - Cite specific relationships from the context when possible
        - Be concise but thorough
        - If you're uncertain, express that uncertainty
        """

    /// Template for injecting graph context into prompts.
    public static func contextTemplate(context: String, question: String) -> String {
        """
        Graph Context:
        \(context)

        Question: \(question)

        Based on the graph context above, please answer the question.
        """
    }
}
