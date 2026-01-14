import Foundation

/// A text splitter that recursively splits text into chunks of a specified size.
///
/// This implementation mirrors LangChain's `RecursiveCharacterTextSplitter`:
/// 1. Try to split on the largest separator first (e.g., double newlines)
/// 2. If chunks are still too large, split on smaller separators
/// 3. As a last resort, split on individual characters
/// 4. Optionally apply overlap between chunks
///
/// - Note: This matches the Python implementation used in the hypergraph generation pipeline.
public struct RecursiveTextSplitter: Sendable {

    /// Maximum size of each chunk in characters.
    public let chunkSize: Int

    /// Number of characters to overlap between consecutive chunks.
    public let chunkOverlap: Int

    /// Separators to try, in order from most preferred to least preferred.
    public let separators: [String]

    /// Whether to keep the separator at the end of chunks.
    public let keepSeparator: Bool

    /// Default separators matching LangChain's behavior.
    public static let defaultSeparators = ["\n\n", "\n", ". ", " ", ""]

    /// Creates a new recursive text splitter.
    ///
    /// - Parameters:
    ///   - chunkSize: Maximum size of each chunk in characters. Default is 2500.
    ///   - chunkOverlap: Characters to overlap between chunks. Default is 0.
    ///   - separators: Separators to use. Default is paragraph, line, sentence, word, character.
    ///   - keepSeparator: Whether to keep the separator. Default is true.
    public init(
        chunkSize: Int = 2500,
        chunkOverlap: Int = 0,
        separators: [String] = defaultSeparators,
        keepSeparator: Bool = true
    ) {
        precondition(chunkSize > 0, "chunkSize must be positive")
        precondition(chunkOverlap >= 0, "chunkOverlap cannot be negative")
        precondition(chunkOverlap < chunkSize, "chunkOverlap must be less than chunkSize")

        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.separators = separators
        self.keepSeparator = keepSeparator
    }

    /// Splits text into chunks.
    ///
    /// - Parameter text: The text to split.
    /// - Returns: An array of text chunks with auto-generated IDs.
    public func split(_ text: String) -> [TextChunk] {
        let parts = splitText(text, separators: separators)
        let merged = mergeSplits(parts)
        return merged.map { TextChunk(text: $0) }
    }

    /// Splits text into raw strings (without creating TextChunk objects).
    ///
    /// - Parameter text: The text to split.
    /// - Returns: An array of strings.
    public func splitToStrings(_ text: String) -> [String] {
        let parts = splitText(text, separators: separators)
        return mergeSplits(parts)
    }

    // MARK: - Private Implementation

    /// Recursively splits text using the given separators.
    private func splitText(_ text: String, separators: [String]) -> [String] {
        var finalChunks = [String]()

        // Find the appropriate separator
        var separator = separators.last ?? ""
        var newSeparators = [String]()

        for (i, sep) in separators.enumerated() {
            if sep.isEmpty {
                separator = sep
                newSeparators = []
                break
            }
            if text.contains(sep) {
                separator = sep
                newSeparators = Array(separators.dropFirst(i + 1))
                break
            }
        }

        // Split on the chosen separator
        let splits: [String]
        if separator.isEmpty {
            // Character-by-character split
            splits = text.map { String($0) }
        } else {
            splits = splitWithSeparator(text, separator: separator)
        }

        // Process each split
        var goodSplits = [String]()

        for split in splits {
            if split.count < chunkSize {
                goodSplits.append(split)
            } else {
                // Merge any accumulated good splits
                if !goodSplits.isEmpty {
                    let merged = mergeSplits(goodSplits)
                    finalChunks.append(contentsOf: merged)
                    goodSplits.removeAll()
                }

                // Recursively split the large chunk
                if newSeparators.isEmpty {
                    // Can't split further, keep as is
                    finalChunks.append(split)
                } else {
                    let subChunks = splitText(split, separators: newSeparators)
                    finalChunks.append(contentsOf: subChunks)
                }
            }
        }

        // Merge remaining good splits
        if !goodSplits.isEmpty {
            let merged = mergeSplits(goodSplits)
            finalChunks.append(contentsOf: merged)
        }

        return finalChunks
    }

    /// Splits text on a separator, optionally keeping the separator.
    private func splitWithSeparator(_ text: String, separator: String) -> [String] {
        let components = text.components(separatedBy: separator)

        if keepSeparator && !separator.isEmpty {
            // Append separator to all but the last component
            return components.enumerated().map { index, component in
                if index < components.count - 1 {
                    return component + separator
                }
                return component
            }.filter { !$0.isEmpty }
        } else {
            return components.filter { !$0.isEmpty }
        }
    }

    /// Merges small splits into chunks of appropriate size.
    private func mergeSplits(_ splits: [String]) -> [String] {
        var chunks = [String]()
        var currentChunk = ""

        for split in splits {
            let potentialLength = currentChunk.count + split.count

            if potentialLength <= chunkSize {
                currentChunk += split
            } else {
                // Save current chunk if non-empty
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                }

                // Handle overlap
                if chunkOverlap > 0 && !chunks.isEmpty {
                    let overlap = getOverlapText(from: currentChunk)
                    currentChunk = overlap + split
                } else {
                    currentChunk = split
                }

                // If the new split itself is too large, just add it
                if currentChunk.count > chunkSize {
                    chunks.append(currentChunk)
                    currentChunk = ""
                }
            }
        }

        // Don't forget the last chunk
        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }

        return chunks
    }

    /// Gets the overlap text from the end of a string.
    private func getOverlapText(from text: String) -> String {
        if text.count <= chunkOverlap {
            return text
        }
        let startIndex = text.index(text.endIndex, offsetBy: -chunkOverlap)
        return String(text[startIndex...])
    }
}

// MARK: - CustomStringConvertible

extension RecursiveTextSplitter: CustomStringConvertible {
    public var description: String {
        "RecursiveTextSplitter(chunkSize: \(chunkSize), overlap: \(chunkOverlap))"
    }
}
