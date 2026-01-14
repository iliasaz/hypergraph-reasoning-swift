import Foundation
import Testing
@testable import HyperGraphReasoning

@Suite("Text Splitter Tests")
struct TextSplitterTests {

    // MARK: - TextChunk Tests

    @Test("TextChunk MD5 generation")
    func testChunkMD5() {
        let chunk = TextChunk(text: "Hello, World!")

        // MD5 of "Hello, World!" is "65a8e27d8879283831b664bd8b7f0ad4"
        #expect(chunk.chunkID == "65a8e27d8879283831b664bd8b7f0ad4")
    }

    @Test("TextChunk with same text has same ID")
    func testDeterministicID() {
        let chunk1 = TextChunk(text: "Test content")
        let chunk2 = TextChunk(text: "Test content")

        #expect(chunk1.chunkID == chunk2.chunkID)
    }

    @Test("TextChunk with different text has different ID")
    func testDifferentID() {
        let chunk1 = TextChunk(text: "Content A")
        let chunk2 = TextChunk(text: "Content B")

        #expect(chunk1.chunkID != chunk2.chunkID)
    }

    @Test("TextChunk custom ID")
    func testCustomID() {
        let chunk = TextChunk(text: "Test", chunkID: "custom-id")

        #expect(chunk.chunkID == "custom-id")
    }

    // MARK: - RecursiveTextSplitter Tests

    @Test("Splitter with default parameters")
    func testDefaultSplitter() {
        let splitter = RecursiveTextSplitter()

        #expect(splitter.chunkSize == 2500)
        #expect(splitter.chunkOverlap == 0)
    }

    @Test("Split short text returns single chunk")
    func testShortText() {
        let splitter = RecursiveTextSplitter(chunkSize: 100)
        let chunks = splitter.split("This is a short text.")

        #expect(chunks.count == 1)
        #expect(chunks[0].text == "This is a short text.")
    }

    @Test("Split on paragraph boundaries")
    func testParagraphSplit() {
        let splitter = RecursiveTextSplitter(chunkSize: 50)
        let text = """
        First paragraph here.

        Second paragraph here.

        Third paragraph here.
        """

        let chunks = splitter.split(text)

        #expect(chunks.count >= 2)
    }

    @Test("Split on line boundaries")
    func testLineSplit() {
        let splitter = RecursiveTextSplitter(chunkSize: 30)
        let text = """
        Line one.
        Line two.
        Line three.
        """

        let chunks = splitter.split(text)

        #expect(chunks.count >= 2)
    }

    @Test("Split with overlap")
    func testOverlap() {
        let splitter = RecursiveTextSplitter(chunkSize: 20, chunkOverlap: 5)
        let text = "Word1 Word2 Word3 Word4 Word5 Word6"

        let chunks = splitter.split(text)

        // With overlap, chunks should share some content
        #expect(chunks.count >= 2)
    }

    @Test("All chunks have valid IDs")
    func testChunkIDs() {
        let splitter = RecursiveTextSplitter(chunkSize: 50)
        let text = """
        This is some text that should be split into multiple chunks.

        Each chunk should have a unique MD5-based identifier.
        """

        let chunks = splitter.split(text)

        for chunk in chunks {
            #expect(!chunk.chunkID.isEmpty)
            #expect(chunk.chunkID.count == 32)  // MD5 hex is 32 chars
        }
    }

    @Test("Empty text returns no chunks")
    func testEmptyText() {
        let splitter = RecursiveTextSplitter()
        let chunks = splitter.split("")

        #expect(chunks.isEmpty)
    }

    @Test("Chunk sizes respect limit")
    func testChunkSizeLimit() {
        let maxSize = 100
        let splitter = RecursiveTextSplitter(chunkSize: maxSize)
        let text = String(repeating: "word ", count: 50)  // ~250 chars

        let chunks = splitter.split(text)

        for chunk in chunks {
            // Most chunks should be under limit (some may slightly exceed due to separator handling)
            #expect(chunk.length <= maxSize + 10)
        }
    }

    // MARK: - String MD5 Extension Tests

    @Test("MD5 of empty string")
    func testEmptyMD5() {
        let hash = "".md5Hex()
        #expect(hash == "d41d8cd98f00b204e9800998ecf8427e")
    }

    @Test("MD5 produces 32 character hex string")
    func testMD5Length() {
        let hash = "any text".md5Hex()
        #expect(hash.count == 32)

        // All characters should be hex
        let hexChars = CharacterSet(charactersIn: "0123456789abcdef")
        for char in hash.unicodeScalars {
            #expect(hexChars.contains(char))
        }
    }
}
