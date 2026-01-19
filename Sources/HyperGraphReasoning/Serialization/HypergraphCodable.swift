import Foundation

// MARK: - Hypergraph Codable

extension Hypergraph: Codable where NodeID: Codable, EdgeID: Codable {

    enum CodingKeys: String, CodingKey {
        case incidenceDict = "incidence_dict"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Decode as dictionary of arrays, then convert to sets
        let dictOfArrays = try container.decode([EdgeID: [NodeID]].self, forKey: .incidenceDict)
        self.incidenceDict = dictOfArrays.mapValues { Set($0) }
        self._adjacencyDict = nil
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        // Encode sets as sorted arrays for consistent output
        let dictOfArrays = incidenceDict.mapValues { nodes -> [NodeID] in
            if let stringNodes = nodes as? Set<String> {
                return Array(stringNodes.sorted()) as! [NodeID]
            }
            return Array(nodes)
        }
        try container.encode(dictOfArrays, forKey: .incidenceDict)
    }
}

// MARK: - File Operations

extension Hypergraph where NodeID: Codable, EdgeID: Codable {

    /// Saves the hypergraph to a JSON file.
    ///
    /// - Parameter url: The file URL to save to.
    /// - Throws: Encoding or file system errors.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Loads a hypergraph from a JSON file.
    ///
    /// - Parameter url: The file URL to load from.
    /// - Returns: The loaded hypergraph.
    /// - Throws: Decoding or file system errors.
    public static func load(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(Self.self, from: data)
    }

    /// Saves the hypergraph to a binary plist file.
    ///
    /// This format is more compact than JSON.
    ///
    /// - Parameter url: The file URL to save to.
    public func saveBinary(to url: URL) throws {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Loads a hypergraph from a binary plist file.
    ///
    /// - Parameter url: The file URL to load from.
    /// - Returns: The loaded hypergraph.
    public static func loadBinary(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        let decoder = PropertyListDecoder()
        return try decoder.decode(Self.self, from: data)
    }
}

// MARK: - Metadata File Operations

extension Array where Element == ChunkMetadata {

    /// Saves metadata to a JSON file.
    ///
    /// - Parameter url: The file URL to save to.
    /// - Throws: Encoding or file system errors.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Loads metadata from a JSON file.
    ///
    /// - Parameter url: The file URL to load from.
    /// - Returns: The loaded metadata array.
    /// - Throws: Decoding or file system errors.
    public static func load(from url: URL) throws -> [ChunkMetadata] {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode([ChunkMetadata].self, from: data)
    }
}

// MARK: - Python Compatibility

extension Hypergraph where NodeID == String, EdgeID == String {

    /// Exports to a format compatible with HyperNetX's incidence dict.
    ///
    /// This produces a JSON structure that can be loaded in Python as:
    /// ```python
    /// import hypernetx as hnx
    /// with open('graph.json') as f:
    ///     data = json.load(f)
    /// H = hnx.Hypergraph(data['incidence_dict'])
    /// ```
    public func exportForHyperNetX() throws -> Data {
        let export = ["incidence_dict": incidenceDict.mapValues { Array($0.sorted()) }]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(export)
    }

    /// Exports to a HyperNetX-compatible JSON file.
    ///
    /// - Parameter url: The file URL to save to.
    public func saveForHyperNetX(to url: URL) throws {
        let data = try exportForHyperNetX()
        try data.write(to: url)
    }
}
