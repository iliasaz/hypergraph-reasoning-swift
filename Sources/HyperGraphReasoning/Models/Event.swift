import Foundation

/// Represents a single relationship (hyperedge) extracted from text.
///
/// An event captures a semantic relationship between source and target entities,
/// connected by a relation. This supports n-ary relationships where multiple
/// sources can relate to multiple targets.
///
/// Example:
/// ```json
/// {
///     "source": ["chitosan", "hydroxyapatite"],
///     "relation": "compose",
///     "target": ["chitosan/hydroxyapatite nanocomposite"]
/// }
/// ```
public struct Event: Codable, Sendable, Hashable {

    /// Source entities of the relationship.
    /// Always an array, even for binary relations with a single source.
    public let source: [String]

    /// Target entities of the relationship.
    /// Always an array, even for binary relations with a single target.
    public let target: [String]

    /// The semantic relationship between source and target.
    /// Examples: "is", "has", "compose", "leads to", "used for"
    public let relation: String

    /// Creates a new event.
    ///
    /// - Parameters:
    ///   - source: Source entities.
    ///   - target: Target entities.
    ///   - relation: The relationship between them.
    public init(source: [String], target: [String], relation: String) {
        self.source = source
        self.target = target
        self.relation = relation
    }

    /// Creates a binary event with single source and target.
    ///
    /// - Parameters:
    ///   - source: Single source entity.
    ///   - target: Single target entity.
    ///   - relation: The relationship between them.
    public init(source: String, target: String, relation: String) {
        self.source = [source]
        self.target = [target]
        self.relation = relation
    }

    /// All nodes involved in this event (union of source and target).
    public var allNodes: Set<String> {
        Set(source).union(Set(target))
    }

    /// Whether this is a binary relation (single source, single target).
    public var isBinary: Bool {
        source.count == 1 && target.count == 1
    }
}

// MARK: - CustomStringConvertible

extension Event: CustomStringConvertible {
    public var description: String {
        let srcStr = source.count == 1 ? source[0] : "[\(source.joined(separator: ", "))]"
        let tgtStr = target.count == 1 ? target[0] : "[\(target.joined(separator: ", "))]"
        return "\(srcStr) --[\(relation)]--> \(tgtStr)"
    }
}
