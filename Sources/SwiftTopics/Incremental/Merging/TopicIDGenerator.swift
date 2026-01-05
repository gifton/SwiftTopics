// TopicIDGenerator.swift
// SwiftTopics
//
// Manages stable topic ID generation across model updates

import Foundation

// MARK: - Topic ID Generator

/// Manages stable topic ID generation.
///
/// Tracks the highest used topic ID to ensure new topics receive unique IDs
/// that don't conflict with existing or previously-used IDs. This is crucial
/// for topic stability across retrains.
///
/// ## Usage
///
/// ```swift
/// var generator = TopicIDGenerator()
///
/// // Observe existing topic IDs
/// for topic in existingTopics {
///     generator.observe(topic.id)
/// }
///
/// // Generate new IDs for unmatched topics
/// let newID1 = generator.next()  // Won't conflict with observed IDs
/// let newID2 = generator.next()
/// ```
///
/// ## Thread Safety
///
/// `TopicIDGenerator` is a value type. Each mutation creates a new value,
/// so it's safe to use in concurrent contexts when properly synchronized.
///
/// ## Persistence
///
/// Conforms to `Codable` for storage. The generator state should be persisted
/// alongside the model state to maintain ID continuity across sessions.
public struct TopicIDGenerator: Sendable, Codable, Hashable {

    /// The next ID to be generated.
    private var nextID: Int

    // MARK: - Initialization

    /// Creates a generator starting from a specific ID.
    ///
    /// - Parameter startingFrom: The first ID that will be generated.
    ///   Defaults to 0.
    public init(startingFrom: Int = 0) {
        self.nextID = max(startingFrom, 0)
    }

    /// Creates a generator initialized with existing topics.
    ///
    /// The generator will start from an ID higher than any existing topic ID.
    ///
    /// - Parameter existingTopics: Topics whose IDs should be observed.
    public init(observing existingTopics: [Topic]) {
        self.nextID = 0
        for topic in existingTopics {
            observe(topic.id)
        }
    }

    /// Creates a generator initialized with existing topic IDs.
    ///
    /// - Parameter existingIDs: Topic IDs to observe.
    public init(observing existingIDs: [TopicID]) {
        self.nextID = 0
        for id in existingIDs {
            observe(id)
        }
    }

    // MARK: - ID Generation

    /// Generates the next unique topic ID.
    ///
    /// This increments the internal counter and returns a new, unique ID.
    ///
    /// - Returns: A new topic ID guaranteed to be unique within this generator's scope.
    public mutating func next() -> TopicID {
        let id = TopicID(value: nextID)
        nextID += 1
        return id
    }

    /// Generates multiple unique topic IDs.
    ///
    /// - Parameter count: Number of IDs to generate.
    /// - Returns: Array of new unique topic IDs.
    public mutating func next(count: Int) -> [TopicID] {
        var ids = [TopicID]()
        ids.reserveCapacity(count)
        for _ in 0..<count {
            ids.append(next())
        }
        return ids
    }

    /// Peeks at the next ID without consuming it.
    ///
    /// - Returns: The ID that would be returned by `next()`.
    public func peek() -> TopicID {
        TopicID(value: nextID)
    }

    // MARK: - Observation

    /// Updates the generator to account for an observed topic ID.
    ///
    /// If the observed ID is >= the next ID, the next ID is updated to
    /// be higher than the observed ID. This ensures no conflicts with
    /// existing topic IDs.
    ///
    /// - Parameter id: A topic ID that already exists in the system.
    public mutating func observe(_ id: TopicID) {
        // Don't track outlier IDs
        guard !id.isOutlier else { return }

        // Ensure nextID is always higher than observed IDs
        if id.value >= nextID {
            nextID = id.value + 1
        }
    }

    /// Updates the generator to account for multiple observed topic IDs.
    ///
    /// - Parameter ids: Topic IDs that already exist in the system.
    public mutating func observe(_ ids: [TopicID]) {
        for id in ids {
            observe(id)
        }
    }

    /// Updates the generator to account for topics in an array.
    ///
    /// - Parameter topics: Topics whose IDs should be observed.
    public mutating func observe(topics: [Topic]) {
        for topic in topics {
            observe(topic.id)
        }
    }

    // MARK: - Queries

    /// The value of the next ID that will be generated.
    public var nextIDValue: Int {
        nextID
    }

    /// The number of topic IDs that have been or could be in use.
    ///
    /// This is the count of IDs from 0 to nextID-1.
    public var usedIDCount: Int {
        nextID
    }
}

// MARK: - Topic ID Generator Extensions

extension TopicIDGenerator {

    /// Creates ID mappings from a match result.
    ///
    /// This is the primary way to convert a `TopicMatcher.MatchResult` into
    /// stable topic IDs for the new model.
    ///
    /// - Parameters:
    ///   - matchResult: The result from `TopicMatcher.match()`.
    ///   - preferExistingOnMerge: When topics merge, whether to keep the
    ///     lowest-numbered existing ID. Defaults to true.
    /// - Returns: Dictionary mapping new topic indices to their stable IDs.
    public mutating func createIDMappings(
        from matchResult: TopicMatcher.MatchResult,
        preferExistingOnMerge: Bool = true
    ) -> [Int: TopicID] {
        var mappings = [Int: TopicID]()

        // First, handle matched topics (keep their existing IDs)
        for (newIndex, oldID) in matchResult.newToOld {
            if let oldID = oldID {
                mappings[newIndex] = oldID
                observe(oldID)
            }
        }

        // Handle merges: use the lowest existing ID for the merged topic
        if preferExistingOnMerge {
            for merge in matchResult.merges {
                let sortedIDs = merge.oldIDs.sorted()
                if let primaryID = sortedIDs.first {
                    mappings[merge.newIndex] = primaryID
                }
            }
        }

        // Generate new IDs for unmatched topics
        for newIndex in matchResult.newTopicIndices {
            if mappings[newIndex] == nil {
                mappings[newIndex] = next()
            }
        }

        return mappings
    }
}
