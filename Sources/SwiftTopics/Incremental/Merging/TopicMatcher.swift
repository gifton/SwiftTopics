// TopicMatcher.swift
// SwiftTopics
//
// Matches new topics to existing topics for ID stability

import Foundation

// MARK: - Topic Matcher

/// Matches new topics to existing topics for ID stability.
///
/// When a topic model is retrained (micro-retrain or full refresh), the new
/// clustering may produce topics that are semantically equivalent to existing
/// topics but with different internal indices. `TopicMatcher` solves this by
/// finding the optimal mapping between new and old topics based on centroid
/// similarity.
///
/// ## Algorithm
///
/// 1. Compute cosine similarity between all new/old centroid pairs
/// 2. Use Hungarian algorithm for optimal bipartite matching
/// 3. Filter matches below similarity threshold
/// 4. Classify results: matches, merges, splits, new topics, retired topics
///
/// ## Example
///
/// ```swift
/// let matcher = TopicMatcher()
/// let result = matcher.match(
///     newCentroids: newTopics.compactMap(\.centroid),
///     oldTopics: existingTopics,
///     configuration: .default
/// )
///
/// // Map new topic indices to stable IDs
/// for (newIndex, oldID) in result.newToOld {
///     if let oldID = oldID {
///         // Use existing topic ID
///     } else {
///         // Assign new topic ID
///     }
/// }
/// ```
///
/// ## Thread Safety
///
/// `TopicMatcher` is `Sendable` and stateless.
public struct TopicMatcher: Sendable {

    // MARK: - Configuration

    /// Configuration for topic matching.
    public struct Configuration: Sendable {

        /// Minimum cosine similarity to consider a match.
        ///
        /// Matches below this threshold are treated as new topics.
        /// - Default: 0.7
        /// - Range: [0, 1] where higher = stricter matching
        public let similarityThreshold: Float

        /// Whether to detect and report merge events.
        ///
        /// A merge occurs when multiple old topics best-match to a single new topic.
        /// This typically happens when the model consolidates similar topics.
        public let detectMerges: Bool

        /// Whether to detect and report split events.
        ///
        /// A split occurs when one old topic best-matches multiple new topics.
        /// This typically happens when a broad topic becomes more granular.
        public let detectSplits: Bool

        /// Default configuration.
        public static let `default` = Configuration(
            similarityThreshold: 0.7,
            detectMerges: true,
            detectSplits: true
        )

        /// Strict configuration requiring higher similarity.
        public static let strict = Configuration(
            similarityThreshold: 0.85,
            detectMerges: true,
            detectSplits: true
        )

        /// Lenient configuration allowing lower similarity matches.
        public static let lenient = Configuration(
            similarityThreshold: 0.5,
            detectMerges: true,
            detectSplits: true
        )

        /// Creates a custom configuration.
        ///
        /// - Parameters:
        ///   - similarityThreshold: Minimum similarity for a match (default: 0.7).
        ///   - detectMerges: Whether to detect merges (default: true).
        ///   - detectSplits: Whether to detect splits (default: true).
        public init(
            similarityThreshold: Float = 0.7,
            detectMerges: Bool = true,
            detectSplits: Bool = true
        ) {
            self.similarityThreshold = similarityThreshold
            self.detectMerges = detectMerges
            self.detectSplits = detectSplits
        }
    }

    // MARK: - Match Result

    /// Result of topic matching.
    public struct MatchResult: Sendable {

        /// Mapping from new topic index to matched old topic ID.
        ///
        /// - Key: Index in the new topics array (0-indexed)
        /// - Value: The matched old `TopicID`, or `nil` if this is a new topic
        public let newToOld: [Int: TopicID?]

        /// Topics that were merged (multiple old → one new).
        ///
        /// Each entry represents one merge event where multiple old topics
        /// now map to a single new topic.
        public let merges: [MergeEvent]

        /// Topics that were split (one old → multiple new).
        ///
        /// Each entry represents one split event where one old topic
        /// now maps to multiple new topics.
        public let splits: [SplitEvent]

        /// Indices of genuinely new topics (no match found).
        ///
        /// These are new topics that have no similar counterpart in
        /// the old model and need new IDs.
        public let newTopicIndices: [Int]

        /// IDs of old topics that have no matching new topic.
        ///
        /// These topics "disappeared" in the new model, possibly because
        /// the underlying documents were reassigned or removed.
        public let retiredTopicIDs: [TopicID]

        /// The similarity matrix used for matching.
        ///
        /// `similarityMatrix[newIndex][oldIndex]` = cosine similarity
        public let similarityMatrix: [[Float]]

        /// Summary statistics about the matching.
        public var summary: String {
            var lines = [String]()
            lines.append("Match Result:")
            lines.append("  Matched: \(newToOld.values.compactMap { $0 }.count)")
            lines.append("  New topics: \(newTopicIndices.count)")
            lines.append("  Retired: \(retiredTopicIDs.count)")
            if !merges.isEmpty {
                lines.append("  Merges: \(merges.count)")
            }
            if !splits.isEmpty {
                lines.append("  Splits: \(splits.count)")
            }
            return lines.joined(separator: "\n")
        }
    }

    /// A merge event where multiple old topics become one new topic.
    public struct MergeEvent: Sendable {
        /// IDs of the old topics that were merged.
        public let oldIDs: [TopicID]

        /// Index of the new topic they merged into.
        public let newIndex: Int

        /// Similarity scores for each old topic.
        public let similarities: [Float]
    }

    /// A split event where one old topic becomes multiple new topics.
    public struct SplitEvent: Sendable {
        /// ID of the old topic that was split.
        public let oldID: TopicID

        /// Indices of the new topics it split into.
        public let newIndices: [Int]

        /// Similarity scores for each new topic.
        public let similarities: [Float]
    }

    // MARK: - Initialization

    private let hungarian: HungarianMatcher

    /// Creates a new topic matcher.
    public init() {
        self.hungarian = HungarianMatcher()
    }

    // MARK: - Public API

    /// Matches new topics to existing topics.
    ///
    /// Uses the Hungarian algorithm for optimal bipartite matching based on
    /// centroid cosine similarity.
    ///
    /// - Parameters:
    ///   - newCentroids: Centroids from the new training run.
    ///   - oldTopics: Existing topics with their IDs and centroids.
    ///   - configuration: Matching configuration.
    /// - Returns: Match result with ID mappings and event detection.
    public func match(
        newCentroids: [Embedding],
        oldTopics: [Topic],
        configuration: Configuration = .default
    ) -> MatchResult {
        // Handle edge cases
        if newCentroids.isEmpty {
            return MatchResult(
                newToOld: [:],
                merges: [],
                splits: [],
                newTopicIndices: [],
                retiredTopicIDs: oldTopics.map(\.id),
                similarityMatrix: []
            )
        }

        if oldTopics.isEmpty {
            let newToOld = Dictionary(
                uniqueKeysWithValues: (0..<newCentroids.count).map { ($0, nil as TopicID?) }
            )
            return MatchResult(
                newToOld: newToOld,
                merges: [],
                splits: [],
                newTopicIndices: Array(0..<newCentroids.count),
                retiredTopicIDs: [],
                similarityMatrix: []
            )
        }

        // Compute similarity matrix
        let oldCentroids = oldTopics.compactMap(\.centroid)
        guard oldCentroids.count == oldTopics.count else {
            // Some old topics lack centroids - fall back to treating all as new
            let newToOld = Dictionary(
                uniqueKeysWithValues: (0..<newCentroids.count).map { ($0, nil as TopicID?) }
            )
            return MatchResult(
                newToOld: newToOld,
                merges: [],
                splits: [],
                newTopicIndices: Array(0..<newCentroids.count),
                retiredTopicIDs: [],
                similarityMatrix: []
            )
        }

        let similarityMatrix = computeSimilarityMatrix(
            newCentroids: newCentroids,
            oldCentroids: oldCentroids
        )

        // Convert to cost matrix for Hungarian algorithm
        let costMatrix = HungarianMatcher.costMatrixFromSimilarities(similarityMatrix)

        // Find optimal assignment
        let assignment = hungarian.solve(costs: costMatrix)

        // Build results
        return buildMatchResult(
            assignment: assignment,
            similarityMatrix: similarityMatrix,
            newCount: newCentroids.count,
            oldTopics: oldTopics,
            configuration: configuration
        )
    }

    /// Matches topics using pre-computed similarity matrix.
    ///
    /// Useful when you've already computed similarities and want to
    /// avoid redundant computation.
    ///
    /// - Parameters:
    ///   - similarityMatrix: Pre-computed similarity matrix [newIndex][oldIndex].
    ///   - oldTopics: Existing topics.
    ///   - configuration: Matching configuration.
    /// - Returns: Match result.
    public func match(
        similarityMatrix: [[Float]],
        oldTopics: [Topic],
        configuration: Configuration = .default
    ) -> MatchResult {
        guard !similarityMatrix.isEmpty else {
            return MatchResult(
                newToOld: [:],
                merges: [],
                splits: [],
                newTopicIndices: [],
                retiredTopicIDs: oldTopics.map(\.id),
                similarityMatrix: []
            )
        }

        let costMatrix = HungarianMatcher.costMatrixFromSimilarities(similarityMatrix)
        let assignment = hungarian.solve(costs: costMatrix)

        return buildMatchResult(
            assignment: assignment,
            similarityMatrix: similarityMatrix,
            newCount: similarityMatrix.count,
            oldTopics: oldTopics,
            configuration: configuration
        )
    }

    // MARK: - Private Implementation

    /// Computes the cosine similarity matrix between new and old centroids.
    private func computeSimilarityMatrix(
        newCentroids: [Embedding],
        oldCentroids: [Embedding]
    ) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(newCentroids.count)

        for newCentroid in newCentroids {
            var row = [Float]()
            row.reserveCapacity(oldCentroids.count)

            for oldCentroid in oldCentroids {
                let similarity = newCentroid.cosineSimilarity(oldCentroid)
                row.append(similarity)
            }

            matrix.append(row)
        }

        return matrix
    }

    /// Builds the match result from the Hungarian assignment.
    private func buildMatchResult(
        assignment: [(row: Int, col: Int)],
        similarityMatrix: [[Float]],
        newCount: Int,
        oldTopics: [Topic],
        configuration: Configuration
    ) -> MatchResult {
        let oldCount = oldTopics.count

        // Build mapping from new to old, applying threshold
        var newToOld = [Int: TopicID?]()
        var matchedOldIndices = Set<Int>()
        var matchedNewIndices = Set<Int>()

        for (newIdx, oldIdx) in assignment {
            let similarity = similarityMatrix[newIdx][oldIdx]

            if similarity >= configuration.similarityThreshold {
                newToOld[newIdx] = oldTopics[oldIdx].id
                matchedOldIndices.insert(oldIdx)
                matchedNewIndices.insert(newIdx)
            } else {
                // Below threshold - treat as new topic
                newToOld[newIdx] = nil
            }
        }

        // Fill in unmatched new topics
        for newIdx in 0..<newCount where newToOld[newIdx] == nil {
            newToOld[newIdx] = nil
        }

        // Find new topic indices (no match)
        let newTopicIndices = (0..<newCount).filter { newToOld[$0] == nil }

        // Find retired topic IDs (no matching new topic)
        let retiredTopicIDs = (0..<oldCount)
            .filter { !matchedOldIndices.contains($0) }
            .map { oldTopics[$0].id }

        // Detect merges and splits
        var merges = [MergeEvent]()
        var splits = [SplitEvent]()

        if configuration.detectMerges || configuration.detectSplits {
            (merges, splits) = detectMergesAndSplits(
                similarityMatrix: similarityMatrix,
                oldTopics: oldTopics,
                threshold: configuration.similarityThreshold,
                detectMerges: configuration.detectMerges,
                detectSplits: configuration.detectSplits
            )
        }

        return MatchResult(
            newToOld: newToOld,
            merges: merges,
            splits: splits,
            newTopicIndices: newTopicIndices,
            retiredTopicIDs: retiredTopicIDs,
            similarityMatrix: similarityMatrix
        )
    }

    /// Detects merge and split events from the similarity matrix.
    ///
    /// A merge is detected when multiple old topics have their best match
    /// as the same new topic (above threshold).
    ///
    /// A split is detected when multiple new topics have their best match
    /// as the same old topic (above threshold).
    private func detectMergesAndSplits(
        similarityMatrix: [[Float]],
        oldTopics: [Topic],
        threshold: Float,
        detectMerges: Bool,
        detectSplits: Bool
    ) -> (merges: [MergeEvent], splits: [SplitEvent]) {
        let newCount = similarityMatrix.count
        guard newCount > 0 else { return ([], []) }
        let oldCount = similarityMatrix[0].count

        var merges = [MergeEvent]()
        var splits = [SplitEvent]()

        // For each new topic, find all old topics that best-match to it
        if detectMerges {
            var newToBestOlds = [Int: [(oldIdx: Int, similarity: Float)]]()

            for oldIdx in 0..<oldCount {
                // Find best new match for this old topic
                var bestNewIdx = -1
                var bestSimilarity: Float = -1

                for newIdx in 0..<newCount {
                    let sim = similarityMatrix[newIdx][oldIdx]
                    if sim > bestSimilarity {
                        bestSimilarity = sim
                        bestNewIdx = newIdx
                    }
                }

                if bestNewIdx >= 0 && bestSimilarity >= threshold {
                    if newToBestOlds[bestNewIdx] == nil {
                        newToBestOlds[bestNewIdx] = []
                    }
                    newToBestOlds[bestNewIdx]?.append((oldIdx, bestSimilarity))
                }
            }

            // Report merges (multiple old → one new)
            for (newIdx, olds) in newToBestOlds where olds.count > 1 {
                merges.append(MergeEvent(
                    oldIDs: olds.map { oldTopics[$0.oldIdx].id },
                    newIndex: newIdx,
                    similarities: olds.map(\.similarity)
                ))
            }
        }

        // For each old topic, find all new topics that best-match to it
        if detectSplits {
            var oldToBestNews = [Int: [(newIdx: Int, similarity: Float)]]()

            for newIdx in 0..<newCount {
                // Find best old match for this new topic
                var bestOldIdx = -1
                var bestSimilarity: Float = -1

                for oldIdx in 0..<oldCount {
                    let sim = similarityMatrix[newIdx][oldIdx]
                    if sim > bestSimilarity {
                        bestSimilarity = sim
                        bestOldIdx = oldIdx
                    }
                }

                if bestOldIdx >= 0 && bestSimilarity >= threshold {
                    if oldToBestNews[bestOldIdx] == nil {
                        oldToBestNews[bestOldIdx] = []
                    }
                    oldToBestNews[bestOldIdx]?.append((newIdx, bestSimilarity))
                }
            }

            // Report splits (one old → multiple new)
            for (oldIdx, news) in oldToBestNews where news.count > 1 {
                splits.append(SplitEvent(
                    oldID: oldTopics[oldIdx].id,
                    newIndices: news.map(\.newIdx),
                    similarities: news.map(\.similarity)
                ))
            }
        }

        return (merges, splits)
    }
}
