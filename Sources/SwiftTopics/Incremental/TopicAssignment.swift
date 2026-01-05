// TopicAssignment.swift
// SwiftTopics
//
// Result type for incremental document-to-topic assignment

import Foundation

// MARK: - Incremental Topic Assignment

/// Result of assigning a document to a topic in incremental mode.
///
/// When a document is processed by `IncrementalTopicUpdater`, it receives
/// an immediate assignment to the nearest topic centroid. This struct
/// captures the assignment details for display and analysis.
///
/// This type is distinct from `TopicAssignment` in the core module because
/// it includes additional metadata specific to incremental updates:
/// - Whether this was a transform (fast) vs training assignment
/// - Topic keywords for display
/// - Confidence based on similarity margin
///
/// ## Assignment Types
///
/// - **Transform assignment**: Fast centroid-based assignment (<1ms)
/// - **Training assignment**: Assignment from full training run
/// - **Outlier**: No suitable topic found (topicID.isOutlier = true)
///
/// ## Example
///
/// ```swift
/// let assignment = try await updater.processDocument(document, embedding: embedding)
///
/// if assignment.isOutlier {
///     print("Document is an outlier")
/// } else {
///     print("Assigned to topic \(assignment.topicID)")
///     print("Keywords: \(assignment.topicKeywords.joined(separator: ", "))")
///     print("Confidence: \(String(format: "%.1f%%", assignment.confidence * 100))")
/// }
/// ```
public struct IncrementalTopicAssignment: Sendable, Hashable {

    // MARK: - Properties

    /// The assigned topic ID.
    ///
    /// For outliers, this is `TopicID.outlier` (value = -1).
    /// For assigned documents, this is a non-negative topic ID.
    public let topicID: TopicID

    /// Confidence in the assignment (0-1).
    ///
    /// Computed based on:
    /// - Similarity to the assigned topic's centroid
    /// - Margin over the second-best topic
    ///
    /// Higher confidence indicates clearer assignment.
    /// Outliers always have confidence = 0.
    public let confidence: Float

    /// Cosine similarity to the assigned topic's centroid.
    ///
    /// Range: [-1, 1] where:
    /// - 1.0 = identical direction
    /// - 0.0 = orthogonal
    /// - -1.0 = opposite direction
    ///
    /// For outliers, this is the similarity to the best (but insufficient) topic.
    public let similarity: Float

    /// Whether this was assigned via transform (true) or full training (false).
    ///
    /// Transform assignments are fast but may be less accurate than
    /// assignments from full training.
    public let isTransformAssignment: Bool

    /// Top keywords for the assigned topic.
    ///
    /// Useful for displaying the topic theme to users.
    /// Empty for outliers.
    public let topicKeywords: [String]

    /// Whether this document is an outlier.
    ///
    /// Outliers are documents that couldn't be confidently assigned
    /// to any existing topic.
    public var isOutlier: Bool { topicID.isOutlier }

    /// Distance to the centroid (1 - similarity).
    ///
    /// Convenience property for algorithms that prefer distance over similarity.
    public var distanceToCentroid: Float { 1 - similarity }

    // MARK: - Initialization

    /// Creates an incremental topic assignment.
    ///
    /// - Parameters:
    ///   - topicID: The assigned topic ID.
    ///   - confidence: Confidence in the assignment (0-1).
    ///   - similarity: Cosine similarity to the centroid.
    ///   - isTransformAssignment: Whether this is a transform assignment.
    ///   - topicKeywords: Keywords for the assigned topic.
    public init(
        topicID: TopicID,
        confidence: Float,
        similarity: Float,
        isTransformAssignment: Bool,
        topicKeywords: [String]
    ) {
        self.topicID = topicID
        self.confidence = confidence
        self.similarity = similarity
        self.isTransformAssignment = isTransformAssignment
        self.topicKeywords = topicKeywords
    }

    // MARK: - Factory Methods

    /// Creates an outlier assignment.
    ///
    /// - Parameters:
    ///   - bestSimilarity: Similarity to the best (but insufficient) topic.
    ///   - isTransformAssignment: Whether this is a transform assignment.
    /// - Returns: An outlier assignment.
    public static func outlier(
        bestSimilarity: Float = 0,
        isTransformAssignment: Bool = true
    ) -> IncrementalTopicAssignment {
        IncrementalTopicAssignment(
            topicID: .outlier,
            confidence: 0,
            similarity: bestSimilarity,
            isTransformAssignment: isTransformAssignment,
            topicKeywords: []
        )
    }

    /// Creates an assignment during cold start (no model yet).
    ///
    /// - Returns: A cold start outlier assignment.
    public static func coldStart() -> IncrementalTopicAssignment {
        IncrementalTopicAssignment(
            topicID: .outlier,
            confidence: 0,
            similarity: 0,
            isTransformAssignment: true,
            topicKeywords: []
        )
    }
}

// MARK: - CustomStringConvertible

extension IncrementalTopicAssignment: CustomStringConvertible {

    public var description: String {
        if isOutlier {
            return "IncrementalTopicAssignment(outlier, similarity=\(String(format: "%.3f", similarity)))"
        }

        let keywordStr = topicKeywords.prefix(3).joined(separator: ", ")
        return "IncrementalTopicAssignment(topic=\(topicID.value), confidence=\(String(format: "%.2f", confidence)), keywords=[\(keywordStr)])"
    }
}

// MARK: - Codable

extension IncrementalTopicAssignment: Codable {

    private enum CodingKeys: String, CodingKey {
        case topicID
        case confidence
        case similarity
        case isTransformAssignment
        case topicKeywords
    }
}

// MARK: - Batch Assignment Result

/// Result of processing multiple documents incrementally.
///
/// Provides aggregate statistics alongside individual assignments.
public struct IncrementalBatchAssignmentResult: Sendable {

    /// Individual assignments for each document.
    public let assignments: [IncrementalTopicAssignment]

    /// Number of documents assigned to topics (non-outliers).
    public var assignedCount: Int {
        assignments.filter { !$0.isOutlier }.count
    }

    /// Number of outlier documents.
    public var outlierCount: Int {
        assignments.filter { $0.isOutlier }.count
    }

    /// Average confidence across non-outlier assignments.
    public var averageConfidence: Float {
        let nonOutliers = assignments.filter { !$0.isOutlier }
        guard !nonOutliers.isEmpty else { return 0 }
        return nonOutliers.reduce(0) { $0 + $1.confidence } / Float(nonOutliers.count)
    }

    /// Average similarity across all assignments.
    public var averageSimilarity: Float {
        guard !assignments.isEmpty else { return 0 }
        return assignments.reduce(0) { $0 + $1.similarity } / Float(assignments.count)
    }

    /// Topic distribution: topicID -> count.
    public var topicDistribution: [TopicID: Int] {
        var distribution = [TopicID: Int]()
        for assignment in assignments {
            distribution[assignment.topicID, default: 0] += 1
        }
        return distribution
    }

    /// Creates a batch assignment result.
    public init(assignments: [IncrementalTopicAssignment]) {
        self.assignments = assignments
    }
}
