// Topic.swift
// SwiftTopics
//
// Topic representation with keywords and quality metrics

import Foundation

// MARK: - Topic

/// A discovered topic in the corpus.
///
/// Topics emerge from clustering similar document embeddings and are characterized
/// by representative keywords extracted via c-TF-IDF scoring.
///
/// ## Identification
/// Each topic has a unique `TopicID` within the model. The special ID `-1` is
/// reserved for outliers (documents that don't belong to any coherent topic).
///
/// ## Keywords
/// Keywords are ranked by their c-TF-IDF scores, which measure how distinctive
/// a term is for this topic compared to the overall corpus.
///
/// ## Quality Metrics
/// - `coherenceScore`: NPMI-based measure of keyword co-occurrence (higher is better)
/// - `size`: Number of documents assigned to this topic
///
/// ## Thread Safety
/// `Topic` is `Sendable` and can be safely shared across concurrency domains.
public struct Topic: Sendable, Codable, Hashable, Identifiable {

    /// Unique identifier for this topic.
    public let id: TopicID

    /// Keywords that characterize this topic, ranked by relevance.
    public let keywords: [TopicKeyword]

    /// Number of documents assigned to this topic.
    public let size: Int

    /// NPMI coherence score for this topic.
    ///
    /// Ranges from -1 (incoherent) to +1 (perfectly coherent).
    /// A score above 0 indicates meaningful word associations.
    public let coherenceScore: Float?

    /// IDs of documents that are most representative of this topic.
    ///
    /// These are the documents closest to the topic centroid.
    public let representativeDocuments: [DocumentID]

    /// The centroid embedding of this topic.
    ///
    /// Computed as the mean of all document embeddings in this topic.
    public let centroid: Embedding?

    /// Creates a new topic.
    ///
    /// - Parameters:
    ///   - id: Unique topic identifier.
    ///   - keywords: Ranked keywords for this topic.
    ///   - size: Number of documents in this topic.
    ///   - coherenceScore: Optional NPMI coherence score.
    ///   - representativeDocuments: IDs of representative documents.
    ///   - centroid: Optional centroid embedding.
    public init(
        id: TopicID,
        keywords: [TopicKeyword],
        size: Int,
        coherenceScore: Float? = nil,
        representativeDocuments: [DocumentID] = [],
        centroid: Embedding? = nil
    ) {
        self.id = id
        self.keywords = keywords
        self.size = size
        self.coherenceScore = coherenceScore
        self.representativeDocuments = representativeDocuments
        self.centroid = centroid
    }

    /// Whether this represents the outlier "topic" (documents not in any cluster).
    public var isOutlierTopic: Bool {
        id.isOutlier
    }

    /// Returns the top N keywords as a comma-separated string.
    ///
    /// - Parameter count: Maximum number of keywords to include.
    /// - Returns: A human-readable keyword summary.
    public func keywordSummary(count: Int = 5) -> String {
        keywords.prefix(count).map(\.term).joined(separator: ", ")
    }

    /// Gets the keyword at the specified rank (0 = top keyword).
    ///
    /// - Parameter rank: The keyword rank (0-indexed).
    /// - Returns: The keyword at that rank, or nil if out of bounds.
    public func keyword(at rank: Int) -> TopicKeyword? {
        guard rank >= 0 && rank < keywords.count else { return nil }
        return keywords[rank]
    }
}

// MARK: - Topic ID

/// A unique identifier for a topic within a model.
///
/// Topic IDs are integers where:
/// - `-1` represents outliers (no topic assignment)
/// - `0` and above represent actual topics
public struct TopicID: Sendable, Codable, Hashable, CustomStringConvertible {

    /// The integer value of this topic ID.
    public let value: Int

    /// The outlier topic ID (-1).
    public static let outlier = TopicID(value: -1)

    /// Creates a topic ID from an integer.
    ///
    /// - Parameter value: The topic ID value (-1 or greater).
    public init(value: Int) {
        precondition(value >= -1, "Topic ID must be -1 or greater")
        self.value = value
    }

    /// Creates a topic ID from an integer (convenience).
    ///
    /// - Parameter intValue: The topic ID value.
    public init(_ intValue: Int) {
        self.init(value: intValue)
    }

    /// Whether this represents the outlier cluster.
    public var isOutlier: Bool {
        value == -1
    }

    public var description: String {
        if isOutlier {
            return "outlier"
        }
        return "topic_\(value)"
    }
}

extension TopicID: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(value: value)
    }
}

extension TopicID: Comparable {
    public static func < (lhs: TopicID, rhs: TopicID) -> Bool {
        lhs.value < rhs.value
    }
}

// MARK: - Topic Keyword

/// A keyword that characterizes a topic.
///
/// Keywords are ranked by their c-TF-IDF scores, which measure how distinctive
/// the term is for the topic compared to the overall corpus.
public struct TopicKeyword: Sendable, Codable, Hashable {

    /// The keyword term.
    public let term: String

    /// The c-TF-IDF score indicating relevance to the topic.
    ///
    /// Higher scores indicate more distinctive/relevant terms.
    public let score: Float

    /// The raw term frequency within the topic's documents.
    public let frequency: Int?

    /// Creates a topic keyword.
    ///
    /// - Parameters:
    ///   - term: The keyword text.
    ///   - score: The c-TF-IDF score.
    ///   - frequency: Optional raw frequency count.
    public init(term: String, score: Float, frequency: Int? = nil) {
        self.term = term
        self.score = score
        self.frequency = frequency
    }
}

extension TopicKeyword: Comparable {
    public static func < (lhs: TopicKeyword, rhs: TopicKeyword) -> Bool {
        lhs.score < rhs.score
    }
}

// MARK: - Topic Statistics

/// Aggregate statistics about topics in a model.
public struct TopicStatistics: Sendable, Codable {

    /// Total number of topics (excluding outliers).
    public let topicCount: Int

    /// Number of documents assigned to topics.
    public let assignedDocumentCount: Int

    /// Number of outlier documents.
    public let outlierCount: Int

    /// Total number of documents.
    public var totalDocumentCount: Int {
        assignedDocumentCount + outlierCount
    }

    /// Outlier rate as a fraction (0-1).
    public var outlierRate: Float {
        guard totalDocumentCount > 0 else { return 0 }
        return Float(outlierCount) / Float(totalDocumentCount)
    }

    /// Average topic size.
    public var averageTopicSize: Float {
        guard topicCount > 0 else { return 0 }
        return Float(assignedDocumentCount) / Float(topicCount)
    }

    /// Minimum topic size.
    public let minTopicSize: Int

    /// Maximum topic size.
    public let maxTopicSize: Int

    /// Mean coherence score across topics.
    public let meanCoherence: Float?

    /// Creates topic statistics.
    public init(
        topicCount: Int,
        assignedDocumentCount: Int,
        outlierCount: Int,
        minTopicSize: Int,
        maxTopicSize: Int,
        meanCoherence: Float?
    ) {
        self.topicCount = topicCount
        self.assignedDocumentCount = assignedDocumentCount
        self.outlierCount = outlierCount
        self.minTopicSize = minTopicSize
        self.maxTopicSize = maxTopicSize
        self.meanCoherence = meanCoherence
    }
}

// MARK: - Topic Comparison

/// Result of comparing two topic models.
public struct TopicComparison: Sendable {

    /// Number of topics in the first model.
    public let topicCount1: Int

    /// Number of topics in the second model.
    public let topicCount2: Int

    /// Similarity matrix between topics (rows = model1, cols = model2).
    ///
    /// Values are cosine similarities between topic centroids, or
    /// Jaccard similarities between document assignments.
    public let similarityMatrix: [[Float]]

    /// Best match for each topic in model 1.
    public let bestMatches: [TopicID: TopicID]

    /// Average similarity of best matches.
    public var averageBestMatchSimilarity: Float {
        guard !similarityMatrix.isEmpty else { return 0 }
        var sum: Float = 0
        for row in similarityMatrix {
            if let maxSim = row.max() {
                sum += maxSim
            }
        }
        return sum / Float(similarityMatrix.count)
    }

    public init(
        topicCount1: Int,
        topicCount2: Int,
        similarityMatrix: [[Float]],
        bestMatches: [TopicID: TopicID]
    ) {
        self.topicCount1 = topicCount1
        self.topicCount2 = topicCount2
        self.similarityMatrix = similarityMatrix
        self.bestMatches = bestMatches
    }
}
