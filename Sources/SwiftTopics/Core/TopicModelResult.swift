// TopicModelResult.swift
// SwiftTopics
//
// Complete output from the topic modeling pipeline

import Foundation

// MARK: - Topic Model Result

/// The complete output of a topic modeling pipeline run.
///
/// Contains discovered topics, document-topic assignments, quality metrics,
/// and processing metadata.
///
/// ## Topics
/// The `topics` array contains all discovered topics, sorted by ID.
/// Each topic includes keywords, size, and optional coherence scores.
///
/// ## Assignments
/// Use `topicAssignment(for:)` or `documentTopics` to get the topic
/// assignment for a specific document.
///
/// ## Quality Metrics
/// - `coherenceScore`: Aggregate NPMI coherence (higher = better)
/// - `statistics`: Detailed statistics about topic distribution
///
/// ## Thread Safety
/// `TopicModelResult` is `Sendable` and can be safely shared across concurrency domains.
///
/// ## Serialization
/// The result is `Codable` for persistence. Note that the full embeddings
/// are not stored to reduce size; only topic centroids are included.
public struct TopicModelResult: Sendable, Codable {

    /// The discovered topics, sorted by ID.
    public let topics: [Topic]

    /// Mapping from document ID to topic assignment.
    public let documentTopics: [DocumentID: TopicAssignment]

    /// Aggregate NPMI coherence score for the model.
    ///
    /// Ranges from -1 to +1, where higher is better.
    /// A score above 0.1 typically indicates reasonable topics.
    public let coherenceScore: Float?

    /// Aggregate statistics about the topics.
    public let statistics: TopicStatistics

    /// Metadata about the pipeline run.
    public let metadata: TopicModelMetadata

    /// Creates a topic model result.
    ///
    /// - Parameters:
    ///   - topics: The discovered topics.
    ///   - documentTopics: Document to topic assignment map.
    ///   - coherenceScore: Optional aggregate coherence.
    ///   - statistics: Topic statistics.
    ///   - metadata: Pipeline metadata.
    public init(
        topics: [Topic],
        documentTopics: [DocumentID: TopicAssignment],
        coherenceScore: Float?,
        statistics: TopicStatistics,
        metadata: TopicModelMetadata
    ) {
        self.topics = topics.sorted { $0.id < $1.id }
        self.documentTopics = documentTopics
        self.coherenceScore = coherenceScore
        self.statistics = statistics
        self.metadata = metadata
    }

    // MARK: - Accessors

    /// The number of topics (excluding outlier topic).
    public var topicCount: Int {
        topics.filter { !$0.isOutlierTopic }.count
    }

    /// The number of documents processed.
    public var documentCount: Int {
        documentTopics.count
    }

    /// IDs of all outlier documents.
    public var outlierDocuments: [DocumentID] {
        documentTopics.compactMap { docID, assignment in
            assignment.topicID.isOutlier ? docID : nil
        }
    }

    /// Gets the topic assignment for a document.
    ///
    /// - Parameter documentID: The document ID.
    /// - Returns: The topic assignment, or nil if document not found.
    public func topicAssignment(for documentID: DocumentID) -> TopicAssignment? {
        documentTopics[documentID]
    }

    /// Gets the topic for a document.
    ///
    /// - Parameter documentID: The document ID.
    /// - Returns: The topic, or nil if document not found or is an outlier.
    public func topic(for documentID: DocumentID) -> Topic? {
        guard let assignment = documentTopics[documentID],
              !assignment.topicID.isOutlier else {
            return nil
        }
        return topics.first { $0.id == assignment.topicID }
    }

    /// Gets a topic by ID.
    ///
    /// - Parameter topicID: The topic ID.
    /// - Returns: The topic, or nil if not found.
    public func topic(id: TopicID) -> Topic? {
        topics.first { $0.id == id }
    }

    /// Documents assigned to a specific topic.
    ///
    /// - Parameter topicID: The topic ID.
    /// - Returns: Document IDs assigned to that topic.
    public func documents(for topicID: TopicID) -> [DocumentID] {
        documentTopics.compactMap { docID, assignment in
            assignment.topicID == topicID ? docID : nil
        }
    }

    /// Topics sorted by size (largest first).
    public var topicsBySizeDescending: [Topic] {
        topics.filter { !$0.isOutlierTopic }.sorted { $0.size > $1.size }
    }

    /// Topics sorted by coherence (most coherent first).
    public var topicsByCoherenceDescending: [Topic] {
        topics.filter { !$0.isOutlierTopic && $0.coherenceScore != nil }
            .sorted { ($0.coherenceScore ?? 0) > ($1.coherenceScore ?? 0) }
    }

    /// Gets the most representative documents for a topic.
    ///
    /// - Parameters:
    ///   - topicID: The topic ID.
    ///   - count: Maximum number of documents to return.
    /// - Returns: Document IDs ranked by representation quality.
    public func representativeDocuments(for topicID: TopicID, count: Int = 5) -> [DocumentID] {
        // First try to get from topic's stored representatives
        if let topic = topic(id: topicID) {
            return Array(topic.representativeDocuments.prefix(count))
        }
        // Fall back to documents sorted by probability
        return documentTopics
            .filter { $0.value.topicID == topicID }
            .sorted { $0.value.probability > $1.value.probability }
            .prefix(count)
            .map(\.key)
    }
}

// MARK: - Topic Assignment

/// A document's assignment to a topic.
///
/// Contains the topic ID, confidence probability, and optional additional
/// topic matches for soft clustering scenarios.
public struct TopicAssignment: Sendable, Codable, Hashable {

    /// The primary topic ID assigned to this document.
    public let topicID: TopicID

    /// Confidence probability for the assignment.
    ///
    /// Range: [0, 1] where 1 = highly confident.
    public let probability: Float

    /// Distance to the topic centroid.
    ///
    /// Lower distances indicate closer match to topic.
    public let distanceToCentroid: Float?

    /// Alternative topic assignments (for soft clustering).
    ///
    /// Contains other topics this document could belong to,
    /// sorted by probability descending.
    public let alternatives: [AlternativeAssignment]?

    /// Whether this document is assigned to the outlier cluster.
    public var isOutlier: Bool {
        topicID.isOutlier
    }

    /// Creates a topic assignment.
    ///
    /// - Parameters:
    ///   - topicID: The assigned topic ID.
    ///   - probability: Confidence probability.
    ///   - distanceToCentroid: Optional distance to topic centroid.
    ///   - alternatives: Optional alternative assignments.
    public init(
        topicID: TopicID,
        probability: Float,
        distanceToCentroid: Float? = nil,
        alternatives: [AlternativeAssignment]? = nil
    ) {
        self.topicID = topicID
        self.probability = probability
        self.distanceToCentroid = distanceToCentroid
        self.alternatives = alternatives
    }

    /// Creates an outlier assignment.
    public static var outlier: TopicAssignment {
        TopicAssignment(topicID: .outlier, probability: 0)
    }
}

// MARK: - Alternative Assignment

/// An alternative topic assignment for soft clustering.
public struct AlternativeAssignment: Sendable, Codable, Hashable {

    /// The alternative topic ID.
    public let topicID: TopicID

    /// Probability of this alternative assignment.
    public let probability: Float

    /// Creates an alternative assignment.
    public init(topicID: TopicID, probability: Float) {
        self.topicID = topicID
        self.probability = probability
    }
}

// MARK: - Topic Model Metadata

/// Metadata about a topic model training run.
public struct TopicModelMetadata: Sendable, Codable {

    /// Version of SwiftTopics used.
    public let libraryVersion: String

    /// When the model was trained.
    public let trainedAt: Date

    /// Duration of the training process.
    public let trainingDuration: TimeInterval

    /// Configuration used for training.
    public let configuration: TopicModelConfigurationSnapshot

    /// Number of documents used for training.
    public let documentCount: Int

    /// Embedding dimension used.
    public let embeddingDimension: Int

    /// Reduced dimension (after PCA/UMAP).
    public let reducedDimension: Int

    /// Random seed used for reproducibility.
    public let randomSeed: UInt64?

    /// Creates topic model metadata.
    public init(
        libraryVersion: String,
        trainedAt: Date = Date(),
        trainingDuration: TimeInterval,
        configuration: TopicModelConfigurationSnapshot,
        documentCount: Int,
        embeddingDimension: Int,
        reducedDimension: Int,
        randomSeed: UInt64?
    ) {
        self.libraryVersion = libraryVersion
        self.trainedAt = trainedAt
        self.trainingDuration = trainingDuration
        self.configuration = configuration
        self.documentCount = documentCount
        self.embeddingDimension = embeddingDimension
        self.reducedDimension = reducedDimension
        self.randomSeed = randomSeed
    }
}

// MARK: - Configuration Snapshot

/// A snapshot of the configuration used for training.
///
/// Stored with results to ensure reproducibility and debugging.
public struct TopicModelConfigurationSnapshot: Sendable, Codable {

    /// Dimension reduction method used.
    public let reductionMethod: String

    /// Target dimensions after reduction.
    public let reducedDimensions: Int

    /// Clustering algorithm used.
    public let clusteringAlgorithm: String

    /// Minimum cluster size parameter.
    public let minClusterSize: Int

    /// Minimum samples parameter.
    public let minSamples: Int

    /// Cluster selection method.
    public let clusterSelectionMethod: String

    /// Number of keywords per topic.
    public let keywordsPerTopic: Int

    /// Creates a configuration snapshot.
    public init(
        reductionMethod: String,
        reducedDimensions: Int,
        clusteringAlgorithm: String,
        minClusterSize: Int,
        minSamples: Int,
        clusterSelectionMethod: String,
        keywordsPerTopic: Int
    ) {
        self.reductionMethod = reductionMethod
        self.reducedDimensions = reducedDimensions
        self.clusteringAlgorithm = clusteringAlgorithm
        self.minClusterSize = minClusterSize
        self.minSamples = minSamples
        self.clusterSelectionMethod = clusterSelectionMethod
        self.keywordsPerTopic = keywordsPerTopic
    }
}

// MARK: - Result Builder

/// Builder for constructing `TopicModelResult` incrementally.
///
/// Used by the `TopicModel` orchestrator to assemble results from
/// the various pipeline stages.
public final class TopicModelResultBuilder: @unchecked Sendable {

    private var topics: [Topic] = []
    private var documentTopics: [DocumentID: TopicAssignment] = [:]
    private var coherenceScore: Float?
    private var metadata: TopicModelMetadata?

    public init() {}

    /// Sets the discovered topics.
    public func setTopics(_ topics: [Topic]) {
        self.topics = topics
    }

    /// Adds a document-topic assignment.
    public func addAssignment(documentID: DocumentID, assignment: TopicAssignment) {
        documentTopics[documentID] = assignment
    }

    /// Sets all document-topic assignments.
    public func setAssignments(_ assignments: [DocumentID: TopicAssignment]) {
        self.documentTopics = assignments
    }

    /// Sets the aggregate coherence score.
    public func setCoherenceScore(_ score: Float) {
        self.coherenceScore = score
    }

    /// Sets the metadata.
    public func setMetadata(_ metadata: TopicModelMetadata) {
        self.metadata = metadata
    }

    /// Builds the final result.
    ///
    /// - Returns: The constructed `TopicModelResult`.
    /// - Precondition: Metadata must be set.
    public func build() -> TopicModelResult {
        guard let metadata = metadata else {
            preconditionFailure("Metadata must be set before building result")
        }

        // Calculate statistics
        let nonOutlierTopics = topics.filter { !$0.isOutlierTopic }
        let topicCount = nonOutlierTopics.count
        let outlierCount = documentTopics.values.filter(\.isOutlier).count
        let assignedCount = documentTopics.count - outlierCount
        let sizes = nonOutlierTopics.map(\.size)
        let minSize = sizes.min() ?? 0
        let maxSize = sizes.max() ?? 0
        let coherences = nonOutlierTopics.compactMap(\.coherenceScore)
        let meanCoherence = coherences.isEmpty ? nil : coherences.reduce(0, +) / Float(coherences.count)

        let statistics = TopicStatistics(
            topicCount: topicCount,
            assignedDocumentCount: assignedCount,
            outlierCount: outlierCount,
            minTopicSize: minSize,
            maxTopicSize: maxSize,
            meanCoherence: meanCoherence
        )

        return TopicModelResult(
            topics: topics,
            documentTopics: documentTopics,
            coherenceScore: coherenceScore,
            statistics: statistics,
            metadata: metadata
        )
    }
}
