// IncrementalTopicModelState.swift
// SwiftTopics
//
// Extended state for incremental topic model updates

import Foundation

// MARK: - Incremental Topic Model State

/// Extended state for incremental topic modeling.
///
/// This extends the base `TopicModelState` with additional information
/// needed for incremental updates:
///
/// - **Vocabulary**: Term frequencies for c-TF-IDF updates
/// - **Drift Statistics**: For detecting when full refresh is needed
/// - **Training History**: When and how the model was last updated
///
/// ## State Lifecycle
///
/// 1. **Cold Start**: No state exists; buffer documents until threshold
/// 2. **Initial Training**: First model created from buffered documents
/// 3. **Transform-Only**: New documents assigned via centroids
/// 4. **Micro-Retrain**: Buffer incorporated via model merging
/// 5. **Full Refresh**: Complete retraining from all embeddings
///
/// ## Thread Safety
///
/// `IncrementalTopicModelState` is `Sendable` and `Codable`.
public struct IncrementalTopicModelState: Sendable, Codable {

    // MARK: - Version

    /// Current state format version.
    public static let currentVersion = 1

    /// Version number for compatibility checking.
    public let version: Int

    // MARK: - Base Model State

    /// Configuration used to train the model.
    public let configuration: TopicModelConfiguration

    /// Discovered topics with keywords.
    public let topics: [Topic]

    /// Document-to-topic assignments.
    ///
    /// Note: For incremental models, this may not include all documents.
    /// Use storage to get authoritative document assignments.
    public let assignments: ClusterAssignment

    /// Topic centroids in original embedding space.
    ///
    /// Used for transform-only assignment of new documents.
    public let centroids: [Embedding]

    /// Input embedding dimension.
    public let inputDimension: Int

    /// Reduced embedding dimension (after UMAP/PCA).
    public let reducedDimension: Int

    // MARK: - Incremental State

    /// Vocabulary with term frequencies per topic.
    public let vocabulary: IncrementalVocabulary

    /// Total document count (including all processed documents).
    public let totalDocumentCount: Int

    /// When the model was last updated (any update type).
    public let lastUpdatedAt: Date

    /// When the model was last fully retrained.
    public let lastFullRetrainAt: Date?

    /// Document count at last full retrain.
    ///
    /// Used with `totalDocumentCount` to determine if full refresh is needed.
    public let documentCountAtLastRetrain: Int

    /// Running statistics for drift detection.
    public let driftStatistics: DriftStatistics

    // MARK: - Optional Large Data Flags

    /// Whether k-NN graph is stored (separate file due to size).
    public let hasKNNGraph: Bool

    /// Whether reduced embeddings are stored (separate file due to size).
    public let hasReducedEmbeddings: Bool

    // MARK: - Initialization

    /// Creates an incremental topic model state.
    public init(
        version: Int = IncrementalTopicModelState.currentVersion,
        configuration: TopicModelConfiguration,
        topics: [Topic],
        assignments: ClusterAssignment,
        centroids: [Embedding],
        inputDimension: Int,
        reducedDimension: Int,
        vocabulary: IncrementalVocabulary,
        totalDocumentCount: Int,
        lastUpdatedAt: Date = Date(),
        lastFullRetrainAt: Date? = nil,
        documentCountAtLastRetrain: Int = 0,
        driftStatistics: DriftStatistics = .initial,
        hasKNNGraph: Bool = false,
        hasReducedEmbeddings: Bool = false
    ) {
        self.version = version
        self.configuration = configuration
        self.topics = topics
        self.assignments = assignments
        self.centroids = centroids
        self.inputDimension = inputDimension
        self.reducedDimension = reducedDimension
        self.vocabulary = vocabulary
        self.totalDocumentCount = totalDocumentCount
        self.lastUpdatedAt = lastUpdatedAt
        self.lastFullRetrainAt = lastFullRetrainAt
        self.documentCountAtLastRetrain = documentCountAtLastRetrain
        self.driftStatistics = driftStatistics
        self.hasKNNGraph = hasKNNGraph
        self.hasReducedEmbeddings = hasReducedEmbeddings
    }

    // MARK: - Factory Methods

    /// Creates initial state from a first training run.
    public static func initial(
        configuration: TopicModelConfiguration,
        topics: [Topic],
        assignments: ClusterAssignment,
        centroids: [Embedding],
        vocabulary: IncrementalVocabulary,
        inputDimension: Int,
        reducedDimension: Int,
        documentCount: Int
    ) -> IncrementalTopicModelState {
        let now = Date()
        return IncrementalTopicModelState(
            configuration: configuration,
            topics: topics,
            assignments: assignments,
            centroids: centroids,
            inputDimension: inputDimension,
            reducedDimension: reducedDimension,
            vocabulary: vocabulary,
            totalDocumentCount: documentCount,
            lastUpdatedAt: now,
            lastFullRetrainAt: now,
            documentCountAtLastRetrain: documentCount,
            driftStatistics: .initial,
            hasKNNGraph: false,
            hasReducedEmbeddings: false
        )
    }

    // MARK: - Update Methods

    /// Creates a new state after micro-retrain.
    public func afterMicroRetrain(
        topics: [Topic],
        assignments: ClusterAssignment,
        centroids: [Embedding],
        vocabulary: IncrementalVocabulary,
        newDocumentCount: Int,
        driftStatistics: DriftStatistics
    ) -> IncrementalTopicModelState {
        IncrementalTopicModelState(
            version: version,
            configuration: configuration,
            topics: topics,
            assignments: assignments,
            centroids: centroids,
            inputDimension: inputDimension,
            reducedDimension: reducedDimension,
            vocabulary: vocabulary,
            totalDocumentCount: totalDocumentCount + newDocumentCount,
            lastUpdatedAt: Date(),
            lastFullRetrainAt: lastFullRetrainAt,
            documentCountAtLastRetrain: documentCountAtLastRetrain,
            driftStatistics: driftStatistics,
            hasKNNGraph: hasKNNGraph,
            hasReducedEmbeddings: hasReducedEmbeddings
        )
    }

    /// Creates a new state after full refresh.
    public func afterFullRefresh(
        topics: [Topic],
        assignments: ClusterAssignment,
        centroids: [Embedding],
        vocabulary: IncrementalVocabulary,
        documentCount: Int,
        hasKNNGraph: Bool = false,
        hasReducedEmbeddings: Bool = false
    ) -> IncrementalTopicModelState {
        let now = Date()
        return IncrementalTopicModelState(
            version: version,
            configuration: configuration,
            topics: topics,
            assignments: assignments,
            centroids: centroids,
            inputDimension: inputDimension,
            reducedDimension: reducedDimension,
            vocabulary: vocabulary,
            totalDocumentCount: documentCount,
            lastUpdatedAt: now,
            lastFullRetrainAt: now,
            documentCountAtLastRetrain: documentCount,
            driftStatistics: .initial,
            hasKNNGraph: hasKNNGraph,
            hasReducedEmbeddings: hasReducedEmbeddings
        )
    }

    /// Creates a new state with updated drift statistics.
    public func withDriftStatistics(_ stats: DriftStatistics) -> IncrementalTopicModelState {
        IncrementalTopicModelState(
            version: version,
            configuration: configuration,
            topics: topics,
            assignments: assignments,
            centroids: centroids,
            inputDimension: inputDimension,
            reducedDimension: reducedDimension,
            vocabulary: vocabulary,
            totalDocumentCount: totalDocumentCount,
            lastUpdatedAt: lastUpdatedAt,
            lastFullRetrainAt: lastFullRetrainAt,
            documentCountAtLastRetrain: documentCountAtLastRetrain,
            driftStatistics: stats,
            hasKNNGraph: hasKNNGraph,
            hasReducedEmbeddings: hasReducedEmbeddings
        )
    }

    // MARK: - Queries

    /// Growth ratio since last full retrain.
    public var growthRatio: Float {
        guard documentCountAtLastRetrain > 0 else { return 0 }
        return Float(totalDocumentCount) / Float(documentCountAtLastRetrain)
    }

    /// Time since last full retrain.
    public var timeSinceLastRetrain: TimeInterval? {
        guard let lastRetrain = lastFullRetrainAt else { return nil }
        return Date().timeIntervalSince(lastRetrain)
    }

    /// Human-readable summary of the state.
    public var summary: String {
        var lines = [String]()
        lines.append("IncrementalTopicModelState v\(version)")
        lines.append("  Topics: \(topics.count)")
        lines.append("  Documents: \(totalDocumentCount)")
        lines.append("  Dimensions: \(inputDimension) → \(reducedDimension)")
        lines.append("  Vocabulary: \(vocabulary.termCount) terms")
        lines.append("  Growth: \(String(format: "%.1f", growthRatio))x since last retrain")

        if let elapsed = timeSinceLastRetrain {
            let days = elapsed / (24 * 60 * 60)
            lines.append("  Last retrain: \(String(format: "%.1f", days)) days ago")
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - Drift Statistics

/// Statistics for detecting model drift.
///
/// Drift occurs when new documents diverge from the training distribution,
/// causing transform-only assignments to become less accurate.
///
/// ## Detection Metrics
///
/// - **Centroid Distance**: Average distance from new documents to their assigned topic
/// - **Outlier Rate**: Percentage of new documents that can't be confidently assigned
///
/// ## Thresholds
///
/// When drift metrics exceed thresholds, a full refresh is recommended.
public struct DriftStatistics: Sendable, Codable {

    /// Average distance to assigned centroid (recent entries).
    ///
    /// Computed over the most recent `recentWindowSize` entries.
    /// Increasing trend indicates drift.
    public var recentAverageDistance: Float

    /// Average distance to assigned centroid (all time).
    ///
    /// Baseline for comparison with recent distance.
    public var overallAverageDistance: Float

    /// Percentage of recent entries marked as outliers.
    ///
    /// An outlier is a document that couldn't be confidently assigned
    /// to any existing topic.
    public var recentOutlierRate: Float

    /// Number of entries in the recent window.
    public var recentWindowSize: Int

    /// Number of entries included in overall statistics.
    public var totalEntriesTracked: Int

    // MARK: - Initialization

    /// Creates drift statistics.
    public init(
        recentAverageDistance: Float,
        overallAverageDistance: Float,
        recentOutlierRate: Float,
        recentWindowSize: Int,
        totalEntriesTracked: Int
    ) {
        self.recentAverageDistance = recentAverageDistance
        self.overallAverageDistance = overallAverageDistance
        self.recentOutlierRate = recentOutlierRate
        self.recentWindowSize = recentWindowSize
        self.totalEntriesTracked = totalEntriesTracked
    }

    /// Initial drift statistics (no data yet).
    public static let initial = DriftStatistics(
        recentAverageDistance: 0,
        overallAverageDistance: 0,
        recentOutlierRate: 0,
        recentWindowSize: 0,
        totalEntriesTracked: 0
    )

    // MARK: - Update Methods

    /// Updates statistics with a new observation.
    public mutating func observe(
        distance: Float,
        isOutlier: Bool,
        windowSize: Int = 100
    ) {
        totalEntriesTracked += 1

        // Update overall average
        let n = Float(totalEntriesTracked)
        overallAverageDistance = overallAverageDistance * ((n - 1) / n) + distance / n

        // Update recent window (exponential moving average for efficiency)
        let alpha: Float = 2.0 / (Float(windowSize) + 1.0)
        recentAverageDistance = alpha * distance + (1 - alpha) * recentAverageDistance

        // Update outlier rate in recent window
        let outlierValue: Float = isOutlier ? 1.0 : 0.0
        recentOutlierRate = alpha * outlierValue + (1 - alpha) * recentOutlierRate

        recentWindowSize = min(recentWindowSize + 1, windowSize)
    }

    // MARK: - Analysis

    /// Drift ratio: how much higher recent distance is vs overall.
    ///
    /// A ratio > 1.0 indicates drift. Higher values = more drift.
    public var driftRatio: Float {
        guard overallAverageDistance > 0 else { return 0 }
        return recentAverageDistance / overallAverageDistance
    }

    /// Whether drift metrics suggest a full refresh is needed.
    ///
    /// - Parameters:
    ///   - driftThreshold: Maximum acceptable drift ratio (default 1.5)
    ///   - outlierThreshold: Maximum acceptable outlier rate (default 0.2)
    public func needsRefresh(
        driftThreshold: Float = 1.5,
        outlierThreshold: Float = 0.2
    ) -> Bool {
        // Need enough data to make a judgment
        guard recentWindowSize >= 30 else { return false }

        return driftRatio > driftThreshold || recentOutlierRate > outlierThreshold
    }
}

// MARK: - Incremental Vocabulary

/// Vocabulary for incremental c-TF-IDF computation.
///
/// Stores term frequencies per topic and global IDF values.
/// Supports incremental updates when new documents are added.
///
/// Note: This differs from the base `Vocabulary` type in that it is
/// mutable and Codable, designed for persistent storage and updates.
public struct IncrementalVocabulary: Sendable, Codable {

    /// Term to index mapping.
    public private(set) var termToIndex: [String: Int]

    /// Index to term mapping.
    public private(set) var indexToTerm: [String]

    /// Term frequencies per topic: [topicIndex][termIndex] -> frequency
    public private(set) var topicTermFrequencies: [[Int]]

    /// Document frequency per term (number of documents containing term).
    public private(set) var documentFrequencies: [Int]

    /// Total number of documents used to compute IDF.
    public private(set) var totalDocuments: Int

    /// Number of terms in vocabulary.
    public var termCount: Int { indexToTerm.count }

    /// Number of topics.
    public var topicCount: Int { topicTermFrequencies.count }

    // MARK: - Initialization

    /// Creates an empty vocabulary.
    public init() {
        self.termToIndex = [:]
        self.indexToTerm = []
        self.topicTermFrequencies = []
        self.documentFrequencies = []
        self.totalDocuments = 0
    }

    /// Creates a vocabulary from term data.
    public init(
        termToIndex: [String: Int],
        indexToTerm: [String],
        topicTermFrequencies: [[Int]],
        documentFrequencies: [Int],
        totalDocuments: Int
    ) {
        self.termToIndex = termToIndex
        self.indexToTerm = indexToTerm
        self.topicTermFrequencies = topicTermFrequencies
        self.documentFrequencies = documentFrequencies
        self.totalDocuments = totalDocuments
    }

    // MARK: - Access

    /// Gets the index for a term, adding it if necessary.
    public mutating func getOrAddTerm(_ term: String) -> Int {
        if let index = termToIndex[term] {
            return index
        }

        let index = indexToTerm.count
        termToIndex[term] = index
        indexToTerm.append(term)
        documentFrequencies.append(0)

        // Expand topic-term matrices
        for i in 0..<topicTermFrequencies.count {
            topicTermFrequencies[i].append(0)
        }

        return index
    }

    /// Gets IDF value for a term.
    ///
    /// IDF = log(N / (df + 1)) where N = total docs, df = doc frequency
    public func idf(for termIndex: Int) -> Float {
        guard termIndex < documentFrequencies.count else { return 0 }
        let df = Float(documentFrequencies[termIndex])
        let n = Float(max(totalDocuments, 1))
        return log(n / (df + 1))
    }

    /// Computes c-TF-IDF score for a term in a topic.
    ///
    /// c-TF-IDF = (tf / topic_size) * idf
    public func ctfidf(termIndex: Int, topicIndex: Int, topicSize: Int) -> Float {
        guard topicIndex < topicTermFrequencies.count,
              termIndex < topicTermFrequencies[topicIndex].count else {
            return 0
        }

        let tf = Float(topicTermFrequencies[topicIndex][termIndex])
        let size = Float(max(topicSize, 1))
        return (tf / size) * idf(for: termIndex)
    }

    // MARK: - Updates

    /// Adds a topic to the vocabulary.
    public mutating func addTopic() {
        topicTermFrequencies.append([Int](repeating: 0, count: termCount))
    }

    /// Increments term frequency in a topic.
    public mutating func incrementFrequency(term: String, topic: Int) {
        let termIndex = getOrAddTerm(term)

        // Expand topics if needed
        while topicTermFrequencies.count <= topic {
            addTopic()
        }

        topicTermFrequencies[topic][termIndex] += 1
    }

    /// Records a document with its terms.
    public mutating func addDocument(terms: [String], topic: Int) {
        totalDocuments += 1

        // Track unique terms for document frequency
        var seenTerms = Set<String>()

        for term in terms {
            incrementFrequency(term: term, topic: topic)

            if !seenTerms.contains(term) {
                seenTerms.insert(term)
                let termIndex = getOrAddTerm(term)
                documentFrequencies[termIndex] += 1
            }
        }
    }
}
