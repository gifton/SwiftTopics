// CoherenceEvaluator.swift
// SwiftTopics
//
// Topic coherence evaluation using NPMI

import Foundation

// MARK: - Coherence Configuration

/// Configuration for coherence evaluation.
public struct CoherenceConfiguration: Sendable, Codable, Hashable {

    /// Window size for sliding window co-occurrence counting.
    ///
    /// Ignored when `useDocumentCooccurrence` is true.
    public let windowSize: Int

    /// Whether to use document-level co-occurrence instead of sliding window.
    ///
    /// - `true`: Count pairs that appear in the same document (boolean)
    /// - `false`: Use sliding window counting
    public let useDocumentCooccurrence: Bool

    /// Number of top keywords to consider per topic.
    public let topKeywords: Int

    /// Smoothing epsilon for probability calculations.
    public let epsilon: Float

    /// Creates a coherence configuration.
    ///
    /// - Parameters:
    ///   - windowSize: Sliding window size (default: 10).
    ///   - useDocumentCooccurrence: Use document-level counting (default: false).
    ///   - topKeywords: Keywords per topic to evaluate (default: 10).
    ///   - epsilon: Smoothing epsilon (default: 1e-12).
    public init(
        windowSize: Int = 10,
        useDocumentCooccurrence: Bool = false,
        topKeywords: Int = 10,
        epsilon: Float = 1e-12
    ) {
        precondition(windowSize >= 2, "windowSize must be at least 2")
        precondition(topKeywords >= 2, "topKeywords must be at least 2")
        precondition(epsilon > 0, "epsilon must be positive")

        self.windowSize = windowSize
        self.useDocumentCooccurrence = useDocumentCooccurrence
        self.topKeywords = topKeywords
        self.epsilon = epsilon
    }

    /// Default configuration: sliding window of 10, top 10 keywords.
    public static let `default` = CoherenceConfiguration()

    /// Configuration for document-level co-occurrence.
    public static let document = CoherenceConfiguration(
        useDocumentCooccurrence: true
    )

    /// Configuration with a larger window (better for capturing semantic relationships).
    public static let semantic = CoherenceConfiguration(
        windowSize: 50,
        topKeywords: 10
    )

    /// Configuration for fewer keywords (faster, less noise).
    public static let concise = CoherenceConfiguration(
        windowSize: 10,
        topKeywords: 5
    )

    /// The co-occurrence mode derived from configuration.
    internal var cooccurrenceMode: CooccurrenceMode {
        if useDocumentCooccurrence {
            return .document
        } else {
            return .slidingWindow(size: windowSize)
        }
    }
}

// MARK: - Coherence Result

/// Result of coherence evaluation.
public struct CoherenceResult: Sendable {

    /// Per-topic coherence scores (NPMI mean).
    ///
    /// One score per topic, in the same order as the input topics.
    public let topicScores: [Float]

    /// Detailed NPMI results per topic (optional).
    ///
    /// Contains per-word-pair scores for debugging/analysis.
    public let detailedResults: [TopicNPMIResult]?

    /// Mean coherence across all topics.
    public let meanCoherence: Float

    /// Median coherence across all topics.
    public let medianCoherence: Float

    /// Minimum coherence (worst topic).
    public let minCoherence: Float

    /// Maximum coherence (best topic).
    public let maxCoherence: Float

    /// Standard deviation of coherence scores.
    public let stdCoherence: Float

    /// Number of topics evaluated.
    public let topicCount: Int

    /// Number of topics with positive coherence (> 0).
    public let positiveCoherenceCount: Int

    /// Creates a coherence result.
    public init(
        topicScores: [Float],
        detailedResults: [TopicNPMIResult]?,
        meanCoherence: Float,
        medianCoherence: Float,
        minCoherence: Float,
        maxCoherence: Float,
        stdCoherence: Float,
        topicCount: Int,
        positiveCoherenceCount: Int
    ) {
        self.topicScores = topicScores
        self.detailedResults = detailedResults
        self.meanCoherence = meanCoherence
        self.medianCoherence = medianCoherence
        self.minCoherence = minCoherence
        self.maxCoherence = maxCoherence
        self.stdCoherence = stdCoherence
        self.topicCount = topicCount
        self.positiveCoherenceCount = positiveCoherenceCount
    }

    /// Fraction of topics with positive coherence.
    public var positiveCoherenceRatio: Float {
        guard topicCount > 0 else { return 0 }
        return Float(positiveCoherenceCount) / Float(topicCount)
    }

    /// Whether all topics have positive coherence.
    public var allPositive: Bool {
        positiveCoherenceCount == topicCount
    }

    /// Gets topics sorted by coherence (descending).
    ///
    /// - Returns: Array of (topicIndex, coherenceScore) pairs.
    public func topicsByCoherence() -> [(index: Int, score: Float)] {
        topicScores.enumerated()
            .map { (index: $0.offset, score: $0.element) }
            .sorted { $0.score > $1.score }
    }

    /// Gets the indices of low-coherence topics.
    ///
    /// - Parameter threshold: Maximum coherence to be considered "low" (default: 0).
    /// - Returns: Indices of topics with coherence below threshold.
    public func lowCoherenceTopics(threshold: Float = 0) -> [Int] {
        topicScores.enumerated()
            .filter { $0.element < threshold }
            .map { $0.offset }
    }
}

// MARK: - Coherence Evaluator Protocol

/// Protocol for topic coherence evaluation.
public protocol CoherenceEvaluatorProtocol: Sendable {

    /// Configuration for evaluation.
    var configuration: CoherenceConfiguration { get }

    /// Evaluates coherence of topics against a corpus.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - documents: Corpus documents for co-occurrence counting.
    /// - Returns: Coherence evaluation result.
    func evaluate(
        topics: [Topic],
        documents: [Document]
    ) async -> CoherenceResult

    /// Evaluates coherence using pre-computed co-occurrence counts.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - counts: Pre-computed co-occurrence counts.
    /// - Returns: Coherence evaluation result.
    func evaluate(
        topics: [Topic],
        counts: CooccurrenceCounts
    ) -> CoherenceResult
}

// MARK: - NPMI Coherence Evaluator

/// Evaluates topic coherence using Normalized Pointwise Mutual Information.
///
/// ## Algorithm
///
/// 1. **Build co-occurrence counts**: Count word pairs that appear within
///    a sliding window (or same document) across the corpus.
///
/// 2. **For each topic**: Take top-K keywords, compute NPMI for all pairs,
///    and average to get the topic's coherence score.
///
/// 3. **Aggregate**: Compute mean, median, min, max coherence across topics.
///
/// ## Interpretation
///
/// - **High coherence (> 0.3)**: Keywords frequently co-occur; topic is meaningful
/// - **Medium coherence (0-0.3)**: Some association; topic may be useful
/// - **Low coherence (< 0)**: Keywords rarely co-occur; topic may be noisy
///
/// ## Usage
///
/// ```swift
/// let evaluator = NPMICoherenceEvaluator()
/// let result = await evaluator.evaluate(topics: topics, documents: documents)
/// print("Mean coherence: \(result.meanCoherence)")
/// ```
///
/// ## Thread Safety
///
/// `NPMICoherenceEvaluator` is `Sendable` and safe to use from any thread.
public struct NPMICoherenceEvaluator: CoherenceEvaluatorProtocol, Sendable {

    // MARK: - Properties

    /// Configuration for evaluation.
    public let configuration: CoherenceConfiguration

    /// Tokenizer for document processing.
    public let tokenizer: Tokenizer

    /// Whether to include detailed per-pair scores in results.
    public let includeDetailedResults: Bool

    // MARK: - Initialization

    /// Creates an NPMI coherence evaluator.
    ///
    /// - Parameters:
    ///   - configuration: Evaluation configuration.
    ///   - tokenizer: Tokenizer for documents (default: English tokenizer).
    ///   - includeDetailedResults: Include per-pair scores (default: false).
    public init(
        configuration: CoherenceConfiguration = .default,
        tokenizer: Tokenizer = Tokenizer(),
        includeDetailedResults: Bool = false
    ) {
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.includeDetailedResults = includeDetailedResults
    }

    // MARK: - Evaluation

    /// Evaluates coherence of topics against a corpus.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - documents: Corpus documents for co-occurrence counting.
    /// - Returns: Coherence evaluation result.
    public func evaluate(
        topics: [Topic],
        documents: [Document]
    ) async -> CoherenceResult {
        // Handle empty inputs
        guard !topics.isEmpty else {
            return emptyResult()
        }

        guard !documents.isEmpty else {
            return emptyResult(topicCount: topics.count)
        }

        // Step 1: Tokenize documents
        let tokenizedDocs = tokenizer.tokenize(documents: documents)

        // Step 2: Build co-occurrence counts
        let counter = CooccurrenceCounter(mode: configuration.cooccurrenceMode)
        let counts = counter.count(tokenizedDocuments: tokenizedDocs)

        // Step 3: Evaluate using counts
        return evaluate(topics: topics, counts: counts)
    }

    /// Evaluates coherence using pre-computed co-occurrence counts.
    ///
    /// Useful when evaluating multiple topic models against the same corpus.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - counts: Pre-computed co-occurrence counts.
    /// - Returns: Coherence evaluation result.
    public func evaluate(
        topics: [Topic],
        counts: CooccurrenceCounts
    ) -> CoherenceResult {
        guard !topics.isEmpty else {
            return emptyResult()
        }

        guard !counts.isEmpty else {
            return emptyResult(topicCount: topics.count)
        }

        // Score each topic
        let scorer = NPMIScorer(configuration: NPMIConfiguration(epsilon: configuration.epsilon))
        let topicResults = scorer.score(
            topics: topics,
            counts: counts,
            topKeywords: configuration.topKeywords
        )

        let scores = topicResults.map(\.meanNPMI)

        // Compute aggregate statistics
        let stats = computeStatistics(scores)

        return CoherenceResult(
            topicScores: scores,
            detailedResults: includeDetailedResults ? topicResults : nil,
            meanCoherence: stats.mean,
            medianCoherence: stats.median,
            minCoherence: stats.min,
            maxCoherence: stats.max,
            stdCoherence: stats.std,
            topicCount: topics.count,
            positiveCoherenceCount: scores.filter { $0 > 0 }.count
        )
    }

    // MARK: - Statistics

    private struct Statistics {
        let mean: Float
        let median: Float
        let min: Float
        let max: Float
        let std: Float
    }

    private func computeStatistics(_ scores: [Float]) -> Statistics {
        guard !scores.isEmpty else {
            return Statistics(mean: 0, median: 0, min: 0, max: 0, std: 0)
        }

        // Mean
        let sum = scores.reduce(0, +)
        let mean = sum / Float(scores.count)

        // Median
        let sorted = scores.sorted()
        let median: Float
        if sorted.count % 2 == 0 {
            let mid = sorted.count / 2
            median = (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            median = sorted[sorted.count / 2]
        }

        // Min/Max
        let min = sorted.first ?? 0
        let max = sorted.last ?? 0

        // Standard deviation
        let variance = scores.reduce(0.0) { acc, score in
            let diff = score - mean
            return acc + diff * diff
        } / Float(scores.count)
        let std = sqrt(variance)

        return Statistics(mean: mean, median: median, min: min, max: max, std: std)
    }

    private func emptyResult(topicCount: Int = 0) -> CoherenceResult {
        CoherenceResult(
            topicScores: Array(repeating: 0, count: topicCount),
            detailedResults: nil,
            meanCoherence: 0,
            medianCoherence: 0,
            minCoherence: 0,
            maxCoherence: 0,
            stdCoherence: 0,
            topicCount: topicCount,
            positiveCoherenceCount: 0
        )
    }
}

// MARK: - Convenience Extensions

extension Array where Element == Topic {

    /// Evaluates coherence of topics against a corpus.
    ///
    /// - Parameters:
    ///   - documents: Corpus documents.
    ///   - configuration: Evaluation configuration.
    /// - Returns: Coherence evaluation result.
    public func evaluateCoherence(
        documents: [Document],
        configuration: CoherenceConfiguration = .default
    ) async -> CoherenceResult {
        let evaluator = NPMICoherenceEvaluator(configuration: configuration)
        return await evaluator.evaluate(topics: self, documents: documents)
    }

    /// Evaluates coherence using pre-computed co-occurrence counts.
    ///
    /// - Parameters:
    ///   - counts: Pre-computed co-occurrence counts.
    ///   - configuration: Evaluation configuration.
    /// - Returns: Coherence evaluation result.
    public func evaluateCoherence(
        counts: CooccurrenceCounts,
        configuration: CoherenceConfiguration = .default
    ) -> CoherenceResult {
        let evaluator = NPMICoherenceEvaluator(configuration: configuration)
        return evaluator.evaluate(topics: self, counts: counts)
    }
}

// MARK: - Co-occurrence Pre-computation

extension Array where Element == Document {

    /// Pre-computes co-occurrence counts for coherence evaluation.
    ///
    /// Useful when evaluating multiple topic models against the same corpus.
    ///
    /// - Parameters:
    ///   - configuration: Coherence configuration (determines counting mode).
    ///   - tokenizer: Tokenizer to use.
    /// - Returns: Co-occurrence counts.
    public func precomputeCooccurrences(
        configuration: CoherenceConfiguration = .default,
        tokenizer: Tokenizer = Tokenizer()
    ) -> CooccurrenceCounts {
        let tokenizedDocs = tokenizer.tokenize(documents: self)
        let counter = CooccurrenceCounter(mode: configuration.cooccurrenceMode)
        return counter.count(tokenizedDocuments: tokenizedDocs)
    }
}
