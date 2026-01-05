// DiversityMetrics.swift
// SwiftTopics
//
// Topic diversity and redundancy metrics

import Foundation

// MARK: - Diversity Result

/// Result of topic diversity evaluation.
public struct DiversityResult: Sendable {

    /// Percentage of unique keywords across all topics (0-1).
    ///
    /// - `1.0`: All keywords are unique (maximum diversity)
    /// - `0.0`: All topics share the same keywords (no diversity)
    ///
    /// Formula: unique_keywords / total_keywords
    public let diversity: Float

    /// Number of unique keywords across all topics.
    public let uniqueKeywordCount: Int

    /// Total number of keywords (sum across all topics).
    public let totalKeywordCount: Int

    /// Per-topic redundancy with other topics.
    ///
    /// For each topic, the fraction of its keywords that appear in other topics.
    public let topicRedundancy: [Float]

    /// Mean redundancy across topics.
    public let meanRedundancy: Float

    /// Pairwise overlap matrix between topics.
    ///
    /// `overlapMatrix[i][j]` = fraction of topic i's keywords that appear in topic j.
    public let overlapMatrix: [[Float]]?

    /// Creates a diversity result.
    public init(
        diversity: Float,
        uniqueKeywordCount: Int,
        totalKeywordCount: Int,
        topicRedundancy: [Float],
        meanRedundancy: Float,
        overlapMatrix: [[Float]]?
    ) {
        self.diversity = diversity
        self.uniqueKeywordCount = uniqueKeywordCount
        self.totalKeywordCount = totalKeywordCount
        self.topicRedundancy = topicRedundancy
        self.meanRedundancy = meanRedundancy
        self.overlapMatrix = overlapMatrix
    }

    /// Whether topics are highly diverse (diversity > 0.9).
    public var isHighlyDiverse: Bool {
        diversity > 0.9
    }

    /// Whether topics have low redundancy (mean < 0.1).
    public var isLowRedundancy: Bool {
        meanRedundancy < 0.1
    }

    /// Gets the most redundant topics.
    ///
    /// - Parameter threshold: Minimum redundancy to include.
    /// - Returns: Indices of topics with redundancy above threshold.
    public func redundantTopics(threshold: Float = 0.5) -> [Int] {
        topicRedundancy.enumerated()
            .filter { $0.element >= threshold }
            .map { $0.offset }
    }

    /// Gets pairs of highly overlapping topics.
    ///
    /// - Parameter threshold: Minimum overlap to include.
    /// - Returns: Pairs of topic indices with overlap above threshold.
    public func overlappingPairs(threshold: Float = 0.5) -> [(topic1: Int, topic2: Int, overlap: Float)] {
        guard let matrix = overlapMatrix else { return [] }

        var pairs: [(topic1: Int, topic2: Int, overlap: Float)] = []

        for i in 0..<matrix.count {
            for j in (i + 1)..<matrix.count {
                // Use max of both directions for symmetric comparison
                let overlap = max(matrix[i][j], matrix[j][i])
                if overlap >= threshold {
                    pairs.append((topic1: i, topic2: j, overlap: overlap))
                }
            }
        }

        return pairs.sorted { $0.overlap > $1.overlap }
    }
}

// MARK: - Diversity Metrics

/// Computes diversity and redundancy metrics for topics.
///
/// ## Metrics
///
/// ### Diversity
/// Measures what fraction of keywords are unique across all topics:
/// ```
/// diversity = unique_keywords / total_keywords
/// ```
///
/// High diversity (> 0.9) indicates topics have distinct vocabularies.
/// Low diversity (< 0.5) indicates significant keyword overlap.
///
/// ### Redundancy
/// For each topic, measures what fraction of its keywords appear in other topics:
/// ```
/// redundancy(t) = |keywords(t) ∩ keywords(other_topics)| / |keywords(t)|
/// ```
///
/// Low redundancy (< 0.1) indicates topics are well-separated.
/// High redundancy (> 0.5) indicates topics may need to be merged.
///
/// ## Usage
///
/// ```swift
/// let metrics = DiversityMetrics()
/// let result = metrics.evaluate(topics: topics, topKeywords: 10)
/// print("Diversity: \(result.diversity)")
/// ```
///
/// ## Thread Safety
///
/// `DiversityMetrics` is `Sendable` and safe to use from any thread.
public struct DiversityMetrics: Sendable {

    /// Whether to compute the full overlap matrix.
    public let computeOverlapMatrix: Bool

    /// Creates diversity metrics calculator.
    ///
    /// - Parameter computeOverlapMatrix: Whether to compute pairwise overlaps.
    public init(computeOverlapMatrix: Bool = false) {
        self.computeOverlapMatrix = computeOverlapMatrix
    }

    // MARK: - Evaluation

    /// Evaluates diversity metrics for topics.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - topKeywords: Number of top keywords to use per topic.
    /// - Returns: Diversity evaluation result.
    public func evaluate(topics: [Topic], topKeywords: Int = 10) -> DiversityResult {
        guard !topics.isEmpty else {
            return emptyResult()
        }

        // Extract keywords per topic
        let topicKeywords: [[String]] = topics.map { topic in
            topic.keywords
                .prefix(topKeywords)
                .map(\.term)
        }

        return evaluate(topicKeywords: topicKeywords)
    }

    /// Evaluates diversity metrics for keyword lists.
    ///
    /// - Parameter topicKeywords: Keywords per topic.
    /// - Returns: Diversity evaluation result.
    public func evaluate(topicKeywords: [[String]]) -> DiversityResult {
        guard !topicKeywords.isEmpty else {
            return emptyResult()
        }

        // Flatten and count
        let allKeywords = topicKeywords.flatMap { $0 }
        let uniqueKeywords = Set(allKeywords)

        let totalCount = allKeywords.count
        let uniqueCount = uniqueKeywords.count

        // Diversity = unique / total
        let diversity: Float = totalCount > 0 ? Float(uniqueCount) / Float(totalCount) : 1.0

        // Compute per-topic redundancy
        let redundancy = computeRedundancy(topicKeywords: topicKeywords)

        // Mean redundancy
        let meanRedundancy: Float
        if redundancy.isEmpty {
            meanRedundancy = 0
        } else {
            meanRedundancy = redundancy.reduce(0, +) / Float(redundancy.count)
        }

        // Overlap matrix (optional)
        let overlapMatrix: [[Float]]?
        if computeOverlapMatrix {
            overlapMatrix = computeOverlapMatrix(topicKeywords: topicKeywords)
        } else {
            overlapMatrix = nil
        }

        return DiversityResult(
            diversity: diversity,
            uniqueKeywordCount: uniqueCount,
            totalKeywordCount: totalCount,
            topicRedundancy: redundancy,
            meanRedundancy: meanRedundancy,
            overlapMatrix: overlapMatrix
        )
    }

    // MARK: - Redundancy Computation

    private func computeRedundancy(topicKeywords: [[String]]) -> [Float] {
        let k = topicKeywords.count

        // Build keyword sets per topic
        let keywordSets = topicKeywords.map(Set.init)

        // For each topic, count keywords that appear in other topics
        var redundancy = [Float]()
        redundancy.reserveCapacity(k)

        for i in 0..<k {
            let topicKeywordSet = keywordSets[i]

            guard !topicKeywordSet.isEmpty else {
                redundancy.append(0)
                continue
            }

            // Keywords from all other topics
            var otherKeywords = Set<String>()
            for j in 0..<k where j != i {
                otherKeywords.formUnion(keywordSets[j])
            }

            // Count overlap
            let overlap = topicKeywordSet.intersection(otherKeywords).count
            let redundancyScore = Float(overlap) / Float(topicKeywordSet.count)

            redundancy.append(redundancyScore)
        }

        return redundancy
    }

    // MARK: - Overlap Matrix

    private func computeOverlapMatrix(topicKeywords: [[String]]) -> [[Float]] {
        let k = topicKeywords.count
        let keywordSets = topicKeywords.map(Set.init)

        var matrix = [[Float]](repeating: [Float](repeating: 0, count: k), count: k)

        for i in 0..<k {
            for j in 0..<k {
                if i == j {
                    matrix[i][j] = 1.0  // Self-overlap is 1
                } else if keywordSets[i].isEmpty {
                    matrix[i][j] = 0
                } else {
                    // Fraction of i's keywords that appear in j
                    let overlap = keywordSets[i].intersection(keywordSets[j]).count
                    matrix[i][j] = Float(overlap) / Float(keywordSets[i].count)
                }
            }
        }

        return matrix
    }

    private func emptyResult() -> DiversityResult {
        DiversityResult(
            diversity: 1.0,
            uniqueKeywordCount: 0,
            totalKeywordCount: 0,
            topicRedundancy: [],
            meanRedundancy: 0,
            overlapMatrix: nil
        )
    }
}

// MARK: - Convenience Extensions

extension Array where Element == Topic {

    /// Evaluates diversity metrics for topics.
    ///
    /// - Parameters:
    ///   - topKeywords: Number of top keywords to use per topic.
    ///   - computeOverlapMatrix: Whether to compute pairwise overlaps.
    /// - Returns: Diversity evaluation result.
    public func evaluateDiversity(
        topKeywords: Int = 10,
        computeOverlapMatrix: Bool = false
    ) -> DiversityResult {
        let metrics = DiversityMetrics(computeOverlapMatrix: computeOverlapMatrix)
        return metrics.evaluate(topics: self, topKeywords: topKeywords)
    }
}

// MARK: - Combined Evaluation

/// Combined coherence and diversity evaluation result.
public struct TopicQualityResult: Sendable {

    /// Coherence evaluation result.
    public let coherence: CoherenceResult

    /// Diversity evaluation result.
    public let diversity: DiversityResult

    /// Combined quality score (coherence × diversity).
    ///
    /// Balances semantic meaningfulness (coherence) with topic separation (diversity).
    public var qualityScore: Float {
        // Use mean coherence normalized to [0, 1] by mapping [-1, 1] -> [0, 1]
        let normalizedCoherence = (coherence.meanCoherence + 1) / 2
        return normalizedCoherence * diversity.diversity
    }

    /// Creates a topic quality result.
    public init(coherence: CoherenceResult, diversity: DiversityResult) {
        self.coherence = coherence
        self.diversity = diversity
    }
}

extension Array where Element == Topic {

    /// Evaluates both coherence and diversity for topics.
    ///
    /// - Parameters:
    ///   - documents: Corpus documents for coherence evaluation.
    ///   - configuration: Coherence configuration.
    /// - Returns: Combined quality evaluation result.
    public func evaluateQuality(
        documents: [Document],
        configuration: CoherenceConfiguration = .default
    ) async -> TopicQualityResult {
        async let coherence = evaluateCoherence(
            documents: documents,
            configuration: configuration
        )
        let diversity = evaluateDiversity(topKeywords: configuration.topKeywords)

        return TopicQualityResult(
            coherence: await coherence,
            diversity: diversity
        )
    }
}
