// NPMIScorer.swift
// SwiftTopics
//
// Normalized Pointwise Mutual Information scoring

import Foundation

// MARK: - NPMI Configuration

/// Configuration for NPMI scoring.
public struct NPMIConfiguration: Sendable, Codable, Hashable {

    /// Smoothing epsilon to avoid log(0).
    ///
    /// A small value added to probabilities before taking logarithms.
    public let epsilon: Float

    /// Creates an NPMI configuration.
    ///
    /// - Parameter epsilon: Smoothing epsilon (default: 1e-12).
    public init(epsilon: Float = 1e-12) {
        precondition(epsilon > 0, "epsilon must be positive")
        self.epsilon = epsilon
    }

    /// Default configuration.
    public static let `default` = NPMIConfiguration()
}

// MARK: - NPMI Pair Score

/// NPMI score for a single word pair.
public struct NPMIPairScore: Sendable, Hashable {

    /// First word.
    public let word1: String

    /// Second word.
    public let word2: String

    /// NPMI score in range [-1, +1].
    ///
    /// - `+1`: Words always co-occur (perfect positive association)
    /// - `0`: Words are statistically independent
    /// - `-1`: Words never co-occur (perfect negative association)
    public let npmi: Float

    /// PMI score (unnormalized).
    public let pmi: Float

    /// Probability of word1.
    public let pWord1: Float

    /// Probability of word2.
    public let pWord2: Float

    /// Joint probability of word pair.
    public let pPair: Float

    /// Creates an NPMI pair score.
    public init(
        word1: String,
        word2: String,
        npmi: Float,
        pmi: Float,
        pWord1: Float,
        pWord2: Float,
        pPair: Float
    ) {
        self.word1 = word1
        self.word2 = word2
        self.npmi = npmi
        self.pmi = pmi
        self.pWord1 = pWord1
        self.pWord2 = pWord2
        self.pPair = pPair
    }
}

// MARK: - Topic NPMI Result

/// NPMI scores for a topic's keywords.
public struct TopicNPMIResult: Sendable {

    /// The topic's keywords that were scored.
    public let keywords: [String]

    /// NPMI scores for each word pair.
    public let pairScores: [NPMIPairScore]

    /// Mean NPMI across all word pairs.
    ///
    /// This is the topic's coherence score.
    public let meanNPMI: Float

    /// Number of word pairs evaluated.
    public var pairCount: Int {
        pairScores.count
    }

    /// Creates a topic NPMI result.
    public init(keywords: [String], pairScores: [NPMIPairScore], meanNPMI: Float) {
        self.keywords = keywords
        self.pairScores = pairScores
        self.meanNPMI = meanNPMI
    }
}

// MARK: - NPMI Scorer

/// Computes Normalized Pointwise Mutual Information scores.
///
/// ## What is NPMI?
///
/// NPMI measures the association between word pairs, normalized to [-1, +1]:
///
/// ```
/// PMI(w1, w2) = log(P(w1, w2) / (P(w1) × P(w2)))
/// NPMI(w1, w2) = PMI(w1, w2) / -log(P(w1, w2))
/// ```
///
/// ## Interpretation
///
/// - **+1**: Words always co-occur (perfect association)
/// - **0**: Words are statistically independent
/// - **-1**: Words never co-occur
///
/// ## Topic Coherence
///
/// Topic coherence is the average NPMI over all keyword pairs:
///
/// ```
/// coherence = mean(NPMI(w_i, w_j)) for all i < j
/// ```
///
/// Higher coherence indicates that the topic's keywords frequently
/// co-occur in the corpus, suggesting a meaningful concept.
///
/// ## Thread Safety
///
/// `NPMIScorer` is `Sendable` and safe to use from any thread.
public struct NPMIScorer: Sendable {

    /// Configuration for scoring.
    public let configuration: NPMIConfiguration

    /// Creates an NPMI scorer.
    ///
    /// - Parameter configuration: Scoring configuration.
    public init(configuration: NPMIConfiguration = .default) {
        self.configuration = configuration
    }

    // MARK: - Scoring

    /// Computes NPMI for a single word pair.
    ///
    /// - Parameters:
    ///   - word1: First word.
    ///   - word2: Second word.
    ///   - counts: Co-occurrence counts from the corpus.
    /// - Returns: NPMI pair score, or nil if either word is unknown.
    public func score(
        word1: String,
        word2: String,
        counts: CooccurrenceCounts
    ) -> NPMIPairScore? {
        guard counts.totalWindows > 0 else { return nil }
        guard word1 != word2 else { return nil }

        let totalWindows = Float(counts.totalWindows)
        let epsilon = configuration.epsilon

        // Get raw counts
        let countW1 = counts.count(for: word1)
        let countW2 = counts.count(for: word2)
        let countPair = counts.count(for: word1, word2)

        // If either word doesn't appear, we can't compute meaningful NPMI
        guard countW1 > 0 && countW2 > 0 else { return nil }

        // Compute probabilities with smoothing
        let pW1 = (Float(countW1) + epsilon) / (totalWindows + epsilon)
        let pW2 = (Float(countW2) + epsilon) / (totalWindows + epsilon)
        let pPair = (Float(countPair) + epsilon) / (totalWindows + epsilon)

        // Compute PMI = log(P(w1,w2) / (P(w1) × P(w2)))
        let pmi = log(pPair / (pW1 * pW2))

        // Compute NPMI = PMI / -log(P(w1,w2))
        // When pPair is very small, -log(pPair) is large positive
        let denominator = -log(pPair)
        let npmi: Float
        if denominator.isNaN || denominator.isInfinite || denominator == 0 {
            // Edge case: avoid division by zero/infinity
            npmi = 0
        } else {
            npmi = pmi / denominator
        }

        // Clamp to [-1, +1] for numerical stability
        let clampedNPMI = max(-1.0, min(1.0, npmi))

        return NPMIPairScore(
            word1: word1,
            word2: word2,
            npmi: clampedNPMI,
            pmi: pmi,
            pWord1: pW1,
            pWord2: pW2,
            pPair: pPair
        )
    }

    /// Computes NPMI for a topic's keywords.
    ///
    /// Evaluates all pairs (w_i, w_j) where i < j and returns the mean NPMI.
    ///
    /// - Parameters:
    ///   - keywords: The topic's keywords to evaluate.
    ///   - counts: Co-occurrence counts from the corpus.
    /// - Returns: Topic NPMI result with per-pair scores and mean.
    public func score(
        keywords: [String],
        counts: CooccurrenceCounts
    ) -> TopicNPMIResult {
        var pairScores: [NPMIPairScore] = []

        // Compute NPMI for all pairs (i, j) where i < j
        for i in 0..<keywords.count {
            for j in (i + 1)..<keywords.count {
                if let pairScore = score(
                    word1: keywords[i],
                    word2: keywords[j],
                    counts: counts
                ) {
                    pairScores.append(pairScore)
                }
            }
        }

        // Compute mean NPMI
        let meanNPMI: Float
        if pairScores.isEmpty {
            // No valid pairs - return 0 as neutral score
            meanNPMI = 0
        } else {
            let sum = pairScores.reduce(0.0) { $0 + $1.npmi }
            meanNPMI = sum / Float(pairScores.count)
        }

        return TopicNPMIResult(
            keywords: keywords,
            pairScores: pairScores,
            meanNPMI: meanNPMI
        )
    }

    /// Computes NPMI for a `Topic`'s keywords.
    ///
    /// - Parameters:
    ///   - topic: The topic to evaluate.
    ///   - counts: Co-occurrence counts from the corpus.
    ///   - topKeywords: Number of top keywords to use (default: 10).
    /// - Returns: Topic NPMI result.
    public func score(
        topic: Topic,
        counts: CooccurrenceCounts,
        topKeywords: Int = 10
    ) -> TopicNPMIResult {
        let keywords = topic.keywords
            .prefix(topKeywords)
            .map(\.term)
        return score(keywords: Array(keywords), counts: counts)
    }

    /// Computes NPMI for multiple topics.
    ///
    /// - Parameters:
    ///   - topics: Topics to evaluate.
    ///   - counts: Co-occurrence counts from the corpus.
    ///   - topKeywords: Number of top keywords to use per topic.
    /// - Returns: Array of topic NPMI results (one per topic).
    public func score(
        topics: [Topic],
        counts: CooccurrenceCounts,
        topKeywords: Int = 10
    ) -> [TopicNPMIResult] {
        topics.map { topic in
            score(topic: topic, counts: counts, topKeywords: topKeywords)
        }
    }
}

// MARK: - Convenience Extensions

extension Topic {

    /// Computes NPMI coherence for this topic.
    ///
    /// - Parameters:
    ///   - counts: Co-occurrence counts from the corpus.
    ///   - topKeywords: Number of top keywords to use.
    /// - Returns: Topic NPMI result.
    public func computeNPMI(
        counts: CooccurrenceCounts,
        topKeywords: Int = 10
    ) -> TopicNPMIResult {
        NPMIScorer().score(
            topic: self,
            counts: counts,
            topKeywords: topKeywords
        )
    }
}

extension Array where Element == Topic {

    /// Computes NPMI coherence for all topics.
    ///
    /// - Parameters:
    ///   - counts: Co-occurrence counts from the corpus.
    ///   - topKeywords: Number of top keywords to use per topic.
    /// - Returns: Array of topic NPMI results.
    public func computeNPMI(
        counts: CooccurrenceCounts,
        topKeywords: Int = 10
    ) -> [TopicNPMIResult] {
        NPMIScorer().score(
            topics: self,
            counts: counts,
            topKeywords: topKeywords
        )
    }
}
