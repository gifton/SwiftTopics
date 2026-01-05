// cTFIDF.swift
// SwiftTopics
//
// Class-based TF-IDF for topic keyword extraction

import Foundation

// MARK: - c-TF-IDF Computer

/// Computes class-based TF-IDF scores for topic keyword extraction.
///
/// ## What is c-TF-IDF?
///
/// Class-based TF-IDF (c-TF-IDF) is a variant of TF-IDF that treats each cluster
/// as a single "document". This produces keywords that distinguish one topic from
/// others in the corpus.
///
/// ## Formula
///
/// ```
/// c-TF-IDF(t, c) = tf(t, c) × log(1 + A / tf(t, corpus))
/// ```
///
/// Where:
/// - `tf(t, c)` = frequency of term t in cluster c (sum across all documents in c)
/// - `A` = average number of words per cluster
/// - `tf(t, corpus)` = total frequency of term t across all clusters
///
/// ## Why c-TF-IDF for Topic Modeling?
///
/// - Standard TF-IDF compares documents; c-TF-IDF compares clusters
/// - Finds terms that are distinctive to each topic, not just frequent
/// - The log(1 + A/tf) formulation (BERTopic style) provides smoother weighting
///   than traditional IDF, avoiding issues with very rare terms
///
/// ## Thread Safety
///
/// `CTFIDFComputer` is immutable and safe to use from any thread.
public struct CTFIDFComputer: Sendable {

    /// Minimum score to consider a term as a keyword.
    public let minScore: Float

    /// Whether to normalize scores to [0, 1] range per cluster.
    public let normalizeScores: Bool

    /// Creates a c-TF-IDF computer.
    ///
    /// - Parameters:
    ///   - minScore: Minimum score threshold (default: 0.0).
    ///   - normalizeScores: Whether to normalize scores (default: true).
    public init(minScore: Float = 0.0, normalizeScores: Bool = true) {
        self.minScore = minScore
        self.normalizeScores = normalizeScores
    }

    // MARK: - Compute

    /// Computes c-TF-IDF scores for all clusters.
    ///
    /// - Parameters:
    ///   - clusterTokens: Array of token arrays per cluster (cluster index → tokens).
    ///   - vocabulary: The vocabulary to use.
    /// - Returns: c-TF-IDF scores per cluster (cluster index → term scores).
    public func compute(
        clusterTokens: [[String]],
        vocabulary: Vocabulary
    ) -> [ClusterTermScores] {
        let k = clusterTokens.count
        guard k > 0 && vocabulary.size > 0 else {
            return []
        }

        // Step 1: Compute term frequencies per cluster
        var clusterTermFreqs: [[Int: Int]] = []
        clusterTermFreqs.reserveCapacity(k)

        var totalTokensPerCluster = [Int](repeating: 0, count: k)
        var corpusTermFreq = [Int: Int]()

        for (clusterIdx, tokens) in clusterTokens.enumerated() {
            var termFreq = [Int: Int]()
            for token in tokens {
                if let termIdx = vocabulary.index(for: token) {
                    termFreq[termIdx, default: 0] += 1
                    corpusTermFreq[termIdx, default: 0] += 1
                }
            }
            clusterTermFreqs.append(termFreq)
            totalTokensPerCluster[clusterIdx] = tokens.count
        }

        // Step 2: Compute average tokens per cluster (A in the formula)
        let totalTokens = totalTokensPerCluster.reduce(0, +)
        let avgTokensPerCluster = Float(totalTokens) / Float(max(1, k))

        // Step 3: Compute c-TF-IDF for each cluster
        var results = [ClusterTermScores]()
        results.reserveCapacity(k)

        for clusterIdx in 0..<k {
            let termFreq = clusterTermFreqs[clusterIdx]
            var scores = [TermScore]()
            scores.reserveCapacity(termFreq.count)

            for (termIdx, tf) in termFreq {
                guard let term = vocabulary.term(at: termIdx) else { continue }

                let tfCorpus = corpusTermFreq[termIdx, default: 1]

                // c-TF-IDF formula: tf(t,c) × log(1 + A / tf(t,corpus))
                let idf = log(1.0 + avgTokensPerCluster / Float(tfCorpus))
                let score = Float(tf) * idf

                if score >= minScore {
                    scores.append(TermScore(
                        term: term,
                        termIndex: termIdx,
                        score: score,
                        frequency: tf
                    ))
                }
            }

            // Sort by score descending
            scores.sort { $0.score > $1.score }

            // Normalize if requested
            if normalizeScores && !scores.isEmpty {
                let maxScore = scores[0].score
                if maxScore > 0 {
                    scores = scores.map { score in
                        TermScore(
                            term: score.term,
                            termIndex: score.termIndex,
                            score: score.score / maxScore,
                            frequency: score.frequency
                        )
                    }
                }
            }

            results.append(ClusterTermScores(
                clusterIndex: clusterIdx,
                scores: scores,
                tokenCount: totalTokensPerCluster[clusterIdx]
            ))
        }

        return results
    }

    /// Computes c-TF-IDF from documents and cluster assignments.
    ///
    /// This is a convenience method that handles the aggregation of documents
    /// into clusters.
    ///
    /// - Parameters:
    ///   - documents: The documents.
    ///   - assignment: Cluster assignments.
    ///   - tokenizer: Tokenizer to use.
    ///   - vocabulary: Vocabulary to use (if nil, one is built).
    /// - Returns: c-TF-IDF scores per cluster.
    public func compute(
        documents: [Document],
        assignment: ClusterAssignment,
        tokenizer: Tokenizer,
        vocabulary: Vocabulary? = nil
    ) -> CTFIDFResult {
        let k = assignment.clusterCount

        // Tokenize all documents
        let allTokens = tokenizer.tokenize(documents: documents)

        // Build vocabulary if not provided
        let vocab = vocabulary ?? VocabularyBuilder().build(from: allTokens)

        // Aggregate tokens by cluster
        var clusterTokens = [[String]](repeating: [], count: k)

        for (docIdx, tokens) in allTokens.enumerated() {
            let label = assignment.label(for: docIdx)
            guard label >= 0 && label < k else { continue }  // Skip outliers
            clusterTokens[label].append(contentsOf: tokens)
        }

        // Compute c-TF-IDF
        let scores = compute(clusterTokens: clusterTokens, vocabulary: vocab)

        return CTFIDFResult(
            clusterScores: scores,
            vocabulary: vocab
        )
    }
}

// MARK: - Term Score

/// Score for a single term in a cluster.
public struct TermScore: Sendable, Hashable {

    /// The term string.
    public let term: String

    /// Index in the vocabulary.
    public let termIndex: Int

    /// The c-TF-IDF score.
    public let score: Float

    /// Raw frequency of the term in the cluster.
    public let frequency: Int

    /// Creates a term score.
    public init(term: String, termIndex: Int, score: Float, frequency: Int) {
        self.term = term
        self.termIndex = termIndex
        self.score = score
        self.frequency = frequency
    }
}

extension TermScore: Comparable {
    public static func < (lhs: TermScore, rhs: TermScore) -> Bool {
        lhs.score < rhs.score
    }
}

// MARK: - Cluster Term Scores

/// c-TF-IDF scores for all terms in a cluster.
public struct ClusterTermScores: Sendable {

    /// The cluster index.
    public let clusterIndex: Int

    /// Term scores sorted by score (descending).
    public let scores: [TermScore]

    /// Total token count in this cluster.
    public let tokenCount: Int

    /// Creates cluster term scores.
    public init(clusterIndex: Int, scores: [TermScore], tokenCount: Int) {
        self.clusterIndex = clusterIndex
        self.scores = scores
        self.tokenCount = tokenCount
    }

    /// Gets the top-K terms.
    ///
    /// - Parameter k: Number of terms to return.
    /// - Returns: Top-K term scores.
    public func topK(_ k: Int) -> [TermScore] {
        Array(scores.prefix(k))
    }

    /// Gets the top-K terms as strings.
    ///
    /// - Parameter k: Number of terms to return.
    /// - Returns: Top-K term strings.
    public func topKTerms(_ k: Int) -> [String] {
        topK(k).map(\.term)
    }

    /// Converts to TopicKeyword array for Topic creation.
    ///
    /// - Parameter k: Number of keywords to return.
    /// - Returns: Array of TopicKeyword.
    public func toKeywords(_ k: Int) -> [TopicKeyword] {
        topK(k).map { score in
            TopicKeyword(
                term: score.term,
                score: score.score,
                frequency: score.frequency
            )
        }
    }
}

// MARK: - c-TF-IDF Result

/// Complete result of c-TF-IDF computation.
public struct CTFIDFResult: Sendable {

    /// Scores for each cluster.
    public let clusterScores: [ClusterTermScores]

    /// The vocabulary used.
    public let vocabulary: Vocabulary

    /// Number of clusters.
    public var clusterCount: Int {
        clusterScores.count
    }

    /// Creates a c-TF-IDF result.
    public init(clusterScores: [ClusterTermScores], vocabulary: Vocabulary) {
        self.clusterScores = clusterScores
        self.vocabulary = vocabulary
    }

    /// Gets scores for a specific cluster.
    ///
    /// - Parameter clusterIndex: The cluster index.
    /// - Returns: Cluster term scores, or nil if out of bounds.
    public func scores(for clusterIndex: Int) -> ClusterTermScores? {
        guard clusterIndex >= 0 && clusterIndex < clusterScores.count else {
            return nil
        }
        return clusterScores[clusterIndex]
    }

    /// Gets top-K keywords for a cluster.
    ///
    /// - Parameters:
    ///   - clusterIndex: The cluster index.
    ///   - k: Number of keywords.
    /// - Returns: Top-K keywords, or empty array if cluster not found.
    public func topKeywords(for clusterIndex: Int, k: Int) -> [TopicKeyword] {
        scores(for: clusterIndex)?.toKeywords(k) ?? []
    }

    /// Gets all keywords for all clusters.
    ///
    /// - Parameter k: Number of keywords per cluster.
    /// - Returns: Dictionary mapping cluster index to keywords.
    public func allKeywords(k: Int) -> [Int: [TopicKeyword]] {
        var result = [Int: [TopicKeyword]]()
        for clusterScore in clusterScores {
            result[clusterScore.clusterIndex] = clusterScore.toKeywords(k)
        }
        return result
    }
}

// MARK: - Convenience Extensions

extension Array where Element == [String] {

    /// Computes c-TF-IDF scores using a vocabulary.
    ///
    /// - Parameters:
    ///   - vocabulary: The vocabulary to use.
    ///   - normalizeScores: Whether to normalize scores.
    /// - Returns: c-TF-IDF scores per cluster.
    public func computeCTFIDF(
        vocabulary: Vocabulary,
        normalizeScores: Bool = true
    ) -> [ClusterTermScores] {
        let computer = CTFIDFComputer(normalizeScores: normalizeScores)
        return computer.compute(clusterTokens: self, vocabulary: vocabulary)
    }
}
