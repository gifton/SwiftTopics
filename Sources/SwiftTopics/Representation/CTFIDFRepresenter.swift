// CTFIDFRepresenter.swift
// SwiftTopics
//
// TopicRepresenter implementation using c-TF-IDF

import Foundation

// MARK: - c-TF-IDF Representer

/// Extracts topic keywords using class-based TF-IDF.
///
/// ## Algorithm
///
/// 1. **Tokenize**: Split each document into tokens (lowercase, filter stop words)
/// 2. **Build vocabulary**: Create term→index mapping, compute document frequencies
/// 3. **Aggregate by cluster**: Concatenate all tokens from documents in each cluster
/// 4. **Compute c-TF-IDF**: Apply the formula `tf(t,c) × log(1 + A / tf(t,corpus))`
/// 5. **Extract top-K**: Sort by score and return top-K keywords per cluster
///
/// ## Why c-TF-IDF?
///
/// - Standard TF-IDF weighs terms by document importance
/// - c-TF-IDF weighs terms by cluster importance
/// - Produces keywords that distinguish one topic from others
///
/// ## Thread Safety
///
/// `CTFIDFRepresenter` is `Sendable` and safe to use from any thread.
///
/// ## Usage
///
/// ```swift
/// let representer = CTFIDFRepresenter()
/// let topics = try await representer.represent(
///     documents: documents,
///     assignment: clusterAssignment
/// )
/// ```
public struct CTFIDFRepresenter: TopicRepresenter, Sendable {

    // MARK: - Properties

    /// The configuration for this representer.
    public let configuration: CTFIDFConfiguration

    /// The tokenizer to use.
    public let tokenizer: Tokenizer

    // MARK: - Initialization

    /// Creates a c-TF-IDF representer with configuration.
    ///
    /// - Parameter configuration: c-TF-IDF configuration.
    public init(configuration: CTFIDFConfiguration = .default) {
        self.configuration = configuration

        // Build tokenizer from configuration
        var stopWords = EnglishStopWords.standard
        stopWords.formUnion(configuration.customStopWords)

        self.tokenizer = Tokenizer(configuration: TokenizerConfiguration(
            stopWords: stopWords,
            minTokenLength: configuration.minTermLength,
            maxTokenLength: 50,
            lowercase: true,
            removeNumbers: false,
            useBigrams: configuration.useBigrams
        ))
    }

    /// Creates a c-TF-IDF representer with custom tokenizer.
    ///
    /// - Parameters:
    ///   - configuration: c-TF-IDF configuration.
    ///   - tokenizer: Custom tokenizer to use.
    public init(configuration: CTFIDFConfiguration, tokenizer: Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    // MARK: - TopicRepresenter Protocol

    /// Extracts topic representations from clustered documents.
    ///
    /// - Parameters:
    ///   - documents: The original documents.
    ///   - assignment: Cluster assignments from clustering.
    /// - Returns: Topics with keywords for each cluster.
    /// - Throws: `RepresentationError` if extraction fails.
    public func represent(
        documents: [Document],
        assignment: ClusterAssignment
    ) async throws -> [Topic] {
        try await represent(documents: documents, embeddings: [], assignment: assignment)
    }

    /// Extracts topic representations with embeddings for centroid computation.
    ///
    /// - Parameters:
    ///   - documents: The original documents.
    ///   - embeddings: Document embeddings (for centroid computation).
    ///   - assignment: Cluster assignments from clustering.
    /// - Returns: Topics with keywords and centroids.
    /// - Throws: `RepresentationError` if extraction fails.
    public func represent(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) async throws -> [Topic] {
        // Validation
        try validateInput(documents: documents, assignment: assignment)

        let k = assignment.clusterCount

        // Handle case with no clusters (all outliers)
        guard k > 0 else {
            return []
        }

        // Step 1: Tokenize all documents
        let allTokens = tokenizer.tokenize(documents: documents)

        // Step 2: Build vocabulary with filtering
        let vocabConfig = VocabularyConfiguration(
            minDocumentFrequency: configuration.minDocumentFrequency,
            maxDocumentFrequencyRatio: configuration.maxDocumentFrequencyRatio
        )
        let vocabulary = VocabularyBuilder(configuration: vocabConfig).build(from: allTokens)

        // Step 3: Aggregate tokens by cluster
        var clusterTokens = [[String]](repeating: [], count: k)
        var clusterDocIndices = [[Int]](repeating: [], count: k)

        for (docIdx, tokens) in allTokens.enumerated() {
            let label = assignment.label(for: docIdx)
            guard label >= 0 && label < k else { continue }  // Skip outliers
            clusterTokens[label].append(contentsOf: tokens)
            clusterDocIndices[label].append(docIdx)
        }

        // Step 4: Compute c-TF-IDF
        let computer = CTFIDFComputer(normalizeScores: true)
        let ctfidfScores = computer.compute(clusterTokens: clusterTokens, vocabulary: vocabulary)

        // Step 5: Build topics
        var topics = [Topic]()
        topics.reserveCapacity(k)

        for clusterIdx in 0..<k {
            let clusterScore = ctfidfScores.first { $0.clusterIndex == clusterIdx }
            let keywords = clusterScore?.toKeywords(configuration.keywordsPerTopic) ?? []

            // Apply MMR diversification if configured
            let finalKeywords: [TopicKeyword]
            if configuration.diversify && keywords.count > 1 {
                finalKeywords = diversifyKeywords(
                    keywords,
                    count: configuration.keywordsPerTopic,
                    weight: configuration.diversityWeight
                )
            } else {
                finalKeywords = keywords
            }

            // Get representative documents
            let docIndices = clusterDocIndices[clusterIdx]
            let representativeDocIDs = selectRepresentativeDocuments(
                docIndices: docIndices,
                documents: documents,
                embeddings: embeddings,
                maxCount: 3
            )

            // Compute centroid if embeddings provided
            let centroid = computeCentroid(
                docIndices: docIndices,
                embeddings: embeddings
            )

            let topic = Topic(
                id: TopicID(value: clusterIdx),
                keywords: finalKeywords,
                size: docIndices.count,
                coherenceScore: nil,  // Could compute NPMI here
                representativeDocuments: representativeDocIDs,
                centroid: centroid
            )

            topics.append(topic)
        }

        return topics
    }

    // MARK: - Validation

    private func validateInput(
        documents: [Document],
        assignment: ClusterAssignment
    ) throws {
        guard !documents.isEmpty else {
            throw RepresentationError.emptyDocuments
        }

        guard documents.count == assignment.pointCount else {
            throw RepresentationError.countMismatch(
                documents: documents.count,
                assignments: assignment.pointCount
            )
        }
    }

    // MARK: - MMR Diversification

    /// Diversifies keywords using Maximal Marginal Relevance.
    ///
    /// MMR balances relevance (high c-TF-IDF score) with diversity (dissimilar to
    /// already selected keywords).
    ///
    /// - Parameters:
    ///   - keywords: Original keywords sorted by score.
    ///   - count: Number of keywords to select.
    ///   - weight: Diversity weight (0 = only relevance, 1 = only diversity).
    /// - Returns: Diversified keywords.
    private func diversifyKeywords(
        _ keywords: [TopicKeyword],
        count: Int,
        weight: Float
    ) -> [TopicKeyword] {
        guard keywords.count > 1 else { return keywords }

        var selected = [TopicKeyword]()
        var remaining = keywords

        // Always select the top keyword first
        selected.append(remaining.removeFirst())

        while selected.count < count && !remaining.isEmpty {
            var bestIdx = 0
            var bestScore: Float = -.infinity

            for (idx, candidate) in remaining.enumerated() {
                // Relevance component (normalized score)
                let relevance = candidate.score

                // Diversity component (minimum similarity to selected)
                let maxSimilarity = selected.map { selected in
                    stringSimilarity(candidate.term, selected.term)
                }.max() ?? 0

                let diversity = 1.0 - maxSimilarity

                // MMR score
                let mmrScore = (1.0 - weight) * relevance + weight * diversity

                if mmrScore > bestScore {
                    bestScore = mmrScore
                    bestIdx = idx
                }
            }

            selected.append(remaining.remove(at: bestIdx))
        }

        return selected
    }

    /// Computes string similarity using Jaccard on character n-grams.
    private func stringSimilarity(_ a: String, _ b: String) -> Float {
        let ngramSize = 3

        func ngrams(_ s: String) -> Set<String> {
            guard s.count >= ngramSize else { return [s] }
            var result = Set<String>()
            let chars = Array(s)
            for i in 0...(chars.count - ngramSize) {
                result.insert(String(chars[i..<(i + ngramSize)]))
            }
            return result
        }

        let aNgrams = ngrams(a)
        let bNgrams = ngrams(b)

        let intersection = aNgrams.intersection(bNgrams).count
        let union = aNgrams.union(bNgrams).count

        guard union > 0 else { return 0 }
        return Float(intersection) / Float(union)
    }

    // MARK: - Representative Documents

    /// Selects representative documents for a cluster.
    ///
    /// If embeddings are provided, selects documents closest to the centroid.
    /// Otherwise, returns the first few documents.
    private func selectRepresentativeDocuments(
        docIndices: [Int],
        documents: [Document],
        embeddings: [Embedding],
        maxCount: Int
    ) -> [DocumentID] {
        guard !docIndices.isEmpty else { return [] }

        let count = min(maxCount, docIndices.count)

        // If no embeddings, just take first documents
        guard !embeddings.isEmpty && embeddings.count == documents.count else {
            return docIndices.prefix(count).map { documents[$0].id }
        }

        // Compute centroid
        guard let centroid = computeCentroid(docIndices: docIndices, embeddings: embeddings) else {
            return docIndices.prefix(count).map { documents[$0].id }
        }

        // Sort by distance to centroid
        let sorted = docIndices.sorted { idxA, idxB in
            let distA = euclideanDistance(embeddings[idxA], centroid)
            let distB = euclideanDistance(embeddings[idxB], centroid)
            return distA < distB
        }

        return sorted.prefix(count).map { documents[$0].id }
    }

    /// Computes the centroid embedding for a cluster.
    private func computeCentroid(
        docIndices: [Int],
        embeddings: [Embedding]
    ) -> Embedding? {
        guard !embeddings.isEmpty && !docIndices.isEmpty else { return nil }
        guard docIndices.allSatisfy({ $0 < embeddings.count }) else { return nil }

        let dimension = embeddings[docIndices[0]].dimension
        var sum = [Float](repeating: 0, count: dimension)

        for idx in docIndices {
            let emb = embeddings[idx]
            for d in 0..<dimension {
                sum[d] += emb.vector[d]
            }
        }

        let scale = 1.0 / Float(docIndices.count)
        for d in 0..<dimension {
            sum[d] *= scale
        }

        return Embedding(vector: sum)
    }

    /// Euclidean distance between an embedding and a centroid.
    private func euclideanDistance(_ embedding: Embedding, _ centroid: Embedding) -> Float {
        var sum: Float = 0
        let dim = min(embedding.dimension, centroid.dimension)
        for d in 0..<dim {
            let diff = embedding.vector[d] - centroid.vector[d]
            sum += diff * diff
        }
        return sum.squareRoot()
    }
}

// MARK: - Builder

/// Builder for creating c-TF-IDF representers.
public struct CTFIDFRepresenterBuilder: Sendable {

    private var keywordsPerTopic: Int = 10
    private var minDocumentFrequency: Int = 1
    private var maxDocumentFrequencyRatio: Float = 0.95
    private var minTermLength: Int = 2
    private var useBigrams: Bool = false
    private var customStopWords: [String] = []
    private var diversify: Bool = false
    private var diversityWeight: Float = 0.3

    /// Creates a new builder with default settings.
    public init() {}

    /// Sets the number of keywords per topic.
    public func keywordsPerTopic(_ count: Int) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.keywordsPerTopic = count
        return copy
    }

    /// Sets the minimum document frequency.
    public func minDocumentFrequency(_ freq: Int) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.minDocumentFrequency = freq
        return copy
    }

    /// Sets the maximum document frequency ratio.
    public func maxDocumentFrequencyRatio(_ ratio: Float) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.maxDocumentFrequencyRatio = ratio
        return copy
    }

    /// Sets the minimum term length.
    public func minTermLength(_ length: Int) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.minTermLength = length
        return copy
    }

    /// Enables bigram generation.
    public func useBigrams(_ enable: Bool = true) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.useBigrams = enable
        return copy
    }

    /// Adds custom stop words.
    public func stopWords(_ words: [String]) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.customStopWords = words
        return copy
    }

    /// Enables keyword diversification.
    public func diversify(_ enable: Bool = true) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.diversify = enable
        return copy
    }

    /// Sets the diversity weight for MMR.
    public func diversityWeight(_ weight: Float) -> CTFIDFRepresenterBuilder {
        var copy = self
        copy.diversityWeight = weight
        return copy
    }

    /// Builds the configuration.
    public func buildConfiguration() -> CTFIDFConfiguration {
        CTFIDFConfiguration(
            keywordsPerTopic: keywordsPerTopic,
            minDocumentFrequency: minDocumentFrequency,
            maxDocumentFrequencyRatio: maxDocumentFrequencyRatio,
            minTermLength: minTermLength,
            useBigrams: useBigrams,
            customStopWords: customStopWords,
            diversify: diversify,
            diversityWeight: diversityWeight
        )
    }

    /// Builds the representer.
    public func build() -> CTFIDFRepresenter {
        CTFIDFRepresenter(configuration: buildConfiguration())
    }
}

// MARK: - Convenience Extensions

extension Array where Element == Document {

    /// Extracts topic keywords using c-TF-IDF.
    ///
    /// - Parameters:
    ///   - assignment: Cluster assignments.
    ///   - configuration: c-TF-IDF configuration.
    /// - Returns: Topics with keywords.
    public func extractTopics(
        assignment: ClusterAssignment,
        configuration: CTFIDFConfiguration = .default
    ) async throws -> [Topic] {
        let representer = CTFIDFRepresenter(configuration: configuration)
        return try await representer.represent(documents: self, assignment: assignment)
    }

    /// Extracts topic keywords with embeddings for centroids.
    ///
    /// - Parameters:
    ///   - embeddings: Document embeddings.
    ///   - assignment: Cluster assignments.
    ///   - configuration: c-TF-IDF configuration.
    /// - Returns: Topics with keywords and centroids.
    public func extractTopics(
        embeddings: [Embedding],
        assignment: ClusterAssignment,
        configuration: CTFIDFConfiguration = .default
    ) async throws -> [Topic] {
        let representer = CTFIDFRepresenter(configuration: configuration)
        return try await representer.represent(
            documents: self,
            embeddings: embeddings,
            assignment: assignment
        )
    }
}
