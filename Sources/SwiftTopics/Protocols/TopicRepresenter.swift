// TopicRepresenter.swift
// SwiftTopics
//
// Protocol for topic representation extraction

import Foundation

// MARK: - Topic Representer Protocol

/// A component that extracts interpretable keywords for each topic.
///
/// After clustering, we need to understand what each topic is "about".
/// The representer analyzes documents in each cluster and extracts
/// keywords that characterize the topic.
///
/// ## Available Implementations
/// - `CTFIDFRepresenter`: Class-based TF-IDF scoring
///
/// ## How c-TF-IDF Works
/// Traditional TF-IDF compares terms across documents. c-TF-IDF treats each
/// cluster as a single "mega-document" and finds terms that are:
/// - Frequent within the cluster (high tf in cluster)
/// - Rare across other clusters (high inverse frequency)
///
/// ## Thread Safety
/// Implementations must be safe to call from any thread.
public protocol TopicRepresenter: Sendable {

    /// The type of configuration used by this representer.
    associatedtype Configuration: RepresentationConfiguration

    /// The configuration for this representer.
    var configuration: Configuration { get }

    /// Extracts topic representations from clustered documents.
    ///
    /// - Parameters:
    ///   - documents: The original documents.
    ///   - assignment: Cluster assignments from clustering.
    /// - Returns: Topics with keywords for each cluster.
    /// - Throws: `RepresentationError` if extraction fails.
    func represent(
        documents: [Document],
        assignment: ClusterAssignment
    ) async throws -> [Topic]

    /// Extracts topic representations with document embeddings for centroid computation.
    ///
    /// - Parameters:
    ///   - documents: The original documents.
    ///   - embeddings: Document embeddings (for centroid computation).
    ///   - assignment: Cluster assignments from clustering.
    /// - Returns: Topics with keywords and centroids.
    /// - Throws: `RepresentationError` if extraction fails.
    func represent(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) async throws -> [Topic]
}

// MARK: - Representation Configuration Protocol

/// Configuration for a topic representer.
public protocol RepresentationConfiguration: Sendable, Codable {}

// MARK: - Representation Error

/// Errors that can occur during topic representation.
public enum RepresentationError: Error, Sendable {

    /// No documents provided.
    case emptyDocuments

    /// Document count doesn't match assignment count.
    case countMismatch(documents: Int, assignments: Int)

    /// No clusters to represent (all outliers).
    case noClusters

    /// A cluster has no documents (inconsistent assignment).
    case emptyCluster(clusterID: Int)

    /// Tokenization failed.
    case tokenizationFailed(String)

    /// Unknown error.
    case unknown(String)
}

extension RepresentationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyDocuments:
            return "Cannot represent topics without documents"
        case .countMismatch(let documents, let assignments):
            return "Document count (\(documents)) doesn't match assignment count (\(assignments))"
        case .noClusters:
            return "No clusters to represent (all points are outliers)"
        case .emptyCluster(let clusterID):
            return "Cluster \(clusterID) has no documents"
        case .tokenizationFailed(let message):
            return "Tokenization failed: \(message)"
        case .unknown(let message):
            return "Representation error: \(message)"
        }
    }
}

// MARK: - c-TF-IDF Configuration

/// Configuration for c-TF-IDF topic representation.
public struct CTFIDFConfiguration: RepresentationConfiguration {

    /// Number of keywords to extract per topic.
    public let keywordsPerTopic: Int

    /// Minimum document frequency for a term to be considered.
    ///
    /// Terms appearing in fewer documents are ignored.
    public let minDocumentFrequency: Int

    /// Maximum document frequency ratio for a term.
    ///
    /// Terms appearing in more than this fraction of documents are ignored.
    /// Helps filter common words that weren't in the stop list.
    public let maxDocumentFrequencyRatio: Float

    /// Minimum term length.
    public let minTermLength: Int

    /// Whether to use bigrams in addition to unigrams.
    public let useBigrams: Bool

    /// Custom stop words to filter (in addition to default).
    public let customStopWords: [String]

    /// Whether to apply MMR diversification to keywords.
    public let diversify: Bool

    /// Diversity weight for MMR (0 = only relevance, 1 = only diversity).
    public let diversityWeight: Float

    /// Creates c-TF-IDF configuration.
    public init(
        keywordsPerTopic: Int = 10,
        minDocumentFrequency: Int = 1,
        maxDocumentFrequencyRatio: Float = 0.95,
        minTermLength: Int = 2,
        useBigrams: Bool = false,
        customStopWords: [String] = [],
        diversify: Bool = false,
        diversityWeight: Float = 0.3
    ) {
        precondition(keywordsPerTopic > 0, "Must extract at least one keyword")
        precondition(minDocumentFrequency >= 1, "minDocumentFrequency must be at least 1")
        precondition(maxDocumentFrequencyRatio > 0 && maxDocumentFrequencyRatio <= 1, "maxDocumentFrequencyRatio must be in (0, 1]")
        precondition(diversityWeight >= 0 && diversityWeight <= 1, "diversityWeight must be in [0, 1]")

        self.keywordsPerTopic = keywordsPerTopic
        self.minDocumentFrequency = minDocumentFrequency
        self.maxDocumentFrequencyRatio = maxDocumentFrequencyRatio
        self.minTermLength = minTermLength
        self.useBigrams = useBigrams
        self.customStopWords = customStopWords
        self.diversify = diversify
        self.diversityWeight = diversityWeight
    }

    /// Default configuration.
    public static let `default` = CTFIDFConfiguration()

    /// Configuration for more diverse keywords.
    public static let diverse = CTFIDFConfiguration(
        keywordsPerTopic: 10,
        diversify: true,
        diversityWeight: 0.5
    )

    /// Configuration with bigrams.
    public static let withBigrams = CTFIDFConfiguration(
        keywordsPerTopic: 15,
        useBigrams: true
    )
}

// MARK: - Representation Result

/// Result from topic representation with additional diagnostics.
public struct RepresentationResult: Sendable {

    /// The extracted topics.
    public let topics: [Topic]

    /// Vocabulary size used.
    public let vocabularySize: Int

    /// Processing time in seconds.
    public let processingTime: TimeInterval

    /// Creates a representation result.
    public init(
        topics: [Topic],
        vocabularySize: Int,
        processingTime: TimeInterval = 0
    ) {
        self.topics = topics
        self.vocabularySize = vocabularySize
        self.processingTime = processingTime
    }
}

// MARK: - Topic Representation Options

/// Options for how topics should be represented.
public struct TopicRepresentationOptions: Sendable {

    /// Include representative documents for each topic.
    public let includeRepresentativeDocuments: Bool

    /// Maximum number of representative documents per topic.
    public let maxRepresentativeDocuments: Int

    /// Include topic centroids.
    public let includeCentroids: Bool

    /// Creates topic representation options.
    public init(
        includeRepresentativeDocuments: Bool = true,
        maxRepresentativeDocuments: Int = 3,
        includeCentroids: Bool = true
    ) {
        self.includeRepresentativeDocuments = includeRepresentativeDocuments
        self.maxRepresentativeDocuments = maxRepresentativeDocuments
        self.includeCentroids = includeCentroids
    }

    public static let `default` = TopicRepresentationOptions()

    public static let minimal = TopicRepresentationOptions(
        includeRepresentativeDocuments: false,
        includeCentroids: false
    )
}

// MARK: - Any Topic Representer

/// Type-erased wrapper for topic representers.
public struct AnyTopicRepresenter: Sendable {

    private let _represent: @Sendable ([Document], ClusterAssignment) async throws -> [Topic]
    private let _representWithEmbeddings: @Sendable ([Document], [Embedding], ClusterAssignment) async throws -> [Topic]

    /// Creates a type-erased topic representer.
    public init<R: TopicRepresenter>(_ representer: R) {
        self._represent = { documents, assignment in
            try await representer.represent(documents: documents, assignment: assignment)
        }
        self._representWithEmbeddings = { documents, embeddings, assignment in
            try await representer.represent(documents: documents, embeddings: embeddings, assignment: assignment)
        }
    }

    /// Extracts topic representations.
    public func represent(
        documents: [Document],
        assignment: ClusterAssignment
    ) async throws -> [Topic] {
        try await _represent(documents, assignment)
    }

    /// Extracts topic representations with embeddings.
    public func represent(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) async throws -> [Topic] {
        try await _representWithEmbeddings(documents, embeddings, assignment)
    }
}
