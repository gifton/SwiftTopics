// TopicModelConfiguration.swift
// SwiftTopics
//
// Configuration for the TopicModel orchestrator

import Foundation

// MARK: - Topic Model Configuration

/// Configuration for the complete topic modeling pipeline.
///
/// ## Components
///
/// The configuration combines settings for each pipeline stage:
/// - **Reduction**: Dimensionality reduction (PCA/UMAP) before clustering
/// - **Clustering**: HDBSCAN parameters for topic discovery
/// - **Representation**: c-TF-IDF keyword extraction settings
/// - **Coherence**: Optional NPMI coherence evaluation
///
/// ## Presets
///
/// Three presets are provided for common use cases:
/// - `.default`: Balanced quality and speed
/// - `.fast`: Prioritizes speed (fewer components, skip coherence)
/// - `.quality`: Prioritizes topic quality (more components, stricter clustering)
///
/// ## Custom Configuration
///
/// ```swift
/// let config = TopicModelConfiguration(
///     reduction: ReductionConfiguration(outputDimension: 30, method: .pca),
///     clustering: HDBSCANConfiguration(minClusterSize: 8),
///     representation: CTFIDFConfiguration(keywordsPerTopic: 12, diversify: true),
///     coherence: .default,
///     seed: 42
/// )
/// ```
///
/// ## Thread Safety
///
/// `TopicModelConfiguration` is `Sendable` and `Codable`.
public struct TopicModelConfiguration: Sendable, Codable {

    // MARK: - Properties

    /// Dimensionality reduction configuration.
    ///
    /// Controls how high-dimensional embeddings are reduced before clustering.
    /// PCA is faster, UMAP preserves local structure better.
    public let reduction: ReductionConfiguration

    /// Clustering configuration (HDBSCAN parameters).
    ///
    /// Controls how documents are grouped into topics.
    public let clustering: HDBSCANConfiguration

    /// Topic representation configuration (c-TF-IDF).
    ///
    /// Controls how keywords are extracted for each topic.
    public let representation: CTFIDFConfiguration

    /// Coherence evaluation configuration.
    ///
    /// If nil, coherence evaluation is skipped.
    public let coherence: CoherenceConfiguration?

    /// Random seed for reproducibility.
    ///
    /// When set, the pipeline produces deterministic results.
    public let seed: UInt64?

    // MARK: - Initialization

    /// Creates a topic model configuration.
    ///
    /// - Parameters:
    ///   - reduction: Dimensionality reduction settings.
    ///   - clustering: HDBSCAN clustering settings.
    ///   - representation: c-TF-IDF representation settings.
    ///   - coherence: Coherence evaluation settings (nil to skip).
    ///   - seed: Random seed for reproducibility.
    public init(
        reduction: ReductionConfiguration = .default,
        clustering: HDBSCANConfiguration = .default,
        representation: CTFIDFConfiguration = .default,
        coherence: CoherenceConfiguration? = .default,
        seed: UInt64? = nil
    ) {
        self.reduction = reduction
        self.clustering = clustering
        self.representation = representation
        self.coherence = coherence
        self.seed = seed
    }

    // MARK: - Presets

    /// Default configuration: balanced quality and speed.
    ///
    /// - Reduction: PCA to 15 dimensions
    /// - Clustering: minClusterSize = 5
    /// - Representation: 10 keywords per topic
    /// - Coherence: Enabled with default settings
    public static let `default` = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
        clustering: HDBSCANConfiguration(minClusterSize: 5),
        representation: CTFIDFConfiguration(keywordsPerTopic: 10),
        coherence: .default,
        seed: nil
    )

    /// Fast configuration: prioritizes speed over quality.
    ///
    /// - Reduction: PCA to 10 dimensions
    /// - Clustering: minClusterSize = 3, minSamples = 2
    /// - Representation: 5 keywords per topic
    /// - Coherence: Disabled (skip for speed)
    public static let fast = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 10, method: .pca),
        clustering: HDBSCANConfiguration(minClusterSize: 3, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,  // Skip coherence for speed
        seed: nil
    )

    /// Quality configuration: prioritizes topic quality.
    ///
    /// - Reduction: PCA to 25 dimensions
    /// - Clustering: minClusterSize = 10, EOM selection
    /// - Representation: 15 keywords with diversification
    /// - Coherence: Enabled with larger window
    public static let quality = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 25, method: .pca),
        clustering: HDBSCANConfiguration(
            minClusterSize: 10,
            minSamples: 5,
            clusterSelectionMethod: .eom
        ),
        representation: CTFIDFConfiguration(
            keywordsPerTopic: 15,
            diversify: true,
            diversityWeight: 0.3
        ),
        coherence: CoherenceConfiguration(windowSize: 20, topKeywords: 15),
        seed: nil
    )

    /// Configuration for small corpora (< 100 documents).
    ///
    /// Uses smaller cluster sizes to find topics in limited data.
    public static let smallCorpus = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 10, method: .pca),
        clustering: HDBSCANConfiguration(
            minClusterSize: 3,
            minSamples: 2,
            allowSingleCluster: true
        ),
        representation: CTFIDFConfiguration(keywordsPerTopic: 8),
        coherence: .default,
        seed: nil
    )

    /// Configuration for large corpora (> 10,000 documents).
    ///
    /// Uses larger cluster sizes for more coherent topics.
    public static let largeCorpus = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 30, method: .pca),
        clustering: HDBSCANConfiguration(
            minClusterSize: 20,
            minSamples: 10,
            clusterSelectionMethod: .eom
        ),
        representation: CTFIDFConfiguration(
            keywordsPerTopic: 15,
            minDocumentFrequency: 3
        ),
        coherence: CoherenceConfiguration(windowSize: 15, topKeywords: 10),
        seed: nil
    )

    // MARK: - Validation

    /// Validates the configuration for consistency.
    ///
    /// - Throws: `TopicModelError.invalidConfiguration` if validation fails.
    public func validate() throws {
        // Reduction dimension must be positive
        guard reduction.outputDimension > 0 else {
            throw TopicModelError.invalidConfiguration(
                "Reduction output dimension must be positive"
            )
        }

        // Min cluster size must be at least 2
        guard clustering.minClusterSize >= 2 else {
            throw TopicModelError.invalidConfiguration(
                "Minimum cluster size must be at least 2"
            )
        }

        // Keywords per topic must be positive
        guard representation.keywordsPerTopic > 0 else {
            throw TopicModelError.invalidConfiguration(
                "Keywords per topic must be positive"
            )
        }

        // Coherence topKeywords should not exceed representation keywords
        if let coherenceConfig = coherence {
            if coherenceConfig.topKeywords > representation.keywordsPerTopic {
                throw TopicModelError.invalidConfiguration(
                    "Coherence topKeywords (\(coherenceConfig.topKeywords)) exceeds representation keywordsPerTopic (\(representation.keywordsPerTopic))"
                )
            }
        }
    }

    // MARK: - Snapshot

    /// Creates a snapshot for storage with results.
    public func toSnapshot() -> TopicModelConfigurationSnapshot {
        TopicModelConfigurationSnapshot(
            reductionMethod: reduction.method.rawValue,
            reducedDimensions: reduction.outputDimension,
            clusteringAlgorithm: "HDBSCAN",
            minClusterSize: clustering.minClusterSize,
            minSamples: clustering.effectiveMinSamples,
            clusterSelectionMethod: clustering.clusterSelectionMethod.rawValue,
            keywordsPerTopic: representation.keywordsPerTopic
        )
    }
}

// MARK: - Builder

/// Builder for creating topic model configurations.
///
/// Provides a fluent interface for configuration:
/// ```swift
/// let config = TopicModelConfigurationBuilder()
///     .reductionDimension(20)
///     .minClusterSize(8)
///     .keywordsPerTopic(12)
///     .enableCoherence(true)
///     .build()
/// ```
public struct TopicModelConfigurationBuilder: Sendable {

    private var reductionDimension: Int = 15
    private var reductionMethod: ReductionMethod = .pca
    private var minClusterSize: Int = 5
    private var minSamples: Int? = nil
    private var clusterSelectionMethod: ClusterSelectionMethod = .eom
    private var keywordsPerTopic: Int = 10
    private var diversify: Bool = false
    private var coherenceEnabled: Bool = true
    private var coherenceTopKeywords: Int = 10
    private var seed: UInt64? = nil

    /// Creates a new configuration builder.
    public init() {}

    /// Sets the reduction output dimension.
    public func reductionDimension(_ dimension: Int) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.reductionDimension = dimension
        return copy
    }

    /// Sets the reduction method.
    public func reductionMethod(_ method: ReductionMethod) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.reductionMethod = method
        return copy
    }

    /// Sets the minimum cluster size.
    public func minClusterSize(_ size: Int) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.minClusterSize = size
        return copy
    }

    /// Sets the minimum samples for core points.
    public func minSamples(_ samples: Int) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.minSamples = samples
        return copy
    }

    /// Sets the cluster selection method.
    public func clusterSelectionMethod(_ method: ClusterSelectionMethod) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.clusterSelectionMethod = method
        return copy
    }

    /// Sets the number of keywords per topic.
    public func keywordsPerTopic(_ count: Int) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.keywordsPerTopic = count
        return copy
    }

    /// Enables keyword diversification.
    public func diversify(_ enable: Bool = true) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.diversify = enable
        return copy
    }

    /// Enables or disables coherence evaluation.
    public func enableCoherence(_ enable: Bool) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.coherenceEnabled = enable
        return copy
    }

    /// Sets the number of keywords for coherence evaluation.
    public func coherenceTopKeywords(_ count: Int) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.coherenceTopKeywords = count
        return copy
    }

    /// Sets the random seed.
    public func seed(_ s: UInt64) -> TopicModelConfigurationBuilder {
        var copy = self
        copy.seed = s
        return copy
    }

    /// Builds the configuration.
    public func build() -> TopicModelConfiguration {
        TopicModelConfiguration(
            reduction: ReductionConfiguration(
                outputDimension: reductionDimension,
                method: reductionMethod,
                seed: seed
            ),
            clustering: HDBSCANConfiguration(
                minClusterSize: minClusterSize,
                minSamples: minSamples,
                clusterSelectionMethod: clusterSelectionMethod,
                seed: seed
            ),
            representation: CTFIDFConfiguration(
                keywordsPerTopic: keywordsPerTopic,
                diversify: diversify
            ),
            coherence: coherenceEnabled
                ? CoherenceConfiguration(topKeywords: coherenceTopKeywords)
                : nil,
            seed: seed
        )
    }
}

// MARK: - Topic Model Error

/// Errors that can occur during topic modeling.
public enum TopicModelError: Error, Sendable {

    /// The model has not been fitted yet.
    case notFitted

    /// Invalid input provided.
    case invalidInput(String)

    /// Embedding dimension mismatch.
    case embeddingDimensionMismatch(expected: Int, got: Int)

    /// No topics were discovered.
    case noTopicsDiscovered

    /// Configuration is invalid.
    case invalidConfiguration(String)

    /// Serialization failed.
    case serializationFailed(String)

    /// Pipeline stage failed.
    case pipelineError(stage: String, underlying: Error)

    /// No embedding provider is available.
    ///
    /// This occurs when calling methods like `findTopics(for:)` on a model
    /// that was fitted with pre-computed embeddings rather than an embedding provider.
    case noEmbeddingProvider
}

extension TopicModelError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .notFitted:
            return "Topic model has not been fitted"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .embeddingDimensionMismatch(let expected, let got):
            return "Embedding dimension mismatch: expected \(expected), got \(got)"
        case .noTopicsDiscovered:
            return "No topics were discovered in the data"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .serializationFailed(let message):
            return "Serialization failed: \(message)"
        case .pipelineError(let stage, let underlying):
            return "Pipeline error in \(stage): \(underlying.localizedDescription)"
        case .noEmbeddingProvider:
            return "No embedding provider available. Model was fitted with pre-computed embeddings."
        }
    }
}
