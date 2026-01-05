// DimensionReducer.swift
// SwiftTopics
//
// Protocol for dimensionality reduction algorithms

import Foundation

// MARK: - Dimension Reducer Protocol

/// A dimensionality reduction algorithm.
///
/// Reduces high-dimensional embeddings to a lower-dimensional representation
/// that preserves important structure for clustering.
///
/// ## Available Implementations
/// - `PCAReducer`: Principal Component Analysis (fast, linear)
/// - `UMAPReducer`: UMAP (slower, preserves local structure)
///
/// ## Why Reduce Dimensions?
/// 1. **Curse of Dimensionality**: Distance metrics become less meaningful in high dimensions
/// 2. **Performance**: Lower dimensions = faster clustering
/// 3. **Structure**: UMAP/t-SNE can reveal cluster structure not visible in high-dim
///
/// ## Thread Safety
/// Implementations must be safe to call from any thread.
public protocol DimensionReducer: Sendable {

    /// The output dimension after reduction.
    var outputDimension: Int { get }

    /// Whether this reducer has been fitted to data.
    var isFitted: Bool { get }

    /// Fits the reducer to training data and returns transformed embeddings.
    ///
    /// - Parameter embeddings: The training embeddings.
    /// - Returns: Reduced embeddings.
    /// - Throws: `ReductionError` if fitting fails.
    func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding]

    /// Transforms new embeddings using the fitted model.
    ///
    /// - Parameter embeddings: Embeddings to transform.
    /// - Returns: Reduced embeddings.
    /// - Throws: `ReductionError` if not fitted or transformation fails.
    func transform(_ embeddings: [Embedding]) async throws -> [Embedding]

    /// Fits the reducer without returning transformed data.
    ///
    /// - Parameter embeddings: The training embeddings.
    /// - Throws: `ReductionError` if fitting fails.
    mutating func fit(_ embeddings: [Embedding]) async throws
}

// MARK: - Reduction Error

/// Errors that can occur during dimensionality reduction.
public enum ReductionError: Error, Sendable {

    /// Input embeddings are empty.
    case emptyInput

    /// Input embeddings have inconsistent dimensions.
    case inconsistentDimensions

    /// The reducer has not been fitted yet.
    case notFitted

    /// The output dimension is larger than input dimension.
    case outputDimensionTooLarge(requested: Int, maxAllowed: Int)

    /// The input has too few samples for reduction.
    case insufficientSamples(required: Int, provided: Int)

    /// Numerical instability occurred (e.g., singular matrix).
    case numericalInstability(String)

    /// GPU acceleration failed.
    case gpuError(underlying: Error)

    /// Unknown error.
    case unknown(String)
}

extension ReductionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Cannot reduce empty embedding set"
        case .inconsistentDimensions:
            return "All embeddings must have the same dimension"
        case .notFitted:
            return "Dimension reducer must be fitted before transforming"
        case .outputDimensionTooLarge(let requested, let maxAllowed):
            return "Output dimension \(requested) exceeds maximum \(maxAllowed)"
        case .insufficientSamples(let required, let provided):
            return "Insufficient samples: need at least \(required), have \(provided)"
        case .numericalInstability(let message):
            return "Numerical instability: \(message)"
        case .gpuError(let underlying):
            return "GPU error during reduction: \(underlying.localizedDescription)"
        case .unknown(let message):
            return "Reduction error: \(message)"
        }
    }
}

// MARK: - Reduction Configuration

/// Configuration for dimensionality reduction.
public struct ReductionConfiguration: Sendable, Codable {

    /// The target output dimension.
    public let outputDimension: Int

    /// The reduction method to use.
    public let method: ReductionMethod

    /// Random seed for reproducibility.
    public let seed: UInt64?

    /// PCA-specific settings.
    public let pcaConfig: PCAConfiguration?

    /// UMAP-specific settings.
    public let umapConfig: UMAPConfiguration?

    /// Creates a reduction configuration.
    public init(
        outputDimension: Int = 15,
        method: ReductionMethod = .pca,
        seed: UInt64? = nil,
        pcaConfig: PCAConfiguration? = nil,
        umapConfig: UMAPConfiguration? = nil
    ) {
        self.outputDimension = outputDimension
        self.method = method
        self.seed = seed
        self.pcaConfig = pcaConfig
        self.umapConfig = umapConfig
    }

    /// Default configuration for fast processing.
    public static let fast = ReductionConfiguration(
        outputDimension: 10,
        method: .pca
    )

    /// Default configuration for quality.
    public static let quality = ReductionConfiguration(
        outputDimension: 15,
        method: .umap,
        umapConfig: .default
    )

    /// Default balanced configuration.
    public static let `default` = ReductionConfiguration(
        outputDimension: 15,
        method: .pca
    )
}

// MARK: - Reduction Method

/// The algorithm used for dimensionality reduction.
public enum ReductionMethod: String, Sendable, Codable {
    /// Principal Component Analysis - fast, linear.
    case pca

    /// UMAP - preserves local structure, non-linear.
    case umap

    /// No reduction - pass through original embeddings.
    case none
}

// MARK: - PCA Configuration

/// Configuration specific to PCA reduction.
public struct PCAConfiguration: Sendable, Codable {

    /// Whether to whiten the output (scale by eigenvalues).
    public let whiten: Bool

    /// Minimum variance ratio to retain (alternative to fixed dimensions).
    public let varianceRatio: Float?

    /// Creates PCA configuration.
    public init(whiten: Bool = false, varianceRatio: Float? = nil) {
        self.whiten = whiten
        self.varianceRatio = varianceRatio
    }

    public static let `default` = PCAConfiguration()
}

// MARK: - UMAP Configuration

/// Configuration specific to UMAP reduction.
public struct UMAPConfiguration: Sendable, Codable {

    /// Number of neighbors to consider for manifold approximation.
    /// Higher = more global structure, lower = more local structure.
    public let nNeighbors: Int

    /// Minimum distance between points in output space.
    /// Lower = tighter clusters, higher = more spread out.
    public let minDist: Float

    /// Distance metric for high-dimensional space.
    public let metric: DistanceMetricType

    /// Number of optimization epochs (nil = auto).
    public let nEpochs: Int?

    /// Learning rate for optimization.
    public let learningRate: Float

    /// Creates UMAP configuration.
    public init(
        nNeighbors: Int = 15,
        minDist: Float = 0.1,
        metric: DistanceMetricType = .euclidean,
        nEpochs: Int? = nil,
        learningRate: Float = 1.0
    ) {
        self.nNeighbors = nNeighbors
        self.minDist = minDist
        self.metric = metric
        self.nEpochs = nEpochs
        self.learningRate = learningRate
    }

    public static let `default` = UMAPConfiguration()

    /// Configuration prioritizing speed.
    public static let fast = UMAPConfiguration(
        nNeighbors: 10,
        nEpochs: 100
    )

    /// Configuration prioritizing quality.
    public static let quality = UMAPConfiguration(
        nNeighbors: 30,
        minDist: 0.05,
        nEpochs: 500
    )
}

// MARK: - Distance Metric Type

/// Distance metric types for various algorithms.
public enum DistanceMetricType: String, Sendable, Codable {
    /// Euclidean (L2) distance.
    case euclidean

    /// Cosine distance (1 - cosine similarity).
    case cosine

    /// Manhattan (L1) distance.
    case manhattan

    /// Dot product (for normalized vectors).
    case dotProduct
}

// MARK: - Reduction Result

/// Result of a dimensionality reduction operation.
public struct ReductionResult: Sendable {

    /// The reduced embeddings.
    public let embeddings: [Embedding]

    /// Original embedding dimension.
    public let originalDimension: Int

    /// Reduced embedding dimension.
    public let reducedDimension: Int

    /// Explained variance ratio (for PCA).
    public let explainedVarianceRatio: Float?

    /// Processing time in seconds.
    public let processingTime: TimeInterval

    /// Creates a reduction result.
    public init(
        embeddings: [Embedding],
        originalDimension: Int,
        reducedDimension: Int,
        explainedVarianceRatio: Float? = nil,
        processingTime: TimeInterval = 0
    ) {
        self.embeddings = embeddings
        self.originalDimension = originalDimension
        self.reducedDimension = reducedDimension
        self.explainedVarianceRatio = explainedVarianceRatio
        self.processingTime = processingTime
    }
}

// MARK: - Identity Reducer

/// A reducer that passes through embeddings unchanged.
///
/// Useful when you don't want dimensionality reduction, or for testing.
public struct IdentityReducer: DimensionReducer {

    private var fittedDimension: Int?

    public var outputDimension: Int {
        fittedDimension ?? 0
    }

    public var isFitted: Bool {
        fittedDimension != nil
    }

    public init() {}

    public func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        guard !embeddings.isEmpty else {
            throw ReductionError.emptyInput
        }
        return embeddings
    }

    public func transform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        embeddings
    }

    public mutating func fit(_ embeddings: [Embedding]) async throws {
        guard !embeddings.isEmpty else {
            throw ReductionError.emptyInput
        }
        fittedDimension = embeddings[0].dimension
    }
}
