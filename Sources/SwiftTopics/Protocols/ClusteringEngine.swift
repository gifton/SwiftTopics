// ClusteringEngine.swift
// SwiftTopics
//
// Protocol for clustering algorithms

import Foundation

// MARK: - Clustering Engine Protocol

/// A clustering algorithm that groups embeddings into topics.
///
/// The clustering engine is the core of topic discovery. It takes reduced
/// embeddings and assigns each to a cluster (topic) or marks it as an outlier.
///
/// ## Available Implementations
/// - `HDBSCANEngine`: Density-based clustering with automatic cluster count
///
/// ## Why HDBSCAN?
/// - **No predefined K**: Topic count emerges from data
/// - **Outlier detection**: Documents that don't fit are marked, not forced
/// - **Varying density**: Handles clusters of different densities
/// - **Arbitrary shapes**: Not limited to spherical clusters
///
/// ## Thread Safety
/// Implementations must be safe to call from any thread.
public protocol ClusteringEngine: Sendable {

    /// The type of configuration used by this engine.
    associatedtype Configuration: ClusteringConfiguration

    /// The configuration for this engine.
    var configuration: Configuration { get }

    /// Clusters the given embeddings.
    ///
    /// - Parameter embeddings: The embeddings to cluster.
    /// - Returns: Cluster assignments for each embedding.
    /// - Throws: `ClusteringError` if clustering fails.
    func fit(_ embeddings: [Embedding]) async throws -> ClusterAssignment

    /// Predicts cluster assignments for new embeddings.
    ///
    /// Uses the model fitted by `fit(_:)` to assign new points to clusters.
    /// Points that don't fit any cluster well are marked as outliers.
    ///
    /// - Parameter embeddings: New embeddings to assign.
    /// - Returns: Cluster assignments for each embedding.
    /// - Throws: `ClusteringError` if prediction fails.
    func predict(_ embeddings: [Embedding]) async throws -> ClusterAssignment
}

// MARK: - Clustering Configuration Protocol

/// Configuration for a clustering engine.
public protocol ClusteringConfiguration: Sendable, Codable {}

// MARK: - Clustering Error

/// Errors that can occur during clustering.
public enum ClusteringError: Error, Sendable {

    /// Input embeddings are empty.
    case emptyInput

    /// Input embeddings have inconsistent dimensions.
    case inconsistentDimensions

    /// Not enough points for the configured parameters.
    case insufficientPoints(required: Int, provided: Int)

    /// No clusters were found.
    case noClustersFound

    /// The clustering model has not been fitted.
    case notFitted

    /// Memory allocation failed for large dataset.
    case memoryAllocationFailed(required: Int)

    /// GPU acceleration failed.
    case gpuError(underlying: Error)

    /// Unknown error.
    case unknown(String)
}

extension ClusteringError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Cannot cluster empty embedding set"
        case .inconsistentDimensions:
            return "All embeddings must have the same dimension"
        case .insufficientPoints(let required, let provided):
            return "Insufficient points: need at least \(required), have \(provided)"
        case .noClustersFound:
            return "No clusters found in the data"
        case .notFitted:
            return "Clustering model must be fitted before prediction"
        case .memoryAllocationFailed(let required):
            return "Failed to allocate \(required) bytes for clustering"
        case .gpuError(let underlying):
            return "GPU error during clustering: \(underlying.localizedDescription)"
        case .unknown(let message):
            return "Clustering error: \(message)"
        }
    }
}

// MARK: - HDBSCAN Configuration

/// Configuration for HDBSCAN clustering.
///
/// ## Key Parameters
/// - `minClusterSize`: Minimum documents to form a topic (default: 5)
/// - `minSamples`: Core point threshold (default: same as minClusterSize)
/// - `clusterSelectionMethod`: How to extract flat clusters from hierarchy
///
/// ## Tuning Guide
/// - **Small corpus (<100 docs)**: Use `minClusterSize: 3`
/// - **Medium corpus (100-1000 docs)**: Use `minClusterSize: 5-10`
/// - **Large corpus (>1000 docs)**: Use `minClusterSize: 10-20`
public struct HDBSCANConfiguration: ClusteringConfiguration {

    /// Minimum number of points to form a cluster.
    ///
    /// Smaller = more clusters, larger = fewer, more robust clusters.
    public let minClusterSize: Int

    /// Minimum number of neighbors for a point to be considered a core point.
    ///
    /// If nil, defaults to `minClusterSize`.
    public let minSamples: Int?

    /// Distance threshold for merging clusters.
    ///
    /// Set to 0 for standard HDBSCAN behavior.
    /// Higher values merge small clusters into larger ones.
    public let clusterSelectionEpsilon: Float

    /// Method for extracting flat clusters from hierarchy.
    public let clusterSelectionMethod: ClusterSelectionMethod

    /// Whether to allow single-point clusters.
    public let allowSingleCluster: Bool

    /// Distance metric to use.
    public let metric: DistanceMetricType

    /// Random seed for reproducibility.
    public let seed: UInt64?

    /// Whether to log detailed timing information.
    ///
    /// When enabled, logs per-phase timing breakdown to the HDBSCAN logger.
    /// Useful for performance analysis and debugging.
    public let logTiming: Bool

    /// Creates HDBSCAN configuration.
    public init(
        minClusterSize: Int = 5,
        minSamples: Int? = nil,
        clusterSelectionEpsilon: Float = 0.0,
        clusterSelectionMethod: ClusterSelectionMethod = .eom,
        allowSingleCluster: Bool = false,
        metric: DistanceMetricType = .euclidean,
        seed: UInt64? = nil,
        logTiming: Bool = false
    ) {
        precondition(minClusterSize >= 2, "minClusterSize must be at least 2")
        precondition(minSamples ?? minClusterSize >= 1, "minSamples must be at least 1")
        precondition(clusterSelectionEpsilon >= 0, "clusterSelectionEpsilon must be non-negative")

        self.minClusterSize = minClusterSize
        self.minSamples = minSamples
        self.clusterSelectionEpsilon = clusterSelectionEpsilon
        self.clusterSelectionMethod = clusterSelectionMethod
        self.allowSingleCluster = allowSingleCluster
        self.metric = metric
        self.seed = seed
        self.logTiming = logTiming
    }

    /// The effective minSamples value.
    public var effectiveMinSamples: Int {
        minSamples ?? minClusterSize
    }

    /// Default configuration.
    public static let `default` = HDBSCANConfiguration()

    /// Configuration for small corpora.
    public static let small = HDBSCANConfiguration(
        minClusterSize: 3,
        minSamples: 2
    )

    /// Configuration for large corpora.
    public static let large = HDBSCANConfiguration(
        minClusterSize: 15,
        minSamples: 10
    )
}

// MARK: - Cluster Selection Method

/// Method for extracting flat clusters from the HDBSCAN hierarchy.
public enum ClusterSelectionMethod: String, Sendable, Codable {

    /// Excess of Mass - selects most stable clusters.
    ///
    /// Recommended for most use cases. Balances cluster persistence and size.
    case eom

    /// Leaf clustering - selects all leaf clusters.
    ///
    /// Produces more fine-grained clusters. May produce many small clusters.
    case leaf
}

// MARK: - Clustering Result

/// Extended result from clustering with additional diagnostics.
public struct ClusteringResult: Sendable {

    /// The cluster assignments.
    public let assignment: ClusterAssignment

    /// The cluster hierarchy (if available).
    public let hierarchy: ClusterHierarchy?

    /// Core distances for each point.
    public let coreDistances: [Float]?

    /// Processing time in seconds.
    public let processingTime: TimeInterval

    /// Detailed timing breakdown for each phase.
    ///
    /// Provides per-phase timing information for performance analysis.
    /// Only available when clustering was performed with timing enabled.
    public let timingBreakdown: HDBSCANTimingBreakdown?

    /// Creates a clustering result.
    public init(
        assignment: ClusterAssignment,
        hierarchy: ClusterHierarchy? = nil,
        coreDistances: [Float]? = nil,
        processingTime: TimeInterval = 0,
        timingBreakdown: HDBSCANTimingBreakdown? = nil
    ) {
        self.assignment = assignment
        self.hierarchy = hierarchy
        self.coreDistances = coreDistances
        self.processingTime = processingTime
        self.timingBreakdown = timingBreakdown
    }
}

// MARK: - Any Clustering Engine

/// Type-erased wrapper for clustering engines.
///
/// Allows storing different clustering engine types uniformly.
public struct AnyClusteringEngine: Sendable {

    private let _fit: @Sendable ([Embedding]) async throws -> ClusterAssignment
    private let _predict: @Sendable ([Embedding]) async throws -> ClusterAssignment

    /// Creates a type-erased clustering engine.
    public init<Engine: ClusteringEngine>(_ engine: Engine) {
        self._fit = { embeddings in
            try await engine.fit(embeddings)
        }
        self._predict = { embeddings in
            try await engine.predict(embeddings)
        }
    }

    /// Clusters the given embeddings.
    public func fit(_ embeddings: [Embedding]) async throws -> ClusterAssignment {
        try await _fit(embeddings)
    }

    /// Predicts cluster assignments for new embeddings.
    public func predict(_ embeddings: [Embedding]) async throws -> ClusterAssignment {
        try await _predict(embeddings)
    }
}
