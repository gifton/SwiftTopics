// HDBSCAN.swift
// SwiftTopics
//
// HDBSCAN clustering algorithm orchestrator

import Foundation
import os

/// Logger for HDBSCAN operations.
private let hdbscanLogger = Logger(subsystem: "SwiftTopics", category: "HDBSCAN")

// MARK: - HDBSCAN Engine

/// HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
/// clustering engine.
///
/// ## Algorithm Overview
///
/// HDBSCAN builds a hierarchy of clusters at all density levels, then extracts
/// a flat clustering that maximizes cluster stability. The algorithm:
///
/// 1. **Core Distance**: Compute k-th nearest neighbor distance for each point
/// 2. **Mutual Reachability**: Transform distances to be density-aware
/// 3. **Minimum Spanning Tree**: Build MST on mutual reachability graph
/// 4. **Cluster Hierarchy**: Convert MST to dendrogram with stability scores
/// 5. **Cluster Extraction**: Select stable clusters via EOM or leaf method
///
/// ## Advantages
///
/// - **No predefined K**: Cluster count emerges from data
/// - **Outlier detection**: Points in sparse regions are marked, not forced
/// - **Varying densities**: Handles clusters with different densities
/// - **Arbitrary shapes**: Not limited to spherical clusters
/// - **Soft clustering**: Provides membership probabilities
///
/// ## GPU Acceleration
///
/// Core distance computation uses GPU when available via `GPUBatchKNN`.
/// This provides significant speedup for n > 100 points.
///
/// ## Usage
///
/// ```swift
/// let config = HDBSCANConfiguration(minClusterSize: 5, minSamples: 3)
/// let engine = try await HDBSCANEngine(configuration: config)
/// let assignment = try await engine.fit(embeddings)
/// ```
///
/// ## Thread Safety
///
/// `HDBSCANEngine` is an actor and all methods are isolated.
public actor HDBSCANEngine: ClusteringEngine {

    // MARK: - Types

    public typealias Configuration = HDBSCANConfiguration

    // MARK: - Properties

    /// The configuration for this engine.
    public nonisolated let configuration: HDBSCANConfiguration

    /// GPU context for accelerated computation.
    private let gpuContext: TopicsGPUContext?

    /// Cached fitted model for prediction.
    private var fittedModel: FittedHDBSCANModel?

    // MARK: - Initialization

    /// Creates an HDBSCAN engine with configuration.
    ///
    /// - Parameters:
    ///   - configuration: HDBSCAN configuration.
    ///   - gpuContext: Optional GPU context (will attempt to create one if nil).
    public init(
        configuration: HDBSCANConfiguration = .default,
        gpuContext: TopicsGPUContext? = nil
    ) async throws {
        self.configuration = configuration

        // Try to create GPU context if not provided
        if let context = gpuContext {
            self.gpuContext = context
        } else {
            // Attempt to create GPU context, but don't fail if unavailable
            self.gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        }
    }

    /// Creates an HDBSCAN engine with default configuration.
    public init() async throws {
        try await self.init(configuration: .default)
    }

    // MARK: - Fit

    /// Clusters the given embeddings using HDBSCAN.
    ///
    /// This is the main entry point for clustering. It runs the full HDBSCAN
    /// pipeline and returns cluster assignments.
    ///
    /// - Parameter embeddings: The embeddings to cluster.
    /// - Returns: Cluster assignments for each embedding.
    /// - Throws: `ClusteringError` if clustering fails.
    public func fit(_ embeddings: [Embedding]) async throws -> ClusterAssignment {
        let result = try await fitWithDetails(embeddings)
        return result.assignment
    }

    /// Clusters embeddings and returns detailed results.
    ///
    /// - Parameter embeddings: The embeddings to cluster.
    /// - Returns: Clustering result with hierarchy and metadata.
    public func fitWithDetails(_ embeddings: [Embedding]) async throws -> ClusteringResult {
        let startTime = Date()

        // Validate input
        try validateInput(embeddings)

        let n = embeddings.count

        // Handle trivial cases
        if n == 1 {
            let assignment = ClusterAssignment(
                labels: [-1],
                probabilities: [0],
                outlierScores: [1],
                clusterCount: 0
            )
            return ClusteringResult(
                assignment: assignment,
                hierarchy: nil,
                coreDistances: [0],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        // Handle case where we have fewer points than minClusterSize
        // In this case, all points become outliers
        if n < configuration.minClusterSize {
            let assignment = ClusterAssignment(
                labels: [Int](repeating: -1, count: n),
                probabilities: [Float](repeating: 0, count: n),
                outlierScores: [Float](repeating: 1, count: n),
                clusterCount: 0
            )
            return ClusteringResult(
                assignment: assignment,
                hierarchy: nil,
                coreDistances: nil,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        // Steps 1-3: Compute core distances + mutual reachability + MST
        // GPU path handles all three steps in one efficient pipeline
        let coreDistances: [Float]
        let mst: MinimumSpanningTree
        var usedGPU = false

        // Get GPU threshold from configuration (default: 100)
        let gpuThreshold = gpuContext?.configuration.gpuMinPointsThreshold ?? 100

        let mstStartTime = Date()

        if let gpu = gpuContext, n >= gpuThreshold {
            // GPU path: Fused pipeline via VectorAccelerate's HDBSCANDistanceModule
            // Uses Borůvka's algorithm for O(log N) parallel MST construction
            //
            // Note: GPU HDBSCAN MST computation is atomic - it completes in a single
            // operation without intermediate checkpoints. For typical datasets (< 10K points),
            // this takes < 1 second, making checkpointing unnecessary.
            do {
                let gpuResult = try await gpu.computeHDBSCANMSTWithCoreDistances(
                    embeddings,
                    minSamples: configuration.effectiveMinSamples
                )
                coreDistances = gpuResult.coreDistances
                mst = gpuResult.mst
                usedGPU = true
            } catch {
                // GPU failed - fall back to CPU with warning
                hdbscanLogger.warning(
                    "GPU HDBSCAN computation failed, falling back to CPU: \(error.localizedDescription)"
                )
                (coreDistances, mst) = try await computeCoreDistancesAndMSTOnCPU(embeddings)
            }
        } else {
            // CPU path: For small datasets or when GPU unavailable
            (coreDistances, mst) = try await computeCoreDistancesAndMSTOnCPU(embeddings)
        }

        let mstTime = Date().timeIntervalSince(mstStartTime)

        // Step 4: Build cluster hierarchy
        let hierarchyStartTime = Date()
        let hierarchyBuilder = ClusterHierarchyBuilder(
            minClusterSize: configuration.minClusterSize
        )
        let hierarchy = hierarchyBuilder.build(
            from: mst,
            allowSingleCluster: configuration.allowSingleCluster
        )
        let hierarchyTime = Date().timeIntervalSince(hierarchyStartTime)

        // Step 5: Extract clusters
        let extractionStartTime = Date()
        let extractor = ClusterExtractor(
            method: configuration.clusterSelectionMethod,
            minClusterSize: configuration.minClusterSize,
            epsilon: configuration.clusterSelectionEpsilon,
            allowSingleCluster: configuration.allowSingleCluster
        )
        let assignment = extractor.extract(
            from: hierarchy,
            pointCount: n,
            coreDistances: coreDistances
        )
        let extractionTime = Date().timeIntervalSince(extractionStartTime)

        // Create timing breakdown
        let timingBreakdown = HDBSCANTimingBreakdown(
            coreDistanceTime: usedGPU ? mstTime * 0.3 : mstTime * 0.4,  // Estimated split
            mutualReachabilityTime: usedGPU ? mstTime * 0.2 : mstTime * 0.3,
            mstConstructionTime: usedGPU ? mstTime * 0.5 : mstTime * 0.3,
            hierarchyBuildTime: hierarchyTime,
            clusterExtractionTime: extractionTime,
            usedGPU: usedGPU,
            pointCount: n
        )

        // Log timing if enabled
        if configuration.logTiming {
            hdbscanLogger.info("\(timingBreakdown.summary)")
        }

        // Cache fitted model for prediction
        fittedModel = FittedHDBSCANModel(
            embeddings: embeddings,
            coreDistances: coreDistances,
            hierarchy: hierarchy,
            assignment: assignment
        )

        let processingTime = Date().timeIntervalSince(startTime)

        return ClusteringResult(
            assignment: assignment,
            hierarchy: hierarchy,
            coreDistances: coreDistances,
            processingTime: processingTime,
            timingBreakdown: timingBreakdown
        )
    }

    // MARK: - Predict

    /// Predicts cluster assignments for new embeddings.
    ///
    /// Uses the model fitted by `fit(_:)` to assign new points to clusters.
    /// Points that don't fit any cluster are marked as outliers.
    ///
    /// - Parameter embeddings: New embeddings to assign.
    /// - Returns: Cluster assignments.
    /// - Throws: `ClusteringError.notFitted` if model not fitted.
    public func predict(_ embeddings: [Embedding]) async throws -> ClusterAssignment {
        guard let model = fittedModel else {
            throw ClusteringError.notFitted
        }

        try validateInput(embeddings)

        // For each new point, find its nearest cluster
        var labels = [Int]()
        var probabilities = [Float]()
        var outlierScores = [Float]()

        labels.reserveCapacity(embeddings.count)
        probabilities.reserveCapacity(embeddings.count)
        outlierScores.reserveCapacity(embeddings.count)

        // Compute cluster centroids from fitted model
        let centroids = computeClusterCentroids(model: model)

        for embedding in embeddings {
            let (label, probability, outlierScore) = assignToCluster(
                embedding: embedding,
                model: model,
                centroids: centroids
            )
            labels.append(label)
            probabilities.append(probability)
            outlierScores.append(outlierScore)
        }

        return ClusterAssignment(
            labels: labels,
            probabilities: probabilities,
            outlierScores: outlierScores,
            clusterCount: model.assignment.clusterCount
        )
    }

    // MARK: - Validation

    private func validateInput(_ embeddings: [Embedding]) throws {
        guard !embeddings.isEmpty else {
            throw ClusteringError.emptyInput
        }

        let dimension = embeddings[0].dimension
        guard embeddings.allSatisfy({ $0.dimension == dimension }) else {
            throw ClusteringError.inconsistentDimensions
        }

        // Note: We don't throw for insufficient points here.
        // Instead, we handle small datasets gracefully by returning all points as outliers.
        // This allows prediction to work on any size input.
    }

    // MARK: - Core Distance and MST Computation

    /// Computes core distances and MST using the CPU path.
    ///
    /// This is the fallback path used when GPU is unavailable, for small datasets,
    /// or when GPU computation fails.
    ///
    /// - Parameter embeddings: The embeddings to process.
    /// - Returns: Tuple of (coreDistances, MST).
    private func computeCoreDistancesAndMSTOnCPU(
        _ embeddings: [Embedding]
    ) async throws -> ([Float], MinimumSpanningTree) {
        // Step 1: Compute core distances (may still use GPU for k-NN if available)
        let k = configuration.effectiveMinSamples
        let computer = CoreDistanceComputer(minSamples: k, preferGPU: true)
        let coreDistances = try await computer.compute(
            embeddings: embeddings,
            gpuContext: gpuContext
        )

        // Step 2: Build mutual reachability graph
        let mrGraph = MutualReachabilityGraph(
            embeddings: embeddings,
            coreDistances: coreDistances
        )

        // Step 3: Build MST using Prim's algorithm
        let mstBuilder = PrimMSTBuilder()
        let mst = mstBuilder.build(from: mrGraph)

        return (coreDistances, mst)
    }

    /// Computes core distances only (legacy method for compatibility).
    private func computeCoreDistances(_ embeddings: [Embedding]) async throws -> [Float] {
        let k = configuration.effectiveMinSamples
        let computer = CoreDistanceComputer(minSamples: k, preferGPU: true)

        return try await computer.compute(
            embeddings: embeddings,
            gpuContext: gpuContext
        )
    }

    // MARK: - Prediction Helpers

    private func computeClusterCentroids(model: FittedHDBSCANModel) -> [[Float]] {
        let k = model.assignment.clusterCount
        guard k > 0 else { return [] }

        var centroids = [[Float]](repeating: [], count: k)
        var counts = [Int](repeating: 0, count: k)

        for (i, embedding) in model.embeddings.enumerated() {
            let label = model.assignment.label(for: i)
            guard label >= 0 else { continue }

            if centroids[label].isEmpty {
                centroids[label] = [Float](repeating: 0, count: embedding.dimension)
            }

            for d in 0..<embedding.dimension {
                centroids[label][d] += embedding.vector[d]
            }
            counts[label] += 1
        }

        // Normalize
        for c in 0..<k {
            if counts[c] > 0 {
                for d in 0..<centroids[c].count {
                    centroids[c][d] /= Float(counts[c])
                }
            }
        }

        return centroids
    }

    private func assignToCluster(
        embedding: Embedding,
        model: FittedHDBSCANModel,
        centroids: [[Float]]
    ) -> (label: Int, probability: Float, outlierScore: Float) {
        guard !centroids.isEmpty else {
            return (-1, 0, 1)
        }

        // Find nearest centroid
        var minDist: Float = .infinity
        var nearestCluster = -1

        for (c, centroid) in centroids.enumerated() {
            guard !centroid.isEmpty else { continue }

            var dist: Float = 0
            for d in 0..<embedding.dimension {
                let diff = embedding.vector[d] - centroid[d]
                dist += diff * diff
            }
            dist = dist.squareRoot()

            if dist < minDist {
                minDist = dist
                nearestCluster = c
            }
        }

        // Determine if point should be an outlier
        // Compare distance to nearest cluster with cluster's core distances
        let avgCoreDistance = model.coreDistances.reduce(0, +) / Float(model.coreDistances.count)

        let isOutlier = minDist > avgCoreDistance * 2.0  // Heuristic threshold

        if isOutlier {
            let outlierScore = min(1.0, minDist / (avgCoreDistance * 2.0))
            return (-1, 0, outlierScore)
        } else {
            // Probability based on distance to centroid
            let probability = max(0.1, 1.0 - minDist / (avgCoreDistance * 2.0))
            let outlierScore = minDist / (avgCoreDistance * 2.0)
            return (nearestCluster, probability, outlierScore)
        }
    }
}

// MARK: - Fitted Model

/// A fitted HDBSCAN model for prediction.
private struct FittedHDBSCANModel: Sendable {

    /// Original embeddings used for fitting.
    let embeddings: [Embedding]

    /// Core distances for each point.
    let coreDistances: [Float]

    /// The cluster hierarchy.
    let hierarchy: ClusterHierarchy

    /// The cluster assignment.
    let assignment: ClusterAssignment
}

// MARK: - HDBSCAN Convenience Functions

/// Convenience function for one-shot HDBSCAN clustering.
///
/// Creates an engine, fits, and returns the assignment in one call.
///
/// - Parameters:
///   - embeddings: The embeddings to cluster.
///   - configuration: HDBSCAN configuration (default: .default).
/// - Returns: Cluster assignments.
public func hdbscan(
    _ embeddings: [Embedding],
    configuration: HDBSCANConfiguration = .default
) async throws -> ClusterAssignment {
    let engine = try await HDBSCANEngine(configuration: configuration)
    return try await engine.fit(embeddings)
}

/// Convenience function for one-shot HDBSCAN with detailed results.
///
/// - Parameters:
///   - embeddings: The embeddings to cluster.
///   - configuration: HDBSCAN configuration.
/// - Returns: Clustering result with full details.
public func hdbscanWithDetails(
    _ embeddings: [Embedding],
    configuration: HDBSCANConfiguration = .default
) async throws -> ClusteringResult {
    let engine = try await HDBSCANEngine(configuration: configuration)
    return try await engine.fitWithDetails(embeddings)
}

// MARK: - HDBSCAN Builder

/// Builder for configuring and creating HDBSCAN engines.
///
/// Provides a fluent interface for setting HDBSCAN parameters.
///
/// ```swift
/// let engine = try await HDBSCANBuilder()
///     .minClusterSize(10)
///     .minSamples(5)
///     .selectionMethod(.eom)
///     .build()
/// ```
public struct HDBSCANBuilder: Sendable {

    private var minClusterSize: Int = 5
    private var minSamples: Int? = nil
    private var epsilon: Float = 0.0
    private var selectionMethod: ClusterSelectionMethod = .eom
    private var allowSingleCluster: Bool = false
    private var metric: DistanceMetricType = .euclidean
    private var seed: UInt64? = nil

    /// Creates a new builder with default settings.
    public init() {}

    /// Sets the minimum cluster size.
    public func minClusterSize(_ size: Int) -> HDBSCANBuilder {
        var copy = self
        copy.minClusterSize = size
        return copy
    }

    /// Sets the minimum samples for core points.
    public func minSamples(_ samples: Int) -> HDBSCANBuilder {
        var copy = self
        copy.minSamples = samples
        return copy
    }

    /// Sets the cluster selection epsilon.
    public func epsilon(_ eps: Float) -> HDBSCANBuilder {
        var copy = self
        copy.epsilon = eps
        return copy
    }

    /// Sets the cluster selection method.
    public func selectionMethod(_ method: ClusterSelectionMethod) -> HDBSCANBuilder {
        var copy = self
        copy.selectionMethod = method
        return copy
    }

    /// Sets whether to allow a single cluster result.
    public func allowSingleCluster(_ allow: Bool) -> HDBSCANBuilder {
        var copy = self
        copy.allowSingleCluster = allow
        return copy
    }

    /// Sets the distance metric.
    public func metric(_ m: DistanceMetricType) -> HDBSCANBuilder {
        var copy = self
        copy.metric = m
        return copy
    }

    /// Sets the random seed for reproducibility.
    public func seed(_ s: UInt64) -> HDBSCANBuilder {
        var copy = self
        copy.seed = s
        return copy
    }

    /// Builds the configuration.
    public func buildConfiguration() -> HDBSCANConfiguration {
        HDBSCANConfiguration(
            minClusterSize: minClusterSize,
            minSamples: minSamples,
            clusterSelectionEpsilon: epsilon,
            clusterSelectionMethod: selectionMethod,
            allowSingleCluster: allowSingleCluster,
            metric: metric,
            seed: seed
        )
    }

    /// Builds the HDBSCAN engine.
    public func build() async throws -> HDBSCANEngine {
        try await HDBSCANEngine(configuration: buildConfiguration())
    }
}

// MARK: - Embeddings Extension

extension Array where Element == Embedding {

    /// Clusters these embeddings using HDBSCAN.
    ///
    /// - Parameter configuration: HDBSCAN configuration.
    /// - Returns: Cluster assignments.
    public func cluster(
        using configuration: HDBSCANConfiguration = .default
    ) async throws -> ClusterAssignment {
        try await hdbscan(self, configuration: configuration)
    }

    /// Clusters these embeddings and returns detailed results.
    ///
    /// - Parameter configuration: HDBSCAN configuration.
    /// - Returns: Clustering result with hierarchy.
    public func clusterWithDetails(
        using configuration: HDBSCANConfiguration = .default
    ) async throws -> ClusteringResult {
        try await hdbscanWithDetails(self, configuration: configuration)
    }
}
