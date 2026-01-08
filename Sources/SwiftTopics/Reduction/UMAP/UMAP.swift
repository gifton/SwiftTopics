// UMAP.swift
// SwiftTopics
//
// UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction

import Foundation
import os

/// Logger for UMAP operations.
private let umapLogger = Logger(subsystem: "SwiftTopics", category: "UMAP")

// MARK: - UMAP Reducer

/// UMAP dimensionality reduction algorithm.
///
/// UMAP is a non-linear dimensionality reduction technique that preserves
/// both local and global structure of the data. It's particularly effective
/// as a preprocessing step before HDBSCAN clustering.
///
/// ## Algorithm Overview
///
/// 1. **k-NN Graph**: Find k nearest neighbors for each point
/// 2. **Fuzzy Set**: Convert distances to membership strengths
/// 3. **Initialization**: Spectral embedding or random positions
/// 4. **Optimization**: SGD with attractive/repulsive forces
///
/// ## Why UMAP for Topic Modeling?
///
/// - **Preserves clusters**: Documents in the same topic stay together
/// - **Handles high dimensions**: Works well with 768+ dim embeddings
/// - **Non-linear**: Can "unfold" complex manifold structures
/// - **Faster than t-SNE**: Better scalability to large datasets
///
/// ## Usage
///
/// ```swift
/// // Basic usage
/// var umap = UMAPReducer(configuration: .default)
/// let reduced = try await umap.fitTransform(embeddings)
///
/// // Custom configuration with GPU acceleration
/// let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
/// let config = UMAPConfiguration(
///     nNeighbors: 20,
///     minDist: 0.1,
///     nComponents: 15
/// )
/// var umap = UMAPReducer(configuration: config, gpuContext: gpuContext)
/// let reduced = try await umap.fitTransform(embeddings)
/// ```
///
/// ## GPU Acceleration
///
/// When a `gpuContext` is provided, UMAP uses VectorAccelerate's GPU kernels for
/// the optimization step, providing 10-50x speedup for datasets with 100+ points.
/// GPU is automatically used for optimization when:
/// - A valid `gpuContext` is provided
/// - The dataset has ≥100 points
///
/// If GPU fails, UMAP automatically falls back to CPU with a warning log.
///
/// ## Thread Safety
/// `UMAPReducer` is `Sendable`. The mutable fitting state is managed
/// through the protocol's mutating methods.
public struct UMAPReducer: DimensionReducer, Sendable {

    // MARK: - Properties

    /// UMAP configuration.
    public let configuration: UMAPConfiguration

    /// The fitted state (nil before fitting).
    private var fittedState: FittedUMAPState?

    /// GPU context for acceleration (optional).
    public let gpuContext: TopicsGPUContext?

    // MARK: - DimensionReducer Protocol

    /// The output dimension after reduction.
    public var outputDimension: Int {
        fittedState?.outputDimension ?? (_nComponents ?? 15)
    }

    /// Effective number of output components.
    private var effectiveNComponents: Int {
        _nComponents ?? 15
    }

    /// Effective random seed.
    private var effectiveSeed: UInt64? {
        _seed
    }

    /// Whether this reducer has been fitted to data.
    public var isFitted: Bool {
        fittedState != nil
    }

    // MARK: - Initialization

    /// Creates a UMAP reducer with the specified configuration.
    ///
    /// - Parameters:
    ///   - configuration: UMAP configuration.
    ///   - nComponents: Output dimensionality (default: 15).
    ///   - seed: Random seed for reproducibility.
    ///   - gpuContext: Optional GPU context for acceleration.
    ///
    /// - Important: When using GPU acceleration, use `.gpuOptimized` instead of `.default`
    ///   for full speedup. The default configuration uses spectral initialization which is
    ///   O(n³) CPU-only and will bottleneck performance. Example:
    ///   ```swift
    ///   let reducer = UMAPReducer(
    ///       configuration: .gpuOptimized,  // NOT .default
    ///       gpuContext: gpuContext
    ///   )
    ///   ```
    public init(
        configuration: UMAPConfiguration = .default,
        nComponents: Int = 15,
        seed: UInt64? = nil,
        gpuContext: TopicsGPUContext? = nil
    ) {
        self.configuration = configuration
        self._nComponents = nComponents
        self._seed = seed
        self.gpuContext = gpuContext
    }

    /// Creates a UMAP reducer with individual parameters.
    ///
    /// - Parameters:
    ///   - nNeighbors: Number of neighbors for manifold approximation.
    ///   - minDist: Minimum distance between points in output space.
    ///   - nComponents: Output dimensionality.
    ///   - metric: Distance metric.
    ///   - nEpochs: Number of optimization epochs (nil = auto).
    ///   - seed: Random seed for reproducibility.
    ///   - gpuContext: Optional GPU context for acceleration.
    ///   - initialization: Embedding initialization strategy. Use `.pca` or `.random`
    ///     with GPU for best performance. Defaults to `.spectral` for backward compatibility.
    ///
    /// - Note: When providing `gpuContext`, consider using `.pca` initialization
    ///   to avoid the O(n³) spectral embedding bottleneck.
    public init(
        nNeighbors: Int = 15,
        minDist: Float = 0.1,
        nComponents: Int = 15,
        metric: DistanceMetricType = .euclidean,
        nEpochs: Int? = nil,
        seed: UInt64? = nil,
        gpuContext: TopicsGPUContext? = nil,
        initialization: UMAPInitialization = .spectral
    ) {
        self.configuration = UMAPConfiguration(
            nNeighbors: nNeighbors,
            minDist: minDist,
            metric: metric,
            nEpochs: nEpochs,
            learningRate: 1.0,
            initialization: initialization
        )
        self._nComponents = nComponents
        self._seed = seed
        self.gpuContext = gpuContext
    }

    /// Output dimension override (since UMAPConfiguration doesn't have this).
    private let _nComponents: Int?

    /// Seed override (since UMAPConfiguration doesn't have this).
    private let _seed: UInt64?

    // MARK: - Fit Transform

    /// Fits UMAP to the data and returns the reduced embeddings.
    ///
    /// This is the main entry point for UMAP. It performs all steps:
    /// 1. Build k-NN graph
    /// 2. Construct fuzzy simplicial set
    /// 3. Initialize embedding (spectral or random)
    /// 4. Optimize with SGD
    ///
    /// - Parameter embeddings: Input embeddings.
    /// - Returns: Reduced embeddings.
    /// - Throws: `ReductionError` if reduction fails.
    public func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        var mutableSelf = self
        try await mutableSelf.fit(embeddings)
        return try await mutableSelf.transform(embeddings)
    }

    // MARK: - Fit

    /// Fits UMAP to the training data.
    ///
    /// After fitting, `transform` can be called on new data.
    ///
    /// - Parameter embeddings: Training embeddings.
    /// - Throws: `ReductionError` if fitting fails.
    public mutating func fit(_ embeddings: [Embedding]) async throws {
        // Validate input
        guard !embeddings.isEmpty else {
            throw ReductionError.emptyInput
        }

        // Warn if GPU is provided but spectral initialization is used
        // Spectral is O(n³) and CPU-only, negating most GPU benefits
        if gpuContext != nil && configuration.initialization == .spectral {
            umapLogger.warning("""
                UMAP using spectral initialization with GPU context. \
                Spectral embedding is O(n³) CPU-only and will bottleneck performance. \
                For full GPU acceleration, use .gpuOptimized or set initialization to .pca or .random.
                """)
        }

        let n = embeddings.count
        let inputDimension = embeddings[0].dimension

        guard embeddings.allSatisfy({ $0.dimension == inputDimension }) else {
            throw ReductionError.inconsistentDimensions
        }

        // Need at least k+1 points
        let actualK = min(configuration.nNeighbors, n - 1)
        guard actualK >= 1 else {
            throw ReductionError.insufficientSamples(
                required: configuration.nNeighbors + 1,
                provided: n
            )
        }

        // Convert distance metric
        let metric = convertMetric(configuration.metric)

        // Step 1: Build k-NN graph (GPU accelerated when gpuContext available)
        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: actualK,
            metric: metric,
            gpuContext: gpuContext
        )

        // Step 2: Construct fuzzy simplicial set
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

        // Step 3: Initialize embedding based on configuration
        var initialEmbedding: [[Float]]

        switch configuration.initialization {
        case .spectral:
            // O(n³) but best quality - use for small datasets
            initialEmbedding = try SpectralEmbedding.compute(
                adjacency: fuzzySet.memberships,
                nComponents: effectiveNComponents,
                seed: effectiveSeed
            )
            // Handle disconnected components for spectral init
            SpectralEmbedding.handleDisconnectedComponents(
                embedding: &initialEmbedding,
                graph: knnGraph,
                seed: effectiveSeed
            )

        case .pca:
            // O(n×d²) - good balance of quality and speed
            initialEmbedding = try await SpectralEmbedding.pcaInitialization(
                embeddings: embeddings,
                nComponents: effectiveNComponents
            )

        case .random:
            // O(n) - fastest, may need more epochs
            initialEmbedding = SpectralEmbedding.randomInitialization(
                pointCount: n,
                nComponents: effectiveNComponents,
                seed: effectiveSeed
            )
        }

        // Step 4: Optimize (GPU or CPU)
        let nEpochs = configuration.nEpochs ?? computeAutoEpochs(n: n)

        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: configuration.minDist,
            seed: effectiveSeed
        )

        let optimizedEmbedding: [[Float]]

        // Use GPU for datasets >= threshold when GPU context available
        let gpuThreshold = gpuContext?.configuration.gpuMinPointsThreshold ?? 100
        if let gpu = gpuContext, n >= gpuThreshold {
            do {
                optimizedEmbedding = try await optimizer.optimizeGPU(
                    fuzzySet: fuzzySet,
                    nEpochs: nEpochs,
                    learningRate: configuration.learningRate,
                    negativeSampleRate: 5,
                    gpuContext: gpu
                )
            } catch {
                // GPU failed - fall back to CPU with warning
                umapLogger.warning(
                    "GPU UMAP optimization failed, falling back to CPU: \(error.localizedDescription)"
                )
                optimizedEmbedding = await optimizer.optimize(
                    fuzzySet: fuzzySet,
                    nEpochs: nEpochs,
                    learningRate: configuration.learningRate,
                    negativeSampleRate: 5
                )
            }
        } else {
            // CPU path: For small datasets or when GPU unavailable
            optimizedEmbedding = await optimizer.optimize(
                fuzzySet: fuzzySet,
                nEpochs: nEpochs,
                learningRate: configuration.learningRate,
                negativeSampleRate: 5
            )
        }

        // Store fitted state for transform
        fittedState = FittedUMAPState(
            knnGraph: knnGraph,
            trainingEmbeddings: embeddings,
            reducedEmbeddings: optimizedEmbedding,
            inputDimension: inputDimension,
            outputDimension: effectiveNComponents
        )
    }

    // MARK: - Transform

    /// Transforms embeddings using the fitted UMAP model.
    ///
    /// For new points, UMAP uses an approximate transform based on
    /// finding the nearest training points and interpolating their
    /// low-dimensional positions.
    ///
    /// - Parameter embeddings: Embeddings to transform.
    /// - Returns: Reduced embeddings.
    /// - Throws: `ReductionError` if not fitted or dimensions mismatch.
    public func transform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        guard let state = fittedState else {
            throw ReductionError.notFitted
        }

        guard !embeddings.isEmpty else {
            return []
        }

        let d = embeddings[0].dimension
        guard d == state.inputDimension else {
            throw ReductionError.inconsistentDimensions
        }

        guard embeddings.allSatisfy({ $0.dimension == d }) else {
            throw ReductionError.inconsistentDimensions
        }

        // Check if these are the training embeddings
        if embeddings.count == state.trainingEmbeddings.count {
            // Quick check: first and last match
            let firstMatch = embeddings[0].vector == state.trainingEmbeddings[0].vector
            let lastMatch = embeddings[embeddings.count - 1].vector ==
                           state.trainingEmbeddings[embeddings.count - 1].vector
            if firstMatch && lastMatch {
                // Return the fitted embedding
                return state.reducedEmbeddings.map { Embedding(vector: $0) }
            }
        }

        // Transform new points using nearest neighbor interpolation
        return await transformNewPoints(embeddings, state: state)
    }

    // MARK: - New Point Transform

    /// Transforms new points using nearest neighbor weighted average.
    private func transformNewPoints(
        _ embeddings: [Embedding],
        state: FittedUMAPState
    ) async -> [Embedding] {
        let metric = convertMetric(configuration.metric)
        let k = min(configuration.nNeighbors, state.trainingEmbeddings.count)

        // Build spatial index for training data
        let trainingPoints = state.trainingEmbeddings.map { $0.vector }

        let ballTree: BallTree
        do {
            let config = SpatialIndexConfiguration(metric: metric)
            ballTree = try BallTree.build(points: trainingPoints, configuration: config)
        } catch {
            // Fallback: use brute force
            return embeddings.map { embedding in
                bruteForceTransform(
                    embedding: embedding,
                    training: state.trainingEmbeddings,
                    reduced: state.reducedEmbeddings,
                    k: k,
                    metric: metric
                )
            }
        }

        // Transform each new point
        return embeddings.map { embedding in
            let neighbors = ballTree.query(point: embedding.vector, k: k)
            return interpolateFromNeighbors(
                neighbors: neighbors,
                reducedTraining: state.reducedEmbeddings
            )
        }
    }

    /// Interpolates low-dim position from k nearest neighbors.
    private func interpolateFromNeighbors(
        neighbors: [(index: Int, distance: Float)],
        reducedTraining: [[Float]]
    ) -> Embedding {
        guard !neighbors.isEmpty else {
            return Embedding(vector: [Float](repeating: 0, count: effectiveNComponents))
        }

        // Weighted average based on inverse distance
        let nComponents = reducedTraining[0].count
        var result = [Float](repeating: 0, count: nComponents)
        var totalWeight: Float = 0

        for (idx, dist) in neighbors {
            let weight = 1.0 / (dist + 0.001)
            totalWeight += weight

            for d in 0..<nComponents {
                result[d] += weight * reducedTraining[idx][d]
            }
        }

        if totalWeight > 0 {
            for d in 0..<nComponents {
                result[d] /= totalWeight
            }
        }

        return Embedding(vector: result)
    }

    /// Brute force transform fallback.
    private func bruteForceTransform(
        embedding: Embedding,
        training: [Embedding],
        reduced: [[Float]],
        k: Int,
        metric: DistanceMetric
    ) -> Embedding {
        // Compute all distances
        var distances: [(index: Int, distance: Float)] = []
        for (i, train) in training.enumerated() {
            let dist = metric.distance(embedding.vector, train.vector)
            distances.append((index: i, distance: dist))
        }

        // Get k nearest
        distances.sort { $0.distance < $1.distance }
        let neighbors = Array(distances.prefix(k))

        return interpolateFromNeighbors(neighbors: neighbors, reducedTraining: reduced)
    }

    // MARK: - Helpers

    /// Converts DistanceMetricType to DistanceMetric.
    private func convertMetric(_ type: DistanceMetricType) -> DistanceMetric {
        switch type {
        case .euclidean:
            return .euclidean
        case .cosine:
            return .cosine
        case .manhattan:
            return .manhattan
        case .dotProduct:
            return .cosine  // Approximate with cosine
        }
    }

    /// Computes automatic number of epochs based on dataset size.
    private func computeAutoEpochs(n: Int) -> Int {
        if n <= 200 {
            return 500
        } else if n <= 2000 {
            return 200
        } else if n <= 10000 {
            return 200
        } else {
            return 100
        }
    }
}

// MARK: - Fitted State

/// The fitted state of a UMAP model.
private struct FittedUMAPState: Sendable {
    /// The k-NN graph of training data.
    let knnGraph: NearestNeighborGraph

    /// Original training embeddings.
    let trainingEmbeddings: [Embedding]

    /// Reduced training embeddings.
    let reducedEmbeddings: [[Float]]

    /// Original input dimension.
    let inputDimension: Int

    /// Reduced output dimension.
    let outputDimension: Int
}

// MARK: - Convenience Functions

/// Convenience function for one-shot UMAP reduction.
///
/// - Parameters:
///   - embeddings: Input embeddings.
///   - nComponents: Output dimensions (default: 15).
///   - nNeighbors: Number of neighbors (default: 15).
///   - minDist: Minimum distance (default: 0.1).
/// - Returns: Reduced embeddings.
public func umap(
    _ embeddings: [Embedding],
    nComponents: Int = 15,
    nNeighbors: Int = 15,
    minDist: Float = 0.1
) async throws -> [Embedding] {
    let config = UMAPConfiguration(
        nNeighbors: nNeighbors,
        minDist: minDist
    )
    let reducer = UMAPReducer(
        configuration: config,
        nComponents: nComponents
    )
    return try await reducer.fitTransform(embeddings)
}

// MARK: - Array Extension

extension Array where Element == Embedding {

    /// Reduces dimensionality using UMAP.
    ///
    /// - Parameters:
    ///   - nComponents: Output dimensions.
    ///   - nNeighbors: Number of neighbors.
    ///   - minDist: Minimum distance.
    /// - Returns: Reduced embeddings.
    public func reduceUMAP(
        nComponents: Int = 15,
        nNeighbors: Int = 15,
        minDist: Float = 0.1
    ) async throws -> [Embedding] {
        try await umap(
            self,
            nComponents: nComponents,
            nNeighbors: nNeighbors,
            minDist: minDist
        )
    }
}

// MARK: - UMAP Builder

/// Builder pattern for UMAP configuration.
public struct UMAPBuilder: Sendable {

    private var nNeighbors: Int = 15
    private var minDist: Float = 0.1
    private var nComponents: Int = 15
    private var metric: DistanceMetricType = .euclidean
    private var nEpochs: Int? = nil
    private var learningRate: Float = 1.0
    private var seed: UInt64? = nil
    private var gpuContext: TopicsGPUContext? = nil
    private var initialization: UMAPInitialization = .spectral

    /// Creates a new UMAP builder with default settings.
    public init() {}

    /// Creates a UMAP builder optimized for GPU acceleration.
    ///
    /// Uses PCA initialization to bypass O(n³) spectral embedding.
    public static func gpuOptimized() -> UMAPBuilder {
        UMAPBuilder().initialization(.pca)
    }

    /// Sets the number of neighbors.
    public func neighbors(_ k: Int) -> UMAPBuilder {
        var copy = self
        copy.nNeighbors = k
        return copy
    }

    /// Sets the minimum distance.
    public func minDist(_ dist: Float) -> UMAPBuilder {
        var copy = self
        copy.minDist = dist
        return copy
    }

    /// Sets the output dimension.
    public func components(_ n: Int) -> UMAPBuilder {
        var copy = self
        copy.nComponents = n
        return copy
    }

    /// Sets the distance metric.
    public func metric(_ m: DistanceMetricType) -> UMAPBuilder {
        var copy = self
        copy.metric = m
        return copy
    }

    /// Sets the number of epochs.
    public func epochs(_ n: Int) -> UMAPBuilder {
        var copy = self
        copy.nEpochs = n
        return copy
    }

    /// Sets the learning rate.
    public func learningRate(_ lr: Float) -> UMAPBuilder {
        var copy = self
        copy.learningRate = lr
        return copy
    }

    /// Sets the random seed.
    public func seed(_ s: UInt64) -> UMAPBuilder {
        var copy = self
        copy.seed = s
        return copy
    }

    /// Sets the GPU context for acceleration.
    public func gpu(_ context: TopicsGPUContext?) -> UMAPBuilder {
        var copy = self
        copy.gpuContext = context
        return copy
    }

    /// Sets the initialization strategy.
    ///
    /// - Parameter strategy: The initialization method to use.
    /// - Returns: Updated builder.
    ///
    /// ## Performance Impact
    /// - `.spectral`: Best quality, O(n³) - use for <500 points
    /// - `.pca`: Good quality, O(n×d²) - recommended for GPU
    /// - `.random`: Fast, O(n) - for large datasets or speed-critical cases
    public func initialization(_ strategy: UMAPInitialization) -> UMAPBuilder {
        var copy = self
        copy.initialization = strategy
        return copy
    }

    /// Builds the UMAP reducer.
    public func build() -> UMAPReducer {
        let config = UMAPConfiguration(
            nNeighbors: nNeighbors,
            minDist: minDist,
            metric: metric,
            nEpochs: nEpochs,
            learningRate: learningRate,
            initialization: initialization
        )
        return UMAPReducer(
            configuration: config,
            nComponents: nComponents,
            seed: seed,
            gpuContext: gpuContext
        )
    }
}
