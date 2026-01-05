// GPUBatchKNN.swift
// SwiftTopics
//
// GPU-accelerated batch k-NN using VectorAccelerate

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - GPU Batch k-NN

/// GPU-accelerated batch k-nearest neighbor search.
///
/// Uses VectorAccelerate's `FusedL2TopKKernel` for efficient parallel computation.
/// Optimal for large batch queries where GPU overhead is amortized.
///
/// ## When to Use
///
/// | Scenario | Recommendation |
/// |----------|----------------|
/// | Single query | Use BallTree (CPU) |
/// | Batch < 100 queries | Use BallTree (CPU) |
/// | Batch >= 100 queries | Use GPUBatchKNN |
/// | Core distance computation (HDBSCAN) | Use GPUBatchKNN |
/// | Radius queries | Use BallTree (CPU) |
///
/// ## Performance Characteristics
///
/// - **Memory**: O(Q × K) output, O(Q × N) internal (managed by kernel)
/// - **Complexity**: O(Q × N × D / GPU_parallelism)
/// - **Latency**: Higher fixed cost, better throughput for large batches
///
/// ## GPU Availability
///
/// If GPU is unavailable, the wrapper falls back to CPU-based brute-force
/// computation. Use `isGPUAvailable` to check availability.
///
/// ## Usage
/// ```swift
/// let gpu = try await TopicsGPUContext()
/// let knn = GPUBatchKNN(context: gpu, dataset: embeddings.map { $0.vector })
///
/// // Batch query all points against themselves (for core distances)
/// let neighbors = try await knn.queryBatch(points: dataset, k: minSamples)
/// ```
///
/// ## Thread Safety
/// `GPUBatchKNN` is an actor and all methods are isolated.
public actor GPUBatchKNN {

    // MARK: - Properties

    /// The GPU context for Metal operations.
    private let gpuContext: TopicsGPUContext

    /// The dataset to search against.
    private let dataset: [[Float]]

    /// Number of points in the dataset.
    public nonisolated let count: Int

    /// Dimensionality of points.
    public nonisolated let dimension: Int

    /// Maximum k value supported by the GPU kernel.
    public nonisolated static let maxK: Int = 128

    // MARK: - Initialization

    /// Creates a GPU batch k-NN searcher.
    ///
    /// - Parameters:
    ///   - context: The GPU context for Metal operations.
    ///   - dataset: The dataset to search against.
    /// - Throws: `SpatialIndexError` if dataset is empty or inconsistent.
    public init(context: TopicsGPUContext, dataset: [[Float]]) throws {
        guard !dataset.isEmpty else {
            throw SpatialIndexError.emptyPoints
        }

        let dim = dataset[0].count
        guard dataset.allSatisfy({ $0.count == dim }) else {
            throw SpatialIndexError.constructionFailed("All points must have same dimension")
        }

        self.gpuContext = context
        self.dataset = dataset
        self.count = dataset.count
        self.dimension = dim
    }

    /// Creates a GPU batch k-NN searcher from embeddings.
    ///
    /// - Parameters:
    ///   - context: The GPU context for Metal operations.
    ///   - embeddings: The embeddings to search against.
    /// - Throws: `SpatialIndexError` if embeddings are empty.
    public init(context: TopicsGPUContext, embeddings: [Embedding]) throws {
        try self.init(context: context, dataset: embeddings.map { $0.vector })
    }

    // MARK: - Batch Query

    /// Finds k nearest neighbors for each query point.
    ///
    /// Uses GPU-accelerated fused L2 distance + top-K selection.
    ///
    /// - Parameters:
    ///   - points: Query points (must have same dimension as dataset).
    ///   - k: Number of neighbors to find per query.
    /// - Returns: Array of neighbor arrays, one per query point.
    /// - Throws: `TopicsGPUError` on GPU failure.
    public func queryBatch(
        points: [[Float]],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] {
        guard !points.isEmpty else { return [] }

        // Validate dimensions
        guard let firstDim = points.first?.count, firstDim == dimension else {
            throw SpatialIndexError.dimensionMismatch(
                query: points.first?.count ?? 0,
                indexed: dimension
            )
        }

        // Validate k
        let actualK = min(k, count, Self.maxK)
        guard actualK > 0 else {
            return points.map { _ in [] }
        }

        // Use the GPU context's k-NN method which wraps FusedL2TopKKernel
        do {
            let context = try await gpuContext.getContext()
            let kernel = try await FusedL2TopKKernel(context: context)

            let results = try await kernel.findNearestNeighbors(
                queries: points,
                dataset: dataset,
                k: actualK,
                includeDistances: true
            )

            // Convert squared distances to actual distances
            return results.map { neighbors in
                neighbors.map { (index: $0.index, distance: sqrt($0.distance)) }
            }
        } catch {
            // Fallback to CPU if GPU fails
            return cpuBatchQuery(queries: points, k: actualK)
        }
    }

    /// Finds k nearest neighbors for each embedding.
    ///
    /// - Parameters:
    ///   - embeddings: Query embeddings.
    ///   - k: Number of neighbors to find.
    /// - Returns: Array of neighbor arrays.
    public func queryBatch(
        embeddings: [Embedding],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] {
        try await queryBatch(points: embeddings.map { $0.vector }, k: k)
    }

    // MARK: - Self-Query (for Core Distances)

    /// Computes k-NN for all points in the dataset against themselves.
    ///
    /// This is the core operation for HDBSCAN's core distance computation.
    /// Each point finds its k nearest neighbors (including itself at distance 0).
    ///
    /// - Parameter k: Number of neighbors to find.
    /// - Returns: Array of neighbor arrays, one per dataset point.
    public func selfQuery(k: Int) async throws -> [[(index: Int, distance: Float)]] {
        try await queryBatch(points: dataset, k: k)
    }

    /// Computes core distances (k-th nearest neighbor distance) for all points.
    ///
    /// Used by HDBSCAN for density estimation. The core distance is the
    /// distance to the k-th nearest neighbor (not including self).
    ///
    /// - Parameter k: The k value (typically minSamples in HDBSCAN).
    /// - Returns: Core distance for each point.
    public func computeCoreDistances(k: Int) async throws -> [Float] {
        // Query k+1 neighbors to account for self (distance 0)
        let knn = try await selfQuery(k: k + 1)

        return knn.map { neighbors in
            // Find k-th neighbor (excluding self)
            let nonSelfNeighbors = neighbors.filter { $0.distance > Float.ulpOfOne }

            if nonSelfNeighbors.count >= k {
                return nonSelfNeighbors[k - 1].distance
            } else if !nonSelfNeighbors.isEmpty {
                return nonSelfNeighbors.last!.distance
            } else if neighbors.count >= k {
                // All neighbors are at distance 0 (identical points)
                return neighbors[k - 1].distance
            } else {
                return Float.infinity
            }
        }
    }

    // MARK: - CPU Fallback

    private func cpuBatchQuery(
        queries: [[Float]],
        k: Int
    ) -> [[(index: Int, distance: Float)]] {
        queries.map { query in
            // Compute all distances
            var distances: [(index: Int, distance: Float)] = []
            distances.reserveCapacity(dataset.count)

            for (i, point) in dataset.enumerated() {
                var sumSq: Float = 0
                for d in 0..<dimension {
                    let diff = query[d] - point[d]
                    sumSq += diff * diff
                }
                distances.append((index: i, distance: sqrt(sumSq)))
            }

            // Sort and take top k
            distances.sort { $0.distance < $1.distance }
            return Array(distances.prefix(k))
        }
    }
}

// MARK: - GPU Batch k-NN Builder

/// Factory for creating GPU batch k-NN instances.
public enum GPUBatchKNNBuilder {

    /// Creates a GPU batch k-NN searcher with automatic fallback.
    ///
    /// If GPU is unavailable, returns nil and caller should use BallTree instead.
    ///
    /// - Parameters:
    ///   - dataset: The dataset to search against.
    ///   - preferHighPerformance: Whether to prefer high-performance GPU.
    /// - Returns: GPU batch k-NN instance, or nil if GPU unavailable.
    public static func create(
        dataset: [[Float]],
        preferHighPerformance: Bool = true
    ) async -> GPUBatchKNN? {
        guard let context = await TopicsGPUContext.create(allowCPUFallback: false) else {
            return nil
        }

        do {
            return try GPUBatchKNN(context: context, dataset: dataset)
        } catch {
            return nil
        }
    }

    /// Creates a GPU batch k-NN searcher from embeddings.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to search against.
    ///   - preferHighPerformance: Whether to prefer high-performance GPU.
    /// - Returns: GPU batch k-NN instance, or nil if GPU unavailable.
    public static func create(
        embeddings: [Embedding],
        preferHighPerformance: Bool = true
    ) async -> GPUBatchKNN? {
        await create(dataset: embeddings.map { $0.vector }, preferHighPerformance: preferHighPerformance)
    }
}

// MARK: - Hybrid k-NN Strategy

/// Automatically selects between GPU and CPU k-NN based on query characteristics.
///
/// Uses heuristics to choose the optimal strategy:
/// - GPU: Large batch queries, core distance computation
/// - CPU: Single queries, small batches, radius queries
public struct HybridKNNStrategy: Sendable {

    /// Threshold for switching to GPU (number of queries).
    public let gpuBatchThreshold: Int

    /// Threshold for switching to GPU (query × dataset size).
    public let gpuComputeThreshold: Int

    /// Creates a hybrid k-NN strategy.
    ///
    /// - Parameters:
    ///   - gpuBatchThreshold: Minimum queries for GPU (default: 100).
    ///   - gpuComputeThreshold: Minimum compute for GPU (default: 1_000_000).
    public init(
        gpuBatchThreshold: Int = 100,
        gpuComputeThreshold: Int = 1_000_000
    ) {
        self.gpuBatchThreshold = gpuBatchThreshold
        self.gpuComputeThreshold = gpuComputeThreshold
    }

    /// Default strategy with balanced thresholds.
    public static let `default` = HybridKNNStrategy()

    /// Strategy that prefers GPU for most operations.
    public static let preferGPU = HybridKNNStrategy(
        gpuBatchThreshold: 10,
        gpuComputeThreshold: 100_000
    )

    /// Strategy that prefers CPU for most operations.
    public static let preferCPU = HybridKNNStrategy(
        gpuBatchThreshold: 1000,
        gpuComputeThreshold: 10_000_000
    )

    /// Determines whether to use GPU for the given query parameters.
    ///
    /// - Parameters:
    ///   - queryCount: Number of queries.
    ///   - datasetSize: Size of the dataset.
    ///   - gpuAvailable: Whether GPU is available.
    /// - Returns: True if GPU should be used.
    public func shouldUseGPU(
        queryCount: Int,
        datasetSize: Int,
        gpuAvailable: Bool
    ) -> Bool {
        guard gpuAvailable else { return false }

        if queryCount >= gpuBatchThreshold {
            return true
        }

        let computeSize = queryCount * datasetSize
        return computeSize >= gpuComputeThreshold
    }
}

// MARK: - Unified k-NN Interface

/// Unified interface for k-NN queries that automatically selects GPU or CPU.
public actor UnifiedKNN {

    private let ballTree: BallTree?
    private let gpuKNN: GPUBatchKNN?
    private let dataset: [[Float]]
    private let strategy: HybridKNNStrategy

    /// Creates a unified k-NN searcher.
    ///
    /// - Parameters:
    ///   - dataset: The dataset to search against.
    ///   - gpuContext: Optional GPU context (uses CPU if nil).
    ///   - strategy: Strategy for selecting GPU vs CPU.
    public init(
        dataset: [[Float]],
        gpuContext: TopicsGPUContext?,
        strategy: HybridKNNStrategy = .default
    ) async throws {
        self.dataset = dataset
        self.strategy = strategy

        // Build Ball Tree for CPU queries
        self.ballTree = try BallTree.build(points: dataset)

        // Create GPU k-NN if context available
        if let context = gpuContext {
            self.gpuKNN = try GPUBatchKNN(context: context, dataset: dataset)
        } else {
            self.gpuKNN = nil
        }
    }

    /// Creates a unified k-NN searcher from embeddings.
    public init(
        embeddings: [Embedding],
        gpuContext: TopicsGPUContext?,
        strategy: HybridKNNStrategy = .default
    ) async throws {
        try await self.init(
            dataset: embeddings.map { $0.vector },
            gpuContext: gpuContext,
            strategy: strategy
        )
    }

    /// Number of points in the dataset.
    public nonisolated var count: Int { dataset.count }

    /// Dimensionality of points.
    public nonisolated var dimension: Int { dataset.first?.count ?? 0 }

    /// Whether GPU is available.
    public var isGPUAvailable: Bool { gpuKNN != nil }

    /// Performs a single k-NN query.
    ///
    /// Always uses CPU (Ball Tree) for single queries.
    public func query(point: [Float], k: Int) -> [(index: Int, distance: Float)] {
        ballTree?.query(point: point, k: k) ?? []
    }

    /// Performs batch k-NN queries.
    ///
    /// Automatically selects GPU or CPU based on strategy.
    public func queryBatch(
        points: [[Float]],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] {
        let useGPU = strategy.shouldUseGPU(
            queryCount: points.count,
            datasetSize: dataset.count,
            gpuAvailable: gpuKNN != nil
        )

        if useGPU, let gpu = gpuKNN {
            return try await gpu.queryBatch(points: points, k: k)
        } else if let tree = ballTree {
            return tree.queryBatch(points: points, k: k)
        } else {
            return []
        }
    }

    /// Computes core distances for all points.
    ///
    /// Prefers GPU for this computation-heavy operation.
    public func computeCoreDistances(k: Int) async throws -> [Float] {
        if let gpu = gpuKNN {
            return try await gpu.computeCoreDistances(k: k)
        } else if let tree = ballTree {
            // CPU fallback using batch query
            let knn = tree.queryBatch(points: dataset, k: k + 1)
            return knn.map { neighbors in
                let nonSelf = neighbors.filter { $0.distance > Float.ulpOfOne }
                if nonSelf.count >= k {
                    return nonSelf[k - 1].distance
                } else if !nonSelf.isEmpty {
                    return nonSelf.last!.distance
                } else {
                    return neighbors.count >= k ? neighbors[k - 1].distance : .infinity
                }
            }
        } else {
            return []
        }
    }

    /// Performs a radius query.
    ///
    /// Always uses CPU (Ball Tree) for radius queries.
    public func queryRadius(point: [Float], radius: Float) -> [(index: Int, distance: Float)] {
        ballTree?.queryRadius(point: point, radius: radius) ?? []
    }
}
