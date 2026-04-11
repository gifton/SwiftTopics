// GPUContext.swift
// SwiftTopics
//
// GPU context wrapper for VectorAccelerate Metal 4 operations

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - Topics GPU Context

/// GPU acceleration context for SwiftTopics.
///
/// Wraps VectorAccelerate's `Metal4Context` and provides topic-modeling-specific
/// operations like pairwise distances, k-NN search, and matrix operations.
///
/// ## Usage
/// ```swift
/// let gpu = try await TopicsGPUContext()
/// let distances = try await gpu.computePairwiseDistances(embeddings)
/// ```
///
/// ## Thread Safety
/// `TopicsGPUContext` is an actor and all methods are isolated.
///
/// ## GPU Availability
/// If GPU is unavailable, methods throw `TopicsGPUError.gpuUnavailable`.
/// Use `isAvailable` to check before heavy GPU work.
public actor TopicsGPUContext {

    // MARK: - Properties

    /// The underlying Metal 4 context.
    private var metal4Context: Metal4Context?

    /// GPU health monitor for tracking failures and automatic CPU fallback.
    private var healthMonitor: GPUHealthMonitor?

    /// Whether GPU acceleration is available.
    public nonisolated var isAvailable: Bool {
        Metal4Context.isAvailable
    }

    /// Configuration for this context.
    public let configuration: TopicsGPUConfiguration

    // MARK: - Initialization

    /// Creates a GPU context with default configuration.
    public init() async throws {
        self.configuration = .default
        self.metal4Context = try await Self.initializeContext(configuration: configuration)
        self.healthMonitor = Self.createHealthMonitor(configuration: configuration)
    }

    /// Creates a GPU context with custom configuration.
    ///
    /// - Parameter configuration: Custom GPU configuration.
    public init(configuration: TopicsGPUConfiguration) async throws {
        self.configuration = configuration
        self.metal4Context = try await Self.initializeContext(configuration: configuration)
        self.healthMonitor = Self.createHealthMonitor(configuration: configuration)
    }

    /// Creates the health monitor based on configuration.
    private static func createHealthMonitor(
        configuration: TopicsGPUConfiguration
    ) -> GPUHealthMonitor? {
        guard configuration.enableHealthMonitoring else { return nil }

        if let healthConfig = configuration.healthMonitorConfiguration {
            return GPUHealthMonitor(configuration: healthConfig)
        } else {
            return GPUHealthMonitor()
        }
    }

    /// Creates a GPU context that may gracefully degrade if GPU unavailable.
    ///
    /// - Parameter allowCPUFallback: If true, doesn't throw when GPU unavailable.
    /// - Returns: GPU context, or nil if unavailable and fallback allowed.
    public static func create(allowCPUFallback: Bool = true) async -> TopicsGPUContext? {
        do {
            return try await TopicsGPUContext()
        } catch {
            if allowCPUFallback {
                return nil
            }
            return nil
        }
    }

    private static func initializeContext(
        configuration: TopicsGPUConfiguration
    ) async throws -> Metal4Context? {
        guard Metal4Context.isAvailable else {
            throw TopicsGPUError.gpuUnavailable
        }

        let metalConfig = Metal4Configuration(
            preferHighPerformanceDevice: configuration.preferHighPerformance,
            maxBufferPoolMemory: configuration.maxBufferPoolMemory,
            enableProfiling: configuration.enableProfiling,
            commandQueueLabel: "SwiftTopics.GPU"
        )

        return try await Metal4Context(configuration: metalConfig)
    }

    // MARK: - Context Access

    /// Gets the underlying Metal 4 context.
    ///
    /// - Throws: `TopicsGPUError.gpuUnavailable` if context not initialized.
    public func getContext() throws -> Metal4Context {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }
        return context
    }

    /// Ensures the context is initialized and ready.
    public func ensureReady() async throws {
        if metal4Context == nil {
            metal4Context = try await Self.initializeContext(configuration: configuration)
        }
    }

    // MARK: - Pairwise Distances

    /// Computes pairwise L2 distances between all embeddings.
    ///
    /// Given n embeddings of dimension d, returns an n×n distance matrix.
    /// Uses GPU when available and healthy, with automatic CPU fallback.
    ///
    /// - Parameter embeddings: The embeddings to compute distances for.
    /// - Returns: 2D distance matrix [n][n].
    /// - Complexity: O(n² × d) computation.
    public func computePairwiseL2Distances(_ embeddings: [Embedding]) async throws -> [[Float]] {
        guard !embeddings.isEmpty else { return [] }

        return try await executeWithHealthTracking(
            operation: .pairwiseL2Distance,
            gpu: { [self] in
                guard let context = metal4Context else {
                    throw TopicsGPUError.gpuUnavailable
                }

                let vectors = embeddings.map { $0.vector }
                let kernel = try await L2DistanceKernel(context: context)
                return try await kernel.compute(
                    queries: vectors,
                    database: vectors,
                    computeSqrt: true
                )
            },
            cpuFallback: { [self] in
                computePairwiseL2DistancesCPU(embeddings)
            }
        )
    }

    /// Computes pairwise cosine similarities between all embeddings.
    ///
    /// Uses GPU when available and healthy, with automatic CPU fallback.
    ///
    /// - Parameter embeddings: The embeddings to compute similarities for.
    /// - Returns: 2D similarity matrix [n][n].
    public func computePairwiseCosineSimilarity(_ embeddings: [Embedding]) async throws -> [[Float]] {
        guard !embeddings.isEmpty else { return [] }

        return try await executeWithHealthTracking(
            operation: .pairwiseCosineSimilarity,
            gpu: { [self] in
                guard let context = metal4Context else {
                    throw TopicsGPUError.gpuUnavailable
                }

                let vectors = embeddings.map { $0.vector }
                let kernel = try await CosineSimilarityKernel(context: context)
                return try await kernel.compute(
                    queries: vectors,
                    database: vectors
                )
            },
            cpuFallback: { [self] in
                computePairwiseCosineSimilarityCPU(embeddings)
            }
        )
    }

    // MARK: - K-Nearest Neighbors

    /// Computes k-nearest neighbors for all embeddings.
    ///
    /// Uses VectorAccelerate's `FusedL2TopKKernel` for efficient GPU computation,
    /// with automatic CPU fallback on GPU health issues.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to find neighbors for.
    ///   - k: Number of neighbors to find.
    /// - Returns: For each embedding: array of (index, distance) pairs.
    public func computeBatchKNN(
        _ embeddings: [Embedding],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] {
        let n = embeddings.count
        guard n > 0 else { return [] }
        guard k > 0 && k <= n else {
            throw TopicsGPUError.invalidParameter("k must be between 1 and \(n)")
        }

        return try await executeWithHealthTracking(
            operation: .batchKNN,
            gpu: { [self] in
                guard let context = metal4Context else {
                    throw TopicsGPUError.gpuUnavailable
                }

                let vectors = embeddings.map { $0.vector }
                let kernel = try await FusedL2TopKKernel(context: context)
                return try await kernel.findNearestNeighbors(
                    queries: vectors,
                    dataset: vectors,
                    k: k
                )
            },
            cpuFallback: { [self] in
                computeBatchKNNCPU(embeddings, k: k)
            }
        )
    }

    /// Computes the k-th nearest neighbor distance for each embedding (core distance).
    ///
    /// Used by HDBSCAN for density estimation.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings.
    ///   - k: The k value (typically minSamples in HDBSCAN).
    /// - Returns: Core distance for each embedding.
    public func computeCoreDistances(
        _ embeddings: [Embedding],
        k: Int
    ) async throws -> [Float] {
        let knn = try await computeBatchKNN(embeddings, k: k)
        return knn.map { neighbors in
            // The k-th neighbor distance (last in sorted list)
            neighbors.last?.distance ?? Float.infinity
        }
    }

    // MARK: - HDBSCAN GPU Acceleration

    /// Computes HDBSCAN core distances and MST entirely on GPU using VectorAccelerate.
    ///
    /// This method fuses core distance computation, mutual reachability distance calculation,
    /// and MST construction into an efficient GPU pipeline using Borůvka's algorithm.
    /// It provides 25-125x speedup over CPU for datasets with 1K+ points.
    ///
    /// ## Algorithm
    ///
    /// The GPU pipeline performs:
    /// 1. **Core distances**: k-NN search to find k-th nearest neighbor distance per point
    /// 2. **Mutual reachability**: `max(core_dist[a], core_dist[b], euclidean_dist(a,b))`
    /// 3. **MST construction**: Borůvka's parallel algorithm in O(log N) iterations
    ///
    /// ## Performance
    ///
    /// | Dataset Size | Expected Time |
    /// |--------------|---------------|
    /// | 500 points   | ~50ms         |
    /// | 1,000 points | ~150ms        |
    /// | 5,000 points | ~2s           |
    ///
    /// ## Health Tracking
    ///
    /// This operation is tracked by the health monitor. On repeated failures,
    /// the caller should fall back to CPU-based HDBSCAN.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to cluster.
    ///   - minSamples: The k value for core distance (typically HDBSCAN's minSamples).
    /// - Returns: Core distances and minimum spanning tree for cluster hierarchy construction.
    /// - Throws: `TopicsGPUError.gpuUnavailable` if GPU context not initialized.
    public func computeHDBSCANMSTWithCoreDistances(
        _ embeddings: [Embedding],
        minSamples: Int
    ) async throws -> GPUHDBSCANResult {
        return try await executeWithHealthTracking(
            operation: .hdbscanMST,
            gpu: { [self] in
                guard let context = metal4Context else {
                    throw TopicsGPUError.gpuUnavailable
                }

                let vectors = embeddings.map { $0.vector }

                let module = try await HDBSCANDistanceModule(context: context)
                let result = try await module.computeMST(
                    embeddings: vectors,
                    minSamples: minSamples
                )

                // Convert VectorAccelerate MSTResult to SwiftTopics MinimumSpanningTree
                let edges = result.mst.edges.map { edge in
                    MSTEdge(
                        source: edge.source,
                        target: edge.target,
                        weight: edge.weight
                    )
                }

                let mst = MinimumSpanningTree(edges: edges, pointCount: embeddings.count)

                return GPUHDBSCANResult(
                    coreDistances: result.coreDistances,
                    mst: mst
                )
            },
            cpuFallback: nil  // CPU fallback handled at HDBSCAN level
        )
    }

    // MARK: - Matrix Operations

    /// Computes the covariance matrix of centered embeddings.
    ///
    /// X^T × X / (n-1) where X is the centered data matrix.
    ///
    /// - Parameter embeddings: The embeddings (will be centered).
    /// - Returns: Covariance matrix as 2D array (d×d).
    public func computeCovarianceMatrix(_ embeddings: [Embedding]) async throws -> [[Float]] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        let n = embeddings.count
        guard n > 1 else {
            throw TopicsGPUError.invalidParameter("Need at least 2 embeddings for covariance")
        }

        let d = embeddings[0].dimension

        // First, center the data (CPU for simplicity, could use GPU)
        let centered = centerEmbeddingsCPU(embeddings)

        // Create matrices for multiplication
        // X is n×d, X^T is d×n
        // X^T × X gives d×d covariance
        let xData = centered.flatMap { $0.vector }
        let matrixX = Matrix(rows: n, columns: d, values: xData)

        let kernel = try await MatrixMultiplyKernel(context: context)

        // Multiply X^T × X using transpose config
        let config = Metal4MatrixMultiplyConfig(
            alpha: 1.0 / Float(n - 1),  // Normalize by (n-1)
            beta: 0.0,
            transposeA: true,
            transposeB: false
        )

        let result = try await kernel.multiply(matrixX, matrixX, config: config)

        // Convert result to 2D array
        return result.asMatrix().values.chunked(into: d)
    }

    /// Projects embeddings using a transformation matrix.
    ///
    /// Y = X × W where X is n×d and W is d×k, resulting in Y of n×k.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to project (n×d).
    ///   - transformation: The projection matrix (d×k) as 2D array.
    /// - Returns: Projected embeddings.
    public func projectEmbeddings(
        _ embeddings: [Embedding],
        transformation: [[Float]]
    ) async throws -> [Embedding] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        let n = embeddings.count
        guard n > 0 else { return [] }

        let d = embeddings[0].dimension
        let k = transformation[0].count

        guard transformation.count == d else {
            throw TopicsGPUError.invalidParameter(
                "Transformation rows (\(transformation.count)) must equal embedding dimension (\(d))"
            )
        }

        // Create matrices
        let xData = embeddings.flatMap { $0.vector }
        let matrixX = Matrix(rows: n, columns: d, values: xData)

        let wData = transformation.flatMap { $0 }
        let matrixW = Matrix(rows: d, columns: k, values: wData)

        let kernel = try await MatrixMultiplyKernel(context: context)

        let result = try await kernel.multiply(matrixX, matrixW)

        // Convert back to embeddings
        let projected = result.asMatrix()
        return (0..<n).map { i in
            let start = i * k
            let end = start + k
            return Embedding(vector: Array(projected.values[start..<end]))
        }
    }

    // MARK: - Statistics

    /// Computes the mean embedding using CPU (simple implementation).
    ///
    /// - Parameter embeddings: The embeddings.
    /// - Returns: The mean embedding.
    public func computeMean(_ embeddings: [Embedding]) async throws -> Embedding {
        guard !embeddings.isEmpty else {
            throw TopicsGPUError.invalidParameter("Cannot compute mean of empty embeddings")
        }

        let d = embeddings[0].dimension
        var mean = [Float](repeating: 0, count: d)

        for embedding in embeddings {
            for i in 0..<d {
                mean[i] += embedding.vector[i]
            }
        }

        let n = Float(embeddings.count)
        for i in 0..<d {
            mean[i] /= n
        }

        return Embedding(vector: mean)
    }

    /// Centers embeddings by subtracting the mean (CPU implementation).
    private func centerEmbeddingsCPU(_ embeddings: [Embedding]) -> [Embedding] {
        guard !embeddings.isEmpty else { return [] }

        let d = embeddings[0].dimension
        var mean = [Float](repeating: 0, count: d)

        // Compute mean
        for embedding in embeddings {
            for i in 0..<d {
                mean[i] += embedding.vector[i]
            }
        }
        let n = Float(embeddings.count)
        for i in 0..<d {
            mean[i] /= n
        }

        // Subtract mean
        return embeddings.map { embedding in
            let centered = zip(embedding.vector, mean).map { $0 - $1 }
            return Embedding(vector: centered)
        }
    }

    /// Centers embeddings by subtracting the mean.
    ///
    /// - Parameter embeddings: The embeddings to center.
    /// - Returns: Centered embeddings.
    public func centerEmbeddings(_ embeddings: [Embedding]) async throws -> [Embedding] {
        return centerEmbeddingsCPU(embeddings)
    }

    // MARK: - Normalization

    /// L2-normalizes embeddings.
    ///
    /// Uses GPU when available and healthy, with automatic CPU fallback.
    ///
    /// - Parameter embeddings: The embeddings to normalize.
    /// - Returns: Normalized embeddings (unit length).
    public func normalizeL2(_ embeddings: [Embedding]) async throws -> [Embedding] {
        guard !embeddings.isEmpty else { return [] }

        return try await executeWithHealthTracking(
            operation: .normalization,
            gpu: { [self] in
                guard let context = metal4Context else {
                    throw TopicsGPUError.gpuUnavailable
                }

                let vectors = embeddings.map { $0.vector }
                let kernel = try await L2NormalizationKernel(context: context)
                let result = try await kernel.normalize(vectors)
                return result.asArrays().map { Embedding(vector: $0) }
            },
            cpuFallback: { [self] in
                normalizeL2CPU(embeddings)
            }
        )
    }

    // MARK: - Health Monitoring

    /// Operation identifiers for health tracking.
    private enum GPUOperationID: String {
        case pairwiseL2Distance = "pairwise_l2_distance"
        case pairwiseCosineSimilarity = "pairwise_cosine_similarity"
        case batchKNN = "batch_knn"
        case coreDistances = "core_distances"
        case hdbscanMST = "hdbscan_mst"
        case covariance = "covariance_matrix"
        case projection = "projection"
        case normalization = "normalization"
        case umapGradient = "umap_gradient"
    }

    /// Executes a GPU operation with health tracking and optional CPU fallback.
    ///
    /// This wrapper:
    /// 1. Checks if CPU fallback should be used due to GPU health issues
    /// 2. Executes the GPU operation
    /// 3. Records success/failure for health tracking
    /// 4. Falls back to CPU if GPU fails and fallback is available
    ///
    /// - Parameters:
    ///   - operation: The operation identifier for tracking.
    ///   - gpu: The GPU operation to execute.
    ///   - cpuFallback: Optional CPU fallback implementation.
    /// - Returns: The result of either GPU or CPU operation.
    private func executeWithHealthTracking<T>(
        operation: GPUOperationID,
        gpu: () async throws -> T,
        cpuFallback: (() async throws -> T)? = nil
    ) async throws -> T {
        // Check if we should skip GPU due to health issues
        if let monitor = healthMonitor,
           await monitor.shouldFallbackToCPU(operation: operation.rawValue),
           let fallback = cpuFallback {
            return try await fallback()
        }

        do {
            let result = try await gpu()
            // Record success
            await healthMonitor?.recordSuccess(operation: operation.rawValue)
            return result
        } catch {
            // Record failure
            await healthMonitor?.recordFailure(operation: operation.rawValue, error: error)

            // Try CPU fallback if available
            if let fallback = cpuFallback {
                return try await fallback()
            }

            throw error
        }
    }

    /// Returns the current GPU health status.
    ///
    /// This provides visibility into:
    /// - Whether the GPU is healthy
    /// - Per-operation failure counts and degradation levels
    /// - Disabled operations that have fallen back to CPU
    ///
    /// - Returns: Health status, or nil if health monitoring is disabled.
    public func getHealthStatus() async -> GPUHealthStatus? {
        await healthMonitor?.getHealthStatus()
    }

    /// Resets all GPU health tracking state.
    ///
    /// This clears failure counts and re-enables any disabled operations.
    /// Use when you know GPU issues have been resolved.
    public func resetHealthStatus() async {
        await healthMonitor?.reset()
    }

    /// Checks if the GPU is currently healthy.
    ///
    /// - Returns: `true` if healthy or monitoring disabled, `false` if degraded.
    public func isGPUHealthy() async -> Bool {
        guard let monitor = healthMonitor else { return true }
        return await monitor.isHealthy()
    }

    // MARK: - CPU Fallback Implementations

    /// CPU fallback for pairwise L2 distance computation.
    private func computePairwiseL2DistancesCPU(_ embeddings: [Embedding]) -> [[Float]] {
        let n = embeddings.count
        guard n > 0 else { return [] }

        var distances = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        let d = embeddings[0].dimension

        for i in 0..<n {
            for j in (i + 1)..<n {
                var sum: Float = 0
                for k in 0..<d {
                    let diff = embeddings[i].vector[k] - embeddings[j].vector[k]
                    sum += diff * diff
                }
                let dist = sqrt(sum)
                distances[i][j] = dist
                distances[j][i] = dist
            }
        }

        return distances
    }

    /// CPU fallback for pairwise cosine similarity computation.
    private func computePairwiseCosineSimilarityCPU(_ embeddings: [Embedding]) -> [[Float]] {
        let n = embeddings.count
        guard n > 0 else { return [] }

        var similarities = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        let d = embeddings[0].dimension

        // Precompute norms
        var norms = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var normSq: Float = 0
            for k in 0..<d {
                normSq += embeddings[i].vector[k] * embeddings[i].vector[k]
            }
            norms[i] = sqrt(normSq)
        }

        // Compute similarities
        for i in 0..<n {
            similarities[i][i] = 1.0  // Self-similarity
            for j in (i + 1)..<n {
                var dot: Float = 0
                for k in 0..<d {
                    dot += embeddings[i].vector[k] * embeddings[j].vector[k]
                }
                let sim = (norms[i] > 0 && norms[j] > 0) ? dot / (norms[i] * norms[j]) : 0
                similarities[i][j] = sim
                similarities[j][i] = sim
            }
        }

        return similarities
    }

    /// CPU fallback for batch k-NN computation.
    private func computeBatchKNNCPU(
        _ embeddings: [Embedding],
        k: Int
    ) -> [[(index: Int, distance: Float)]] {
        let n = embeddings.count
        guard n > 0 else { return [] }
        let d = embeddings[0].dimension

        var results = [[(index: Int, distance: Float)]](repeating: [], count: n)

        for i in 0..<n {
            // Compute distances to all other points
            var distances: [(index: Int, distance: Float)] = []
            for j in 0..<n {
                var sum: Float = 0
                for dim in 0..<d {
                    let diff = embeddings[i].vector[dim] - embeddings[j].vector[dim]
                    sum += diff * diff
                }
                distances.append((index: j, distance: sqrt(sum)))
            }

            // Sort and take top k
            distances.sort { $0.distance < $1.distance }
            results[i] = Array(distances.prefix(k))
        }

        return results
    }

    /// CPU fallback for L2 normalization.
    private func normalizeL2CPU(_ embeddings: [Embedding]) -> [Embedding] {
        embeddings.map { embedding in
            var normSq: Float = 0
            for val in embedding.vector {
                normSq += val * val
            }
            let norm = sqrt(normSq)
            guard norm > 0 else { return embedding }

            let normalized = embedding.vector.map { $0 / norm }
            return Embedding(vector: normalized)
        }
    }

    // MARK: - Cleanup

    /// Releases GPU resources.
    public func cleanup() async {
        if let context = metal4Context {
            await context.cleanup()
        }
        metal4Context = nil
    }

    // MARK: - Performance

    /// Gets performance statistics.
    public func getPerformanceStats() async -> TopicsGPUStats? {
        guard let context = metal4Context else { return nil }
        let stats = await context.getPerformanceStats()
        return TopicsGPUStats(
            totalComputeTime: stats.totalComputeTime,
            operationCount: stats.operationCount,
            averageOperationTime: stats.averageOperationTime
        )
    }

    /// Resets performance counters.
    public func resetPerformanceStats() async {
        await metal4Context?.resetPerformanceStats()
    }
}

// MARK: - GPU Configuration

/// Configuration for the SwiftTopics GPU context.
public struct TopicsGPUConfiguration: Sendable {

    /// Whether to prefer high-performance GPU device.
    public let preferHighPerformance: Bool

    /// Maximum memory for buffer pool (bytes).
    public let maxBufferPoolMemory: Int?

    /// Whether to enable profiling.
    public let enableProfiling: Bool

    /// Minimum number of points to use GPU acceleration.
    ///
    /// For datasets smaller than this threshold, CPU computation may be faster
    /// due to GPU overhead (kernel launch, memory transfer). Default is 100.
    ///
    /// ## Tuning Guidelines
    ///
    /// - **< 50 points**: CPU is usually faster
    /// - **50-100 points**: GPU overhead may exceed benefit
    /// - **> 100 points**: GPU provides significant speedup
    /// - **> 1000 points**: GPU provides 10-50x speedup
    public let gpuMinPointsThreshold: Int

    /// Whether to enable GPU health monitoring with automatic CPU fallback.
    ///
    /// When enabled, the GPU context tracks consecutive failures per operation
    /// and automatically falls back to CPU implementations when GPU operations
    /// fail repeatedly.
    public let enableHealthMonitoring: Bool

    /// Configuration for the health monitor (when enabled).
    ///
    /// Controls how failures are tracked and when CPU fallback is triggered.
    public let healthMonitorConfiguration: GPUHealthMonitorConfiguration?

    /// Creates GPU configuration.
    ///
    /// - Parameters:
    ///   - preferHighPerformance: Prefer high-performance GPU device.
    ///   - maxBufferPoolMemory: Maximum memory for buffer pool (bytes).
    ///   - enableProfiling: Enable performance profiling.
    ///   - gpuMinPointsThreshold: Minimum points to use GPU (default: 100).
    ///   - enableHealthMonitoring: Enable GPU health monitoring (default: true).
    ///   - healthMonitorConfiguration: Custom health monitor config (default: .default).
    public init(
        preferHighPerformance: Bool = true,
        maxBufferPoolMemory: Int? = nil,
        enableProfiling: Bool = false,
        gpuMinPointsThreshold: Int = 100,
        enableHealthMonitoring: Bool = true,
        healthMonitorConfiguration: GPUHealthMonitorConfiguration? = nil
    ) {
        self.preferHighPerformance = preferHighPerformance
        self.maxBufferPoolMemory = maxBufferPoolMemory
        self.enableProfiling = enableProfiling
        self.gpuMinPointsThreshold = max(1, gpuMinPointsThreshold)
        self.enableHealthMonitoring = enableHealthMonitoring
        self.healthMonitorConfiguration = healthMonitorConfiguration
    }

    /// Default configuration.
    public static let `default` = TopicsGPUConfiguration()

    /// Configuration optimized for batch processing.
    public static let batch = TopicsGPUConfiguration(
        preferHighPerformance: true,
        maxBufferPoolMemory: 1024 * 1024 * 1024, // 1GB
        gpuMinPointsThreshold: 100
    )

    /// Configuration for debugging.
    public static let debug = TopicsGPUConfiguration(
        preferHighPerformance: false,
        enableProfiling: true,
        gpuMinPointsThreshold: 50  // Lower threshold for testing
    )

    /// Configuration for small datasets (lower GPU threshold).
    public static let smallDataset = TopicsGPUConfiguration(
        preferHighPerformance: true,
        gpuMinPointsThreshold: 50
    )

    /// Configuration with aggressive health monitoring for production.
    ///
    /// Uses aggressive settings that quickly fall back to CPU on GPU issues.
    public static let production = TopicsGPUConfiguration(
        preferHighPerformance: true,
        enableHealthMonitoring: true,
        healthMonitorConfiguration: .aggressive
    )
}

// MARK: - GPU Statistics

/// Performance statistics from GPU operations.
public struct TopicsGPUStats: Sendable {

    /// Total compute time in seconds.
    public let totalComputeTime: TimeInterval

    /// Number of operations executed.
    public let operationCount: Int

    /// Average time per operation.
    public let averageOperationTime: TimeInterval
}

// MARK: - Timing Instrumentation

/// Detailed timing breakdown for HDBSCAN GPU operations.
///
/// Provides per-phase timing information for performance analysis and debugging.
public struct HDBSCANTimingBreakdown: Sendable {

    /// Time spent computing core distances (k-NN search).
    public let coreDistanceTime: TimeInterval

    /// Time spent computing mutual reachability distances.
    public let mutualReachabilityTime: TimeInterval

    /// Time spent constructing the minimum spanning tree (Borůvka's algorithm).
    public let mstConstructionTime: TimeInterval

    /// Time spent building the cluster hierarchy.
    public let hierarchyBuildTime: TimeInterval

    /// Time spent extracting clusters.
    public let clusterExtractionTime: TimeInterval

    /// Total time for all operations.
    public var totalTime: TimeInterval {
        coreDistanceTime + mutualReachabilityTime + mstConstructionTime
            + hierarchyBuildTime + clusterExtractionTime
    }

    /// Whether GPU acceleration was used.
    public let usedGPU: Bool

    /// Number of points processed.
    public let pointCount: Int

    /// Human-readable summary.
    public var summary: String {
        let accel = usedGPU ? "GPU" : "CPU"
        return """
        HDBSCAN Timing (\(accel), \(pointCount) points):
          Core distances:      \(String(format: "%.3f", coreDistanceTime))s
          Mutual reachability: \(String(format: "%.3f", mutualReachabilityTime))s
          MST construction:    \(String(format: "%.3f", mstConstructionTime))s
          Hierarchy building:  \(String(format: "%.3f", hierarchyBuildTime))s
          Cluster extraction:  \(String(format: "%.3f", clusterExtractionTime))s
          Total:               \(String(format: "%.3f", totalTime))s
        """
    }
}

/// Detailed timing breakdown for UMAP GPU operations.
///
/// Provides per-phase timing information for performance analysis and debugging.
public struct UMAPTimingBreakdown: Sendable {

    /// Time spent building the k-NN graph.
    public let knnGraphTime: TimeInterval

    /// Time spent computing the fuzzy simplicial set.
    public let fuzzySetTime: TimeInterval

    /// Time spent on spectral initialization.
    public let spectralInitTime: TimeInterval

    /// Time spent on optimization epochs.
    public let optimizationTime: TimeInterval

    /// Number of optimization epochs.
    public let epochCount: Int

    /// Average time per epoch.
    public var averageEpochTime: TimeInterval {
        epochCount > 0 ? optimizationTime / TimeInterval(epochCount) : 0
    }

    /// Total time for all operations.
    public var totalTime: TimeInterval {
        knnGraphTime + fuzzySetTime + spectralInitTime + optimizationTime
    }

    /// Whether GPU acceleration was used for optimization.
    public let usedGPU: Bool

    /// Number of points processed.
    public let pointCount: Int

    /// Human-readable summary.
    public var summary: String {
        let accel = usedGPU ? "GPU" : "CPU"
        return """
        UMAP Timing (\(accel), \(pointCount) points):
          k-NN graph:          \(String(format: "%.3f", knnGraphTime))s
          Fuzzy simplicial:    \(String(format: "%.3f", fuzzySetTime))s
          Spectral init:       \(String(format: "%.3f", spectralInitTime))s
          Optimization:        \(String(format: "%.3f", optimizationTime))s (\(epochCount) epochs)
          Avg epoch time:      \(String(format: "%.4f", averageEpochTime))s
          Total:               \(String(format: "%.3f", totalTime))s
        """
    }
}

// MARK: - Progress Reporting

/// Progress handler for GPU operations with detailed phase information.
public typealias GPUProgressHandler = @Sendable (GPUOperationProgress) -> Void

/// Progress information for GPU operations.
public struct GPUOperationProgress: Sendable {

    /// The operation being performed.
    public let operation: GPUOperation

    /// Current phase within the operation.
    public let phase: String

    /// Progress within the current phase (0.0 to 1.0).
    public let phaseProgress: Float

    /// Overall progress for the operation (0.0 to 1.0).
    public let overallProgress: Float

    /// Optional epoch information (for iterative operations like UMAP).
    public let epochInfo: EpochInfo?

    /// Elapsed time since operation started.
    public let elapsedTime: TimeInterval

    /// Human-readable description.
    public var description: String {
        if let epoch = epochInfo {
            return "\(operation.name): \(phase) - Epoch \(epoch.current)/\(epoch.total) (\(Int(overallProgress * 100))%)"
        } else {
            return "\(operation.name): \(phase) (\(Int(overallProgress * 100))%)"
        }
    }
}

/// GPU operation type.
public enum GPUOperation: String, Sendable {
    case hdbscan = "HDBSCAN"
    case umap = "UMAP"
    case knn = "k-NN"
    case pairwiseDistance = "Pairwise Distance"

    /// Human-readable name.
    public var name: String { rawValue }
}

/// Epoch progress information.
public struct EpochInfo: Sendable {
    /// Current epoch (1-indexed).
    public let current: Int
    /// Total epochs.
    public let total: Int

    /// Progress as a fraction.
    public var progress: Float {
        total > 0 ? Float(current) / Float(total) : 0
    }
}

// MARK: - GPU HDBSCAN Result

/// Result of GPU-accelerated HDBSCAN distance computation.
///
/// Contains both core distances and the minimum spanning tree computed
/// in a single efficient GPU pipeline via VectorAccelerate.
public struct GPUHDBSCANResult: Sendable {

    /// Core distances for each point (k-th nearest neighbor distance).
    ///
    /// The core distance represents the local density around each point.
    /// Points in dense regions have small core distances.
    public let coreDistances: [Float]

    /// Minimum spanning tree over mutual reachability distances.
    ///
    /// This MST is used to build the cluster hierarchy. Edges are sorted
    /// by weight (ascending) to define the merge order.
    public let mst: MinimumSpanningTree
}

// MARK: - UMAP GPU Acceleration

extension TopicsGPUContext {

    /// Creates a UMAP gradient kernel.
    ///
    /// The kernel handles per-edge gradient computation using segmented reduction.
    /// For best performance, create once and reuse across epochs.
    ///
    /// - Returns: The UMAP gradient kernel.
    /// - Throws: `TopicsGPUError.gpuUnavailable` if GPU context not initialized.
    public func createUMAPGradientKernel() async throws -> UMAPGradientKernel {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        return try await UMAPGradientKernel(context: context)
    }

    /// Runs one epoch of UMAP optimization on GPU.
    ///
    /// This method uses VectorAccelerate's `UMAPGradientKernel` for parallel
    /// gradient computation, providing 10-50x speedup over CPU for large datasets.
    ///
    /// ## Algorithm
    ///
    /// Each epoch performs:
    /// 1. **Attractive gradients**: Pull connected points together based on edge weights
    /// 2. **Repulsive gradients**: Push random non-neighbors apart (negative sampling)
    /// 3. **Gradient application**: Update embedding positions with learning rate decay
    ///
    /// ## Performance
    ///
    /// | Dataset Size | Edges    | Expected Time/Epoch |
    /// |--------------|----------|---------------------|
    /// | 500 points   | ~7K      | ~5ms                |
    /// | 1,000 points | ~15K     | ~10ms               |
    /// | 5,000 points | ~75K     | ~50ms               |
    ///
    /// - Parameters:
    ///   - embedding: Current N×D embedding (modified in place).
    ///   - edges: Edge tuples (source, target, weight) - will be sorted internally.
    ///   - learningRate: Current learning rate (typically decayed over epochs).
    ///   - negativeSampleRate: Number of negative samples per positive edge.
    ///   - a: UMAP curve parameter a (derived from minDist).
    ///   - b: UMAP curve parameter b (derived from minDist).
    ///   - kernel: Optional pre-created kernel (creates new one if nil).
    /// - Throws: `TopicsGPUError.gpuUnavailable` if GPU context not initialized.
    public func optimizeUMAPEpoch(
        embedding: inout [[Float]],
        edges: [(source: Int, target: Int, weight: Float)],
        learningRate: Float,
        negativeSampleRate: Int,
        a: Float,
        b: Float,
        kernel: UMAPGradientKernel? = nil
    ) async throws {
        let gradientKernel: UMAPGradientKernel
        if let existingKernel = kernel {
            gradientKernel = existingKernel
        } else {
            gradientKernel = try await createUMAPGradientKernel()
        }

        // Convert edges to VectorAccelerate UMAPEdge format
        var umapEdges = edges.map { edge in
            UMAPEdge(
                source: edge.source,
                target: edge.target,
                weight: edge.weight
            )
        }

        // Sort edges by source (required for segmented reduction)
        umapEdges.sort { $0.source < $1.source }

        // Create parameters
        let params = UMAPParameters(
            a: a,
            b: b,
            learningRate: learningRate,
            negativeSampleRate: negativeSampleRate,
            epsilon: 0.001
        )

        try await gradientKernel.optimizeEpoch(
            embedding: &embedding,
            edges: umapEdges,
            params: params
        )
    }
}

// MARK: - GPU Error

/// Errors from GPU operations.
public enum TopicsGPUError: Error, Sendable {

    /// GPU is not available.
    case gpuUnavailable

    /// Invalid parameter provided.
    case invalidParameter(String)

    /// GPU computation failed.
    case computeFailed(String)

    /// Buffer allocation failed.
    case bufferAllocationFailed(size: Int)

    /// Kernel compilation failed.
    case kernelCompilationFailed(String)
}

extension TopicsGPUError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .gpuUnavailable:
            return "GPU acceleration is not available"
        case .invalidParameter(let message):
            return "Invalid parameter: \(message)"
        case .computeFailed(let message):
            return "GPU computation failed: \(message)"
        case .bufferAllocationFailed(let size):
            return "Failed to allocate GPU buffer of size \(size)"
        case .kernelCompilationFailed(let message):
            return "Kernel compilation failed: \(message)"
        }
    }
}

// MARK: - Array Extension

extension Array {
    /// Splits array into chunks of the specified size.
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
