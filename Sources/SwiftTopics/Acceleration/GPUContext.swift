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
    }

    /// Creates a GPU context with custom configuration.
    ///
    /// - Parameter configuration: Custom GPU configuration.
    public init(configuration: TopicsGPUConfiguration) async throws {
        self.configuration = configuration
        self.metal4Context = try await Self.initializeContext(configuration: configuration)
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
    ///
    /// - Parameter embeddings: The embeddings to compute distances for.
    /// - Returns: 2D distance matrix [n][n].
    /// - Complexity: O(n² × d) computation.
    public func computePairwiseL2Distances(_ embeddings: [Embedding]) async throws -> [[Float]] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        guard !embeddings.isEmpty else { return [] }

        // Convert to 2D array for VectorAccelerate API
        let vectors = embeddings.map { $0.vector }

        // Use L2DistanceKernel from VectorAccelerate
        let kernel = try await L2DistanceKernel(context: context)
        let distances = try await kernel.compute(
            queries: vectors,
            database: vectors,
            computeSqrt: true
        )

        return distances
    }

    /// Computes pairwise cosine similarities between all embeddings.
    ///
    /// - Parameter embeddings: The embeddings to compute similarities for.
    /// - Returns: 2D similarity matrix [n][n].
    public func computePairwiseCosineSimilarity(_ embeddings: [Embedding]) async throws -> [[Float]] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        guard !embeddings.isEmpty else { return [] }

        let vectors = embeddings.map { $0.vector }

        let kernel = try await CosineSimilarityKernel(context: context)
        let similarities = try await kernel.compute(
            queries: vectors,
            database: vectors
        )

        return similarities
    }

    // MARK: - K-Nearest Neighbors

    /// Computes k-nearest neighbors for all embeddings using GPU.
    ///
    /// Uses VectorAccelerate's `FusedL2TopKKernel` for efficient computation.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to find neighbors for.
    ///   - k: Number of neighbors to find.
    /// - Returns: For each embedding: array of (index, distance) pairs.
    public func computeBatchKNN(
        _ embeddings: [Embedding],
        k: Int
    ) async throws -> [[(index: Int, distance: Float)]] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        let n = embeddings.count
        guard n > 0 else { return [] }
        guard k > 0 && k <= n else {
            throw TopicsGPUError.invalidParameter("k must be between 1 and \(n)")
        }

        let vectors = embeddings.map { $0.vector }

        // Use FusedL2TopKKernel from VectorAccelerate
        let kernel = try await FusedL2TopKKernel(context: context)
        let results = try await kernel.findNearestNeighbors(
            queries: vectors,
            dataset: vectors,
            k: k
        )

        return results
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
    /// - Parameters:
    ///   - embeddings: The embeddings to cluster.
    ///   - minSamples: The k value for core distance (typically HDBSCAN's minSamples).
    /// - Returns: Core distances and minimum spanning tree for cluster hierarchy construction.
    /// - Throws: `TopicsGPUError.gpuUnavailable` if GPU context not initialized.
    public func computeHDBSCANMSTWithCoreDistances(
        _ embeddings: [Embedding],
        minSamples: Int
    ) async throws -> GPUHDBSCANResult {
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
    /// - Parameter embeddings: The embeddings to normalize.
    /// - Returns: Normalized embeddings (unit length).
    public func normalizeL2(_ embeddings: [Embedding]) async throws -> [Embedding] {
        guard let context = metal4Context else {
            throw TopicsGPUError.gpuUnavailable
        }

        guard !embeddings.isEmpty else { return [] }

        let vectors = embeddings.map { $0.vector }

        let kernel = try await L2NormalizationKernel(context: context)
        let result = try await kernel.normalize(vectors)

        return result.asArrays().map { Embedding(vector: $0) }
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

    /// Creates GPU configuration.
    ///
    /// - Parameters:
    ///   - preferHighPerformance: Prefer high-performance GPU device.
    ///   - maxBufferPoolMemory: Maximum memory for buffer pool (bytes).
    ///   - enableProfiling: Enable performance profiling.
    ///   - gpuMinPointsThreshold: Minimum points to use GPU (default: 100).
    public init(
        preferHighPerformance: Bool = true,
        maxBufferPoolMemory: Int? = nil,
        enableProfiling: Bool = false,
        gpuMinPointsThreshold: Int = 100
    ) {
        self.preferHighPerformance = preferHighPerformance
        self.maxBufferPoolMemory = maxBufferPoolMemory
        self.enableProfiling = enableProfiling
        self.gpuMinPointsThreshold = max(1, gpuMinPointsThreshold)
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
