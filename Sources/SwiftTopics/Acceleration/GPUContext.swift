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

    /// Creates GPU configuration.
    public init(
        preferHighPerformance: Bool = true,
        maxBufferPoolMemory: Int? = nil,
        enableProfiling: Bool = false
    ) {
        self.preferHighPerformance = preferHighPerformance
        self.maxBufferPoolMemory = maxBufferPoolMemory
        self.enableProfiling = enableProfiling
    }

    /// Default configuration.
    public static let `default` = TopicsGPUConfiguration()

    /// Configuration optimized for batch processing.
    public static let batch = TopicsGPUConfiguration(
        preferHighPerformance: true,
        maxBufferPoolMemory: 1024 * 1024 * 1024 // 1GB
    )

    /// Configuration for debugging.
    public static let debug = TopicsGPUConfiguration(
        preferHighPerformance: false,
        enableProfiling: true
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
