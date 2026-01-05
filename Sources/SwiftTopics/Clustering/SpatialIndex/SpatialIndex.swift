// SpatialIndex.swift
// SwiftTopics
//
// Protocol for spatial indexing and k-NN queries

import Foundation

// MARK: - Spatial Index Protocol

/// A spatial index for efficient k-nearest neighbor queries.
///
/// Spatial indices partition high-dimensional space to avoid O(n) brute-force
/// distance computations for each query. They provide O(log n) average-case
/// query performance at the cost of O(n log n) construction time.
///
/// ## Available Implementations
/// - `BallTree`: Hierarchical hypersphere partitioning (CPU)
/// - `GPUBatchKNN`: GPU-accelerated batch k-NN via VectorAccelerate
///
/// ## Usage
/// ```swift
/// // Build index from embeddings
/// let index = try BallTree.build(points: embeddings.map { $0.vector })
///
/// // Query k nearest neighbors
/// let neighbors = index.query(point: queryVector, k: 10)
/// for (idx, dist) in neighbors {
///     print("Point \(idx) at distance \(dist)")
/// }
///
/// // Radius query
/// let nearby = index.queryRadius(point: queryVector, radius: 0.5)
/// ```
///
/// ## Thread Safety
/// Implementations must be thread-safe for read operations (queries).
/// Construction is typically single-threaded.
///
/// ## Performance Characteristics
/// | Operation | BallTree | GPU Batch |
/// |-----------|----------|-----------|
/// | Build | O(n log n) | N/A (no build) |
/// | Query (single) | O(log n) avg | O(n) |
/// | Query (batch) | O(m log n) | O(m × n / GPU) |
///
/// Use BallTree for single queries or small batches.
/// Use GPUBatchKNN for large batch queries where GPU overhead is amortized.
public protocol SpatialIndex: Sendable {

    /// The number of points in the index.
    var count: Int { get }

    /// The dimensionality of indexed points.
    var dimension: Int { get }

    /// Finds the k nearest neighbors to a query point.
    ///
    /// - Parameters:
    ///   - point: The query point (must have same dimension as indexed points).
    ///   - k: Number of neighbors to find.
    /// - Returns: Array of (index, distance) pairs sorted by distance (ascending).
    /// - Complexity: O(log n) average for tree-based indices, O(n) worst case.
    func query(point: [Float], k: Int) -> [(index: Int, distance: Float)]

    /// Finds all points within a given radius of the query point.
    ///
    /// - Parameters:
    ///   - point: The query point (must have same dimension as indexed points).
    ///   - radius: Maximum distance threshold.
    /// - Returns: Array of (index, distance) pairs for all points within radius.
    /// - Complexity: O(log n + m) where m is the number of results.
    func queryRadius(point: [Float], radius: Float) -> [(index: Int, distance: Float)]
}

// MARK: - Spatial Index Configuration

/// Configuration for building spatial indices.
public struct SpatialIndexConfiguration: Sendable, Codable {

    /// Maximum number of points in a leaf node.
    ///
    /// Larger values reduce tree depth but increase leaf scan time.
    /// Typical range: 10-40.
    public let leafSize: Int

    /// Distance metric to use.
    public let metric: DistanceMetric

    /// Whether to build the index in parallel.
    public let parallelBuild: Bool

    /// Creates spatial index configuration.
    ///
    /// - Parameters:
    ///   - leafSize: Maximum points per leaf node (default: 20).
    ///   - metric: Distance metric (default: .euclidean).
    ///   - parallelBuild: Enable parallel construction (default: true).
    public init(
        leafSize: Int = 20,
        metric: DistanceMetric = .euclidean,
        parallelBuild: Bool = true
    ) {
        precondition(leafSize >= 1, "leafSize must be at least 1")
        self.leafSize = leafSize
        self.metric = metric
        self.parallelBuild = parallelBuild
    }

    /// Default configuration optimized for typical embedding dimensions.
    public static let `default` = SpatialIndexConfiguration()

    /// Configuration optimized for high-dimensional data (>100 dims).
    public static let highDimensional = SpatialIndexConfiguration(
        leafSize: 40,
        metric: .euclidean,
        parallelBuild: true
    )

    /// Configuration optimized for low-dimensional data (<50 dims).
    public static let lowDimensional = SpatialIndexConfiguration(
        leafSize: 10,
        metric: .euclidean,
        parallelBuild: true
    )
}

// MARK: - Distance Metric

/// Distance metrics for spatial indexing.
public enum DistanceMetric: String, Sendable, Codable, CaseIterable {

    /// Euclidean (L2) distance: √(Σ(a_i - b_i)²)
    case euclidean

    /// Squared Euclidean distance: Σ(a_i - b_i)²
    ///
    /// Faster than Euclidean (no sqrt), preserves ordering.
    case squaredEuclidean

    /// Cosine distance: 1 - cos(θ) = 1 - (a·b)/(||a||×||b||)
    case cosine

    /// Manhattan (L1) distance: Σ|a_i - b_i|
    case manhattan

    /// Computes the distance between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector.
    ///   - b: Second vector.
    /// - Returns: Distance according to this metric.
    /// - Precondition: Vectors must have the same dimension.
    public func distance(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have same dimension")

        switch self {
        case .euclidean:
            return Self.squaredEuclideanDistance(a, b).squareRoot()

        case .squaredEuclidean:
            return Self.squaredEuclideanDistance(a, b)

        case .cosine:
            return Self.cosineDistance(a, b)

        case .manhattan:
            return Self.manhattanDistance(a, b)
        }
    }

    /// Computes distances from a query to multiple targets.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - targets: Array of target vectors.
    /// - Returns: Array of distances.
    public func distances(_ query: [Float], _ targets: [[Float]]) -> [Float] {
        targets.map { distance(query, $0) }
    }

    // MARK: - Private Distance Implementations

    @inline(__always)
    private static func squaredEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
    }

    @inline(__always)
    private static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = (normA * normB).squareRoot()
        guard denom > Float.ulpOfOne else { return 1.0 }

        // Clamp to handle numerical errors
        let similarity = max(-1.0, min(1.0, dot / denom))
        return 1.0 - similarity
    }

    @inline(__always)
    private static func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            sum += abs(a[i] - b[i])
        }
        return sum
    }
}

// MARK: - Neighbor Result

/// A single k-NN query result.
public struct NeighborResult: Sendable, Comparable {

    /// Index of the neighbor in the original dataset.
    public let index: Int

    /// Distance from the query point.
    public let distance: Float

    /// Creates a neighbor result.
    public init(index: Int, distance: Float) {
        self.index = index
        self.distance = distance
    }

    public static func < (lhs: NeighborResult, rhs: NeighborResult) -> Bool {
        lhs.distance < rhs.distance
    }

    /// Converts to tuple format.
    public var asTuple: (index: Int, distance: Float) {
        (index: index, distance: distance)
    }
}

// MARK: - Spatial Index Error

/// Errors from spatial index operations.
public enum SpatialIndexError: Error, Sendable {

    /// No points provided to build index.
    case emptyPoints

    /// Dimension mismatch between query and indexed points.
    case dimensionMismatch(query: Int, indexed: Int)

    /// Invalid k value for k-NN query.
    case invalidK(k: Int, count: Int)

    /// Invalid radius for radius query.
    case invalidRadius(Float)

    /// Index construction failed.
    case constructionFailed(String)
}

extension SpatialIndexError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyPoints:
            return "Cannot build spatial index from empty points"
        case .dimensionMismatch(let query, let indexed):
            return "Query dimension (\(query)) doesn't match indexed dimension (\(indexed))"
        case .invalidK(let k, let count):
            return "Invalid k=\(k) for index with \(count) points"
        case .invalidRadius(let radius):
            return "Invalid radius: \(radius)"
        case .constructionFailed(let message):
            return "Index construction failed: \(message)"
        }
    }
}

// MARK: - Batch Query Extension

extension SpatialIndex {

    /// Performs batch k-NN queries.
    ///
    /// Default implementation queries sequentially. GPU-backed implementations
    /// may override for parallel execution.
    ///
    /// - Parameters:
    ///   - points: Query points.
    ///   - k: Number of neighbors per query.
    /// - Returns: Array of neighbor arrays, one per query point.
    public func queryBatch(points: [[Float]], k: Int) -> [[(index: Int, distance: Float)]] {
        points.map { query(point: $0, k: k) }
    }

    /// Performs batch radius queries.
    ///
    /// - Parameters:
    ///   - points: Query points.
    ///   - radius: Maximum distance threshold.
    /// - Returns: Array of neighbor arrays, one per query point.
    public func queryRadiusBatch(points: [[Float]], radius: Float) -> [[(index: Int, distance: Float)]] {
        points.map { queryRadius(point: $0, radius: radius) }
    }
}

// MARK: - Type-Erased Wrapper

/// Type-erased wrapper for spatial indices.
public struct AnySpatialIndex: SpatialIndex {

    private let _count: @Sendable () -> Int
    private let _dimension: @Sendable () -> Int
    private let _query: @Sendable ([Float], Int) -> [(index: Int, distance: Float)]
    private let _queryRadius: @Sendable ([Float], Float) -> [(index: Int, distance: Float)]

    /// Creates a type-erased spatial index.
    public init<S: SpatialIndex>(_ index: S) {
        self._count = { index.count }
        self._dimension = { index.dimension }
        self._query = { index.query(point: $0, k: $1) }
        self._queryRadius = { index.queryRadius(point: $0, radius: $1) }
    }

    public var count: Int { _count() }
    public var dimension: Int { _dimension() }

    public func query(point: [Float], k: Int) -> [(index: Int, distance: Float)] {
        _query(point, k)
    }

    public func queryRadius(point: [Float], radius: Float) -> [(index: Int, distance: Float)] {
        _queryRadius(point, radius)
    }
}
