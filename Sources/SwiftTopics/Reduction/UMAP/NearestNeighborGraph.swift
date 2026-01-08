// NearestNeighborGraph.swift
// SwiftTopics
//
// k-Nearest Neighbor graph construction for UMAP

import Foundation

// MARK: - Nearest Neighbor Graph

/// A k-nearest neighbor graph storing neighbors and distances for each point.
///
/// This is the first step in UMAP: construct a graph where each node is connected
/// to its k nearest neighbors in the high-dimensional space.
///
/// ## Algorithm
/// 1. Build a spatial index (BallTree) from the input points
/// 2. For each point, query k+1 nearest neighbors (including self)
/// 3. Exclude self-loop and store k neighbors
///
/// ## Why k-NN?
/// UMAP is based on the idea that data lies on a manifold. The k-NN graph
/// approximates the local manifold structure around each point. The choice of k
/// controls the trade-off between local (small k) and global (large k) structure.
///
/// ## Thread Safety
/// `NearestNeighborGraph` is immutable and `Sendable`.
public struct NearestNeighborGraph: Sendable {

    // MARK: - Properties

    /// Neighbor indices for each point. Shape: [pointCount][k]
    /// neighbors[i] contains the k nearest neighbor indices for point i.
    public let neighbors: [[Int]]

    /// Distances to neighbors for each point. Shape: [pointCount][k]
    /// distances[i][j] is the distance from point i to neighbors[i][j].
    public let distances: [[Float]]

    /// Number of points in the graph.
    public let pointCount: Int

    /// Number of neighbors per point.
    public let k: Int

    /// Distance metric used for construction.
    public let metric: DistanceMetric

    // MARK: - Initialization

    /// Creates a nearest neighbor graph from precomputed neighbors and distances.
    ///
    /// - Parameters:
    ///   - neighbors: Neighbor indices for each point.
    ///   - distances: Distances to neighbors.
    ///   - k: Number of neighbors per point.
    ///   - metric: Distance metric used.
    public init(
        neighbors: [[Int]],
        distances: [[Float]],
        k: Int,
        metric: DistanceMetric = .euclidean
    ) {
        precondition(!neighbors.isEmpty, "Neighbors cannot be empty")
        precondition(neighbors.count == distances.count, "Neighbors and distances must have same length")

        self.neighbors = neighbors
        self.distances = distances
        self.pointCount = neighbors.count
        self.k = k
        self.metric = metric
    }

    // MARK: - Construction

    /// Builds a k-NN graph from embeddings.
    ///
    /// Uses GPU acceleration when a `gpuContext` is provided and the dataset
    /// is large enough (>= threshold). Falls back to BallTree for CPU path
    /// or when GPU is unavailable.
    ///
    /// ## GPU Acceleration
    ///
    /// When `gpuContext` is provided:
    /// - Uses `FusedL2TopKKernel` for ~27x speedup on k-NN computation
    /// - Automatically falls back to CPU if GPU fails
    /// - Only used when `metric == .euclidean` (GPU kernel limitation)
    ///
    /// ## Performance
    ///
    /// | Points | CPU (BallTree) | GPU (FusedL2TopK) | Speedup |
    /// |--------|----------------|-------------------|---------|
    /// | 500    | ~1.5s          | ~0.1s             | ~15x    |
    /// | 1000   | ~8s            | ~0.3s             | ~27x    |
    /// | 2000   | ~35s           | ~1s               | ~35x    |
    ///
    /// - Parameters:
    ///   - embeddings: Input embeddings.
    ///   - k: Number of neighbors to find for each point.
    ///   - metric: Distance metric to use.
    ///   - gpuContext: Optional GPU context for acceleration.
    /// - Returns: The constructed k-NN graph.
    /// - Throws: `ReductionError` if construction fails.
    public static func build(
        embeddings: [Embedding],
        k: Int,
        metric: DistanceMetric = .euclidean,
        gpuContext: TopicsGPUContext? = nil
    ) async throws -> NearestNeighborGraph {
        guard !embeddings.isEmpty else {
            throw ReductionError.emptyInput
        }

        let n = embeddings.count
        let actualK = min(k, n - 1)  // Can't have more neighbors than points-1

        guard actualK > 0 else {
            throw ReductionError.insufficientSamples(required: 2, provided: n)
        }

        // Convert embeddings to point arrays
        let points = embeddings.map { $0.vector }

        // Validate dimensions
        let dimension = points[0].count
        guard points.allSatisfy({ $0.count == dimension }) else {
            throw ReductionError.inconsistentDimensions
        }

        // Try GPU path if available and beneficial
        let gpuThreshold = gpuContext?.configuration.gpuMinPointsThreshold ?? 100
        if let gpu = gpuContext, n >= gpuThreshold, metric == .euclidean {
            do {
                return try await buildWithGPU(
                    embeddings: embeddings,
                    k: actualK,
                    gpuContext: gpu
                )
            } catch {
                // Log warning and fall back to BallTree
                // GPU failed - continue with CPU path
            }
        }

        // CPU path: Build spatial index
        let configuration = SpatialIndexConfiguration(
            leafSize: 20,
            metric: metric,
            parallelBuild: true
        )

        let ballTree: BallTree
        do {
            ballTree = try BallTree.build(points: points, configuration: configuration)
        } catch let error as SpatialIndexError {
            throw ReductionError.unknown("Failed to build spatial index: \(error)")
        }

        // Query k-NN for each point
        var allNeighbors = [[Int]](repeating: [], count: n)
        var allDistances = [[Float]](repeating: [], count: n)

        // Query k+1 neighbors (to exclude self)
        let queryK = actualK + 1

        for i in 0..<n {
            let result = ballTree.query(point: points[i], k: queryK)

            // Filter out self (distance ~0) and keep k neighbors
            var neighborIndices = [Int]()
            var neighborDistances = [Float]()

            neighborIndices.reserveCapacity(actualK)
            neighborDistances.reserveCapacity(actualK)

            for (idx, dist) in result {
                if idx != i {  // Exclude self
                    neighborIndices.append(idx)
                    neighborDistances.append(dist)
                    if neighborIndices.count >= actualK {
                        break
                    }
                }
            }

            // Pad if we didn't get enough neighbors (edge case)
            while neighborIndices.count < actualK {
                // Find any unused neighbor
                for j in 0..<n where j != i && !neighborIndices.contains(j) {
                    neighborIndices.append(j)
                    neighborDistances.append(metric.distance(points[i], points[j]))
                    if neighborIndices.count >= actualK {
                        break
                    }
                }
            }

            allNeighbors[i] = neighborIndices
            allDistances[i] = neighborDistances
        }

        return NearestNeighborGraph(
            neighbors: allNeighbors,
            distances: allDistances,
            k: actualK,
            metric: metric
        )
    }

    /// Builds k-NN graph using GPU acceleration.
    ///
    /// Uses VectorAccelerate's `FusedL2TopKKernel` for efficient GPU k-NN.
    private static func buildWithGPU(
        embeddings: [Embedding],
        k: Int,
        gpuContext: TopicsGPUContext
    ) async throws -> NearestNeighborGraph {
        // Use existing computeBatchKNN which wraps FusedL2TopKKernel
        // Request k+1 neighbors to account for self-neighbor
        let knnResult = try await gpuContext.computeBatchKNN(embeddings, k: k + 1)

        let n = embeddings.count

        // Convert to NearestNeighborGraph format (exclude self-neighbors)
        var allNeighbors = [[Int]](repeating: [], count: n)
        var allDistances = [[Float]](repeating: [], count: n)

        for i in 0..<n {
            // Filter out self (index == i) and take first k neighbors
            let filtered = knnResult[i].filter { $0.index != i }.prefix(k)
            allNeighbors[i] = filtered.map { $0.index }
            allDistances[i] = filtered.map { $0.distance }

            // Pad if needed (edge case: self was in results)
            while allNeighbors[i].count < k && allNeighbors[i].count < n - 1 {
                // Find any neighbor not already included
                for j in 0..<n where j != i && !allNeighbors[i].contains(j) {
                    allNeighbors[i].append(j)
                    // Compute distance manually for padding
                    let dist = euclideanDistance(embeddings[i].vector, embeddings[j].vector)
                    allDistances[i].append(dist)
                    if allNeighbors[i].count >= k {
                        break
                    }
                }
            }
        }

        return NearestNeighborGraph(
            neighbors: allNeighbors,
            distances: allDistances,
            k: k,
            metric: .euclidean
        )
    }

    /// Computes Euclidean distance between two vectors.
    private static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    /// Builds a k-NN graph from raw point arrays.
    ///
    /// - Parameters:
    ///   - points: Input points as float arrays.
    ///   - k: Number of neighbors.
    ///   - metric: Distance metric.
    /// - Returns: The k-NN graph.
    /// - Throws: `ReductionError` if construction fails.
    public static func build(
        points: [[Float]],
        k: Int,
        metric: DistanceMetric = .euclidean
    ) async throws -> NearestNeighborGraph {
        let embeddings = points.map { Embedding(vector: $0) }
        return try await build(embeddings: embeddings, k: k, metric: metric)
    }

    // MARK: - Graph Properties

    /// Returns the nearest neighbor (excluding self) for each point.
    ///
    /// - Returns: Array of nearest neighbor indices.
    public var nearestNeighbors: [Int] {
        neighbors.map { $0.isEmpty ? -1 : $0[0] }
    }

    /// Returns the distance to the nearest neighbor for each point.
    ///
    /// This is the ρ (rho) value used in UMAP's fuzzy set construction.
    ///
    /// - Returns: Array of distances to nearest neighbors.
    public var nearestDistances: [Float] {
        distances.map { $0.isEmpty ? .infinity : $0[0] }
    }

    /// Checks if the graph is symmetric (if A is neighbor of B, B is neighbor of A).
    ///
    /// Note: k-NN graphs are generally NOT symmetric.
    public var isSymmetric: Bool {
        for i in 0..<pointCount {
            for j in neighbors[i] {
                if !neighbors[j].contains(i) {
                    return false
                }
            }
        }
        return true
    }

    /// Checks for disconnected components.
    ///
    /// Returns true if all points are reachable from point 0 via the
    /// undirected version of the k-NN graph.
    ///
    /// - Returns: True if graph is connected.
    public var isConnected: Bool {
        guard pointCount > 0 else { return true }

        var visited = [Bool](repeating: false, count: pointCount)
        var queue = [0]
        visited[0] = true
        var visitCount = 1

        while !queue.isEmpty {
            let current = queue.removeFirst()

            // Check outgoing edges
            for neighbor in neighbors[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true
                    visitCount += 1
                    queue.append(neighbor)
                }
            }

            // Check incoming edges (treat as undirected)
            for i in 0..<pointCount {
                if !visited[i] && neighbors[i].contains(current) {
                    visited[i] = true
                    visitCount += 1
                    queue.append(i)
                }
            }
        }

        return visitCount == pointCount
    }

    // MARK: - Conversion

    /// Converts to a sparse adjacency matrix.
    ///
    /// The values are the distances (not membership weights).
    ///
    /// - Returns: Sparse matrix of distances.
    public func toSparseMatrix() -> SparseMatrix<Float> {
        var entries: [(row: Int, col: Int, value: Float)] = []
        entries.reserveCapacity(pointCount * k)

        for i in 0..<pointCount {
            for (j, neighborIdx) in neighbors[i].enumerated() {
                entries.append((row: i, col: neighborIdx, value: distances[i][j]))
            }
        }

        return SparseMatrix.fromCOO(
            rows: pointCount,
            cols: pointCount,
            entries: entries
        )
    }

    /// Symmetrizes the graph by adding reverse edges.
    ///
    /// For each edge (i → j) with distance d, adds (j → i) with the same
    /// distance if it doesn't exist, or keeps the minimum distance.
    ///
    /// - Returns: Symmetrized k-NN graph (now undirected).
    public func symmetrized() -> NearestNeighborGraph {
        // Build adjacency map with minimum distances
        var adjacency = [[Int: Float]](repeating: [:], count: pointCount)

        for i in 0..<pointCount {
            for (j, neighborIdx) in neighbors[i].enumerated() {
                let dist = distances[i][j]

                // Add or update i → neighbor
                if let existing = adjacency[i][neighborIdx] {
                    adjacency[i][neighborIdx] = min(existing, dist)
                } else {
                    adjacency[i][neighborIdx] = dist
                }

                // Add reverse edge neighbor → i
                if let existing = adjacency[neighborIdx][i] {
                    adjacency[neighborIdx][i] = min(existing, dist)
                } else {
                    adjacency[neighborIdx][i] = dist
                }
            }
        }

        // Convert back to arrays
        var symNeighbors = [[Int]](repeating: [], count: pointCount)
        var symDistances = [[Float]](repeating: [], count: pointCount)

        for i in 0..<pointCount {
            // Sort by distance
            let sorted = adjacency[i].sorted { $0.value < $1.value }
            symNeighbors[i] = sorted.map { $0.key }
            symDistances[i] = sorted.map { $0.value }
        }

        // Note: After symmetrization, each point may have more than k neighbors
        return NearestNeighborGraph(
            neighbors: symNeighbors,
            distances: symDistances,
            k: symNeighbors.map { $0.count }.max() ?? k,
            metric: metric
        )
    }
}

// MARK: - Statistics

extension NearestNeighborGraph {

    /// Statistics about the k-NN graph.
    public struct Statistics: Sendable {
        /// Number of points.
        public let pointCount: Int

        /// Target number of neighbors per point.
        public let k: Int

        /// Average distance to nearest neighbor.
        public let meanNearestDistance: Float

        /// Maximum distance to nearest neighbor.
        public let maxNearestDistance: Float

        /// Minimum distance to nearest neighbor (excluding zero).
        public let minNearestDistance: Float

        /// Whether the graph is connected.
        public let isConnected: Bool

        /// Number of connected components (1 = fully connected).
        public let componentCount: Int
    }

    /// Computes statistics about this graph.
    ///
    /// - Returns: Graph statistics.
    public func computeStatistics() -> Statistics {
        let nearestDists = nearestDistances.filter { $0 > 0 && $0.isFinite }

        let meanNearest = nearestDists.isEmpty ? 0 :
            nearestDists.reduce(0, +) / Float(nearestDists.count)

        return Statistics(
            pointCount: pointCount,
            k: k,
            meanNearestDistance: meanNearest,
            maxNearestDistance: nearestDists.max() ?? 0,
            minNearestDistance: nearestDists.min() ?? 0,
            isConnected: isConnected,
            componentCount: countComponents()
        )
    }

    /// Counts the number of connected components.
    private func countComponents() -> Int {
        guard pointCount > 0 else { return 0 }

        var visited = [Bool](repeating: false, count: pointCount)
        var componentCount = 0

        for start in 0..<pointCount {
            guard !visited[start] else { continue }

            componentCount += 1

            // BFS from this unvisited node
            var queue = [start]
            visited[start] = true

            while !queue.isEmpty {
                let current = queue.removeFirst()

                for neighbor in neighbors[current] {
                    if !visited[neighbor] {
                        visited[neighbor] = true
                        queue.append(neighbor)
                    }
                }

                // Include reverse edges
                for i in 0..<pointCount {
                    if !visited[i] && neighbors[i].contains(current) {
                        visited[i] = true
                        queue.append(i)
                    }
                }
            }
        }

        return componentCount
    }
}

// MARK: - CustomStringConvertible

extension NearestNeighborGraph: CustomStringConvertible {
    public var description: String {
        "NearestNeighborGraph(n=\(pointCount), k=\(k), connected=\(isConnected))"
    }
}
