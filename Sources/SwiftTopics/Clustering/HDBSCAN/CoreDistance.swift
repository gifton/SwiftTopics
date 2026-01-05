// CoreDistance.swift
// SwiftTopics
//
// Core distance computation for HDBSCAN clustering

import Foundation

// MARK: - Core Distance Computer

/// Computes core distances for HDBSCAN clustering.
///
/// The core distance of a point is the distance to its k-th nearest neighbor,
/// which serves as a measure of local density. Points in dense regions have
/// small core distances; points in sparse regions have large core distances.
///
/// ## Algorithm
///
/// For each point p:
/// 1. Find the k nearest neighbors of p
/// 2. The core distance is the distance to the k-th neighbor
///
/// Where k = `minSamples` in HDBSCAN configuration.
///
/// ## Mathematical Background
///
/// Core distance is the radius of the smallest ball centered at a point
/// that contains at least `minSamples` points (including the point itself).
/// This is used in the mutual reachability distance:
///
/// ```
/// mutual_reach(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
/// ```
///
/// ## Performance
///
/// - GPU path: O(n × k) using FusedL2TopKKernel via GPUBatchKNN
/// - CPU path: O(n² log k) using heap-based selection
///
/// The GPU path is strongly preferred for n > 100 points.
public struct CoreDistanceComputer: Sendable {

    /// The number of neighbors to consider (k value).
    public let minSamples: Int

    /// Whether to prefer GPU computation when available.
    public let preferGPU: Bool

    /// Creates a core distance computer.
    ///
    /// - Parameters:
    ///   - minSamples: The k value (minimum neighbors for core point).
    ///   - preferGPU: Whether to prefer GPU computation (default: true).
    public init(minSamples: Int, preferGPU: Bool = true) {
        precondition(minSamples >= 1, "minSamples must be at least 1")
        self.minSamples = minSamples
        self.preferGPU = preferGPU
    }

    // MARK: - Computation

    /// Computes core distances for all points.
    ///
    /// - Parameters:
    ///   - embeddings: The embeddings to compute core distances for.
    ///   - gpuContext: Optional GPU context for acceleration.
    /// - Returns: Core distance for each point.
    /// - Throws: `ClusteringError` if computation fails.
    public func compute(
        embeddings: [Embedding],
        gpuContext: TopicsGPUContext?
    ) async throws -> [Float] {
        let n = embeddings.count

        guard n > 0 else {
            throw ClusteringError.emptyInput
        }

        // Handle edge case: k > n
        let effectiveK = min(minSamples, n - 1)
        if effectiveK < 1 {
            // All points are identical or only one point
            return [Float](repeating: 0, count: n)
        }

        // Try GPU path if available and preferred
        if preferGPU, let context = gpuContext {
            do {
                return try await computeWithGPU(embeddings: embeddings, k: effectiveK, context: context)
            } catch {
                // Fall back to CPU on GPU failure
            }
        }

        // CPU fallback
        return computeWithCPU(embeddings: embeddings, k: effectiveK)
    }

    /// Computes core distances for pre-computed point vectors.
    ///
    /// - Parameters:
    ///   - points: The point vectors.
    ///   - gpuContext: Optional GPU context.
    /// - Returns: Core distance for each point.
    public func compute(
        points: [[Float]],
        gpuContext: TopicsGPUContext?
    ) async throws -> [Float] {
        let embeddings = points.map { Embedding(vector: $0) }
        return try await compute(embeddings: embeddings, gpuContext: gpuContext)
    }

    // MARK: - GPU Path

    private func computeWithGPU(
        embeddings: [Embedding],
        k: Int,
        context: TopicsGPUContext
    ) async throws -> [Float] {
        let points = embeddings.map { $0.vector }
        let gpuKNN = try GPUBatchKNN(context: context, dataset: points)
        return try await gpuKNN.computeCoreDistances(k: k)
    }

    // MARK: - CPU Path

    private func computeWithCPU(
        embeddings: [Embedding],
        k: Int
    ) -> [Float] {
        let n = embeddings.count
        var coreDistances = [Float](repeating: 0, count: n)

        for i in 0..<n {
            coreDistances[i] = computeKthNeighborDistance(
                pointIndex: i,
                embeddings: embeddings,
                k: k
            )
        }

        return coreDistances
    }

    /// Computes the k-th nearest neighbor distance for a single point.
    ///
    /// Uses a max-heap to maintain the k smallest distances seen.
    private func computeKthNeighborDistance(
        pointIndex: Int,
        embeddings: [Embedding],
        k: Int
    ) -> Float {
        let point = embeddings[pointIndex]
        var maxHeap = BoundedMaxHeapFloat(capacity: k)

        for j in 0..<embeddings.count {
            guard j != pointIndex else { continue }

            let dist = point.euclideanDistance(embeddings[j])
            maxHeap.insert(dist)
        }

        // The k-th nearest neighbor distance is the max in the heap
        return maxHeap.max ?? Float.infinity
    }
}

// MARK: - Bounded Max Heap for Floats

/// A bounded max-heap for finding k smallest elements.
///
/// Maintains the k smallest values seen so far. The maximum value
/// is at the root for O(1) access and O(log k) insertion.
private struct BoundedMaxHeapFloat {

    private var elements: [Float]
    private let capacity: Int

    var isFull: Bool { elements.count >= capacity }
    var max: Float? { elements.first }
    var count: Int { elements.count }

    init(capacity: Int) {
        self.capacity = capacity
        self.elements = []
        self.elements.reserveCapacity(capacity)
    }

    /// Inserts a new element if it's smaller than current max.
    mutating func insert(_ value: Float) {
        if elements.count < capacity {
            elements.append(value)
            siftUp(elements.count - 1)
        } else if value < elements[0] {
            elements[0] = value
            siftDown(0)
        }
    }

    private mutating func siftUp(_ index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            if elements[child] > elements[parent] {
                elements.swapAt(child, parent)
                child = parent
            } else {
                break
            }
        }
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        let count = elements.count

        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var largest = parent

            if left < count && elements[left] > elements[largest] {
                largest = left
            }
            if right < count && elements[right] > elements[largest] {
                largest = right
            }

            if largest == parent {
                break
            }

            elements.swapAt(parent, largest)
            parent = largest
        }
    }
}

// MARK: - Core Distance Result

/// Result of core distance computation with metadata.
public struct CoreDistanceResult: Sendable {

    /// Core distance for each point.
    public let distances: [Float]

    /// The k value used.
    public let k: Int

    /// Whether GPU was used for computation.
    public let usedGPU: Bool

    /// Computation time in seconds.
    public let computeTime: TimeInterval

    /// Minimum core distance (densest region).
    public var minCoreDistance: Float {
        distances.min() ?? 0
    }

    /// Maximum core distance (sparsest region).
    public var maxCoreDistance: Float {
        distances.max() ?? 0
    }

    /// Mean core distance.
    public var meanCoreDistance: Float {
        guard !distances.isEmpty else { return 0 }
        return distances.reduce(0, +) / Float(distances.count)
    }

    /// Creates a core distance result.
    public init(
        distances: [Float],
        k: Int,
        usedGPU: Bool,
        computeTime: TimeInterval
    ) {
        self.distances = distances
        self.k = k
        self.usedGPU = usedGPU
        self.computeTime = computeTime
    }
}
