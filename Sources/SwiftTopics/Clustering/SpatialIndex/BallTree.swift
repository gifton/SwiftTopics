// BallTree.swift
// SwiftTopics
//
// Ball tree spatial index for efficient k-NN queries

import Foundation

// MARK: - Ball Tree

/// A ball tree for efficient k-nearest neighbor queries in high-dimensional space.
///
/// Ball trees partition space into nested hyperspheres. Each internal node stores
/// a bounding ball (center + radius) containing all points in its subtree.
/// Queries prune branches whose bounding balls cannot contain nearer neighbors.
///
/// ## Algorithm
///
/// **Construction** (O(n log n)):
/// 1. Find the dimension with maximum spread
/// 2. Split points at median along that dimension
/// 3. Recursively build left and right subtrees
/// 4. Compute bounding ball from children
///
/// **Query** (O(log n) average):
/// 1. Maintain a max-heap of k best candidates
/// 2. Traverse tree, computing distance to current node's center
/// 3. Prune if `distance(query, center) - radius > current_kth_best`
/// 4. Visit closer child first (better pruning)
///
/// ## Performance Characteristics
///
/// - Build time: O(n log n)
/// - Query time: O(log n) average, O(n) worst case
/// - Memory: O(n) for tree structure
/// - Best for: d < 100, single/small-batch queries
///
/// ## Thread Safety
/// Ball trees are immutable after construction and safe for concurrent queries.
///
/// ## Usage
/// ```swift
/// let tree = try BallTree.build(points: embeddings.map { $0.vector })
/// let neighbors = tree.query(point: queryVector, k: 10)
/// ```
public final class BallTree: SpatialIndex, @unchecked Sendable {

    // MARK: - Node Structure

    /// A node in the ball tree.
    private final class Node {
        /// Center of the bounding ball.
        let center: [Float]

        /// Radius of the bounding ball.
        let radius: Float

        /// Point indices contained in this subtree (for leaf nodes).
        /// Empty for internal nodes.
        let pointIndices: [Int]

        /// Left child (closer to center on split dimension).
        var left: Node?

        /// Right child (farther from center on split dimension).
        var right: Node?

        /// Whether this is a leaf node.
        var isLeaf: Bool { left == nil && right == nil }

        init(center: [Float], radius: Float, pointIndices: [Int]) {
            self.center = center
            self.radius = radius
            self.pointIndices = pointIndices
            self.left = nil
            self.right = nil
        }
    }

    // MARK: - Properties

    /// The root node of the tree.
    private let root: Node

    /// Original points (indexed by point indices in nodes).
    private let points: [[Float]]

    /// Configuration used to build this tree.
    public let configuration: SpatialIndexConfiguration

    public var count: Int { points.count }
    public var dimension: Int { points.isEmpty ? 0 : points[0].count }

    // MARK: - Construction

    /// Builds a ball tree from the given points.
    ///
    /// - Parameters:
    ///   - points: Array of point vectors (must all have same dimension).
    ///   - configuration: Build configuration.
    /// - Returns: The constructed ball tree.
    /// - Throws: `SpatialIndexError` if construction fails.
    public static func build(
        points: [[Float]],
        configuration: SpatialIndexConfiguration = .default
    ) throws -> BallTree {
        guard !points.isEmpty else {
            throw SpatialIndexError.emptyPoints
        }

        let dimension = points[0].count
        guard points.allSatisfy({ $0.count == dimension }) else {
            throw SpatialIndexError.constructionFailed("All points must have same dimension")
        }

        let indices = Array(0..<points.count)
        let root = buildNode(
            points: points,
            indices: indices,
            leafSize: configuration.leafSize,
            metric: configuration.metric
        )

        return BallTree(root: root, points: points, configuration: configuration)
    }

    private init(root: Node, points: [[Float]], configuration: SpatialIndexConfiguration) {
        self.root = root
        self.points = points
        self.configuration = configuration
    }

    // MARK: - Recursive Build

    private static func buildNode(
        points: [[Float]],
        indices: [Int],
        leafSize: Int,
        metric: DistanceMetric
    ) -> Node {
        guard !indices.isEmpty else {
            // Should never happen, but handle gracefully
            return Node(center: [], radius: 0, pointIndices: [])
        }

        let dimension = points[0].count

        // Compute centroid (center of mass)
        var center = [Float](repeating: 0, count: dimension)
        for idx in indices {
            let point = points[idx]
            for d in 0..<dimension {
                center[d] += point[d]
            }
        }
        let n = Float(indices.count)
        for d in 0..<dimension {
            center[d] /= n
        }

        // Compute radius (max distance from center to any point)
        var radius: Float = 0
        for idx in indices {
            let dist = metric.distance(center, points[idx])
            radius = max(radius, dist)
        }

        // Create leaf if small enough
        if indices.count <= leafSize {
            return Node(center: center, radius: radius, pointIndices: indices)
        }

        // Find dimension with maximum spread
        let splitDim = findSplitDimension(points: points, indices: indices)

        // Sort indices by coordinate on split dimension
        var sortedIndices = indices
        sortedIndices.sort { points[$0][splitDim] < points[$1][splitDim] }

        // Split at median
        let mid = sortedIndices.count / 2
        let leftIndices = Array(sortedIndices[0..<mid])
        let rightIndices = Array(sortedIndices[mid...])

        // Build children recursively
        let node = Node(center: center, radius: radius, pointIndices: [])

        if !leftIndices.isEmpty {
            node.left = buildNode(points: points, indices: leftIndices, leafSize: leafSize, metric: metric)
        }
        if !rightIndices.isEmpty {
            node.right = buildNode(points: points, indices: rightIndices, leafSize: leafSize, metric: metric)
        }

        return node
    }

    /// Finds the dimension with maximum spread (variance).
    private static func findSplitDimension(points: [[Float]], indices: [Int]) -> Int {
        guard !indices.isEmpty else { return 0 }

        let dimension = points[0].count
        var bestDim = 0
        var maxSpread: Float = -1

        for d in 0..<dimension {
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity

            for idx in indices {
                let val = points[idx][d]
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }

            let spread = maxVal - minVal
            if spread > maxSpread {
                maxSpread = spread
                bestDim = d
            }
        }

        return bestDim
    }

    // MARK: - k-NN Query

    public func query(point: [Float], k: Int) -> [(index: Int, distance: Float)] {
        guard !points.isEmpty else { return [] }
        guard point.count == dimension else { return [] }

        let actualK = min(k, count)
        guard actualK > 0 else { return [] }

        // Use a bounded priority queue (max-heap by distance)
        var heap = BoundedMaxHeap(capacity: actualK)

        // Recursive search with pruning
        queryNode(root, query: point, heap: &heap)

        // Extract results sorted by distance (ascending)
        return heap.sortedResults()
    }

    private func queryNode(_ node: Node, query: [Float], heap: inout BoundedMaxHeap) {
        let metric = configuration.metric
        let distToCenter = metric.distance(query, node.center)

        // Pruning: if the closest possible point in this ball is farther
        // than our k-th best, skip this subtree
        let closestPossible = max(0, distToCenter - node.radius)
        if heap.isFull && closestPossible >= heap.maxDistance {
            return
        }

        // Leaf node: check all points
        if node.isLeaf {
            for idx in node.pointIndices {
                let dist = metric.distance(query, points[idx])
                heap.insert(index: idx, distance: dist)
            }
            return
        }

        // Internal node: visit children, closer subtree first
        let leftDist = node.left.map { metric.distance(query, $0.center) } ?? .infinity
        let rightDist = node.right.map { metric.distance(query, $0.center) } ?? .infinity

        if leftDist <= rightDist {
            if let left = node.left {
                queryNode(left, query: query, heap: &heap)
            }
            if let right = node.right {
                queryNode(right, query: query, heap: &heap)
            }
        } else {
            if let right = node.right {
                queryNode(right, query: query, heap: &heap)
            }
            if let left = node.left {
                queryNode(left, query: query, heap: &heap)
            }
        }
    }

    // MARK: - Radius Query

    public func queryRadius(point: [Float], radius: Float) -> [(index: Int, distance: Float)] {
        guard !points.isEmpty else { return [] }
        guard point.count == dimension else { return [] }
        guard radius >= 0 else { return [] }

        var results: [(index: Int, distance: Float)] = []
        queryRadiusNode(root, query: point, radius: radius, results: &results)

        // Sort by distance
        results.sort { $0.distance < $1.distance }
        return results
    }

    private func queryRadiusNode(
        _ node: Node,
        query: [Float],
        radius: Float,
        results: inout [(index: Int, distance: Float)]
    ) {
        let metric = configuration.metric
        let distToCenter = metric.distance(query, node.center)

        // Pruning: if the closest possible point is beyond radius, skip
        let closestPossible = max(0, distToCenter - node.radius)
        if closestPossible > radius {
            return
        }

        // Leaf node: check all points
        if node.isLeaf {
            for idx in node.pointIndices {
                let dist = metric.distance(query, points[idx])
                if dist <= radius {
                    results.append((index: idx, distance: dist))
                }
            }
            return
        }

        // Internal node: recurse into both children
        if let left = node.left {
            queryRadiusNode(left, query: query, radius: radius, results: &results)
        }
        if let right = node.right {
            queryRadiusNode(right, query: query, radius: radius, results: &results)
        }
    }

    // MARK: - Batch Query (Optimized)

    /// Performs batch k-NN queries with potential parallelization.
    ///
    /// - Parameters:
    ///   - points: Query points.
    ///   - k: Number of neighbors per query.
    /// - Returns: Array of neighbor arrays.
    public func queryBatch(points queryPoints: [[Float]], k: Int) -> [[(index: Int, distance: Float)]] {
        guard !queryPoints.isEmpty else { return [] }

        // Use sequential processing (concurrent version has Sendable issues in Swift 6)
        // For most use cases, the per-query work dominates and parallelism
        // is better achieved at the algorithm level (e.g., GPU batch k-NN)
        return queryPoints.map { query(point: $0, k: k) }
    }
}

// MARK: - Bounded Max Heap

/// A bounded max-heap for k-NN queries.
///
/// Maintains the k smallest distances seen so far. The maximum distance
/// is always at the root for O(1) pruning decisions.
private struct BoundedMaxHeap {

    private var elements: [(index: Int, distance: Float)]
    private let capacity: Int

    var isFull: Bool { elements.count >= capacity }
    var maxDistance: Float { elements.first?.distance ?? .infinity }
    var count: Int { elements.count }

    init(capacity: Int) {
        self.capacity = capacity
        self.elements = []
        self.elements.reserveCapacity(capacity)
    }

    /// Inserts a new element if it's better than current worst.
    mutating func insert(index: Int, distance: Float) {
        if elements.count < capacity {
            // Not full yet, always insert
            elements.append((index: index, distance: distance))
            siftUp(elements.count - 1)
        } else if distance < elements[0].distance {
            // Better than current worst, replace root
            elements[0] = (index: index, distance: distance)
            siftDown(0)
        }
    }

    /// Returns results sorted by distance (ascending).
    func sortedResults() -> [(index: Int, distance: Float)] {
        elements.sorted { $0.distance < $1.distance }
    }

    // MARK: - Heap Operations

    private mutating func siftUp(_ index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            if elements[child].distance > elements[parent].distance {
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

            if left < count && elements[left].distance > elements[largest].distance {
                largest = left
            }
            if right < count && elements[right].distance > elements[largest].distance {
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

// MARK: - Ball Tree Statistics

extension BallTree {

    /// Statistics about the ball tree structure.
    public struct Statistics {
        /// Total number of nodes.
        public let nodeCount: Int

        /// Number of leaf nodes.
        public let leafCount: Int

        /// Maximum tree depth.
        public let maxDepth: Int

        /// Average points per leaf.
        public let avgPointsPerLeaf: Float

        /// Total number of points.
        public let pointCount: Int
    }

    /// Computes statistics about this ball tree.
    public func computeStatistics() -> Statistics {
        var nodeCount = 0
        var leafCount = 0
        var maxDepth = 0
        var totalLeafPoints = 0

        func traverse(_ node: Node, depth: Int) {
            nodeCount += 1
            maxDepth = max(maxDepth, depth)

            if node.isLeaf {
                leafCount += 1
                totalLeafPoints += node.pointIndices.count
            } else {
                if let left = node.left {
                    traverse(left, depth: depth + 1)
                }
                if let right = node.right {
                    traverse(right, depth: depth + 1)
                }
            }
        }

        traverse(root, depth: 0)

        return Statistics(
            nodeCount: nodeCount,
            leafCount: leafCount,
            maxDepth: maxDepth,
            avgPointsPerLeaf: leafCount > 0 ? Float(totalLeafPoints) / Float(leafCount) : 0,
            pointCount: count
        )
    }
}
