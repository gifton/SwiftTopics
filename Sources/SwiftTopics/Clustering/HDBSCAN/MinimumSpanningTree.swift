// MinimumSpanningTree.swift
// SwiftTopics
//
// Minimum spanning tree construction for HDBSCAN

import Foundation

// MARK: - MST Edge

/// An edge in the minimum spanning tree.
///
/// MST edges connect the cluster hierarchy—each edge represents a merge
/// point where two clusters join at a given distance (density level).
public struct MSTEdge: Sendable, Comparable {

    /// Index of the first point.
    public let source: Int

    /// Index of the second point.
    public let target: Int

    /// The edge weight (mutual reachability distance).
    public let weight: Float

    /// Creates an MST edge.
    public init(source: Int, target: Int, weight: Float) {
        self.source = source
        self.target = target
        self.weight = weight
    }

    public static func < (lhs: MSTEdge, rhs: MSTEdge) -> Bool {
        lhs.weight < rhs.weight
    }
}

// MARK: - Minimum Spanning Tree

/// A minimum spanning tree over points.
///
/// The MST has n-1 edges for n points. When sorted by weight (ascending),
/// the edges define the order in which points/clusters merge as we increase
/// the distance threshold (decrease density).
///
/// ## HDBSCAN Usage
///
/// The MST is converted to a dendrogram by processing edges in order:
/// 1. Start with each point as its own cluster
/// 2. For each edge (in ascending weight order), merge the clusters
/// 3. Track cluster births, deaths, and stability
///
/// ## Properties
///
/// - The MST is unique if all edge weights are distinct
/// - Sorting edges by weight gives a valid merge order
/// - Edge weights define density levels (λ = 1/distance)
public struct MinimumSpanningTree: Sendable {

    /// The MST edges.
    public let edges: [MSTEdge]

    /// Number of points in the tree.
    public let pointCount: Int

    /// Creates an MST.
    public init(edges: [MSTEdge], pointCount: Int) {
        precondition(
            edges.count == pointCount - 1 || pointCount == 0,
            "MST must have exactly n-1 edges for n points"
        )
        self.edges = edges
        self.pointCount = pointCount
    }

    /// Returns edges sorted by weight (ascending).
    ///
    /// This is the merge order for dendrogram construction.
    public var sortedEdges: [MSTEdge] {
        edges.sorted()
    }

    /// The total weight of the MST.
    public var totalWeight: Float {
        edges.reduce(0) { $0 + $1.weight }
    }

    /// The minimum edge weight.
    public var minWeight: Float {
        edges.min()?.weight ?? 0
    }

    /// The maximum edge weight.
    public var maxWeight: Float {
        edges.max()?.weight ?? 0
    }
}

// MARK: - MST Builder (Prim's Algorithm)

/// Builds a minimum spanning tree using Prim's algorithm.
///
/// ## Algorithm
///
/// Prim's algorithm grows the MST one edge at a time:
/// 1. Start with an arbitrary vertex in the tree
/// 2. While tree doesn't span all vertices:
///    a. Find minimum weight edge connecting tree to non-tree vertex
///    b. Add that edge and vertex to tree
///
/// ## Complexity
///
/// - Time: O(n²) using a dense graph approach with `minEdgeWeight` array
/// - Space: O(n) for tracking minimum edge weights and sources
///
/// For dense graphs (like mutual reachability), Prim's with adjacency matrix
/// is O(n²) and avoids the O(n² log n) cost of a binary heap.
public struct PrimMSTBuilder: Sendable {

    /// Creates an MST builder.
    public init() {}

    /// Builds the MST from a mutual reachability graph.
    ///
    /// - Parameter graph: The mutual reachability graph.
    /// - Returns: The minimum spanning tree.
    public func build(from graph: MutualReachabilityGraph) -> MinimumSpanningTree {
        let n = graph.count

        guard n > 0 else {
            return MinimumSpanningTree(edges: [], pointCount: 0)
        }

        guard n > 1 else {
            return MinimumSpanningTree(edges: [], pointCount: 1)
        }

        var mstEdges = [MSTEdge]()
        mstEdges.reserveCapacity(n - 1)

        // Track which points are in the MST
        var inMST = [Bool](repeating: false, count: n)

        // Track the minimum distance to each point from the MST
        var minEdgeWeight = [Float](repeating: Float.infinity, count: n)
        var minEdgeSource = [Int](repeating: -1, count: n)

        // Start from point 0
        inMST[0] = true
        var mstSize = 1
        var currentPoint = 0

        // Grow MST until all points included
        while mstSize < n {
            // Update minEdgeWeight for all points not in MST
            for j in 0..<n {
                if !inMST[j] {
                    let weight = graph.distance(from: currentPoint, to: j)
                    if weight < minEdgeWeight[j] {
                        minEdgeWeight[j] = weight
                        minEdgeSource[j] = currentPoint
                    }
                }
            }

            // Find the next point to add (min weight)
            var minWeight: Float = .infinity
            var nextPoint = -1

            for j in 0..<n {
                if !inMST[j] && minEdgeWeight[j] < minWeight {
                    minWeight = minEdgeWeight[j]
                    nextPoint = j
                }
            }

            guard nextPoint != -1 else { break }

            // Add edge to MST
            mstEdges.append(MSTEdge(
                source: minEdgeSource[nextPoint],
                target: nextPoint,
                weight: minWeight
            ))

            // Mark target as in MST
            inMST[nextPoint] = true
            mstSize += 1
            currentPoint = nextPoint
        }

        return MinimumSpanningTree(edges: mstEdges, pointCount: n)
    }
}

// MARK: - Interruptible MST Builder

/// Interruptible MST builder with checkpoint support.
///
/// This builder supports:
/// - **Interruption**: Checking a `shouldContinue` closure periodically
/// - **Checkpointing**: Saving partial MST state for resumption
/// - **Resumption**: Starting from a previously saved state
///
/// ## Algorithm
///
/// Uses Prim's algorithm, which naturally supports checkpointing because
/// it grows the MST one edge at a time. The state consists of:
/// - `mstEdges`: Edges added to MST so far
/// - `inMST`: Boolean array marking which points are in the MST
///
/// ## Checkpoint Strategy
///
/// Since Prim's explores O(n) edges per point added, we checkpoint after
/// every N edges (default 100) or after a time interval (default 3 seconds).
public struct InterruptibleMSTBuilder: Sendable {

    /// Interval (in edges) between checkpoint callbacks.
    public let checkpointEdgeInterval: Int

    /// Creates an interruptible MST builder.
    ///
    /// - Parameter checkpointEdgeInterval: Number of edges between checkpoints. Default is 100.
    public init(checkpointEdgeInterval: Int = 100) {
        self.checkpointEdgeInterval = checkpointEdgeInterval
    }

    /// Result of interruptible MST construction.
    public struct InterruptibleResult: Sendable {
        /// The MST edges (may be partial if interrupted).
        public let edges: [MSTEdge]

        /// Number of points in the MST.
        public let pointCount: Int

        /// Which points are included in the MST.
        public let inMST: [Bool]

        /// Whether construction is complete.
        public var isComplete: Bool {
            guard pointCount > 0 else { return true }
            return edges.count == pointCount - 1
        }

        /// Progress as a fraction (0.0 to 1.0).
        public var progress: Float {
            guard pointCount > 1 else { return 1.0 }
            return Float(edges.count) / Float(pointCount - 1)
        }

        /// Creates a complete MinimumSpanningTree if construction is complete.
        public func toMST() -> MinimumSpanningTree? {
            guard isComplete else { return nil }
            return MinimumSpanningTree(edges: edges, pointCount: pointCount)
        }
    }

    /// Checkpoint information for MST construction.
    public struct CheckpointInfo: Sendable {
        /// Edges completed so far.
        public let edges: [MSTEdge]

        /// Total points in the graph.
        public let pointCount: Int

        /// Which points are in the MST.
        public let inMST: [Bool]

        /// Progress fraction.
        public var progress: Float {
            guard pointCount > 1 else { return 1.0 }
            return Float(edges.count) / Float(pointCount - 1)
        }
    }

    /// Builds an MST with interruption and checkpoint support.
    ///
    /// - Parameters:
    ///   - graph: The mutual reachability graph.
    ///   - startingEdges: Edges from a previous checkpoint (for resumption).
    ///   - startingInMST: Which points are in MST from checkpoint (for resumption).
    ///   - shouldContinue: Closure called periodically. Return false to interrupt.
    ///   - onCheckpoint: Callback for checkpoint saving.
    /// - Returns: Result containing MST edges and completion state.
    public func build(
        from graph: MutualReachabilityGraph,
        startingEdges: [MSTEdge]? = nil,
        startingInMST: [Bool]? = nil,
        onCheckpoint: (@Sendable (CheckpointInfo) async -> Void)? = nil
    ) async -> InterruptibleResult {
        let n = graph.count

        guard n > 0 else {
            return InterruptibleResult(edges: [], pointCount: 0, inMST: [])
        }

        guard n > 1 else {
            return InterruptibleResult(edges: [], pointCount: 1, inMST: [true])
        }

        // Initialize or restore state
        var mstEdges: [MSTEdge]
        var inMST: [Bool]
        var mstSize: Int

        var minEdgeWeight = [Float](repeating: Float.infinity, count: n)
        var minEdgeSource = [Int](repeating: -1, count: n)

        if let savedEdges = startingEdges,
           let savedInMST = startingInMST,
           savedInMST.count == n {
            // Resume from checkpoint
            mstEdges = savedEdges
            inMST = savedInMST
            mstSize = savedInMST.filter { $0 }.count

            // Recompute minEdgeWeight array for points not in MST
            // O(n * k) which is perfectly fine on resume
            for i in 0..<n where inMST[i] {
                for j in 0..<n where !inMST[j] {
                    let weight = graph.distance(from: i, to: j)
                    if weight < minEdgeWeight[j] {
                        minEdgeWeight[j] = weight
                        minEdgeSource[j] = i
                    }
                }
            }
        } else {
            // Start fresh
            mstEdges = []
            mstEdges.reserveCapacity(n - 1)
            inMST = [Bool](repeating: false, count: n)

            // Start from point 0
            inMST[0] = true
            mstSize = 1

            // Initialize minEdgeWeights from point 0
            for j in 0..<n where !inMST[j] {
                minEdgeWeight[j] = graph.distance(from: 0, to: j)
                minEdgeSource[j] = 0
            }
        }

        var edgesSinceCheckpoint = 0

        // Grow MST until all points included or interrupted
        while mstSize < n {
            // Check if we should continue
            if Task.isCancelled {
                break
            }

            // Find the next point to add (min weight)
            var minWeight: Float = .infinity
            var nextPoint = -1

            for j in 0..<n {
                if !inMST[j] && minEdgeWeight[j] < minWeight {
                    minWeight = minEdgeWeight[j]
                    nextPoint = j
                }
            }

            guard nextPoint != -1 else { break }

            // Add edge to MST
            mstEdges.append(MSTEdge(
                source: minEdgeSource[nextPoint],
                target: nextPoint,
                weight: minWeight
            ))

            // Mark target as in MST
            inMST[nextPoint] = true
            mstSize += 1

            // Update minEdgeWeight for all remaining points
            for j in 0..<n {
                if !inMST[j] {
                    let weight = graph.distance(from: nextPoint, to: j)
                    if weight < minEdgeWeight[j] {
                        minEdgeWeight[j] = weight
                        minEdgeSource[j] = nextPoint
                    }
                }
            }

            edgesSinceCheckpoint += 1

            // Checkpoint at intervals
            if let onCheckpoint = onCheckpoint,
               edgesSinceCheckpoint >= checkpointEdgeInterval {
                let info = CheckpointInfo(
                    edges: mstEdges,
                    pointCount: n,
                    inMST: inMST
                )
                await onCheckpoint(info)
                edgesSinceCheckpoint = 0
            }
        }

        // Final checkpoint if we completed or were interrupted mid-interval
        if let onCheckpoint = onCheckpoint, edgesSinceCheckpoint > 0 {
            let info = CheckpointInfo(
                edges: mstEdges,
                pointCount: n,
                inMST: inMST
            )
            await onCheckpoint(info)
        }

        return InterruptibleResult(
            edges: mstEdges,
            pointCount: n,
            inMST: inMST
        )
    }
}

// MARK: - MST Edge Serialization Support

extension MSTEdge: Codable {

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        source = try container.decode(Int.self, forKey: .source)
        target = try container.decode(Int.self, forKey: .target)
        weight = try container.decode(Float.self, forKey: .weight)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(source, forKey: .source)
        try container.encode(target, forKey: .target)
        try container.encode(weight, forKey: .weight)
    }

    private enum CodingKeys: String, CodingKey {
        case source, target, weight
    }
}

// MARK: - Union-Find (for alternative Kruskal's algorithm)

/// A disjoint set union (Union-Find) data structure.
///
/// Used by Kruskal's algorithm to detect cycles. Each element starts
/// in its own set, and union operations merge sets.
///
/// ## Complexity
///
/// - Find: O(α(n)) amortized (nearly constant)
/// - Union: O(α(n)) amortized
///
/// Where α is the inverse Ackermann function.
public struct UnionFind: Sendable {

    /// Parent pointers (negative means root with size).
    private var parent: [Int]

    /// Rank (for union by rank).
    private var rank: [Int]

    /// Number of disjoint sets.
    public private(set) var setCount: Int

    /// Creates a Union-Find with n elements.
    public init(count: Int) {
        self.parent = Array(0..<count)
        self.rank = [Int](repeating: 0, count: count)
        self.setCount = count
    }

    /// Finds the representative (root) of the set containing element.
    ///
    /// Uses path compression for efficiency.
    public mutating func find(_ element: Int) -> Int {
        if parent[element] != element {
            parent[element] = find(parent[element])
        }
        return parent[element]
    }

    /// Unions the sets containing two elements.
    ///
    /// Uses union by rank for efficiency.
    ///
    /// - Returns: True if a union occurred (elements were in different sets).
    @discardableResult
    public mutating func union(_ a: Int, _ b: Int) -> Bool {
        let rootA = find(a)
        let rootB = find(b)

        guard rootA != rootB else { return false }

        // Union by rank
        if rank[rootA] < rank[rootB] {
            parent[rootA] = rootB
        } else if rank[rootA] > rank[rootB] {
            parent[rootB] = rootA
        } else {
            parent[rootB] = rootA
            rank[rootA] += 1
        }

        setCount -= 1
        return true
    }

    /// Returns true if two elements are in the same set.
    public mutating func connected(_ a: Int, _ b: Int) -> Bool {
        find(a) == find(b)
    }
}

// MARK: - Kruskal's Algorithm (Alternative)

/// Builds MST using Kruskal's algorithm.
///
/// Kruskal's sorts all edges and greedily adds non-cycle-forming edges.
/// This can be faster for sparse graphs but requires all edges upfront.
///
/// ## Complexity
///
/// - Time: O(E log E) where E = n(n-1)/2 for complete graph
/// - Space: O(E) for edge storage
///
/// For dense graphs like mutual reachability, Prim's is usually better
/// because Kruskal's requires materializing all O(n²) edges.
public struct KruskalMSTBuilder: Sendable {

    /// Creates a Kruskal MST builder.
    public init() {}

    /// Builds MST from sorted edges.
    ///
    /// - Parameters:
    ///   - edges: All graph edges, will be sorted by weight.
    ///   - pointCount: Number of points.
    /// - Returns: The minimum spanning tree.
    public func build(edges: [MutualReachabilityEdge], pointCount: Int) -> MinimumSpanningTree {
        guard pointCount > 1 else {
            return MinimumSpanningTree(edges: [], pointCount: pointCount)
        }

        let sortedEdges = edges.sorted()
        var mstEdges = [MSTEdge]()
        mstEdges.reserveCapacity(pointCount - 1)

        var uf = UnionFind(count: pointCount)

        for edge in sortedEdges {
            // Check if adding this edge would create a cycle
            if uf.union(edge.pointA, edge.pointB) {
                mstEdges.append(MSTEdge(
                    source: edge.pointA,
                    target: edge.pointB,
                    weight: edge.distance
                ))

                // Stop when we have n-1 edges
                if mstEdges.count == pointCount - 1 {
                    break
                }
            }
        }

        return MinimumSpanningTree(edges: mstEdges, pointCount: pointCount)
    }
}
