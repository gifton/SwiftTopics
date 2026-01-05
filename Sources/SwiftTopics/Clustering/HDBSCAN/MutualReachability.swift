// MutualReachability.swift
// SwiftTopics
//
// Mutual reachability distance computation for HDBSCAN

import Foundation

// MARK: - Mutual Reachability Edge

/// An edge in the mutual reachability graph.
///
/// Represents the mutual reachability distance between two points.
public struct MutualReachabilityEdge: Sendable, Comparable {

    /// Index of the first point.
    public let pointA: Int

    /// Index of the second point.
    public let pointB: Int

    /// The mutual reachability distance.
    public let distance: Float

    /// Creates a mutual reachability edge.
    public init(pointA: Int, pointB: Int, distance: Float) {
        // Always store with smaller index first for consistency
        if pointA <= pointB {
            self.pointA = pointA
            self.pointB = pointB
        } else {
            self.pointA = pointB
            self.pointB = pointA
        }
        self.distance = distance
    }

    public static func < (lhs: MutualReachabilityEdge, rhs: MutualReachabilityEdge) -> Bool {
        lhs.distance < rhs.distance
    }
}

// MARK: - Mutual Reachability Graph

/// A sparse representation of the mutual reachability graph.
///
/// Instead of storing the full n×n distance matrix, this stores only the
/// edges needed for MST construction. For HDBSCAN, we don't need all n(n-1)/2
/// edges—we can compute them on-demand during Prim's algorithm.
///
/// ## Mathematical Definition
///
/// The mutual reachability distance between points a and b is:
///
/// ```
/// mutual_reach(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
/// ```
///
/// This transformation makes the distance metric robust to varying densities.
/// In sparse regions (high core distances), points need to be closer to be
/// considered similar. In dense regions, points can be farther apart.
///
/// ## Memory Efficiency
///
/// For n points, a full distance matrix requires O(n²) space.
/// This implementation computes distances on-demand, using O(n) for core
/// distances plus O(n-1) for MST edges during construction.
public struct MutualReachabilityGraph: Sendable {

    /// Core distances for all points.
    public let coreDistances: [Float]

    /// The original embeddings (for distance computation).
    private let embeddings: [Embedding]

    /// Number of points.
    public var count: Int { embeddings.count }

    /// Creates a mutual reachability graph.
    ///
    /// - Parameters:
    ///   - embeddings: The point embeddings.
    ///   - coreDistances: Pre-computed core distances.
    public init(embeddings: [Embedding], coreDistances: [Float]) {
        precondition(
            embeddings.count == coreDistances.count,
            "Embeddings and core distances must have same count"
        )
        self.embeddings = embeddings
        self.coreDistances = coreDistances
    }

    // MARK: - Distance Computation

    /// Computes the mutual reachability distance between two points.
    ///
    /// - Parameters:
    ///   - i: Index of first point.
    ///   - j: Index of second point.
    /// - Returns: The mutual reachability distance.
    public func distance(from i: Int, to j: Int) -> Float {
        guard i != j else { return 0 }

        let euclidean = embeddings[i].euclideanDistance(embeddings[j])
        let coreI = coreDistances[i]
        let coreJ = coreDistances[j]

        return max(coreI, coreJ, euclidean)
    }

    /// Computes the mutual reachability distance given a pre-computed Euclidean distance.
    ///
    /// Use this when you already have the Euclidean distance to avoid recomputation.
    ///
    /// - Parameters:
    ///   - i: Index of first point.
    ///   - j: Index of second point.
    ///   - euclideanDistance: The pre-computed Euclidean distance.
    /// - Returns: The mutual reachability distance.
    public func distance(from i: Int, to j: Int, euclideanDistance: Float) -> Float {
        guard i != j else { return 0 }
        return max(coreDistances[i], coreDistances[j], euclideanDistance)
    }

    // MARK: - Edge Iteration

    /// Creates an edge iterator for use with MST algorithms.
    ///
    /// The iterator yields edges on-demand, avoiding full materialization.
    public func makeEdgeIterator() -> MutualReachabilityEdgeIterator {
        MutualReachabilityEdgeIterator(graph: self)
    }

    /// Returns all edges from a given point.
    ///
    /// Used by Prim's algorithm to explore neighbors of a point.
    ///
    /// - Parameter pointIndex: The source point index.
    /// - Returns: Array of edges from this point to all other points.
    public func edges(from pointIndex: Int) -> [MutualReachabilityEdge] {
        var result = [MutualReachabilityEdge]()
        result.reserveCapacity(count - 1)

        for j in 0..<count {
            guard j != pointIndex else { continue }
            let dist = distance(from: pointIndex, to: j)
            result.append(MutualReachabilityEdge(pointA: pointIndex, pointB: j, distance: dist))
        }

        return result
    }

    // MARK: - Full Graph Materialization (for small datasets)

    /// Returns all edges in the graph.
    ///
    /// **Warning**: This creates n(n-1)/2 edges. Only use for small datasets.
    ///
    /// - Parameter sorted: Whether to return edges sorted by distance.
    /// - Returns: All edges in the graph.
    public func allEdges(sorted: Bool = false) -> [MutualReachabilityEdge] {
        var edges = [MutualReachabilityEdge]()
        let edgeCount = count * (count - 1) / 2
        edges.reserveCapacity(edgeCount)

        for i in 0..<count {
            for j in (i + 1)..<count {
                let dist = distance(from: i, to: j)
                edges.append(MutualReachabilityEdge(pointA: i, pointB: j, distance: dist))
            }
        }

        if sorted {
            edges.sort()
        }

        return edges
    }
}

// MARK: - Edge Iterator

/// Iterator for mutual reachability graph edges.
///
/// Computes edges on-demand to save memory.
public struct MutualReachabilityEdgeIterator: IteratorProtocol, Sequence {

    public typealias Element = MutualReachabilityEdge

    private let graph: MutualReachabilityGraph
    private var i: Int = 0
    private var j: Int = 1

    init(graph: MutualReachabilityGraph) {
        self.graph = graph
    }

    public mutating func next() -> MutualReachabilityEdge? {
        guard i < graph.count - 1 else { return nil }

        let edge = MutualReachabilityEdge(
            pointA: i,
            pointB: j,
            distance: graph.distance(from: i, to: j)
        )

        // Advance to next pair
        j += 1
        if j >= graph.count {
            i += 1
            j = i + 1
        }

        return edge
    }
}

// MARK: - Mutual Reachability Builder

/// Builder for constructing mutual reachability graphs.
public struct MutualReachabilityBuilder: Sendable {

    /// The minimum samples parameter (k for core distance).
    public let minSamples: Int

    /// Creates a builder.
    ///
    /// - Parameter minSamples: The k value for core distance computation.
    public init(minSamples: Int) {
        precondition(minSamples >= 1, "minSamples must be at least 1")
        self.minSamples = minSamples
    }

    /// Builds a mutual reachability graph from embeddings.
    ///
    /// - Parameters:
    ///   - embeddings: The point embeddings.
    ///   - gpuContext: Optional GPU context for core distance computation.
    /// - Returns: The mutual reachability graph.
    public func build(
        embeddings: [Embedding],
        gpuContext: TopicsGPUContext?
    ) async throws -> MutualReachabilityGraph {
        // Compute core distances
        let coreComputer = CoreDistanceComputer(minSamples: minSamples)
        let coreDistances = try await coreComputer.compute(
            embeddings: embeddings,
            gpuContext: gpuContext
        )

        return MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)
    }

    /// Builds a mutual reachability graph with pre-computed core distances.
    ///
    /// - Parameters:
    ///   - embeddings: The point embeddings.
    ///   - coreDistances: Pre-computed core distances.
    /// - Returns: The mutual reachability graph.
    public func build(
        embeddings: [Embedding],
        coreDistances: [Float]
    ) -> MutualReachabilityGraph {
        MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)
    }
}

// MARK: - Distance Matrix (for debugging/testing)

/// A full distance matrix for small datasets or debugging.
///
/// **Warning**: Requires O(n²) memory. Only use for n < 1000.
public struct MutualReachabilityMatrix: Sendable {

    /// The flattened distance matrix (row-major, upper triangular stored).
    private let storage: [Float]

    /// Number of points.
    public let count: Int

    /// Creates a distance matrix from a mutual reachability graph.
    ///
    /// - Parameter graph: The graph to materialize.
    public init(graph: MutualReachabilityGraph) {
        self.count = graph.count

        // Store only upper triangular (including diagonal)
        let storageSize = count * (count + 1) / 2
        var storage = [Float](repeating: 0, count: storageSize)

        var index = 0
        for i in 0..<count {
            for j in i..<count {
                storage[index] = graph.distance(from: i, to: j)
                index += 1
            }
        }

        self.storage = storage
    }

    /// Gets the distance between two points.
    public subscript(i: Int, j: Int) -> Float {
        if i == j { return 0 }

        // Ensure i < j for upper triangular access
        let (row, col) = i < j ? (i, j) : (j, i)

        // Index into upper triangular storage
        // Row i starts at position: sum(count - k for k in 0..<i) = i*count - i*(i+1)/2
        let rowStart = row * count - row * (row + 1) / 2
        let index = rowStart + (col - row)

        return storage[index]
    }
}
