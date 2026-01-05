// FuzzySimplicialSet.swift
// SwiftTopics
//
// Fuzzy simplicial set construction for UMAP

import Foundation

// MARK: - Fuzzy Simplicial Set

/// A fuzzy simplicial set representing the topological structure of data.
///
/// In UMAP, the fuzzy simplicial set is a weighted graph where edge weights
/// represent "membership strength" - the probability that two points are
/// connected on the underlying manifold.
///
/// ## Mathematical Background
///
/// For each point i, we compute:
/// - **ρ_i** (rho): Distance to the nearest neighbor. This is the "local scale".
/// - **σ_i** (sigma): Bandwidth parameter such that the sum of memberships equals log₂(k).
///
/// The membership from point i to j is:
/// ```
/// μ(i → j) = exp(-(d_ij - ρ_i) / σ_i)   if d_ij > ρ_i
///          = 1.0                         otherwise
/// ```
///
/// This formula ensures:
/// 1. The nearest neighbor always has membership 1.0
/// 2. Membership decreases exponentially with distance beyond ρ
/// 3. Each point "sees" approximately k effective neighbors
///
/// ## Symmetrization
///
/// The directed memberships are symmetrized using fuzzy set union:
/// ```
/// μ(i, j) = μ(i → j) + μ(j → i) - μ(i → j) × μ(j → i)
/// ```
///
/// This ensures the graph is undirected while preserving fuzzy set semantics.
///
/// ## Thread Safety
/// `FuzzySimplicialSet` is immutable and `Sendable`.
public struct FuzzySimplicialSet: Sendable {

    // MARK: - Properties

    /// Sparse matrix of symmetric membership strengths.
    /// Entry [i,j] is the membership weight of edge (i,j) after symmetrization.
    public let memberships: SparseMatrix<Float>

    /// Per-point ρ values (distance to nearest neighbor).
    /// These define the local scale at each point.
    public let rho: [Float]

    /// Per-point σ values (bandwidth/smoothing parameters).
    /// Computed by binary search to satisfy the target sum constraint.
    public let sigma: [Float]

    /// Number of points.
    public let pointCount: Int

    /// Target number of effective neighbors (log₂(k)).
    public let targetSum: Float

    // MARK: - Construction

    /// Builds a fuzzy simplicial set from a k-NN graph.
    ///
    /// This is the core UMAP graph construction step. It converts discrete
    /// k-NN relationships into fuzzy membership weights.
    ///
    /// - Parameters:
    ///   - graph: The k-NN graph with distances.
    ///   - localConnectivity: Number of nearest neighbors that should have
    ///     membership ≈ 1. Default is 1.0 (only the nearest neighbor).
    ///   - bandwidth: Smoothing factor for sigma computation. Default is 1.0.
    /// - Returns: The fuzzy simplicial set.
    public static func build(
        from graph: NearestNeighborGraph,
        localConnectivity: Float = 1.0,
        bandwidth: Float = 1.0
    ) -> FuzzySimplicialSet {
        let n = graph.pointCount
        let k = graph.k

        // Target sum of memberships per point
        let targetSum = log2(Float(k))

        // Step 1: Compute ρ (rho) for each point
        // rho[i] = distance to the localConnectivity-th nearest neighbor
        var rho = [Float](repeating: 0, count: n)

        for i in 0..<n {
            if graph.distances[i].isEmpty {
                rho[i] = 0
            } else if localConnectivity <= 1.0 {
                // rho is distance to nearest neighbor
                rho[i] = graph.distances[i][0]
            } else {
                // Interpolate for fractional local connectivity
                let index = min(Int(localConnectivity) - 1, graph.distances[i].count - 1)
                if index < graph.distances[i].count - 1 {
                    let frac = localConnectivity - Float(Int(localConnectivity))
                    rho[i] = graph.distances[i][index] * (1 - frac) +
                             graph.distances[i][index + 1] * frac
                } else {
                    rho[i] = graph.distances[i][index]
                }
            }
        }

        // Step 2: Compute σ (sigma) for each point using binary search
        var sigma = [Float](repeating: 1.0, count: n)

        for i in 0..<n {
            sigma[i] = computeSigma(
                distances: graph.distances[i],
                rho: rho[i],
                targetSum: targetSum,
                bandwidth: bandwidth
            )
        }

        // Step 3: Compute directed membership weights
        var entries: [(row: Int, col: Int, value: Float)] = []
        entries.reserveCapacity(n * k)

        for i in 0..<n {
            for (j, neighborIdx) in graph.neighbors[i].enumerated() {
                let distance = graph.distances[i][j]
                let membership = computeMembership(
                    distance: distance,
                    rho: rho[i],
                    sigma: sigma[i]
                )

                if membership > 0 {
                    entries.append((row: i, col: neighborIdx, value: membership))
                }
            }
        }

        // Build directed membership matrix
        let directedMemberships = SparseMatrix<Float>.fromCOO(
            rows: n,
            cols: n,
            entries: entries
        )

        // Step 4: Symmetrize using fuzzy set union
        let symmetricMemberships = directedMemberships.fuzzySetUnion()

        return FuzzySimplicialSet(
            memberships: symmetricMemberships,
            rho: rho,
            sigma: sigma,
            pointCount: n,
            targetSum: targetSum
        )
    }

    // MARK: - Private Helpers

    /// Computes sigma using binary search.
    ///
    /// Finds sigma such that:
    /// Σ_j exp(-(d_ij - ρ_i) / σ_i) ≈ targetSum
    ///
    /// This ensures each point has approximately the same number of
    /// "effective" neighbors regardless of local density.
    private static func computeSigma(
        distances: [Float],
        rho: Float,
        targetSum: Float,
        bandwidth: Float,
        tolerance: Float = 1e-5,
        maxIterations: Int = 64
    ) -> Float {
        // Handle edge cases
        if distances.isEmpty {
            return 1.0
        }

        // If all distances are the same, return a default sigma
        let maxDist = distances.max() ?? 0
        let minDist = distances.min() ?? 0
        if maxDist - minDist < Float.ulpOfOne {
            return 1.0
        }

        // Binary search bounds
        var lo: Float = 0.001
        var hi: Float = 1000.0

        // Initial guess based on mean distance
        var sigma = 1.0 * bandwidth

        for _ in 0..<maxIterations {
            // Compute sum of memberships with current sigma
            var sum: Float = 0
            for distance in distances {
                let membership = computeMembership(
                    distance: distance,
                    rho: rho,
                    sigma: sigma
                )
                sum += membership
            }

            // Check convergence
            if abs(sum - targetSum) < tolerance {
                break
            }

            // Binary search update
            if sum > targetSum {
                hi = sigma
                sigma = (lo + hi) / 2
            } else {
                lo = sigma
                if hi >= 1000.0 {
                    sigma *= 2
                } else {
                    sigma = (lo + hi) / 2
                }
            }
        }

        return max(sigma, Float.ulpOfOne)
    }

    /// Computes membership weight for a single distance.
    ///
    /// μ(d) = exp(-(d - ρ) / σ) if d > ρ
    ///      = 1.0               otherwise
    @inline(__always)
    private static func computeMembership(
        distance: Float,
        rho: Float,
        sigma: Float
    ) -> Float {
        if distance <= rho {
            return 1.0
        }

        let normalizedDist = (distance - rho) / sigma
        // Clamp to avoid underflow
        if normalizedDist > 20 {
            return 0
        }
        return exp(-normalizedDist)
    }

    // MARK: - Initialization

    /// Creates a fuzzy simplicial set from components.
    private init(
        memberships: SparseMatrix<Float>,
        rho: [Float],
        sigma: [Float],
        pointCount: Int,
        targetSum: Float
    ) {
        self.memberships = memberships
        self.rho = rho
        self.sigma = sigma
        self.pointCount = pointCount
        self.targetSum = targetSum
    }

    // MARK: - Accessors

    /// Gets the membership weight between two points.
    ///
    /// - Parameters:
    ///   - i: First point index.
    ///   - j: Second point index.
    /// - Returns: Membership weight (0 if not connected).
    public func membership(from i: Int, to j: Int) -> Float {
        memberships[i, j]
    }

    /// Gets all edges with weights above a threshold.
    ///
    /// - Parameter threshold: Minimum membership weight.
    /// - Returns: Array of (source, target, weight) tuples.
    public func edges(aboveThreshold threshold: Float = 0) -> [(source: Int, target: Int, weight: Float)] {
        var result: [(source: Int, target: Int, weight: Float)] = []

        memberships.forEachNonZero { row, col, value in
            if value > threshold {
                result.append((source: row, target: col, weight: value))
            }
        }

        return result
    }

    /// Converts to an edge list for SGD optimization.
    ///
    /// Returns only the upper triangle of the symmetric matrix
    /// (since memberships are symmetric, we only need half).
    ///
    /// - Returns: Array of edges with weights.
    public func toEdgeList() -> [SparseMatrix<Float>.Edge] {
        var edges: [SparseMatrix<Float>.Edge] = []
        edges.reserveCapacity(memberships.nonZeroCount / 2)

        memberships.forEachNonZero { row, col, value in
            // Only include upper triangle to avoid duplicates
            if row < col {
                edges.append(SparseMatrix<Float>.Edge(
                    source: row,
                    target: col,
                    weight: value
                ))
            }
        }

        return edges
    }

    /// Computes the total weight (sum of all membership values).
    public var totalWeight: Float {
        var sum: Float = 0
        memberships.forEachNonZero { _, _, value in
            sum += value
        }
        return sum
    }

    /// Returns the number of edges (non-zero entries).
    public var edgeCount: Int {
        memberships.nonZeroCount
    }
}

// MARK: - Epoch Sampling

extension FuzzySimplicialSet {

    /// Computes the number of epochs each edge should be sampled.
    ///
    /// In UMAP's SGD, edges are sampled proportionally to their weight.
    /// This function pre-computes how many times each edge should be
    /// updated across all epochs.
    ///
    /// - Parameter nEpochs: Total number of training epochs.
    /// - Returns: Array of epoch counts per edge.
    public func epochsPerEdge(nEpochs: Int) -> [Float] {
        let edges = toEdgeList()
        guard !edges.isEmpty else { return [] }

        // Find maximum weight
        let maxWeight = edges.map { $0.weight }.max() ?? 1.0
        guard maxWeight > 0 else { return [Float](repeating: 0, count: edges.count) }

        // Normalize by maximum weight and scale by epochs
        return edges.map { edge in
            Float(nEpochs) * edge.weight / maxWeight
        }
    }

    /// Creates a sampling schedule for SGD optimization.
    ///
    /// Returns an array indicating at which epoch each edge should next
    /// be sampled. This allows efficient epoch-by-epoch processing.
    ///
    /// - Parameter nEpochs: Total number of epochs.
    /// - Returns: Sampling schedule.
    public func createSamplingSchedule(nEpochs: Int) -> EdgeSamplingSchedule {
        let edges = toEdgeList()
        let epochsPerEdge = self.epochsPerEdge(nEpochs: nEpochs)

        var nextSampleEpoch = [Float](repeating: 0, count: edges.count)

        // Initialize: edges with higher weights sample earlier
        for i in 0..<edges.count {
            if epochsPerEdge[i] > 0 {
                nextSampleEpoch[i] = Float(nEpochs) / epochsPerEdge[i]
            } else {
                nextSampleEpoch[i] = Float(nEpochs + 1)  // Never sample
            }
        }

        return EdgeSamplingSchedule(
            edges: edges,
            epochsPerEdge: epochsPerEdge,
            nextSampleEpoch: nextSampleEpoch,
            totalEpochs: nEpochs
        )
    }
}

// MARK: - Edge Sampling Schedule

/// Pre-computed sampling schedule for UMAP's edge-wise SGD.
public struct EdgeSamplingSchedule: Sendable {

    /// All edges in the graph.
    public let edges: [SparseMatrix<Float>.Edge]

    /// How many epochs each edge should be sampled.
    public let epochsPerEdge: [Float]

    /// Next epoch at which each edge should be sampled.
    public private(set) var nextSampleEpoch: [Float]

    /// Total number of epochs.
    public let totalEpochs: Int

    /// Returns the indices of edges to sample in the given epoch.
    ///
    /// - Parameter epoch: Current epoch (0-indexed).
    /// - Returns: Array of edge indices to sample.
    public mutating func edgesToSample(epoch: Int) -> [Int] {
        var toSample: [Int] = []

        for i in 0..<edges.count {
            if nextSampleEpoch[i] <= Float(epoch + 1) {
                toSample.append(i)
                // Update next sample time
                if epochsPerEdge[i] > 0 {
                    nextSampleEpoch[i] += Float(totalEpochs) / epochsPerEdge[i]
                }
            }
        }

        return toSample
    }
}

// MARK: - Statistics

extension FuzzySimplicialSet {

    /// Statistics about the fuzzy simplicial set.
    public struct Statistics: Sendable {
        /// Number of points.
        public let pointCount: Int

        /// Number of edges (non-zero entries).
        public let edgeCount: Int

        /// Mean rho (distance to nearest neighbor).
        public let meanRho: Float

        /// Mean sigma (bandwidth).
        public let meanSigma: Float

        /// Mean edge weight.
        public let meanWeight: Float

        /// Maximum edge weight.
        public let maxWeight: Float

        /// Graph density.
        public let density: Float
    }

    /// Computes statistics about this fuzzy set.
    public func computeStatistics() -> Statistics {
        let meanRho = rho.isEmpty ? 0 : rho.reduce(0, +) / Float(rho.count)
        let meanSigma = sigma.isEmpty ? 0 : sigma.reduce(0, +) / Float(sigma.count)

        var totalWeight: Float = 0
        var maxWeight: Float = 0
        memberships.forEachNonZero { _, _, value in
            totalWeight += value
            maxWeight = max(maxWeight, value)
        }

        let meanWeight = edgeCount > 0 ? totalWeight / Float(edgeCount) : 0

        return Statistics(
            pointCount: pointCount,
            edgeCount: edgeCount,
            meanRho: meanRho,
            meanSigma: meanSigma,
            meanWeight: meanWeight,
            maxWeight: maxWeight,
            density: memberships.density
        )
    }
}

// MARK: - CustomStringConvertible

extension FuzzySimplicialSet: CustomStringConvertible {
    public var description: String {
        let stats = computeStatistics()
        return "FuzzySimplicialSet(n=\(pointCount), edges=\(edgeCount), " +
               "meanWeight=\(String(format: "%.3f", stats.meanWeight)))"
    }
}
