// SpectralEmbedding.swift
// SwiftTopics
//
// Spectral embedding initialization for UMAP

import Foundation
import Accelerate

// MARK: - Spectral Embedding

/// Spectral embedding for UMAP initialization.
///
/// Computes initial low-dimensional positions using the Laplacian eigenmaps
/// algorithm. This provides a much better starting point than random
/// initialization, leading to faster convergence and better results.
///
/// ## Algorithm
///
/// 1. Build the normalized graph Laplacian: L = I - D^(-1/2) × W × D^(-1/2)
/// 2. Compute the eigenvectors of L with smallest eigenvalues (excluding 0)
/// 3. Use the first `nComponents` eigenvectors as initial coordinates
///
/// ## Why Spectral Initialization?
///
/// - **Geometric intuition**: Laplacian eigenvectors capture the "smoothest"
///   functions on the graph - points connected by high-weight edges get
///   similar coordinates.
/// - **Global structure**: Unlike random init, spectral init preserves the
///   overall topology from the start.
/// - **Faster convergence**: SGD only needs to refine local structure.
///
/// ## Fallback
///
/// For very large or ill-conditioned graphs, spectral embedding may fail.
/// In this case, we fall back to random initialization with appropriate scaling.
public enum SpectralEmbedding {

    // MARK: - Spectral Computation

    /// Computes spectral embedding for UMAP initialization.
    ///
    /// - Parameters:
    ///   - adjacency: Sparse adjacency matrix (symmetric).
    ///   - nComponents: Number of output dimensions.
    ///   - seed: Random seed for fallback initialization.
    /// - Returns: Initial coordinates as [pointCount][nComponents].
    /// - Throws: `ReductionError` if computation fails completely.
    public static func compute(
        adjacency: SparseMatrix<Float>,
        nComponents: Int,
        seed: UInt64? = nil
    ) throws -> [[Float]] {
        let n = adjacency.rows

        guard n >= nComponents + 1 else {
            throw ReductionError.insufficientSamples(
                required: nComponents + 1,
                provided: n
            )
        }

        // For large matrices, fall back to random initialization
        // (full eigendecomposition is O(n³) which is prohibitive)
        if n > 10000 {
            return randomInitialization(
                pointCount: n,
                nComponents: nComponents,
                seed: seed
            )
        }

        // Try spectral embedding
        do {
            return try spectralFromLaplacian(
                adjacency: adjacency,
                nComponents: nComponents
            )
        } catch {
            // Fall back to random initialization
            return randomInitialization(
                pointCount: n,
                nComponents: nComponents,
                seed: seed
            )
        }
    }

    /// Computes spectral embedding from the graph Laplacian.
    private static func spectralFromLaplacian(
        adjacency: SparseMatrix<Float>,
        nComponents: Int
    ) throws -> [[Float]] {
        let n = adjacency.rows

        // Convert to dense (required for LAPACK)
        // Note: For very large matrices, we should use sparse eigensolvers
        let laplacian = adjacency.normalizedLaplacian()
        let denseLaplacian = laplacian.toDense()

        // Compute eigendecomposition
        // The Laplacian is symmetric positive semi-definite
        let eigenResult: Eigendecomposition.Result
        do {
            // Regularize for numerical stability
            let regularized = Eigendecomposition.regularize(
                denseLaplacian,
                dimension: n,
                epsilon: 1e-6
            )
            eigenResult = try Eigendecomposition.symmetric(regularized, dimension: n)
        } catch {
            throw ReductionError.numericalInstability(
                "Eigendecomposition failed: \(error)"
            )
        }

        // Find eigenvectors corresponding to smallest eigenvalues
        // Skip the first (constant) eigenvector with eigenvalue ~0
        // LAPACK returns eigenvalues in ascending order, but our wrapper reverses
        // So we need the LAST eigenvalues (smallest)
        var embedding = [[Float]](repeating: [Float](repeating: 0, count: nComponents), count: n)

        for i in 0..<n {
            for j in 0..<nComponents {
                // Get eigenvector corresponding to (j+1)-th smallest eigenvalue
                // (skip the 0th which is the constant vector)
                let eigIdx = n - 2 - j  // Skip last (smallest) and go backwards
                if eigIdx >= 0 {
                    embedding[i][j] = eigenResult.eigenvectors[i + eigIdx * n]
                }
            }
        }

        // Scale to reasonable range
        normalizeEmbedding(&embedding)

        return embedding
    }

    // MARK: - Random Initialization

    /// Random initialization fallback.
    ///
    /// Generates uniformly distributed points scaled to a reasonable range.
    /// Uses spectral embedding scale (roughly [-10, 10]).
    ///
    /// - Parameters:
    ///   - pointCount: Number of points.
    ///   - nComponents: Number of dimensions.
    ///   - seed: Random seed for reproducibility.
    /// - Returns: Random initial coordinates.
    public static func randomInitialization(
        pointCount: Int,
        nComponents: Int,
        seed: UInt64? = nil
    ) -> [[Float]] {
        var rng = RandomState(seed: seed)

        // Scale based on dataset size (heuristic from umap-learn)
        let scale: Float = 10.0 / sqrt(Float(pointCount))

        var embedding = [[Float]](repeating: [Float](repeating: 0, count: nComponents), count: pointCount)

        for i in 0..<pointCount {
            for j in 0..<nComponents {
                embedding[i][j] = rng.nextFloat(in: -scale...scale)
            }
        }

        return embedding
    }

    /// PCA-based initialization.
    ///
    /// Uses PCA to reduce to the target dimension as initialization.
    /// This is often better than random when the data has clear structure.
    ///
    /// - Parameters:
    ///   - embeddings: Original high-dimensional embeddings.
    ///   - nComponents: Target dimension.
    /// - Returns: PCA-projected coordinates.
    public static func pcaInitialization(
        embeddings: [Embedding],
        nComponents: Int
    ) async throws -> [[Float]] {
        let pca = PCAReducer(components: nComponents)
        let reduced = try await pca.fitTransform(embeddings)

        // Scale to UMAP's expected range
        var embedding = reduced.map { $0.vector }
        normalizeEmbedding(&embedding)

        return embedding
    }

    // MARK: - Normalization

    /// Normalizes embedding to have zero mean and unit variance per dimension.
    private static func normalizeEmbedding(_ embedding: inout [[Float]]) {
        guard !embedding.isEmpty else { return }

        let n = embedding.count
        let d = embedding[0].count

        // Compute mean and std for each dimension
        for j in 0..<d {
            var mean: Float = 0
            for i in 0..<n {
                mean += embedding[i][j]
            }
            mean /= Float(n)

            var variance: Float = 0
            for i in 0..<n {
                let diff = embedding[i][j] - mean
                variance += diff * diff
            }
            variance /= Float(n)
            let std = max(variance.squareRoot(), Float.ulpOfOne)

            // Normalize to zero mean, then scale to [-10, 10] range
            let scale: Float = 10.0 / (3.0 * std)  // 3σ → 10
            for i in 0..<n {
                embedding[i][j] = (embedding[i][j] - mean) * scale
            }
        }
    }

    /// Clips embedding coordinates to prevent extreme values.
    public static func clipEmbedding(
        _ embedding: inout [[Float]],
        maxValue: Float = 100.0
    ) {
        for i in 0..<embedding.count {
            for j in 0..<embedding[i].count {
                embedding[i][j] = max(-maxValue, min(maxValue, embedding[i][j]))
            }
        }
    }
}

// MARK: - Connected Component Handling

extension SpectralEmbedding {

    /// Handles disconnected components in spectral initialization.
    ///
    /// For disconnected graphs, spectral embedding gives constant values
    /// within each component. We detect this and add random noise.
    ///
    /// - Parameters:
    ///   - embedding: Initial embedding (modified in place).
    ///   - graph: The k-NN graph for component detection.
    ///   - seed: Random seed.
    public static func handleDisconnectedComponents(
        embedding: inout [[Float]],
        graph: NearestNeighborGraph,
        seed: UInt64? = nil
    ) {
        // Find connected components
        let components = findConnectedComponents(graph: graph)
        let componentCount = components.max().map { $0 + 1 } ?? 0

        guard componentCount > 1 else { return }

        var rng = RandomState(seed: seed)
        let d = embedding[0].count

        // Add random offset to each component to separate them
        var componentOffsets = [[Float]](repeating: [Float](repeating: 0, count: d), count: componentCount)
        for c in 0..<componentCount {
            for j in 0..<d {
                componentOffsets[c][j] = rng.nextFloat(in: -5...5)
            }
        }

        // Apply offsets
        for i in 0..<embedding.count {
            let comp = components[i]
            for j in 0..<d {
                embedding[i][j] += componentOffsets[comp][j]
            }
        }
    }

    /// Finds connected components in the k-NN graph.
    private static func findConnectedComponents(graph: NearestNeighborGraph) -> [Int] {
        let n = graph.pointCount
        var componentId = [Int](repeating: -1, count: n)
        var currentComponent = 0

        for start in 0..<n {
            guard componentId[start] == -1 else { continue }

            // BFS from this unvisited node
            var queue = [start]
            componentId[start] = currentComponent

            while !queue.isEmpty {
                let current = queue.removeFirst()

                // Outgoing edges
                for neighbor in graph.neighbors[current] {
                    if componentId[neighbor] == -1 {
                        componentId[neighbor] = currentComponent
                        queue.append(neighbor)
                    }
                }

                // Incoming edges (treat as undirected)
                for i in 0..<n {
                    if componentId[i] == -1 && graph.neighbors[i].contains(current) {
                        componentId[i] = currentComponent
                        queue.append(i)
                    }
                }
            }

            currentComponent += 1
        }

        return componentId
    }
}

// MARK: - Initialization Quality Metrics

extension SpectralEmbedding {

    /// Metrics for evaluating initialization quality.
    public struct InitializationMetrics: Sendable {
        /// Mean pairwise distance in the embedding.
        public let meanDistance: Float

        /// Standard deviation of pairwise distances.
        public let stdDistance: Float

        /// Minimum pairwise distance (ideally not too small).
        public let minDistance: Float

        /// Maximum pairwise distance.
        public let maxDistance: Float

        /// Whether the initialization seems reasonable.
        public var isReasonable: Bool {
            // Check for degenerate cases
            guard minDistance > Float.ulpOfOne else { return false }
            guard maxDistance < Float.infinity else { return false }
            guard meanDistance > 0.01 else { return false }
            return true
        }
    }

    /// Computes metrics for an initialization.
    ///
    /// Samples pairwise distances to evaluate quality without O(n²) cost.
    ///
    /// - Parameters:
    ///   - embedding: The initialized coordinates.
    ///   - sampleSize: Number of pairs to sample.
    ///   - seed: Random seed.
    /// - Returns: Quality metrics.
    public static func evaluateInitialization(
        _ embedding: [[Float]],
        sampleSize: Int = 1000,
        seed: UInt64? = nil
    ) -> InitializationMetrics {
        guard embedding.count >= 2 else {
            return InitializationMetrics(
                meanDistance: 0,
                stdDistance: 0,
                minDistance: 0,
                maxDistance: 0
            )
        }

        var rng = RandomState(seed: seed)
        let n = embedding.count
        let d = embedding[0].count
        let actualSamples = min(sampleSize, n * (n - 1) / 2)

        var distances: [Float] = []
        distances.reserveCapacity(actualSamples)

        // Sample random pairs
        for _ in 0..<actualSamples {
            let i = rng.nextInt(upperBound: n)
            var j = rng.nextInt(upperBound: n)
            while j == i {
                j = rng.nextInt(upperBound: n)
            }

            // Compute Euclidean distance
            var sumSq: Float = 0
            for k in 0..<d {
                let diff = embedding[i][k] - embedding[j][k]
                sumSq += diff * diff
            }
            distances.append(sumSq.squareRoot())
        }

        // Compute statistics
        let mean = distances.reduce(0, +) / Float(distances.count)
        let variance = distances.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(distances.count)
        let std = variance.squareRoot()

        return InitializationMetrics(
            meanDistance: mean,
            stdDistance: std,
            minDistance: distances.min() ?? 0,
            maxDistance: distances.max() ?? 0
        )
    }
}
