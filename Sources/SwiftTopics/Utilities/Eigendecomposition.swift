// Eigendecomposition.swift
// SwiftTopics
//
// Pure Swift eigendecomposition using Jacobi algorithm

import Foundation
import Accelerate

// MARK: - Eigendecomposition

/// Symmetric matrix eigendecomposition using Jacobi algorithm.
///
/// Computes eigenvalues and eigenvectors of a real symmetric matrix using
/// the Jacobi rotation method, which is numerically stable and doesn't
/// rely on deprecated LAPACK interfaces.
///
/// ## Usage
/// ```swift
/// // Covariance matrix (3x3, row-major)
/// let matrix: [Float] = [
///     4.0, 2.0, 0.6,
///     2.0, 5.0, 3.0,
///     0.6, 3.0, 6.0
/// ]
///
/// let result = try Eigendecomposition.symmetric(matrix, dimension: 3)
/// // result.eigenvalues: sorted descending
/// // result.eigenvectors: column-major, sorted to match eigenvalues
/// ```
///
/// ## Algorithm
/// The Jacobi method iteratively applies rotation matrices to zero out
/// off-diagonal elements. Each rotation is a 2D Givens rotation in the
/// plane of the two elements being zeroed.
///
/// ## Performance
/// - Complexity: O(n³) per sweep, typically 5-10 sweeps for convergence
/// - Memory: O(n²) for the rotation matrix
///
/// ## Numerical Stability
/// The Jacobi method is unconditionally stable and maintains orthogonality
/// of eigenvectors to machine precision.
public enum Eigendecomposition {

    /// Result of eigendecomposition.
    public struct Result: Sendable {

        /// Eigenvalues sorted in descending order.
        public let eigenvalues: [Float]

        /// Eigenvectors as column-major matrix, columns sorted to match eigenvalues.
        ///
        /// For a d×d matrix, this array has d² elements.
        /// eigenvectors[i + j*d] is the i-th component of the j-th eigenvector.
        public let eigenvectors: [Float]

        /// Dimension of the matrix.
        public let dimension: Int

        /// Gets the i-th eigenvector (corresponding to eigenvalues[i]).
        ///
        /// - Parameter index: The eigenvector index (0 = largest eigenvalue).
        /// - Returns: The eigenvector as a Float array.
        public func eigenvector(at index: Int) -> [Float] {
            precondition(index >= 0 && index < dimension, "Index out of bounds")
            var vector = [Float](repeating: 0, count: dimension)
            for i in 0..<dimension {
                vector[i] = eigenvectors[i + index * dimension]
            }
            return vector
        }

        /// Gets the top k eigenvectors as a matrix (for PCA projection).
        ///
        /// Returns a d×k matrix in column-major format.
        ///
        /// - Parameter k: Number of eigenvectors to return.
        /// - Returns: Column-major matrix of top k eigenvectors.
        public func topKEigenvectors(_ k: Int) -> [Float] {
            precondition(k > 0 && k <= dimension, "k must be in [1, \(dimension)]")
            var result = [Float](repeating: 0, count: dimension * k)
            for j in 0..<k {
                for i in 0..<dimension {
                    result[i + j * dimension] = eigenvectors[i + j * dimension]
                }
            }
            return result
        }

        /// Cumulative explained variance ratio.
        ///
        /// The fraction of total variance explained by the first k components.
        ///
        /// - Parameter k: Number of components.
        /// - Returns: Cumulative variance ratio in [0, 1].
        public func cumulativeVarianceRatio(_ k: Int) -> Float {
            let total = eigenvalues.reduce(0, +)
            guard total > 0 else { return 0 }
            let cumulative = eigenvalues.prefix(k).reduce(0, +)
            return cumulative / total
        }

        /// Individual explained variance ratios.
        public var varianceRatios: [Float] {
            let total = eigenvalues.reduce(0, +)
            guard total > 0 else { return eigenvalues.map { _ in 0 } }
            return eigenvalues.map { $0 / total }
        }

        /// Number of components needed to explain the given variance ratio.
        ///
        /// - Parameter ratio: Target variance ratio (e.g., 0.95 for 95%).
        /// - Returns: Minimum number of components.
        public func componentsForVariance(_ ratio: Float) -> Int {
            var cumulative: Float = 0
            let total = eigenvalues.reduce(0, +)
            guard total > 0 else { return dimension }

            for (i, eigenvalue) in eigenvalues.enumerated() {
                cumulative += eigenvalue / total
                if cumulative >= ratio {
                    return i + 1
                }
            }
            return dimension
        }
    }

    // MARK: - Symmetric Decomposition

    /// Computes eigendecomposition of a symmetric matrix using Jacobi algorithm.
    ///
    /// - Parameters:
    ///   - matrix: Square symmetric matrix in row-major order.
    ///   - dimension: The matrix dimension (n for an n×n matrix).
    ///   - tolerance: Convergence tolerance (default: 1e-6, appropriate for Float32).
    ///   - maxSweeps: Maximum number of full sweeps (default: 30, each sweep visits all pairs).
    /// - Returns: Eigenvalues and eigenvectors.
    /// - Throws: `EigendecompositionError` on failure.
    ///
    /// ## Algorithm
    /// Uses cyclic Jacobi algorithm which systematically sweeps through all off-diagonal
    /// pairs rather than searching for the maximum. This provides more predictable
    /// convergence behavior for random/ill-conditioned matrices.
    ///
    /// ## Convergence Notes
    /// The default tolerance of 1e-6 is chosen to be achievable with Float32 precision
    /// (machine epsilon ~1.19e-7). For ill-conditioned matrices, regularization via
    /// `Eigendecomposition.regularize()` is recommended before decomposition.
    public static func symmetric(
        _ matrix: [Float],
        dimension: Int,
        tolerance: Float = 1e-6,
        maxSweeps: Int = 30
    ) throws -> Result {
        guard dimension > 0 else {
            throw EigendecompositionError.invalidDimension
        }

        guard matrix.count == dimension * dimension else {
            throw EigendecompositionError.matrixSizeMismatch(
                expected: dimension * dimension,
                actual: matrix.count
            )
        }

        // Handle trivial case
        if dimension == 1 {
            return Result(
                eigenvalues: [matrix[0]],
                eigenvectors: [1.0],
                dimension: 1
            )
        }

        // Create working copy of matrix (row-major)
        var A = matrix

        // Initialize eigenvector matrix to identity (row-major)
        var V = [Float](repeating: 0, count: dimension * dimension)
        for i in 0..<dimension {
            V[i * dimension + i] = 1.0
        }

        // Cyclic Jacobi iteration - sweeps through all (i,j) pairs systematically
        // This is more reliable for convergence than classical Jacobi
        var converged = false

        for _ in 0..<maxSweeps {
            // Compute off-diagonal Frobenius norm for convergence check
            var offDiagNorm: Float = 0
            for i in 0..<dimension {
                for j in (i + 1)..<dimension {
                    let val = A[i * dimension + j]
                    offDiagNorm += val * val
                }
            }
            offDiagNorm = sqrt(offDiagNorm * 2) // Factor of 2 for symmetry

            // Check convergence
            if offDiagNorm < tolerance {
                converged = true
                break
            }

            // Perform one sweep through all off-diagonal pairs
            for p in 0..<(dimension - 1) {
                for q in (p + 1)..<dimension {
                    let apq = A[p * dimension + q]

                    // Skip if already small enough
                    if abs(apq) < tolerance * 1e-2 {
                        continue
                    }

                    let app = A[p * dimension + p]
                    let aqq = A[q * dimension + q]

                    // Compute rotation angle using the stable formula
                    let c: Float
                    let s: Float

                    let diff = aqq - app
                    if abs(diff) < Float.ulpOfOne * max(abs(app), abs(aqq), 1.0) {
                        // Diagonal elements are nearly equal
                        c = Float(1.0 / sqrt(2.0))
                        s = apq > 0 ? c : -c
                    } else {
                        // Standard rotation formula with improved numerical stability
                        let tau = diff / (2.0 * apq)
                        let t: Float
                        if tau >= 0 {
                            t = 1.0 / (tau + sqrt(1.0 + tau * tau))
                        } else {
                            t = 1.0 / (tau - sqrt(1.0 + tau * tau))
                        }
                        c = 1.0 / sqrt(1.0 + t * t)
                        s = t * c
                    }

                    // Apply rotation to A: A' = J^T * A * J
                    for k in 0..<dimension {
                        if k != p && k != q {
                            let akp = A[k * dimension + p]
                            let akq = A[k * dimension + q]
                            A[k * dimension + p] = c * akp - s * akq
                            A[k * dimension + q] = s * akp + c * akq
                            // Maintain symmetry
                            A[p * dimension + k] = A[k * dimension + p]
                            A[q * dimension + k] = A[k * dimension + q]
                        }
                    }

                    // Update diagonal and off-diagonal elements
                    let newApp = c * c * app - 2.0 * s * c * apq + s * s * aqq
                    let newAqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
                    A[p * dimension + p] = newApp
                    A[q * dimension + q] = newAqq
                    A[p * dimension + q] = 0
                    A[q * dimension + p] = 0

                    // Update eigenvector matrix
                    for k in 0..<dimension {
                        let vkp = V[k * dimension + p]
                        let vkq = V[k * dimension + q]
                        V[k * dimension + p] = c * vkp - s * vkq
                        V[k * dimension + q] = s * vkp + c * vkq
                    }
                }
            }
        }

        if !converged {
            throw EigendecompositionError.convergenceFailed
        }

        // Extract eigenvalues from diagonal
        var eigenvalues = [Float](repeating: 0, count: dimension)
        for i in 0..<dimension {
            eigenvalues[i] = A[i * dimension + i]
        }

        // Sort eigenvalues and eigenvectors by descending eigenvalue
        var indices = Array(0..<dimension)
        indices.sort { eigenvalues[$0] > eigenvalues[$1] }

        let sortedEigenvalues = indices.map { eigenvalues[$0] }

        // Convert eigenvectors to column-major and reorder
        var sortedEigenvectors = [Float](repeating: 0, count: dimension * dimension)
        for (newIdx, oldIdx) in indices.enumerated() {
            for row in 0..<dimension {
                // V is row-major, output is column-major
                sortedEigenvectors[row + newIdx * dimension] = V[row * dimension + oldIdx]
            }
        }

        return Result(
            eigenvalues: sortedEigenvalues,
            eigenvectors: sortedEigenvectors,
            dimension: dimension
        )
    }

    /// Computes only the top k eigenvalues and eigenvectors.
    ///
    /// - Parameters:
    ///   - matrix: Square symmetric matrix in row-major order.
    ///   - dimension: The matrix dimension.
    ///   - k: Number of top eigenvalues/vectors to compute.
    /// - Returns: Top k eigenvalues and eigenvectors.
    /// - Throws: `EigendecompositionError` on failure.
    public static func topK(
        _ matrix: [Float],
        dimension: Int,
        k: Int
    ) throws -> Result {
        // Compute all and truncate (for now)
        // Could be optimized with partial Jacobi or power iteration for large k
        let full = try symmetric(matrix, dimension: dimension)

        guard k > 0 && k <= dimension else {
            throw EigendecompositionError.invalidK(k: k, dimension: dimension)
        }

        let truncatedEigenvalues = Array(full.eigenvalues.prefix(k))
        let truncatedEigenvectors = full.topKEigenvectors(k)

        return Result(
            eigenvalues: truncatedEigenvalues,
            eigenvectors: truncatedEigenvectors,
            dimension: dimension
        )
    }

    // MARK: - Double Precision

    /// Computes eigendecomposition of a symmetric matrix (double precision).
    ///
    /// - Parameters:
    ///   - matrix: Square symmetric matrix in row-major order.
    ///   - dimension: The matrix dimension.
    ///   - tolerance: Convergence tolerance.
    ///   - maxIterations: Maximum iterations.
    /// - Returns: Eigenvalues and eigenvectors (as Float).
    /// - Throws: `EigendecompositionError` on failure.
    public static func symmetricDouble(
        _ matrix: [Double],
        dimension: Int,
        tolerance: Double = 1e-15,
        maxIterations: Int = 50
    ) throws -> Result {
        guard dimension > 0 else {
            throw EigendecompositionError.invalidDimension
        }

        guard matrix.count == dimension * dimension else {
            throw EigendecompositionError.matrixSizeMismatch(
                expected: dimension * dimension,
                actual: matrix.count
            )
        }

        // Handle trivial case
        if dimension == 1 {
            return Result(
                eigenvalues: [Float(matrix[0])],
                eigenvectors: [1.0],
                dimension: 1
            )
        }

        // Create working copy of matrix
        var A = matrix

        // Initialize eigenvector matrix to identity
        var V = [Double](repeating: 0, count: dimension * dimension)
        for i in 0..<dimension {
            V[i * dimension + i] = 1.0
        }

        // Jacobi iteration
        var converged = false

        for _ in 0..<maxIterations {
            // Find largest off-diagonal element
            var maxVal: Double = 0
            var p = 0
            var q = 1

            for i in 0..<dimension {
                for j in (i + 1)..<dimension {
                    let absVal = abs(A[i * dimension + j])
                    if absVal > maxVal {
                        maxVal = absVal
                        p = i
                        q = j
                    }
                }
            }

            // Check convergence
            if maxVal < tolerance {
                converged = true
                break
            }

            // Compute rotation angle
            let app = A[p * dimension + p]
            let aqq = A[q * dimension + q]
            let apq = A[p * dimension + q]

            let theta: Double
            if abs(app - aqq) < Double.ulpOfOne {
                theta = Double.pi / 4.0 * (apq > 0 ? 1 : -1)
            } else {
                theta = 0.5 * atan(2.0 * apq / (aqq - app))
            }

            let c = cos(theta)
            let s = sin(theta)

            // Apply rotation
            for k in 0..<dimension {
                if k != p && k != q {
                    let akp = A[k * dimension + p]
                    let akq = A[k * dimension + q]
                    A[k * dimension + p] = c * akp - s * akq
                    A[k * dimension + q] = s * akp + c * akq
                    A[p * dimension + k] = A[k * dimension + p]
                    A[q * dimension + k] = A[k * dimension + q]
                }
            }

            let newApp = c * c * app - 2 * s * c * apq + s * s * aqq
            let newAqq = s * s * app + 2 * s * c * apq + c * c * aqq
            A[p * dimension + p] = newApp
            A[q * dimension + q] = newAqq
            A[p * dimension + q] = 0
            A[q * dimension + p] = 0

            for k in 0..<dimension {
                let vkp = V[k * dimension + p]
                let vkq = V[k * dimension + q]
                V[k * dimension + p] = c * vkp - s * vkq
                V[k * dimension + q] = s * vkp + c * vkq
            }
        }

        if !converged {
            throw EigendecompositionError.convergenceFailed
        }

        // Extract and sort
        var eigenvalues = [Double](repeating: 0, count: dimension)
        for i in 0..<dimension {
            eigenvalues[i] = A[i * dimension + i]
        }

        var indices = Array(0..<dimension)
        indices.sort { eigenvalues[$0] > eigenvalues[$1] }

        let sortedEigenvalues = indices.map { Float(eigenvalues[$0]) }

        var sortedEigenvectors = [Float](repeating: 0, count: dimension * dimension)
        for (newIdx, oldIdx) in indices.enumerated() {
            for row in 0..<dimension {
                sortedEigenvectors[row + newIdx * dimension] = Float(V[row * dimension + oldIdx])
            }
        }

        return Result(
            eigenvalues: sortedEigenvalues,
            eigenvectors: sortedEigenvectors,
            dimension: dimension
        )
    }
}

// MARK: - Errors

/// Errors from eigendecomposition operations.
public enum EigendecompositionError: Error, Sendable {

    /// Matrix dimension is invalid (must be > 0).
    case invalidDimension

    /// Matrix size doesn't match expected dimension².
    case matrixSizeMismatch(expected: Int, actual: Int)

    /// LAPACK workspace query failed.
    case workspaceQueryFailed(info: Int)

    /// Invalid argument at the given position.
    case invalidArgument(position: Int)

    /// Algorithm failed to converge.
    case convergenceFailed

    /// Invalid k value for top-k computation.
    case invalidK(k: Int, dimension: Int)
}

extension EigendecompositionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidDimension:
            return "Matrix dimension must be greater than 0"
        case .matrixSizeMismatch(let expected, let actual):
            return "Matrix size mismatch: expected \(expected), got \(actual)"
        case .workspaceQueryFailed(let info):
            return "Workspace query failed with info=\(info)"
        case .invalidArgument(let position):
            return "Invalid argument at position \(position)"
        case .convergenceFailed:
            return "Eigendecomposition failed to converge"
        case .invalidK(let k, let dimension):
            return "Invalid k=\(k) for dimension=\(dimension)"
        }
    }
}

// MARK: - Utility Functions

extension Eigendecomposition {

    /// Creates a symmetric matrix from eigenvalues and eigenvectors.
    ///
    /// Reconstructs A = V × D × V^T where D is diagonal eigenvalues.
    ///
    /// - Parameters:
    ///   - eigenvalues: The eigenvalues.
    ///   - eigenvectors: Column-major eigenvectors.
    ///   - dimension: Matrix dimension.
    /// - Returns: Reconstructed matrix in row-major order.
    public static func reconstruct(
        eigenvalues: [Float],
        eigenvectors: [Float],
        dimension: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: dimension * dimension)

        for i in 0..<dimension {
            for j in 0..<dimension {
                var sum: Float = 0
                for k in 0..<dimension {
                    // V[i,k] * D[k] * V[j,k]
                    let vi_k = eigenvectors[i + k * dimension]
                    let vj_k = eigenvectors[j + k * dimension]
                    sum += vi_k * eigenvalues[k] * vj_k
                }
                result[i * dimension + j] = sum
            }
        }

        return result
    }

    /// Computes the condition number of a symmetric positive definite matrix.
    ///
    /// κ(A) = λ_max / λ_min
    ///
    /// A high condition number indicates potential numerical instability.
    ///
    /// - Parameter result: Eigendecomposition result.
    /// - Returns: Condition number, or .infinity if matrix is singular.
    public static func conditionNumber(_ result: Result) -> Float {
        guard let maxEig = result.eigenvalues.first,
              let minEig = result.eigenvalues.last,
              minEig > 0 else {
            return .infinity
        }
        return maxEig / minEig
    }

    /// Regularizes a covariance matrix by adding a small value to the diagonal.
    ///
    /// This helps with numerical stability when eigenvalues are very small.
    ///
    /// - Parameters:
    ///   - matrix: Square matrix in row-major order.
    ///   - dimension: Matrix dimension.
    ///   - epsilon: Value to add to diagonal (default: 1e-6).
    /// - Returns: Regularized matrix.
    public static func regularize(
        _ matrix: [Float],
        dimension: Int,
        epsilon: Float = 1e-6
    ) -> [Float] {
        var result = matrix
        for i in 0..<dimension {
            result[i * dimension + i] += epsilon
        }
        return result
    }
}
