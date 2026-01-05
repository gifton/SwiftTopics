// PCA.swift
// SwiftTopics
//
// Principal Component Analysis for dimensionality reduction

import Foundation
import Accelerate

// MARK: - PCA Reducer

/// Principal Component Analysis (PCA) dimensionality reducer.
///
/// PCA finds the directions of maximum variance in high-dimensional data and
/// projects onto those directions. This is the standard preprocessing step
/// before HDBSCAN clustering.
///
/// ## Algorithm
/// 1. **Center data**: X_centered = X - mean(X)
/// 2. **Covariance matrix**: C = X^T × X / (n-1)
/// 3. **Eigendecomposition**: C = V × Λ × V^T
/// 4. **Project**: X_reduced = X_centered × V_k (top k eigenvectors)
///
/// ## Why PCA before clustering?
/// - **Curse of dimensionality**: Distance metrics lose meaning in high dimensions
/// - **Noise reduction**: Low-variance dimensions often contain noise
/// - **Performance**: Faster k-NN computation in lower dimensions
///
/// ## Usage
/// ```swift
/// var pca = PCAReducer(components: 50)
/// let reduced = try await pca.fitTransform(embeddings)
/// // reduced embeddings have 50 dimensions
/// ```
///
/// ## Thread Safety
/// `PCAReducer` is `Sendable` and can be used across concurrency domains.
/// The mutable fitting state is managed through the protocol's mutating methods.
public struct PCAReducer: DimensionReducer, Sendable {

    // MARK: - Properties

    /// Number of components to keep.
    public let components: Int

    /// Whether to whiten the output (scale by eigenvalues).
    public let whiten: Bool

    /// Minimum variance ratio to retain a component (overrides components if set).
    public let varianceRatio: Float?

    /// Random seed for reproducibility.
    public let seed: UInt64?

    /// The fitted state (nil before fitting).
    private var fittedState: FittedPCAState?

    // MARK: - DimensionReducer Protocol

    /// The output dimension after reduction.
    public var outputDimension: Int {
        fittedState?.effectiveComponents ?? components
    }

    /// Whether this reducer has been fitted.
    public var isFitted: Bool {
        fittedState != nil
    }

    // MARK: - Initialization

    /// Creates a PCA reducer with specified number of components.
    ///
    /// - Parameters:
    ///   - components: Number of principal components to keep.
    ///   - whiten: Whether to scale output by sqrt(eigenvalues). Default: false.
    ///   - varianceRatio: Optional minimum cumulative variance ratio to retain.
    ///   - seed: Random seed (unused in PCA, kept for protocol consistency).
    public init(
        components: Int = 50,
        whiten: Bool = false,
        varianceRatio: Float? = nil,
        seed: UInt64? = nil
    ) {
        precondition(components > 0, "Components must be positive")
        if let ratio = varianceRatio {
            precondition(ratio > 0 && ratio <= 1, "Variance ratio must be in (0, 1]")
        }
        self.components = components
        self.whiten = whiten
        self.varianceRatio = varianceRatio
        self.seed = seed
    }

    /// Creates a PCA reducer from a PCAConfiguration.
    ///
    /// - Parameters:
    ///   - config: The PCA configuration.
    ///   - outputDimension: Target output dimension.
    public init(config: PCAConfiguration, outputDimension: Int) {
        self.components = outputDimension
        self.whiten = config.whiten
        self.varianceRatio = config.varianceRatio
        self.seed = nil
    }

    // MARK: - Fit Transform

    /// Fits the PCA model and transforms the embeddings.
    ///
    /// This is the main entry point for PCA. It computes the principal components
    /// from the training data and projects onto them.
    ///
    /// - Parameter embeddings: The training embeddings.
    /// - Returns: Reduced-dimension embeddings.
    /// - Throws: `ReductionError` if fitting fails.
    public func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        var mutableSelf = self
        try await mutableSelf.fit(embeddings)
        return try await mutableSelf.transform(embeddings)
    }

    // MARK: - Fit

    /// Fits the PCA model to training data without transforming.
    ///
    /// - Parameter embeddings: The training embeddings.
    /// - Throws: `ReductionError` if fitting fails.
    public mutating func fit(_ embeddings: [Embedding]) async throws {
        // Validate input
        guard !embeddings.isEmpty else {
            throw ReductionError.emptyInput
        }

        let n = embeddings.count
        let d = embeddings[0].dimension

        guard embeddings.allSatisfy({ $0.dimension == d }) else {
            throw ReductionError.inconsistentDimensions
        }

        // Check that we have enough samples
        guard n >= 2 else {
            throw ReductionError.insufficientSamples(required: 2, provided: n)
        }

        // Effective components can't exceed input dimension or sample count - 1
        let maxComponents = min(d, n - 1)
        if components > maxComponents && varianceRatio == nil {
            throw ReductionError.outputDimensionTooLarge(
                requested: components,
                maxAllowed: maxComponents
            )
        }

        // Step 1: Center the data
        let mean = computeMean(embeddings)
        let centered = centerData(embeddings, mean: mean)

        // Step 2: Compute covariance matrix (d×d)
        let covariance = computeCovarianceMatrix(centered, dimension: d)

        // Step 3: Eigendecomposition
        let eigenResult: Eigendecomposition.Result
        do {
            // Regularize to improve numerical stability
            let regularized = Eigendecomposition.regularize(covariance, dimension: d, epsilon: 1e-6)
            eigenResult = try Eigendecomposition.symmetric(regularized, dimension: d)
        } catch let error as EigendecompositionError {
            throw ReductionError.numericalInstability("Eigendecomposition failed: \(error)")
        }

        // Step 4: Determine number of components
        let effectiveK: Int
        if let targetVariance = varianceRatio {
            effectiveK = eigenResult.componentsForVariance(targetVariance)
        } else {
            effectiveK = min(components, maxComponents)
        }

        // Step 5: Extract top-k eigenvectors
        let principalComponents = eigenResult.topKEigenvectors(effectiveK)
        let eigenvalues = Array(eigenResult.eigenvalues.prefix(effectiveK))

        // Compute explained variance ratio
        let totalVariance = eigenResult.eigenvalues.reduce(0, +)
        let explainedVariance = eigenvalues.reduce(0, +)
        let explainedRatio = totalVariance > 0 ? explainedVariance / totalVariance : 0

        // Store fitted state
        fittedState = FittedPCAState(
            mean: mean,
            principalComponents: principalComponents,
            eigenvalues: eigenvalues,
            inputDimension: d,
            effectiveComponents: effectiveK,
            explainedVarianceRatio: explainedRatio
        )
    }

    // MARK: - Transform

    /// Transforms embeddings using the fitted PCA model.
    ///
    /// Projects embeddings onto the principal components computed during fitting.
    ///
    /// - Parameter embeddings: Embeddings to transform.
    /// - Returns: Reduced-dimension embeddings.
    /// - Throws: `ReductionError` if not fitted or dimensions mismatch.
    public func transform(_ embeddings: [Embedding]) async throws -> [Embedding] {
        guard let state = fittedState else {
            throw ReductionError.notFitted
        }

        guard !embeddings.isEmpty else {
            return []
        }

        let d = embeddings[0].dimension
        guard d == state.inputDimension else {
            throw ReductionError.inconsistentDimensions
        }

        guard embeddings.allSatisfy({ $0.dimension == d }) else {
            throw ReductionError.inconsistentDimensions
        }

        // Center using fitted mean
        let centered = centerData(embeddings, mean: state.mean)

        // Project onto principal components
        var results = [Embedding]()
        results.reserveCapacity(embeddings.count)

        let k = state.effectiveComponents

        for embedding in centered {
            var projected = [Float](repeating: 0, count: k)

            // X_reduced = X_centered × V_k
            // Principal components are stored column-major (each column is an eigenvector)
            for j in 0..<k {
                var dot: Float = 0
                for i in 0..<d {
                    dot += embedding.vector[i] * state.principalComponents[i + j * d]
                }

                if whiten, state.eigenvalues[j] > Float.ulpOfOne {
                    // Whitening: divide by sqrt(eigenvalue)
                    projected[j] = dot / state.eigenvalues[j].squareRoot()
                } else {
                    projected[j] = dot
                }
            }

            results.append(Embedding(vector: projected))
        }

        return results
    }

    // MARK: - Accessors

    /// Gets the explained variance ratio for each component.
    ///
    /// - Returns: Array of variance ratios, or nil if not fitted.
    public var explainedVarianceRatios: [Float]? {
        guard let state = fittedState else { return nil }
        let total = state.eigenvalues.reduce(0, +)
        guard total > 0 else { return nil }
        return state.eigenvalues.map { $0 / total }
    }

    /// Gets the cumulative explained variance ratio.
    ///
    /// - Returns: Total variance ratio explained by all components, or nil if not fitted.
    public var cumulativeExplainedVariance: Float? {
        fittedState?.explainedVarianceRatio
    }

    /// Gets the eigenvalues of the principal components.
    ///
    /// - Returns: Eigenvalues (variances along principal axes), or nil if not fitted.
    public var eigenvalues: [Float]? {
        fittedState?.eigenvalues
    }

    /// Gets the principal components (transformation matrix).
    ///
    /// - Returns: Column-major matrix of eigenvectors, or nil if not fitted.
    public var principalComponents: [Float]? {
        fittedState?.principalComponents
    }

    // MARK: - Private Methods

    /// Computes the mean of embeddings.
    private func computeMean(_ embeddings: [Embedding]) -> [Float] {
        let n = embeddings.count
        let d = embeddings[0].dimension
        var mean = [Float](repeating: 0, count: d)

        for embedding in embeddings {
            for i in 0..<d {
                mean[i] += embedding.vector[i]
            }
        }

        let scale = 1.0 / Float(n)
        for i in 0..<d {
            mean[i] *= scale
        }

        return mean
    }

    /// Centers embeddings by subtracting the mean.
    private func centerData(_ embeddings: [Embedding], mean: [Float]) -> [Embedding] {
        embeddings.map { embedding in
            let centered = zip(embedding.vector, mean).map { $0 - $1 }
            return Embedding(vector: centered)
        }
    }

    /// Computes the covariance matrix from centered data.
    ///
    /// C = X^T × X / (n-1)
    ///
    /// Returns a d×d matrix in row-major order.
    private func computeCovarianceMatrix(_ centered: [Embedding], dimension d: Int) -> [Float] {
        let n = centered.count

        // Flatten centered data (row-major: each row is an embedding)
        var flatData = [Float]()
        flatData.reserveCapacity(n * d)
        for embedding in centered {
            flatData.append(contentsOf: embedding.vector)
        }

        // Transpose X to get X^T (d×n in row-major, which is n×d in column-major)
        var transposed = [Float](repeating: 0, count: n * d)
        vDSP_mtrans(flatData, 1, &transposed, 1, vDSP_Length(d), vDSP_Length(n))

        // Compute X^T × X using vDSP_mmul
        // transposed is d×n (row-major), flatData is n×d (row-major)
        // Result is d×d
        var covariance = [Float](repeating: 0, count: d * d)
        vDSP_mmul(
            transposed, 1,      // A = X^T (d×n)
            flatData, 1,        // B = X (n×d)
            &covariance, 1,     // C = result (d×d)
            vDSP_Length(d),     // M: rows of A / result
            vDSP_Length(d),     // N: columns of B / result
            vDSP_Length(n)      // K: columns of A / rows of B
        )

        // Scale by 1/(n-1) for sample covariance
        var scale = 1.0 / Float(n - 1)
        vDSP_vsmul(covariance, 1, &scale, &covariance, 1, vDSP_Length(d * d))

        return covariance
    }
}

// MARK: - Fitted State

/// The fitted state of a PCA model.
private struct FittedPCAState: Sendable {

    /// The mean of the training data (for centering new data).
    let mean: [Float]

    /// Principal components as column-major matrix (d × k).
    /// Each column is an eigenvector.
    let principalComponents: [Float]

    /// Eigenvalues (variances) for each principal component.
    let eigenvalues: [Float]

    /// Original input dimension.
    let inputDimension: Int

    /// Number of components actually used.
    let effectiveComponents: Int

    /// Cumulative explained variance ratio.
    let explainedVarianceRatio: Float
}

// MARK: - Convenience Functions

/// Convenience function for one-shot PCA reduction.
///
/// - Parameters:
///   - embeddings: The embeddings to reduce.
///   - components: Number of components to keep.
///   - whiten: Whether to whiten the output.
/// - Returns: Reduced embeddings.
public func pca(
    _ embeddings: [Embedding],
    components: Int = 50,
    whiten: Bool = false
) async throws -> [Embedding] {
    let reducer = PCAReducer(components: components, whiten: whiten)
    return try await reducer.fitTransform(embeddings)
}

/// Convenience function for PCA with variance threshold.
///
/// - Parameters:
///   - embeddings: The embeddings to reduce.
///   - varianceRatio: Minimum cumulative variance to retain (e.g., 0.95 for 95%).
///   - whiten: Whether to whiten the output.
/// - Returns: Reduced embeddings.
public func pcaWithVariance(
    _ embeddings: [Embedding],
    varianceRatio: Float = 0.95,
    whiten: Bool = false
) async throws -> [Embedding] {
    let reducer = PCAReducer(components: 1, whiten: whiten, varianceRatio: varianceRatio)
    return try await reducer.fitTransform(embeddings)
}

// MARK: - PCA Builder

/// Builder for configuring PCA reducers.
public struct PCABuilder: Sendable {

    private var components: Int = 50
    private var whiten: Bool = false
    private var varianceRatio: Float? = nil
    private var seed: UInt64? = nil

    /// Creates a new PCA builder with default settings.
    public init() {}

    /// Sets the number of components.
    public func components(_ n: Int) -> PCABuilder {
        var copy = self
        copy.components = n
        return copy
    }

    /// Enables whitening.
    public func whiten(_ enable: Bool = true) -> PCABuilder {
        var copy = self
        copy.whiten = enable
        return copy
    }

    /// Sets the minimum variance ratio to retain.
    public func varianceRatio(_ ratio: Float) -> PCABuilder {
        var copy = self
        copy.varianceRatio = ratio
        return copy
    }

    /// Sets the random seed.
    public func seed(_ s: UInt64) -> PCABuilder {
        var copy = self
        copy.seed = s
        return copy
    }

    /// Builds the PCA reducer.
    public func build() -> PCAReducer {
        PCAReducer(
            components: components,
            whiten: whiten,
            varianceRatio: varianceRatio,
            seed: seed
        )
    }
}

// MARK: - Array Extension for PCA

extension Array where Element == Embedding {

    /// Reduces dimensionality using PCA.
    ///
    /// - Parameters:
    ///   - components: Number of principal components.
    ///   - whiten: Whether to whiten the output.
    /// - Returns: Reduced embeddings.
    public func reducePCA(
        components: Int = 50,
        whiten: Bool = false
    ) async throws -> [Embedding] {
        try await pca(self, components: components, whiten: whiten)
    }

    /// Reduces dimensionality to retain the specified variance.
    ///
    /// - Parameters:
    ///   - varianceRatio: Minimum cumulative variance ratio (e.g., 0.95).
    ///   - whiten: Whether to whiten the output.
    /// - Returns: Reduced embeddings.
    public func reducePCAToVariance(
        _ varianceRatio: Float = 0.95,
        whiten: Bool = false
    ) async throws -> [Embedding] {
        try await pcaWithVariance(self, varianceRatio: varianceRatio, whiten: whiten)
    }
}
