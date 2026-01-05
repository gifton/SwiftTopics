// UMAPOptimizer.swift
// SwiftTopics
//
// Stochastic Gradient Descent optimizer for UMAP

import Foundation

// MARK: - UMAP Optimizer

/// Stochastic Gradient Descent optimizer for UMAP embedding.
///
/// This is the core optimization loop that iteratively refines the
/// low-dimensional embedding to match the high-dimensional fuzzy structure.
///
/// ## Optimization Objective
///
/// UMAP minimizes the cross-entropy between:
/// - P: The high-dimensional fuzzy simplicial set (from k-NN)
/// - Q: The low-dimensional fuzzy simplicial set (what we're learning)
///
/// ## Force Model
///
/// Each iteration applies two types of forces:
///
/// **Attractive force** (for connected pairs):
/// Pulls neighbors together based on their membership weight.
/// ```
/// F_attract = -2ab × w_ij × (d²)^(b-1) / (1 + a × d^(2b)) × (y_i - y_j)
/// ```
///
/// **Repulsive force** (for random non-neighbors):
/// Pushes non-connected points apart (negative sampling).
/// ```
/// F_repel = 2b / ((0.001 + d²) × (1 + a × d^(2b))) × (y_i - y_j)
/// ```
///
/// Where:
/// - `a`, `b`: Curve parameters derived from `minDist` (typically a ≈ 1.93, b ≈ 0.79)
/// - `d`: Euclidean distance between points in low-dim space
/// - `w_ij`: Edge weight (membership strength)
///
/// ## Thread Safety
/// `UMAPOptimizer` is an actor for thread-safe optimization state.
public actor UMAPOptimizer {

    // MARK: - Curve Parameters

    /// The `a` parameter for the low-dimensional curve.
    private let a: Float

    /// The `b` parameter for the low-dimensional curve.
    private let b: Float

    /// Small constant to prevent division by zero.
    private let epsilon: Float = 0.001

    /// Maximum gradient magnitude (for clipping).
    private let gradientClip: Float = 4.0

    // MARK: - Optimization State

    /// Current embedding coordinates.
    private var embedding: [[Float]]

    /// Number of dimensions.
    private let nComponents: Int

    /// Random state for negative sampling.
    private var rng: RandomState

    // MARK: - Initialization

    /// Creates a UMAP optimizer.
    ///
    /// - Parameters:
    ///   - initialEmbedding: Starting positions [n × d].
    ///   - minDist: Minimum distance parameter (affects curve shape).
    ///   - seed: Random seed for negative sampling.
    public init(
        initialEmbedding: [[Float]],
        minDist: Float = 0.1,
        seed: UInt64? = nil
    ) {
        precondition(!initialEmbedding.isEmpty, "Initial embedding cannot be empty")
        precondition(!initialEmbedding[0].isEmpty, "Embedding dimension must be > 0")

        self.embedding = initialEmbedding
        self.nComponents = initialEmbedding[0].count
        self.rng = RandomState(seed: seed)

        // Compute curve parameters a and b from minDist
        // These are derived by fitting the curve: (1 + a × d^(2b))^(-1)
        // to approximate: 1 if d ≤ minDist, exp(-(d - minDist)) otherwise
        (self.a, self.b) = Self.findAB(spread: 1.0, minDist: minDist)
    }

    // MARK: - Curve Fitting

    /// Computes the a and b parameters from spread and minDist.
    ///
    /// These parameters define the low-dimensional similarity curve:
    /// φ(d) = 1 / (1 + a × d^(2b))
    ///
    /// The curve is fit to approximate:
    /// ψ(d) = 1 if d ≤ minDist, exp(-(d - minDist) / spread) otherwise
    private static func findAB(spread: Float, minDist: Float) -> (Float, Float) {
        // These are pre-computed values for common minDist settings
        // For minDist = 0.1, spread = 1.0: a ≈ 1.929, b ≈ 0.7915
        // For minDist = 0.0, spread = 1.0: a ≈ 1.577, b ≈ 0.8951
        // For minDist = 0.5, spread = 1.0: a ≈ 2.326, b ≈ 0.6702

        // Use curve fitting (simplified version)
        // In practice, these are computed by scipy.optimize.curve_fit
        // Here we use a simple approximation formula

        if minDist < 0.001 {
            return (1.577, 0.8951)
        }

        // Approximate formula based on curve fitting
        let b = 0.79 - 0.12 * minDist
        let a = 1.93 * pow(1 + minDist, 0.5)

        return (a, max(b, 0.1))
    }

    // MARK: - Optimization

    /// Runs the full optimization loop.
    ///
    /// - Parameters:
    ///   - fuzzySet: The high-dimensional fuzzy simplicial set.
    ///   - nEpochs: Number of optimization epochs.
    ///   - learningRate: Initial learning rate.
    ///   - negativeSampleRate: Number of negative samples per positive.
    ///   - progressHandler: Optional callback for progress updates.
    /// - Returns: Optimized embedding.
    public func optimize(
        fuzzySet: FuzzySimplicialSet,
        nEpochs: Int,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        progressHandler: ((Float) -> Void)? = nil
    ) async -> [[Float]] {
        let n = embedding.count
        let edges = fuzzySet.toEdgeList()

        guard !edges.isEmpty else {
            return embedding
        }

        // Pre-compute sampling schedule
        var schedule = fuzzySet.createSamplingSchedule(nEpochs: nEpochs)

        // Run epochs
        for epoch in 0..<nEpochs {
            // Compute current learning rate (linear decay)
            let alpha = learningRate * (1.0 - Float(epoch) / Float(nEpochs))

            // Get edges to sample this epoch
            let edgesToSample = schedule.edgesToSample(epoch: epoch)

            // Process each edge
            for edgeIdx in edgesToSample {
                let edge = edges[edgeIdx]

                // Apply attractive force
                applyAttractiveForce(
                    source: edge.source,
                    target: edge.target,
                    learningRate: alpha
                )

                // Apply repulsive forces (negative sampling)
                for _ in 0..<negativeSampleRate {
                    let negTarget = rng.nextInt(upperBound: n)
                    if negTarget != edge.source {
                        applyRepulsiveForce(
                            source: edge.source,
                            target: negTarget,
                            learningRate: alpha
                        )
                    }
                }
            }

            // Report progress
            if let handler = progressHandler {
                handler(Float(epoch + 1) / Float(nEpochs))
            }
        }

        return embedding
    }

    // MARK: - Force Application

    /// Applies attractive force between connected points.
    ///
    /// The gradient of the attractive term is:
    /// ∂/∂y_i (w_ij × log(φ(d_ij))) = w_ij × φ'(d_ij) / φ(d_ij) × (y_i - y_j) / d_ij
    ///
    /// Where φ(d) = 1 / (1 + a × d^(2b))
    private func applyAttractiveForce(
        source: Int,
        target: Int,
        learningRate: Float
    ) {
        // Compute squared distance
        var distSq: Float = 0
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            distSq += diff * diff
        }

        // Avoid degenerate case
        guard distSq > epsilon else { return }

        // Compute gradient coefficient
        // grad_coef = -2ab × d^(2b-2) / (1 + a × d^(2b))
        let distPow2b = pow(distSq, b)
        let gradCoef = -2.0 * a * b * pow(distSq, b - 1) / (1.0 + a * distPow2b)

        // Apply gradient to both points (Newton's third law)
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            var grad = gradCoef * diff

            // Clip gradient
            grad = max(-gradientClip, min(gradientClip, grad))

            embedding[source][d] -= learningRate * grad
            embedding[target][d] += learningRate * grad
        }
    }

    /// Applies repulsive force between non-connected points.
    ///
    /// The gradient of the repulsive term is:
    /// ∂/∂y_i ((1-w_ij) × log(1 - φ(d_ij))) ≈ (1 - φ(d_ij))^(-1) × φ'(d_ij) × (y_i - y_j) / d_ij
    private func applyRepulsiveForce(
        source: Int,
        target: Int,
        learningRate: Float
    ) {
        // Compute squared distance
        var distSq: Float = 0
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            distSq += diff * diff
        }

        // Add epsilon to avoid division by zero when points coincide
        distSq = max(distSq, epsilon)

        // Compute gradient coefficient
        // grad_coef = 2b / ((ε + d²) × (1 + a × d^(2b)))
        let distPow2b = pow(distSq, b)
        let denominator = (epsilon + distSq) * (1.0 + a * distPow2b)

        guard denominator > epsilon else { return }

        let gradCoef = 2.0 * b / denominator

        // Apply gradient to source only (target is random, no gradient there)
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            var grad = gradCoef * diff

            // Clip gradient
            grad = max(-gradientClip, min(gradientClip, grad))

            embedding[source][d] += learningRate * grad
        }
    }

    // MARK: - Accessors

    /// Returns the current embedding.
    public func getEmbedding() -> [[Float]] {
        embedding
    }

    /// Returns the number of points.
    public var pointCount: Int {
        embedding.count
    }
}

// MARK: - Batch Optimization

extension UMAPOptimizer {

    /// Optimizes with batch updates for better parallelism.
    ///
    /// This version accumulates gradients for all edges in a batch
    /// before applying updates, which can be more efficient for
    /// parallel execution.
    ///
    /// - Parameters:
    ///   - fuzzySet: The fuzzy simplicial set.
    ///   - nEpochs: Number of epochs.
    ///   - learningRate: Initial learning rate.
    ///   - batchSize: Number of edges per batch.
    ///   - negativeSampleRate: Negative samples per positive edge.
    /// - Returns: Optimized embedding.
    public func optimizeBatched(
        fuzzySet: FuzzySimplicialSet,
        nEpochs: Int,
        learningRate: Float = 1.0,
        batchSize: Int = 100,
        negativeSampleRate: Int = 5
    ) async -> [[Float]] {
        let n = embedding.count
        let edges = fuzzySet.toEdgeList()

        guard !edges.isEmpty else {
            return embedding
        }

        // Create edge indices for shuffling
        var edgeIndices = Array(0..<edges.count)

        for epoch in 0..<nEpochs {
            // Shuffle edges each epoch
            rng.shuffle(&edgeIndices)

            // Compute learning rate with decay
            let alpha = learningRate * (1.0 - Float(epoch) / Float(nEpochs))

            // Process in batches
            var offset = 0
            while offset < edgeIndices.count {
                let batchEnd = min(offset + batchSize, edgeIndices.count)

                // Accumulate gradients for this batch
                var gradients = [[Float]](repeating: [Float](repeating: 0, count: nComponents), count: n)

                for i in offset..<batchEnd {
                    let edgeIdx = edgeIndices[i]
                    let edge = edges[edgeIdx]

                    // Compute attractive gradient
                    let attractGrad = computeAttractiveGradient(
                        source: edge.source,
                        target: edge.target
                    )

                    // Accumulate
                    for d in 0..<nComponents {
                        gradients[edge.source][d] += attractGrad.source[d]
                        gradients[edge.target][d] += attractGrad.target[d]
                    }

                    // Negative sampling
                    for _ in 0..<negativeSampleRate {
                        let negTarget = rng.nextInt(upperBound: n)
                        if negTarget != edge.source {
                            let repelGrad = computeRepulsiveGradient(
                                source: edge.source,
                                target: negTarget
                            )
                            for d in 0..<nComponents {
                                gradients[edge.source][d] += repelGrad[d]
                            }
                        }
                    }
                }

                // Apply accumulated gradients
                for i in 0..<n {
                    for d in 0..<nComponents {
                        let grad = max(-gradientClip, min(gradientClip, gradients[i][d]))
                        embedding[i][d] -= alpha * grad
                    }
                }

                offset = batchEnd
            }
        }

        return embedding
    }

    /// Computes attractive gradient without applying it.
    private func computeAttractiveGradient(
        source: Int,
        target: Int
    ) -> (source: [Float], target: [Float]) {
        var sourceGrad = [Float](repeating: 0, count: nComponents)
        var targetGrad = [Float](repeating: 0, count: nComponents)

        var distSq: Float = 0
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            distSq += diff * diff
        }

        guard distSq > epsilon else {
            return (sourceGrad, targetGrad)
        }

        let distPow2b = pow(distSq, b)
        let gradCoef = -2.0 * a * b * pow(distSq, b - 1) / (1.0 + a * distPow2b)

        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            let grad = gradCoef * diff
            sourceGrad[d] = grad
            targetGrad[d] = -grad
        }

        return (sourceGrad, targetGrad)
    }

    /// Computes repulsive gradient without applying it.
    private func computeRepulsiveGradient(
        source: Int,
        target: Int
    ) -> [Float] {
        var grad = [Float](repeating: 0, count: nComponents)

        var distSq: Float = 0
        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            distSq += diff * diff
        }

        distSq = max(distSq, epsilon)
        let distPow2b = pow(distSq, b)
        let denominator = (epsilon + distSq) * (1.0 + a * distPow2b)

        guard denominator > epsilon else { return grad }

        let gradCoef = 2.0 * b / denominator

        for d in 0..<nComponents {
            let diff = embedding[source][d] - embedding[target][d]
            grad[d] = -gradCoef * diff  // Negative because repulsion pushes apart
        }

        return grad
    }
}

// MARK: - Interruptible Optimization

extension UMAPOptimizer {

    /// Result of interruptible optimization.
    public struct InterruptibleResult: Sendable {
        /// The embedding state (may be partial if interrupted).
        public let embedding: [[Float]]

        /// The epoch completed (last fully processed epoch, 0-indexed).
        public let completedEpoch: Int

        /// Total epochs requested.
        public let totalEpochs: Int

        /// Whether optimization completed all epochs.
        public var isComplete: Bool { completedEpoch >= totalEpochs - 1 }

        /// Progress as a fraction (0.0 to 1.0).
        public var progress: Float {
            guard totalEpochs > 0 else { return 1.0 }
            return Float(completedEpoch + 1) / Float(totalEpochs)
        }
    }

    /// Checkpoint callback parameters.
    public struct CheckpointInfo: Sendable {
        /// Current epoch (0-indexed).
        public let epoch: Int

        /// Total epochs.
        public let totalEpochs: Int

        /// Current embedding state.
        public let embedding: [[Float]]

        /// The sampling schedule state for accurate resumption.
        public let samplingScheduleState: [Float]

        /// Progress fraction.
        public var progress: Float {
            Float(epoch + 1) / Float(totalEpochs)
        }
    }

    /// Runs interruptible optimization with checkpoint support.
    ///
    /// This method supports:
    /// - **Interruption**: The `shouldContinue` closure is checked after each epoch
    /// - **Checkpointing**: The `onCheckpoint` callback is called at specified intervals
    /// - **Resumption**: Pass `startingEpoch` and `samplingScheduleState` from a checkpoint
    ///
    /// ## Resumption
    ///
    /// To resume from a checkpoint:
    /// 1. Create a new `UMAPOptimizer` with `initialEmbedding` set to the checkpoint's embedding
    /// 2. Call this method with `startingEpoch` and `samplingScheduleState` from the checkpoint
    ///
    /// - Parameters:
    ///   - fuzzySet: The high-dimensional fuzzy simplicial set.
    ///   - nEpochs: Total number of optimization epochs.
    ///   - learningRate: Initial learning rate.
    ///   - negativeSampleRate: Number of negative samples per positive.
    ///   - startingEpoch: Epoch to start from (for resumption). Default is 0.
    ///   - samplingScheduleState: The `nextSampleEpoch` array from a checkpoint (for resumption).
    ///   - checkpointInterval: Number of epochs between checkpoints. Default is 50.
    ///   - shouldContinue: Closure called after each epoch. Return false to interrupt.
    ///   - onCheckpoint: Callback for checkpoint saving.
    /// - Returns: Result containing final/partial embedding and completion state.
    public func optimizeInterruptible(
        fuzzySet: FuzzySimplicialSet,
        nEpochs: Int,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        startingEpoch: Int = 0,
        samplingScheduleState: [Float]? = nil,
        checkpointInterval: Int = 50,
        shouldContinue: @escaping @Sendable () -> Bool = { true },
        onCheckpoint: (@Sendable (CheckpointInfo) async -> Void)? = nil
    ) async -> InterruptibleResult {
        let n = embedding.count
        let edges = fuzzySet.toEdgeList()

        guard !edges.isEmpty else {
            return InterruptibleResult(
                embedding: embedding,
                completedEpoch: nEpochs - 1,
                totalEpochs: nEpochs
            )
        }

        // Create or restore sampling schedule
        var schedule: EdgeSamplingSchedule
        if let savedState = samplingScheduleState,
           savedState.count == edges.count {
            // Restore from checkpoint
            schedule = EdgeSamplingSchedule(
                edges: edges,
                epochsPerEdge: fuzzySet.epochsPerEdge(nEpochs: nEpochs),
                nextSampleEpoch: savedState,
                totalEpochs: nEpochs
            )
        } else {
            // Create fresh schedule
            schedule = fuzzySet.createSamplingSchedule(nEpochs: nEpochs)
        }

        var lastCompletedEpoch = startingEpoch > 0 ? startingEpoch - 1 : -1

        // Run epochs
        for epoch in startingEpoch..<nEpochs {
            // Check if we should continue before starting epoch
            guard shouldContinue() else {
                break
            }

            // Compute current learning rate (linear decay from full schedule)
            let alpha = learningRate * (1.0 - Float(epoch) / Float(nEpochs))

            // Get edges to sample this epoch
            let edgesToSample = schedule.edgesToSample(epoch: epoch)

            // Process each edge
            for edgeIdx in edgesToSample {
                let edge = edges[edgeIdx]

                // Apply attractive force
                applyAttractiveForce(
                    source: edge.source,
                    target: edge.target,
                    learningRate: alpha
                )

                // Apply repulsive forces (negative sampling)
                for _ in 0..<negativeSampleRate {
                    let negTarget = rng.nextInt(upperBound: n)
                    if negTarget != edge.source {
                        applyRepulsiveForce(
                            source: edge.source,
                            target: negTarget,
                            learningRate: alpha
                        )
                    }
                }
            }

            lastCompletedEpoch = epoch

            // Call checkpoint callback at intervals
            if let onCheckpoint = onCheckpoint,
               (epoch + 1) % checkpointInterval == 0 || epoch == nEpochs - 1 {
                let info = CheckpointInfo(
                    epoch: epoch,
                    totalEpochs: nEpochs,
                    embedding: embedding,
                    samplingScheduleState: schedule.nextSampleEpoch
                )
                await onCheckpoint(info)
            }
        }

        return InterruptibleResult(
            embedding: embedding,
            completedEpoch: lastCompletedEpoch,
            totalEpochs: nEpochs
        )
    }

    /// Sets the embedding state (for resumption from checkpoint).
    ///
    /// Use this to restore the embedding state before calling `optimizeInterruptible`
    /// with a `startingEpoch > 0`.
    ///
    /// - Parameter newEmbedding: The embedding state to restore.
    public func setEmbedding(_ newEmbedding: [[Float]]) {
        precondition(
            newEmbedding.count == embedding.count,
            "New embedding must have same point count"
        )
        precondition(
            newEmbedding.first?.count == nComponents,
            "New embedding must have same dimension"
        )
        self.embedding = newEmbedding
    }

    /// Creates a fresh UMAP optimizer for resumption from a checkpoint.
    ///
    /// - Parameters:
    ///   - savedEmbedding: The embedding state from the checkpoint.
    ///   - minDist: The minDist parameter (should match original).
    ///   - seed: Random seed for negative sampling.
    /// - Returns: A new optimizer initialized with the saved embedding.
    public static func forResumption(
        savedEmbedding: [[Float]],
        minDist: Float = 0.1,
        seed: UInt64? = nil
    ) -> UMAPOptimizer {
        UMAPOptimizer(
            initialEmbedding: savedEmbedding,
            minDist: minDist,
            seed: seed
        )
    }
}


// MARK: - Optimization Configuration

/// Configuration for UMAP optimization.
public struct UMAPOptimizationConfig: Sendable {

    /// Number of optimization epochs.
    public let nEpochs: Int

    /// Initial learning rate.
    public let learningRate: Float

    /// Learning rate decay type.
    public let decay: LearningRateDecay

    /// Number of negative samples per positive edge.
    public let negativeSampleRate: Int

    /// Gradient clipping threshold.
    public let gradientClip: Float

    /// Creates optimization configuration.
    public init(
        nEpochs: Int = 200,
        learningRate: Float = 1.0,
        decay: LearningRateDecay = .linear,
        negativeSampleRate: Int = 5,
        gradientClip: Float = 4.0
    ) {
        self.nEpochs = nEpochs
        self.learningRate = learningRate
        self.decay = decay
        self.negativeSampleRate = negativeSampleRate
        self.gradientClip = gradientClip
    }

    /// Default configuration.
    public static let `default` = UMAPOptimizationConfig()

    /// Fast configuration with fewer epochs.
    public static let fast = UMAPOptimizationConfig(
        nEpochs: 100,
        learningRate: 1.0
    )

    /// High quality configuration.
    public static let quality = UMAPOptimizationConfig(
        nEpochs: 500,
        learningRate: 1.0
    )
}

/// Learning rate decay schedule.
public enum LearningRateDecay: String, Sendable, Codable {
    /// Linear decay from initial to 0.
    case linear

    /// Cosine annealing.
    case cosine

    /// No decay (constant learning rate).
    case none

    /// Computes learning rate at given epoch.
    public func rate(initial: Float, epoch: Int, totalEpochs: Int) -> Float {
        switch self {
        case .linear:
            return initial * (1.0 - Float(epoch) / Float(totalEpochs))
        case .cosine:
            return initial * 0.5 * (1.0 + cos(Float.pi * Float(epoch) / Float(totalEpochs)))
        case .none:
            return initial
        }
    }
}
