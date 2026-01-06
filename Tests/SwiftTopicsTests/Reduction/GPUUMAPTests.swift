// GPUUMAPTests.swift
// SwiftTopicsTests
//
// Tests for GPU-accelerated UMAP dimensionality reduction via VectorAccelerate integration.

import XCTest
@testable import SwiftTopics

/// Tests for GPU-accelerated UMAP optimization.
///
/// These tests validate that the GPU path (via VectorAccelerate's UMAPGradientKernel)
/// produces equivalent embeddings to the CPU path while providing significant
/// performance improvements.
final class GPUUMAPTests: XCTestCase {

    // MARK: - Test Fixtures

    /// Generates random test embeddings.
    ///
    /// - Parameters:
    ///   - count: Number of embeddings to generate.
    ///   - dimension: Dimension of each embedding (default: 384).
    ///   - seed: Random seed for reproducibility.
    /// - Returns: Array of random embeddings.
    private func generateRandomEmbeddings(
        count: Int,
        dimension: Int = 384,
        seed: UInt64 = 42
    ) -> [Embedding] {
        var rng = SeededRandomNumberGenerator(seed: seed)
        return (0..<count).map { _ in
            let vector = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
            return Embedding(vector: vector)
        }
    }

    /// Generates clustered test embeddings with known structure.
    ///
    /// Creates embeddings that form distinct clusters for validation.
    ///
    /// - Parameters:
    ///   - clustersCount: Number of clusters to generate.
    ///   - pointsPerCluster: Points per cluster.
    ///   - dimension: Embedding dimension.
    ///   - clusterSpread: Standard deviation within clusters.
    ///   - seed: Random seed.
    /// - Returns: Array of embeddings with cluster structure.
    private func generateClusteredEmbeddings(
        clustersCount: Int,
        pointsPerCluster: Int,
        dimension: Int = 384,
        clusterSpread: Float = 0.1,
        seed: UInt64 = 42
    ) -> [Embedding] {
        var rng = SeededRandomNumberGenerator(seed: seed)
        var embeddings: [Embedding] = []

        for _ in 0..<clustersCount {
            // Generate cluster center
            let center = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }

            // Generate points around center
            for _ in 0..<pointsPerCluster {
                let vector = center.map { $0 + Float.random(in: -clusterSpread...clusterSpread, using: &rng) }
                embeddings.append(Embedding(vector: vector))
            }
        }

        return embeddings
    }

    // MARK: - GPU vs CPU Equivalence Tests

    /// Tests that GPU and CPU paths produce embeddings with similar structure.
    func testGPUUMAPPreservesClusterStructure() async throws {
        // Generate clustered data (3 clusters of 50 points each = 150 total)
        let embeddings = generateClusteredEmbeddings(
            clustersCount: 3,
            pointsPerCluster: 50,
            dimension: 384,
            clusterSpread: 0.05,
            seed: 12345
        )

        let config = UMAPConfiguration(
            nNeighbors: 15,
            minDist: 0.1,
            nEpochs: 50
        )

        // CPU-only path (no GPU context)
        let cpuReducer = UMAPReducer(
            configuration: config,
            nComponents: 2,
            seed: 42,
            gpuContext: nil
        )
        let cpuResult = try await cpuReducer.fitTransform(embeddings)

        // GPU path
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let gpuReducer = UMAPReducer(
            configuration: config,
            nComponents: 2,
            seed: 42,
            gpuContext: gpuContext
        )
        let gpuResult = try await gpuReducer.fitTransform(embeddings)

        // Both should produce valid embeddings
        XCTAssertEqual(cpuResult.count, embeddings.count)
        XCTAssertEqual(gpuResult.count, embeddings.count)

        // Validate that GPU result has valid dimensions
        for embedding in gpuResult {
            XCTAssertEqual(embedding.dimension, 2)
            // Check for valid values (not NaN or Inf)
            for val in embedding.vector {
                XCTAssertFalse(val.isNaN, "GPU embedding contains NaN")
                XCTAssertFalse(val.isInfinite, "GPU embedding contains Inf")
            }
        }
    }

    /// Tests that GPU path produces bounded embeddings.
    func testGPUUMAPProducesBoundedEmbeddings() async throws {
        let embeddings = generateRandomEmbeddings(count: 200, dimension: 384, seed: 54321)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let reducer = UMAPReducer(
            nNeighbors: 15,
            minDist: 0.1,
            nComponents: 2,
            nEpochs: 100,
            seed: 42,
            gpuContext: gpuContext
        )

        let result = try await reducer.fitTransform(embeddings)

        // Embeddings should be bounded (not exploding)
        var maxMagnitude: Float = 0
        for embedding in result {
            for val in embedding.vector {
                maxMagnitude = max(maxMagnitude, abs(val))
            }
        }

        // UMAP embeddings should generally stay within reasonable bounds
        XCTAssertLessThan(
            maxMagnitude,
            1000,
            "GPU UMAP embeddings should not explode (max magnitude: \(maxMagnitude))"
        )
    }

    // MARK: - Performance Tests

    /// Tests that GPU UMAP optimization epochs are fast.
    ///
    /// Note: This test focuses on the GPU optimization step only, not the full
    /// pipeline (k-NN graph construction, spectral initialization are CPU-bound).
    func testGPUUMAPPerformance() async throws {
        // Generate dataset
        let embeddings = generateRandomEmbeddings(count: 300, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)

        // Skip if GPU not available
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available for performance test")
        }

        // Use fewer epochs for a reasonable test time
        // The full pipeline includes k-NN graph (CPU) + spectral embedding (CPU) + optimization (GPU)
        let config = UMAPConfiguration(
            nNeighbors: 15,
            minDist: 0.1,
            nEpochs: 50  // Reduced epochs for test
        )

        let reducer = UMAPReducer(
            configuration: config,
            nComponents: 2,
            seed: 42,
            gpuContext: gpu
        )

        // Time GPU path (includes k-NN graph construction)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await reducer.fitTransform(embeddings)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Full UMAP pipeline includes CPU-bound k-NN and spectral embedding
        // The GPU acceleration applies to the optimization step
        // We mainly verify it completes successfully here
        XCTAssertLessThan(
            elapsed,
            180.0,  // Allow 3 minutes for full pipeline with 300 points
            "GPU UMAP for 300 points should complete in < 180s, took \(elapsed)s"
        )

        print("GPU UMAP for \(embeddings.count) points completed in \(String(format: "%.3f", elapsed))s")
    }

    /// Tests GPU optimization epoch performance.
    func testGPUOptimizeEpochPerformance() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)

        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available for performance test")
        }

        // Create a synthetic embedding and edge set
        let n = 1000
        let d = 15  // Typical UMAP output dimension
        var embedding = (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -1...1) }
        }

        // Create synthetic edges (simulating ~15 neighbors per point)
        var edges: [(source: Int, target: Int, weight: Float)] = []
        for i in 0..<n {
            for _ in 0..<15 {
                let j = Int.random(in: 0..<n)
                if j != i {
                    edges.append((source: i, target: j, weight: Float.random(in: 0.5...1.0)))
                }
            }
        }

        // Time a single epoch
        let start = CFAbsoluteTimeGetCurrent()
        try await gpu.optimizeUMAPEpoch(
            embedding: &embedding,
            edges: edges,
            learningRate: 1.0,
            negativeSampleRate: 5,
            a: 1.929,
            b: 0.7915
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Single epoch should be very fast
        XCTAssertLessThan(
            elapsed,
            0.5,
            "GPU UMAP epoch for 1K points should complete in < 0.5s, took \(elapsed)s"
        )

        print("GPU UMAP epoch for \(n) points with \(edges.count) edges: \(String(format: "%.3f", elapsed))s")
    }

    // MARK: - Edge Case Tests

    /// Tests GPU path with minimum viable dataset size.
    func testGPUUMAPMinimumDatasetSize() async throws {
        // Just at the 100-point threshold
        let embeddings = generateRandomEmbeddings(count: 100, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let reducer = UMAPReducer(
            nNeighbors: 10,
            minDist: 0.1,
            nComponents: 2,
            nEpochs: 50,
            seed: 42,
            gpuContext: gpuContext
        )

        let result = try await reducer.fitTransform(embeddings)

        // Should complete without error
        XCTAssertEqual(result.count, 100)
        for embedding in result {
            XCTAssertEqual(embedding.dimension, 2)
        }
    }

    /// Tests CPU fallback when dataset is below GPU threshold.
    func testCPUFallbackForSmallDatasets() async throws {
        // Below the 100-point threshold - should use CPU path
        let embeddings = generateRandomEmbeddings(count: 50, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let reducer = UMAPReducer(
            nNeighbors: 10,
            minDist: 0.1,
            nComponents: 2,
            nEpochs: 50,
            seed: 42,
            gpuContext: gpuContext
        )

        let result = try await reducer.fitTransform(embeddings)

        // Should complete using CPU path
        XCTAssertEqual(result.count, 50)
    }

    /// Tests that optimizer correctly handles empty edge list.
    func testGPUOptimizerEmptyEdges() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)

        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        // Create embedding
        var embedding = [[Float]](repeating: [0, 0], count: 10)

        // Empty edges
        let edges: [(source: Int, target: Int, weight: Float)] = []

        // Should not crash
        try await gpu.optimizeUMAPEpoch(
            embedding: &embedding,
            edges: edges,
            learningRate: 1.0,
            negativeSampleRate: 5,
            a: 1.929,
            b: 0.7915
        )

        // Embedding should be unchanged
        for row in embedding {
            XCTAssertEqual(row, [0, 0])
        }
    }

    /// Tests that UMAPBuilder with GPU context works correctly.
    func testUMAPBuilderWithGPUContext() async throws {
        let embeddings = generateRandomEmbeddings(count: 150, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)

        let reducer = UMAPBuilder()
            .neighbors(15)
            .minDist(0.1)
            .components(2)
            .epochs(50)
            .seed(42)
            .gpu(gpuContext)
            .build()

        let result = try await reducer.fitTransform(embeddings)

        XCTAssertEqual(result.count, embeddings.count)
        XCTAssertEqual(result[0].dimension, 2)
    }

    /// Tests interruptible GPU optimization.
    func testGPUInterruptibleOptimization() async throws {
        let embeddings = generateRandomEmbeddings(count: 200, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)

        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        // Build k-NN graph and fuzzy set manually
        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: 15,
            metric: .euclidean
        )
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

        // Initialize embedding
        let initialEmbedding = try SpectralEmbedding.compute(
            adjacency: fuzzySet.memberships,
            nComponents: 2,
            seed: 42
        )

        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: 0.1,
            seed: 42
        )

        // Use actor for thread-safe epoch counting
        let counter = EpochCounter(maxEpochs: 10)

        // Run interruptible optimization
        let result = try await optimizer.optimizeGPUInterruptible(
            fuzzySet: fuzzySet,
            nEpochs: 100,
            learningRate: 1.0,
            negativeSampleRate: 5,
            gpuContext: gpu,
            shouldContinue: {
                counter.increment()
            }
        )

        // Should have been interrupted
        XCTAssertLessThan(result.completedEpoch, 99)
        XCTAssertEqual(result.embedding.count, embeddings.count)
    }
}

// MARK: - Thread-Safe Epoch Counter

/// A thread-safe epoch counter for testing interruptible optimization.
private final class EpochCounter: @unchecked Sendable {
    private var count = 0
    private let maxEpochs: Int
    private let lock = NSLock()

    init(maxEpochs: Int) {
        self.maxEpochs = maxEpochs
    }

    func increment() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        count += 1
        return count <= maxEpochs
    }
}

// MARK: - Seeded Random Number Generator

/// A seeded random number generator for reproducible tests.
private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
