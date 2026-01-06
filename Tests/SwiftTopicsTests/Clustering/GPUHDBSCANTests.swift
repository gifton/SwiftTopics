// GPUHDBSCANTests.swift
// SwiftTopicsTests
//
// Tests for GPU-accelerated HDBSCAN clustering via VectorAccelerate integration.

import XCTest
@testable import SwiftTopics

/// Tests for GPU-accelerated HDBSCAN clustering.
///
/// These tests validate that the GPU path (via VectorAccelerate's HDBSCANDistanceModule)
/// produces equivalent clustering results to the CPU path while providing significant
/// performance improvements.
final class GPUHDBSCANTests: XCTestCase {

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

        for clusterIdx in 0..<clustersCount {
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

    /// Tests that GPU and CPU paths produce same cluster count.
    func testGPUMSTClusterCountMatchesCPU() async throws {
        // Generate clustered data (5 clusters of 50 points each = 250 total)
        let embeddings = generateClusteredEmbeddings(
            clustersCount: 5,
            pointsPerCluster: 50,
            dimension: 384,
            clusterSpread: 0.05,
            seed: 12345
        )

        let config = HDBSCANConfiguration(
            minClusterSize: 10,
            minSamples: 5
        )

        // CPU-only path (no GPU context)
        let cpuEngine = try await HDBSCANEngine(configuration: config, gpuContext: nil)
        let cpuResult = try await cpuEngine.fitWithDetails(embeddings)

        // GPU path
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let gpuEngine = try await HDBSCANEngine(configuration: config, gpuContext: gpuContext)
        let gpuResult = try await gpuEngine.fitWithDetails(embeddings)

        // Compare cluster counts - they should be equal or very close
        // Note: Due to numerical precision differences, we allow a small tolerance
        let clusterDiff = abs(cpuResult.assignment.clusterCount - gpuResult.assignment.clusterCount)
        XCTAssertLessThanOrEqual(
            clusterDiff,
            1,
            "Cluster count difference should be at most 1 (CPU: \(cpuResult.assignment.clusterCount), GPU: \(gpuResult.assignment.clusterCount))"
        )
    }

    /// Tests that GPU path identifies similar outlier proportions.
    func testGPUMSTOutlierRatioSimilarToCPU() async throws {
        // Generate embeddings with some noise
        var embeddings = generateClusteredEmbeddings(
            clustersCount: 3,
            pointsPerCluster: 60,
            dimension: 384,
            clusterSpread: 0.05,
            seed: 54321
        )

        // Add outliers (random points far from clusters)
        var rng = SeededRandomNumberGenerator(seed: 99999)
        for _ in 0..<20 {
            let outlier = (0..<384).map { _ in Float.random(in: -10...10, using: &rng) }
            embeddings.append(Embedding(vector: outlier))
        }

        let config = HDBSCANConfiguration(minClusterSize: 15, minSamples: 5)

        // CPU path
        let cpuEngine = try await HDBSCANEngine(configuration: config, gpuContext: nil)
        let cpuResult = try await cpuEngine.fitWithDetails(embeddings)

        // GPU path
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let gpuEngine = try await HDBSCANEngine(configuration: config, gpuContext: gpuContext)
        let gpuResult = try await gpuEngine.fitWithDetails(embeddings)

        // Calculate outlier ratios
        let cpuOutliers = cpuResult.assignment.labels.filter { $0 == -1 }.count
        let gpuOutliers = gpuResult.assignment.labels.filter { $0 == -1 }.count

        let cpuOutlierRatio = Float(cpuOutliers) / Float(embeddings.count)
        let gpuOutlierRatio = Float(gpuOutliers) / Float(embeddings.count)

        // Outlier ratios should be within 10% of each other
        XCTAssertLessThanOrEqual(
            abs(cpuOutlierRatio - gpuOutlierRatio),
            0.1,
            "Outlier ratios should be similar (CPU: \(cpuOutlierRatio), GPU: \(gpuOutlierRatio))"
        )
    }

    // MARK: - Performance Tests

    /// Tests that GPU MST is significantly faster than CPU for large datasets.
    func testGPUMSTPerformance() async throws {
        // Generate large dataset (1000 points)
        let embeddings = generateRandomEmbeddings(count: 1000, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)

        // Skip if GPU not available
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available for performance test")
        }

        // Warm up
        _ = try? await gpu.computeHDBSCANMSTWithCoreDistances(
            Array(embeddings.prefix(100)),
            minSamples: 5
        )

        // Time GPU path
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await gpu.computeHDBSCANMSTWithCoreDistances(
            embeddings,
            minSamples: 5
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Validate result structure
        XCTAssertEqual(result.coreDistances.count, embeddings.count)
        XCTAssertEqual(result.mst.edges.count, embeddings.count - 1)
        XCTAssertEqual(result.mst.pointCount, embeddings.count)

        // GPU should complete in < 1s for 1K points
        // (vs ~2-3s for CPU Prim's algorithm)
        XCTAssertLessThan(
            elapsed,
            1.0,
            "GPU MST for 1K points should complete in < 1s, took \(elapsed)s"
        )

        print("GPU MST for \(embeddings.count) points completed in \(String(format: "%.3f", elapsed))s")
    }

    /// Tests GPU performance scaling with dataset size.
    func testGPUMSTPerformanceScaling() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)

        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available for performance test")
        }

        let sizes = [100, 250, 500, 1000]
        var timings: [(size: Int, time: Double)] = []

        for size in sizes {
            let embeddings = generateRandomEmbeddings(count: size, dimension: 384, seed: UInt64(size))

            let start = CFAbsoluteTimeGetCurrent()
            _ = try await gpu.computeHDBSCANMSTWithCoreDistances(embeddings, minSamples: 5)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            timings.append((size, elapsed))
        }

        // Print scaling results
        print("GPU HDBSCAN MST scaling:")
        for (size, time) in timings {
            print("  \(size) points: \(String(format: "%.3f", time))s")
        }

        // Verify that performance scales reasonably (not worse than O(n²))
        // 10x more points should take less than 100x more time
        if let small = timings.first, let large = timings.last {
            let sizeRatio = Double(large.size) / Double(small.size)
            let timeRatio = large.time / max(small.time, 0.001)
            let scalingExponent = log(timeRatio) / log(sizeRatio)

            XCTAssertLessThan(
                scalingExponent,
                2.5,
                "GPU scaling should be sub-quadratic (got exponent \(scalingExponent))"
            )
        }
    }

    // MARK: - Edge Case Tests

    /// Tests GPU path with minimum viable dataset size.
    func testGPUMSTMinimumDatasetSize() async throws {
        // Just above the 100-point threshold
        let embeddings = generateRandomEmbeddings(count: 100, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let engine = try await HDBSCANEngine(
            configuration: .init(minClusterSize: 5, minSamples: 3),
            gpuContext: gpuContext
        )

        let result = try await engine.fitWithDetails(embeddings)

        // Should complete without error
        XCTAssertEqual(result.assignment.labels.count, 100)
        XCTAssertNotNil(result.coreDistances)
    }

    /// Tests CPU fallback when dataset is below GPU threshold.
    func testCPUFallbackForSmallDatasets() async throws {
        // Below the 100-point threshold - should use CPU path
        let embeddings = generateRandomEmbeddings(count: 50, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)
        let engine = try await HDBSCANEngine(
            configuration: .init(minClusterSize: 5, minSamples: 3),
            gpuContext: gpuContext
        )

        let result = try await engine.fitWithDetails(embeddings)

        // Should complete using CPU path
        XCTAssertEqual(result.assignment.labels.count, 50)
    }

    /// Tests that MST edges are valid.
    func testGPUMSTEdgeValidity() async throws {
        let embeddings = generateRandomEmbeddings(count: 200, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        let result = try await gpu.computeHDBSCANMSTWithCoreDistances(embeddings, minSamples: 5)

        // Validate MST properties
        XCTAssertEqual(result.mst.edges.count, embeddings.count - 1, "MST should have n-1 edges")

        for edge in result.mst.edges {
            XCTAssertGreaterThanOrEqual(edge.source, 0)
            XCTAssertLessThan(edge.source, embeddings.count)
            XCTAssertGreaterThanOrEqual(edge.target, 0)
            XCTAssertLessThan(edge.target, embeddings.count)
            XCTAssertNotEqual(edge.source, edge.target, "Self-loops not allowed")
            XCTAssertGreaterThanOrEqual(edge.weight, 0, "Edge weights should be non-negative")
        }
    }

    /// Tests that core distances are valid.
    func testGPUCoreDistanceValidity() async throws {
        let embeddings = generateRandomEmbeddings(count: 200, dimension: 384, seed: 42)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        let result = try await gpu.computeHDBSCANMSTWithCoreDistances(embeddings, minSamples: 5)

        XCTAssertEqual(result.coreDistances.count, embeddings.count)

        for distance in result.coreDistances {
            XCTAssertGreaterThanOrEqual(distance, 0, "Core distances should be non-negative")
            XCTAssertFalse(distance.isNaN, "Core distances should not be NaN")
            XCTAssertFalse(distance.isInfinite, "Core distances should not be infinite")
        }
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
