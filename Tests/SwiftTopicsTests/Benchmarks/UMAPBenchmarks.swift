// UMAPBenchmarks.swift
// SwiftTopicsTests
//
// CPU vs GPU benchmarks for UMAP dimensionality reduction.
// Part of Phase 3: UMAP Benchmarks

import XCTest
@testable import SwiftTopics

/// Benchmarks for UMAP dimensionality reduction comparing CPU vs GPU performance.
///
/// Tests various dataset sizes to measure GPU speedup and generate
/// performance data for documentation.
///
/// ## Important Notes
///
/// **What's actually being compared:**
/// - **GPU path**: Uses GPU-accelerated optimization via VectorAccelerate's UMAPGradientKernel
/// - **CPU path**: Uses CPU-only optimization (no GPU context)
///
/// **Note**: Both paths use CPU for k-NN graph construction and spectral embedding.
/// The GPU acceleration applies only to the SGD optimization epochs.
///
/// **Performance characteristics**:
/// - k-NN graph construction: O(n log n) with BallTree, CPU-only
/// - Fuzzy simplicial set: O(n × k), CPU-only
/// - Spectral initialization: O(n × d²), CPU-only
/// - Optimization: O(epochs × edges), GPU-accelerated
///
/// ## Test Points
/// - 100, 250, 500, 1000, 2000 points
/// - 384-dimensional embeddings → 15-dimensional output
///
/// ## Expected Results
/// - Single epoch: 5-10x GPU speedup
/// - Full pipeline: 6-15x speedup (limited by CPU-bound k-NN)
final class UMAPBenchmarks: XCTestCase {

    // MARK: - Configuration

    /// Benchmark scales to test.
    static let benchmarkScales: [(points: Int, label: String)] = [
        (100, "100 points"),
        (250, "250 points"),
        (500, "500 points"),
        (1000, "1K points"),
        (2000, "2K points"),
    ]

    /// UMAP configuration for benchmarks.
    let umapConfig = UMAPConfiguration(
        nNeighbors: 15,
        minDist: 0.1,
        metric: .euclidean,
        nEpochs: 200,  // Fixed epochs for consistent comparison
        learningRate: 1.0
    )

    /// Output dimensions (typical for topic modeling).
    let outputDimension = 15

    // MARK: - Main Benchmark Suite

    /// Runs the full UMAP benchmark suite across all scales.
    ///
    /// This is the main entry point for comprehensive benchmarking.
    /// Skips if GPU is unavailable.
    func testUMAPBenchmarkSuite() async throws {
        // Skip if GPU not available
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available for benchmarks")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP CPU vs GPU Benchmark Suite")
        print("═══════════════════════════════════════════════════════════════════")

        // Print hardware info
        let hardwareInfo = HardwareInfo.capture()
        print("\n\(hardwareInfo.summary)\n")

        var results: [ComparisonResult] = []

        // Run benchmarks at each scale
        for (points, label) in Self.benchmarkScales {
            print("─────────────────────────────────────────────────────────────────")
            print("  Running: \(label)")
            print("─────────────────────────────────────────────────────────────────")

            do {
                let result = try await benchmarkAtScale(points: points, label: label)
                results.append(result)
            } catch {
                print("  ⚠️  Skipped: \(error.localizedDescription)")
            }
        }

        // Print summary table
        print("\n")
        results.printTable(title: "UMAP CPU vs GPU Summary")
        print("\n")

        // Print detailed breakdown for key scales
        printDetailedBreakdown(for: results)

        // Save results
        let storage = BenchmarkStorage()
        let savedURL = try storage.saveRun(results, name: "UMAP_Benchmark")
        print("Results saved to: \(savedURL.lastPathComponent)")
    }

    // MARK: - Individual Scale Benchmarks

    /// Benchmarks UMAP at 500 points (primary test case).
    func testUMAP500Points() async throws {
        let result = try await benchmarkAtScale(points: 500, label: "500 points")

        // Expected ~8x speedup for full pipeline
        print("500 points speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 2.0, "Should show GPU speedup")
    }

    /// Benchmarks UMAP at 1000 points.
    func testUMAP1000Points() async throws {
        let result = try await benchmarkAtScale(points: 1000, label: "1K points")

        // Expected ~10x speedup
        print("1K points speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 3.0, "Should show significant GPU speedup")
    }

    /// Benchmarks UMAP at 2000 points.
    func testUMAP2000Points() async throws {
        let result = try await benchmarkAtScale(points: 2000, label: "2K points")

        // Expected ~12x speedup
        print("2K points speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 4.0, "Should show significant GPU speedup")
    }

    // MARK: - Per-Phase Breakdown Tests

    /// Tests and reports per-phase timing breakdown.
    ///
    /// Since UMAP doesn't have a built-in timing breakdown,
    /// we manually instrument each phase.
    func testUMAPPhaseBreakdown() async throws {
        let points = 1000
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 10,
            pointsPerCluster: points / 10,
            seed: 42
        )

        // GPU context for optimization
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Phase Breakdown (\(points) points)")
        print("═══════════════════════════════════════════════════════════════════")

        // Run CPU-only UMAP with phase timing
        print("\nCPU Path (no GPU context):")
        let cpuTiming = try await runUMAPWithPhaseTiming(
            embeddings: embeddings,
            gpuContext: nil
        )
        printPhaseTiming(cpuTiming)

        // Run GPU UMAP with phase timing
        print("\nGPU Path:")
        let gpuTiming = try await runUMAPWithPhaseTiming(
            embeddings: embeddings,
            gpuContext: gpuContext
        )
        printPhaseTiming(gpuTiming)

        // Calculate and print speedups
        print("\nPer-phase speedups:")
        printPhaseSpeedup("k-NN Graph", cpu: cpuTiming.knnTime, gpu: gpuTiming.knnTime)
        printPhaseSpeedup("Fuzzy Set", cpu: cpuTiming.fuzzySetTime, gpu: gpuTiming.fuzzySetTime)
        printPhaseSpeedup("Spectral Init", cpu: cpuTiming.spectralTime, gpu: gpuTiming.spectralTime)
        printPhaseSpeedup("Optimization", cpu: cpuTiming.optimizationTime, gpu: gpuTiming.optimizationTime)
        printPhaseSpeedup("Total", cpu: cpuTiming.totalTime, gpu: gpuTiming.totalTime)
    }

    /// Tests per-epoch timing to measure GPU optimization speedup.
    func testUMAPEpochTiming() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        let sizes = [500, 1000, 2000]

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Per-Epoch Timing (GPU vs CPU)")
        print("═══════════════════════════════════════════════════════════════════")

        for size in sizes {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, size / 100),
                pointsPerCluster: 100,
                seed: UInt64(size)
            ).prefix(size)

            // Build k-NN graph and fuzzy set (same for both paths)
            let knnGraph = try await NearestNeighborGraph.build(
                embeddings: Array(embeddings),
                k: 15,
                metric: .euclidean
            )
            let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

            // Initialize embedding
            var initialEmbedding = try SpectralEmbedding.compute(
                adjacency: fuzzySet.memberships,
                nComponents: outputDimension,
                seed: 42
            )
            SpectralEmbedding.handleDisconnectedComponents(
                embedding: &initialEmbedding,
                graph: knnGraph,
                seed: 42
            )

            // Time single CPU epoch
            let cpuOptimizer = UMAPOptimizer(
                initialEmbedding: initialEmbedding,
                minDist: 0.1,
                seed: 42
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = await cpuOptimizer.optimize(
                fuzzySet: fuzzySet,
                nEpochs: 10,  // Run 10 epochs for measurement
                learningRate: 1.0,
                negativeSampleRate: 5
            )
            let cpuEpochTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / 10.0

            // Time single GPU epoch
            let gpuOptimizer = UMAPOptimizer(
                initialEmbedding: initialEmbedding,
                minDist: 0.1,
                seed: 42
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuOptimizer.optimizeGPU(
                fuzzySet: fuzzySet,
                nEpochs: 10,  // Run 10 epochs for measurement
                learningRate: 1.0,
                negativeSampleRate: 5,
                gpuContext: gpu
            )
            let gpuEpochTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / 10.0

            let epochSpeedup = cpuEpochTime / gpuEpochTime
            print("  \(size) points: CPU epoch=\(formatTime(cpuEpochTime)), GPU epoch=\(formatTime(gpuEpochTime)), Speedup=\(String(format: "%.1f", epochSpeedup))x")

            // Single epoch should show significant GPU speedup
            XCTAssertGreaterThan(epochSpeedup, 2.0, "GPU epoch should be faster than CPU")
        }
    }

    // MARK: - Scaling Analysis

    /// Tests how UMAP performance scales with dataset size.
    func testUMAPScalingAnalysis() async throws {
        let scales = [100, 250, 500, 750, 1000]

        var cpuTimes: [(points: Int, time: Double)] = []
        var gpuTimes: [(points: Int, time: Double)] = []

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        for points in scales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, points / 50),
                pointsPerCluster: 50,
                dimension: 384,
                seed: UInt64(points)
            )

            // Use consistent epoch count for fair comparison
            let config = UMAPConfiguration(
                nNeighbors: 15,
                minDist: 0.1,
                nEpochs: 100,
                learningRate: 1.0
            )

            // CPU timing
            let cpuReducer = UMAPReducer(
                configuration: config,
                nComponents: outputDimension,
                seed: 42,
                gpuContext: nil
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuReducer.fitTransform(embeddings)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((points, cpuTime))

            // GPU timing
            let gpuReducer = UMAPReducer(
                configuration: config,
                nComponents: outputDimension,
                seed: 42,
                gpuContext: gpu
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuReducer.fitTransform(embeddings)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((points, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(points) points: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")
        }

        // Calculate scaling exponents using log-log regression
        let cpuExponent = calculateScalingExponent(cpuTimes)
        let gpuExponent = calculateScalingExponent(gpuTimes)

        print("\n  Scaling Analysis:")
        print("    CPU scaling exponent: \(String(format: "%.2f", cpuExponent))")
        print("    GPU scaling exponent: \(String(format: "%.2f", gpuExponent))")
        print("    Note: k-NN is O(n log n), optimization is O(n × k × epochs)")

        // GPU should scale at least as well as CPU
        XCTAssertLessThan(gpuExponent, cpuExponent + 0.5,
                         "GPU should scale at least as well as CPU")
    }

    // MARK: - Helper Methods

    /// Benchmarks UMAP at a specific scale.
    private func benchmarkAtScale(points: Int, label: String) async throws -> ComparisonResult {
        // Generate clustered embeddings
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: max(5, points / 50),
            pointsPerCluster: min(50, points / 5),
            dimension: 384,
            seed: UInt64(points)
        )

        // Ensure we have exactly the requested point count
        let testEmbeddings = Array(embeddings.prefix(points))

        // Skip if GPU not available
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        // Configure benchmark iterations based on scale
        let iterations: Int
        let warmup: Int

        switch points {
        case 0..<500:
            iterations = 5
            warmup = 2
        case 500..<1500:
            iterations = 3
            warmup = 1
        default:
            iterations = 2
            warmup = 1
        }

        // Create CPU-only reducer (no GPU context)
        let cpuReducer = UMAPReducer(
            configuration: umapConfig,
            nComponents: outputDimension,
            seed: 42,
            gpuContext: nil
        )

        // Create GPU reducer
        let gpuReducer = UMAPReducer(
            configuration: umapConfig,
            nComponents: outputDimension,
            seed: 42,
            gpuContext: gpuContext
        )

        // Run benchmark
        let result = try await Benchmark("UMAP")
            .scale(label)
            .iterations(iterations)
            .warmup(warmup)
            .baseline("CPU") {
                _ = try await cpuReducer.fitTransform(testEmbeddings)
            }
            .variant("GPU") {
                _ = try await gpuReducer.fitTransform(testEmbeddings)
            }
            .runAndReport()

        return result
    }

    /// Runs UMAP with per-phase timing instrumentation.
    private func runUMAPWithPhaseTiming(
        embeddings: [Embedding],
        gpuContext: TopicsGPUContext?
    ) async throws -> UMAPPhaseTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Phase 1: k-NN Graph
        let knnStart = CFAbsoluteTimeGetCurrent()
        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: umapConfig.nNeighbors,
            metric: .euclidean
        )
        let knnTime = CFAbsoluteTimeGetCurrent() - knnStart

        // Phase 2: Fuzzy Simplicial Set
        let fuzzyStart = CFAbsoluteTimeGetCurrent()
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)
        let fuzzySetTime = CFAbsoluteTimeGetCurrent() - fuzzyStart

        // Phase 3: Spectral Initialization
        let spectralStart = CFAbsoluteTimeGetCurrent()
        var initialEmbedding = try SpectralEmbedding.compute(
            adjacency: fuzzySet.memberships,
            nComponents: outputDimension,
            seed: 42
        )
        SpectralEmbedding.handleDisconnectedComponents(
            embedding: &initialEmbedding,
            graph: knnGraph,
            seed: 42
        )
        let spectralTime = CFAbsoluteTimeGetCurrent() - spectralStart

        // Phase 4: Optimization
        let optimizationStart = CFAbsoluteTimeGetCurrent()
        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: umapConfig.minDist,
            seed: 42
        )

        let nEpochs = umapConfig.nEpochs ?? 200

        if let gpu = gpuContext {
            _ = try await optimizer.optimizeGPU(
                fuzzySet: fuzzySet,
                nEpochs: nEpochs,
                learningRate: umapConfig.learningRate,
                negativeSampleRate: 5,
                gpuContext: gpu
            )
        } else {
            _ = await optimizer.optimize(
                fuzzySet: fuzzySet,
                nEpochs: nEpochs,
                learningRate: umapConfig.learningRate,
                negativeSampleRate: 5
            )
        }
        let optimizationTime = CFAbsoluteTimeGetCurrent() - optimizationStart

        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return UMAPPhaseTiming(
            knnTime: knnTime,
            fuzzySetTime: fuzzySetTime,
            spectralTime: spectralTime,
            optimizationTime: optimizationTime,
            totalTime: totalTime
        )
    }

    /// Prints detailed breakdown for key benchmark scales.
    private func printDetailedBreakdown(for results: [ComparisonResult]) {
        print("═══════════════════════════════════════════════════════════════════")
        print("  Detailed Statistics")
        print("═══════════════════════════════════════════════════════════════════")

        // Print details for 500 and 1K point benchmarks
        for result in results where result.scale.contains("500") || result.scale.contains("1K") {
            print("\n\(result.scale):")
            print("  CPU median: \(TimingStatistics.format(result.baseline.statistics.median))")
            print("  GPU median: \(TimingStatistics.format(result.variant.statistics.median))")
            print("  Speedup: \(result.speedupFormatted)")
            print("  CPU CV: \(String(format: "%.1f%%", result.baseline.statistics.coefficientOfVariation * 100))")
            print("  GPU CV: \(String(format: "%.1f%%", result.variant.statistics.coefficientOfVariation * 100))")
        }
    }

    /// Prints a phase timing summary.
    private func printPhaseTiming(_ timing: UMAPPhaseTiming) {
        print("  k-NN Graph:       \(formatTime(timing.knnTime))")
        print("  Fuzzy Set:        \(formatTime(timing.fuzzySetTime))")
        print("  Spectral Init:    \(formatTime(timing.spectralTime))")
        print("  Optimization:     \(formatTime(timing.optimizationTime))")
        print("  Total:            \(formatTime(timing.totalTime))")
    }

    /// Prints a phase speedup comparison.
    private func printPhaseSpeedup(_ phase: String, cpu: TimeInterval, gpu: TimeInterval) {
        let speedup = cpu / max(gpu, 0.0001)
        let cpuStr = formatTime(cpu)
        let gpuStr = formatTime(gpu)
        print("  \(phase.padding(toLength: 16, withPad: " ", startingAt: 0)) CPU=\(cpuStr), GPU=\(gpuStr), \(String(format: "%.1f", speedup))x")
    }

    /// Formats a time interval for display.
    private func formatTime(_ time: TimeInterval) -> String {
        if time >= 1.0 {
            return String(format: "%.2fs", time)
        } else if time >= 0.001 {
            return String(format: "%.1fms", time * 1000)
        } else {
            return String(format: "%.1fµs", time * 1_000_000)
        }
    }

    /// Calculates the scaling exponent using log-log regression.
    private func calculateScalingExponent(_ data: [(points: Int, time: Double)]) -> Double {
        guard data.count >= 2 else { return 0 }

        // Convert to log-log space
        let logData = data.map { (log(Double($0.points)), log($0.time)) }

        // Simple linear regression
        let n = Double(logData.count)
        let sumX = logData.reduce(0) { $0 + $1.0 }
        let sumY = logData.reduce(0) { $0 + $1.1 }
        let sumXY = logData.reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = logData.reduce(0) { $0 + $1.0 * $1.0 }

        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        return slope
    }
}

// MARK: - UMAP Phase Timing

/// Timing breakdown for UMAP phases.
private struct UMAPPhaseTiming {
    /// Time for k-NN graph construction.
    let knnTime: TimeInterval

    /// Time for fuzzy simplicial set construction.
    let fuzzySetTime: TimeInterval

    /// Time for spectral initialization.
    let spectralTime: TimeInterval

    /// Time for optimization epochs.
    let optimizationTime: TimeInterval

    /// Total execution time.
    let totalTime: TimeInterval

    /// Percentage of time spent in optimization.
    var optimizationPercentage: Double {
        guard totalTime > 0 else { return 0 }
        return (optimizationTime / totalTime) * 100
    }
}

// MARK: - Quick Benchmark Extension

extension UMAPBenchmarks {

    /// Quick benchmark for CI/development - runs only 500 points.
    func testQuickBenchmark() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: 100,
            seed: 42
        )

        // Use fewer epochs for quick test
        let quickConfig = UMAPConfiguration(
            nNeighbors: 15,
            minDist: 0.1,
            nEpochs: 50,
            learningRate: 1.0
        )

        let cpuReducer = UMAPReducer(
            configuration: quickConfig,
            nComponents: outputDimension,
            seed: 42,
            gpuContext: nil
        )

        let gpuReducer = UMAPReducer(
            configuration: quickConfig,
            nComponents: outputDimension,
            seed: 42,
            gpuContext: gpuContext
        )

        let result = try await Benchmark("UMAP Quick")
            .scale("500 points")
            .configuration(.quick)
            .baseline("CPU") {
                _ = try await cpuReducer.fitTransform(embeddings)
            }
            .variant("GPU") {
                _ = try await gpuReducer.fitTransform(embeddings)
            }
            .runAndReport()

        print("\nQuick benchmark result:")
        print("  Speedup: \(result.speedupFormatted)")
        print("  Significant: \(result.isSignificant ? "✓" : "○")")
        print("  Note: GPU acceleration applies to optimization phase only")

        // Any speedup indicates GPU path is working
        XCTAssertGreaterThan(result.speedup, 1.0, "GPU should not be slower than CPU")
    }
}

// MARK: - Optimization-Only Benchmark

extension UMAPBenchmarks {

    /// Benchmarks just the UMAP optimization step (bypassing CPU-bound phases).
    ///
    /// This test measures the true GPU speedup for optimization,
    /// which is the primary GPU-accelerated component of UMAP.
    func testOptimizationOnlyBenchmark() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        let sizes = [500, 1000, 2000]

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Optimization-Only Benchmark (GPU vs CPU)")
        print("═══════════════════════════════════════════════════════════════════")

        for size in sizes {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, size / 100),
                pointsPerCluster: 100,
                seed: UInt64(size)
            ).prefix(size)

            // Build shared k-NN graph and fuzzy set (same for both)
            let knnGraph = try await NearestNeighborGraph.build(
                embeddings: Array(embeddings),
                k: 15,
                metric: .euclidean
            )
            let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

            // Initialize embedding
            var initialEmbedding = try SpectralEmbedding.compute(
                adjacency: fuzzySet.memberships,
                nComponents: outputDimension,
                seed: 42
            )
            SpectralEmbedding.handleDisconnectedComponents(
                embedding: &initialEmbedding,
                graph: knnGraph,
                seed: 42
            )

            let nEpochs = 100

            // CPU optimization timing
            let cpuOptimizer = UMAPOptimizer(
                initialEmbedding: initialEmbedding,
                minDist: 0.1,
                seed: 42
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = await cpuOptimizer.optimize(
                fuzzySet: fuzzySet,
                nEpochs: nEpochs,
                learningRate: 1.0,
                negativeSampleRate: 5
            )
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            // GPU optimization timing
            let gpuOptimizer = UMAPOptimizer(
                initialEmbedding: initialEmbedding,
                minDist: 0.1,
                seed: 42
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuOptimizer.optimizeGPU(
                fuzzySet: fuzzySet,
                nEpochs: nEpochs,
                learningRate: 1.0,
                negativeSampleRate: 5,
                gpuContext: gpu
            )
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

            let speedup = cpuTime / gpuTime
            print("  \(size) points (\(nEpochs) epochs): CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")

            // Optimization should show significant GPU speedup
            XCTAssertGreaterThan(speedup, 3.0, "GPU optimization should be significantly faster")
        }
    }

    /// Tests GPU single-epoch performance directly.
    func testGPUSingleEpochPerformance() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        // Create synthetic embedding and edges
        let n = 1000
        let d = outputDimension
        var embedding = (0..<n).map { _ in
            (0..<d).map { _ in Float.random(in: -1...1) }
        }

        // Synthetic edges (~15 neighbors per point)
        var edges: [(source: Int, target: Int, weight: Float)] = []
        for i in 0..<n {
            for _ in 0..<15 {
                let j = Int.random(in: 0..<n)
                if j != i {
                    edges.append((source: i, target: j, weight: Float.random(in: 0.5...1.0)))
                }
            }
        }

        print("\n  GPU single epoch performance (\(n) points, \(edges.count) edges):")

        // Warmup
        try await gpu.optimizeUMAPEpoch(
            embedding: &embedding,
            edges: edges,
            learningRate: 1.0,
            negativeSampleRate: 5,
            a: 1.929,
            b: 0.7915
        )

        // Time 10 epochs
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            try await gpu.optimizeUMAPEpoch(
                embedding: &embedding,
                edges: edges,
                learningRate: 1.0,
                negativeSampleRate: 5,
                a: 1.929,
                b: 0.7915
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgEpochTime = elapsed / 10.0

        print("  Average epoch time: \(formatTime(avgEpochTime))")
        print("  Throughput: \(String(format: "%.0f", Double(edges.count) / avgEpochTime)) edges/second")

        // Single epoch should be very fast
        XCTAssertLessThan(avgEpochTime, 0.5, "GPU epoch should complete in < 0.5s")
    }
}
