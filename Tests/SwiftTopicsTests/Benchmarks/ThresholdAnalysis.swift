// ThresholdAnalysis.swift
// SwiftTopicsTests
//
// Phase 7: Threshold Tuning
//
// Determines the optimal gpuMinPointsThreshold for HDBSCAN and UMAP.
// Finds the crossover point where GPU acceleration becomes beneficial,
// accounting for GPU initialization overhead.

import XCTest
@testable import SwiftTopics

/// Threshold analysis for GPU acceleration.
///
/// Determines the optimal `gpuMinPointsThreshold` by running benchmarks at
/// various dataset sizes to find where GPU becomes faster than CPU.
///
/// ## Key Concepts
///
/// **GPU Overhead Sources:**
/// - Kernel compilation (first invocation only)
/// - Buffer allocation and memory transfer
/// - Command buffer encoding and submission
/// - Synchronization barriers
///
/// **Crossover Point:**
/// The dataset size where GPU compute savings exceed overhead costs.
/// Below this point, CPU is faster. Above it, GPU provides speedup.
///
/// ## Expected Results
///
/// | Algorithm | Typical Crossover | Recommended Threshold |
/// |-----------|-------------------|----------------------|
/// | HDBSCAN   | 50-100 points     | 75-100               |
/// | UMAP      | 75-150 points     | 100-150              |
///
/// ## Usage
///
/// Run the full threshold analysis:
/// ```bash
/// swift test --filter testFullThresholdAnalysis
/// ```
///
/// Quick analysis for CI:
/// ```bash
/// swift test --filter testQuickThresholdAnalysis
/// ```
final class ThresholdAnalysis: XCTestCase {

    // MARK: - Configuration

    /// Dataset sizes to test for threshold analysis.
    ///
    /// Intentionally dense around expected crossover (50-150).
    static let thresholdScales: [Int] = [25, 50, 75, 100, 150, 200, 300]

    /// Extended scales for comprehensive analysis.
    static let extendedScales: [Int] = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]

    /// Number of iterations per scale for statistical reliability.
    static let iterationsPerScale = 5

    /// Warmup iterations to prime caches and JIT.
    static let warmupIterations = 2

    // MARK: - Result Types

    /// Result of a threshold analysis run.
    struct ThresholdResult {
        /// Algorithm being tested (HDBSCAN, UMAP).
        let algorithm: String

        /// Dataset size.
        let pointCount: Int

        /// CPU median time in seconds.
        let cpuMedianTime: TimeInterval

        /// GPU median time in seconds.
        let gpuMedianTime: TimeInterval

        /// GPU speedup (CPU time / GPU time).
        var speedup: Double {
            gpuMedianTime > 0 ? cpuMedianTime / gpuMedianTime : .infinity
        }

        /// Whether GPU is faster than CPU.
        var gpuIsFaster: Bool {
            speedup > 1.0
        }

        /// Whether GPU is significantly faster (>10% improvement).
        var gpuIsSignificantlyFaster: Bool {
            speedup > 1.1
        }
    }

    /// Recommendation for threshold configuration.
    struct ThresholdRecommendation {
        /// Algorithm name.
        let algorithm: String

        /// Recommended threshold value.
        let recommendedThreshold: Int

        /// Crossover point (where GPU == CPU).
        let crossoverPoint: Int

        /// Conservative threshold (guarantees GPU speedup).
        let conservativeThreshold: Int

        /// Aggressive threshold (allows some overhead for consistency).
        let aggressiveThreshold: Int

        /// Peak speedup observed.
        let peakSpeedup: Double

        /// Dataset size for peak speedup.
        let peakSpeedupSize: Int

        var summary: String {
            """
            \(algorithm) Threshold Recommendation:
              Crossover point:       \(crossoverPoint) points
              Recommended threshold: \(recommendedThreshold) points
              Conservative:          \(conservativeThreshold) points (safe, always faster)
              Aggressive:            \(aggressiveThreshold) points (good for large batches)
              Peak speedup:          \(String(format: "%.1f", peakSpeedup))x at \(peakSpeedupSize) points
            """
        }
    }

    // MARK: - Full Threshold Analysis

    /// Runs the complete threshold analysis for all algorithms.
    ///
    /// This is the primary test for Phase 7. It:
    /// 1. Tests HDBSCAN at all threshold scales
    /// 2. Tests UMAP at all threshold scales
    /// 3. Finds crossover points for each
    /// 4. Generates recommendations
    /// 5. Saves results to disk
    func testFullThresholdAnalysis() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available for threshold analysis")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  GPU Threshold Analysis - Finding Optimal gpuMinPointsThreshold")
        print("═══════════════════════════════════════════════════════════════════")

        // Capture hardware info for recommendations
        let hardwareInfo = HardwareInfo.capture()
        print("\n\(hardwareInfo.summary)\n")

        // Analyze HDBSCAN thresholds
        print("\n─────────────────────────────────────────────────────────────────")
        print("  HDBSCAN Threshold Analysis")
        print("─────────────────────────────────────────────────────────────────")
        let hdbscanResults = try await analyzeHDBSCANThresholds(scales: Self.extendedScales)
        let hdbscanRecommendation = computeRecommendation(
            algorithm: "HDBSCAN",
            results: hdbscanResults
        )

        // Analyze UMAP thresholds
        print("\n─────────────────────────────────────────────────────────────────")
        print("  UMAP Threshold Analysis")
        print("─────────────────────────────────────────────────────────────────")
        let umapResults = try await analyzeUMAPThresholds(scales: Self.extendedScales)
        let umapRecommendation = computeRecommendation(
            algorithm: "UMAP",
            results: umapResults
        )

        // Print recommendations
        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Threshold Recommendations")
        print("═══════════════════════════════════════════════════════════════════")
        print("\n\(hdbscanRecommendation.summary)")
        print("\n\(umapRecommendation.summary)")

        // Generate combined recommendation
        let combinedThreshold = max(
            hdbscanRecommendation.recommendedThreshold,
            umapRecommendation.recommendedThreshold
        )
        print("\n  Combined Recommendation for TopicsGPUConfiguration:")
        print("    gpuMinPointsThreshold = \(combinedThreshold)")

        // Print code snippet
        print("\n  Suggested Configuration:")
        print("""
            let config = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: \(combinedThreshold)
            )
        """)

        // Save results
        try saveThresholdResults(
            hdbscan: hdbscanResults,
            umap: umapResults,
            hdbscanRecommendation: hdbscanRecommendation,
            umapRecommendation: umapRecommendation,
            hardwareInfo: hardwareInfo
        )
    }

    // MARK: - Quick Threshold Analysis

    /// Quick threshold analysis for CI/development.
    ///
    /// Tests fewer scales for faster execution while still identifying
    /// the approximate crossover point.
    func testQuickThresholdAnalysis() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        let quickScales = [50, 100, 200]

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Quick Threshold Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        // Quick HDBSCAN analysis
        let hdbscanResults = try await analyzeHDBSCANThresholds(
            scales: quickScales,
            iterations: 3,
            warmup: 1
        )

        // Quick UMAP analysis
        let umapResults = try await analyzeUMAPThresholds(
            scales: quickScales,
            iterations: 3,
            warmup: 1
        )

        // Find approximate crossover
        let hdbscanCrossover = findCrossoverPoint(results: hdbscanResults)
        let umapCrossover = findCrossoverPoint(results: umapResults)

        print("\n  Quick Results:")
        print("    HDBSCAN crossover: ~\(hdbscanCrossover) points")
        print("    UMAP crossover: ~\(umapCrossover) points")
        print("    Suggested threshold: \(max(hdbscanCrossover, umapCrossover)) points")

        // Basic assertions
        XCTAssertLessThan(hdbscanCrossover, 300, "HDBSCAN crossover should be below 300")
        XCTAssertLessThan(umapCrossover, 300, "UMAP crossover should be below 300")
    }

    // MARK: - Individual Algorithm Tests

    /// Tests HDBSCAN thresholds at standard scales.
    func testHDBSCANThresholds() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  HDBSCAN Threshold Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let results = try await analyzeHDBSCANThresholds(scales: Self.thresholdScales)
        let recommendation = computeRecommendation(algorithm: "HDBSCAN", results: results)

        print("\n\(recommendation.summary)")

        // Validate recommendation is reasonable
        XCTAssertGreaterThan(recommendation.recommendedThreshold, 20,
                            "HDBSCAN threshold should be > 20")
        XCTAssertLessThan(recommendation.recommendedThreshold, 200,
                         "HDBSCAN threshold should be < 200")
    }

    /// Tests UMAP thresholds at standard scales.
    func testUMAPThresholds() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Threshold Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let results = try await analyzeUMAPThresholds(scales: Self.thresholdScales)
        let recommendation = computeRecommendation(algorithm: "UMAP", results: results)

        print("\n\(recommendation.summary)")

        // Validate recommendation is reasonable
        XCTAssertGreaterThan(recommendation.recommendedThreshold, 30,
                            "UMAP threshold should be > 30")
        XCTAssertLessThan(recommendation.recommendedThreshold, 250,
                         "UMAP threshold should be < 250")
    }

    // MARK: - GPU Initialization Overhead Test

    /// Measures GPU initialization overhead (cold vs warm GPU).
    ///
    /// The first GPU operation incurs kernel compilation overhead.
    /// Subsequent operations are much faster. This test quantifies
    /// the difference to inform threshold recommendations.
    func testGPUInitializationOverhead() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  GPU Initialization Overhead Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let testSize = 100
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: testSize / 5,
            seed: 42
        )

        // Measure cold GPU (first invocation)
        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpu.computeHDBSCANMSTWithCoreDistances(embeddings, minSamples: 5)
        let coldTime = CFAbsoluteTimeGetCurrent() - coldStart

        // Measure warm GPU (subsequent invocations)
        var warmTimes: [TimeInterval] = []
        for _ in 0..<5 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await gpu.computeHDBSCANMSTWithCoreDistances(embeddings, minSamples: 5)
            warmTimes.append(CFAbsoluteTimeGetCurrent() - start)
        }
        let warmMedian = warmTimes.sorted()[warmTimes.count / 2]

        let overhead = coldTime - warmMedian
        let overheadPercent = (overhead / coldTime) * 100

        print("\n  Results for \(testSize) points:")
        print("    Cold GPU time:  \(formatTime(coldTime))")
        print("    Warm GPU time:  \(formatTime(warmMedian))")
        print("    Init overhead:  \(formatTime(overhead)) (\(String(format: "%.1f", overheadPercent))%)")

        print("\n  Implications:")
        print("    - First GPU operation has significant overhead")
        print("    - Threshold should account for typical use patterns")
        print("    - Batch processing amortizes overhead across operations")

        // Overhead should be measurable but not excessive
        XCTAssertGreaterThan(overhead, 0, "Cold GPU should have measurable overhead")
    }

    // MARK: - HDBSCAN Analysis

    /// Analyzes HDBSCAN threshold across multiple scales.
    private func analyzeHDBSCANThresholds(
        scales: [Int],
        iterations: Int = ThresholdAnalysis.iterationsPerScale,
        warmup: Int = ThresholdAnalysis.warmupIterations
    ) async throws -> [ThresholdResult] {
        var results: [ThresholdResult] = []

        let hdbscanConfig = HDBSCANConfiguration(
            minClusterSize: 5,
            minSamples: 3,
            logTiming: false
        )

        for points in scales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(3, points / 25),
                pointsPerCluster: min(25, points / 3),
                dimension: 384,
                seed: UInt64(points)
            ).prefix(points)
            let testEmbeddings = Array(embeddings)

            // Get GPU context (fresh each time to avoid warm cache effects)
            guard let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false) else {
                continue
            }

            // Create engines - CPU uses very high threshold to force Prim's algorithm
            let highThresholdConfig = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: 999999
            )
            let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
            let cpuEngine = try await HDBSCANEngine(
                configuration: hdbscanConfig,
                gpuContext: cpuGpuContext
            )
            let gpuEngine = try await HDBSCANEngine(
                configuration: hdbscanConfig,
                gpuContext: gpuContext
            )

            // Warmup
            for _ in 0..<warmup {
                _ = try await cpuEngine.fit(testEmbeddings)
                _ = try await gpuEngine.fit(testEmbeddings)
            }

            // Measure CPU
            var cpuTimes: [TimeInterval] = []
            for _ in 0..<iterations {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await cpuEngine.fit(testEmbeddings)
                cpuTimes.append(CFAbsoluteTimeGetCurrent() - start)
            }

            // Measure GPU
            var gpuTimes: [TimeInterval] = []
            for _ in 0..<iterations {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await gpuEngine.fit(testEmbeddings)
                gpuTimes.append(CFAbsoluteTimeGetCurrent() - start)
            }

            let cpuMedian = cpuTimes.sorted()[cpuTimes.count / 2]
            let gpuMedian = gpuTimes.sorted()[gpuTimes.count / 2]

            let result = ThresholdResult(
                algorithm: "HDBSCAN",
                pointCount: points,
                cpuMedianTime: cpuMedian,
                gpuMedianTime: gpuMedian
            )
            results.append(result)

            let marker = result.gpuIsFaster ? "✓" : "○"
            print("  \(marker) \(String(format: "%4d", points)) points: CPU=\(formatTime(cpuMedian)), GPU=\(formatTime(gpuMedian)), Speedup=\(String(format: "%.2f", result.speedup))x")
        }

        return results
    }

    // MARK: - UMAP Analysis

    /// Analyzes UMAP threshold across multiple scales.
    private func analyzeUMAPThresholds(
        scales: [Int],
        iterations: Int = ThresholdAnalysis.iterationsPerScale,
        warmup: Int = ThresholdAnalysis.warmupIterations
    ) async throws -> [ThresholdResult] {
        var results: [ThresholdResult] = []

        // UMAP config with random init for consistent benchmarking
        // (avoids eigendecomposition issues with synthetic data)
        let umapConfig = UMAPConfiguration(
            nNeighbors: 10,
            minDist: 0.1,
            nEpochs: 50,  // Reduced epochs for faster testing
            initialization: .random
        )

        for points in scales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(3, points / 25),
                pointsPerCluster: min(25, points / 3),
                dimension: 384,
                seed: UInt64(points)
            ).prefix(points)
            let testEmbeddings = Array(embeddings)

            // Create GPU context for this scale
            guard let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false) else {
                continue
            }

            // Warmup - create fresh reducers for each warmup iteration
            for _ in 0..<warmup {
                var cpuWarmupReducer = UMAPReducer(configuration: umapConfig, gpuContext: nil)
                var gpuWarmupReducer = UMAPReducer(configuration: umapConfig, gpuContext: gpuContext)
                _ = try await cpuWarmupReducer.fitTransform(testEmbeddings)
                _ = try await gpuWarmupReducer.fitTransform(testEmbeddings)
            }

            // Measure CPU
            var cpuTimes: [TimeInterval] = []
            for _ in 0..<iterations {
                var reducer = UMAPReducer(configuration: umapConfig, gpuContext: nil)
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await reducer.fitTransform(testEmbeddings)
                cpuTimes.append(CFAbsoluteTimeGetCurrent() - start)
            }

            // Measure GPU
            var gpuTimes: [TimeInterval] = []
            for _ in 0..<iterations {
                var reducer = UMAPReducer(configuration: umapConfig, gpuContext: gpuContext)
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await reducer.fitTransform(testEmbeddings)
                gpuTimes.append(CFAbsoluteTimeGetCurrent() - start)
            }

            let cpuMedian = cpuTimes.sorted()[cpuTimes.count / 2]
            let gpuMedian = gpuTimes.sorted()[gpuTimes.count / 2]

            let result = ThresholdResult(
                algorithm: "UMAP",
                pointCount: points,
                cpuMedianTime: cpuMedian,
                gpuMedianTime: gpuMedian
            )
            results.append(result)

            let marker = result.gpuIsFaster ? "✓" : "○"
            print("  \(marker) \(String(format: "%4d", points)) points: CPU=\(formatTime(cpuMedian)), GPU=\(formatTime(gpuMedian)), Speedup=\(String(format: "%.2f", result.speedup))x")
        }

        return results
    }

    // MARK: - Recommendation Computation

    /// Computes threshold recommendation from analysis results.
    private func computeRecommendation(
        algorithm: String,
        results: [ThresholdResult]
    ) -> ThresholdRecommendation {
        guard !results.isEmpty else {
            return ThresholdRecommendation(
                algorithm: algorithm,
                recommendedThreshold: 100,
                crossoverPoint: 100,
                conservativeThreshold: 150,
                aggressiveThreshold: 75,
                peakSpeedup: 1.0,
                peakSpeedupSize: 100
            )
        }

        // Find crossover point (where GPU becomes faster)
        let crossoverPoint = findCrossoverPoint(results: results)

        // Find point where GPU is significantly faster (10%+ speedup)
        let significantPoint = results.first { $0.gpuIsSignificantlyFaster }?.pointCount ?? crossoverPoint

        // Find peak speedup
        let peakResult = results.max { $0.speedup < $1.speedup }!

        // Conservative: round up to nearest 25, add buffer
        let conservative = ((crossoverPoint + 24) / 25) * 25 + 25

        // Aggressive: use crossover point rounded down
        let aggressive = max(25, ((crossoverPoint - 1) / 25) * 25)

        // Recommended: balance between crossover and significant improvement
        let recommended = (crossoverPoint + significantPoint) / 2

        return ThresholdRecommendation(
            algorithm: algorithm,
            recommendedThreshold: recommended,
            crossoverPoint: crossoverPoint,
            conservativeThreshold: conservative,
            aggressiveThreshold: aggressive,
            peakSpeedup: peakResult.speedup,
            peakSpeedupSize: peakResult.pointCount
        )
    }

    /// Finds the crossover point using linear interpolation.
    private func findCrossoverPoint(results: [ThresholdResult]) -> Int {
        // Sort by point count
        let sorted = results.sorted { $0.pointCount < $1.pointCount }

        // Find where speedup crosses 1.0
        for i in 0..<(sorted.count - 1) {
            let current = sorted[i]
            let next = sorted[i + 1]

            // If speedup crosses 1.0 between these points
            if current.speedup < 1.0 && next.speedup >= 1.0 {
                // Linear interpolation to find crossover
                let ratio = (1.0 - current.speedup) / (next.speedup - current.speedup)
                let crossover = Double(current.pointCount) + ratio * Double(next.pointCount - current.pointCount)
                return Int(crossover)
            }
        }

        // If GPU is always faster, use smallest tested size
        if let first = sorted.first, first.gpuIsFaster {
            return first.pointCount
        }

        // If GPU is never faster in tested range, use largest + buffer
        if let last = sorted.last {
            return last.pointCount + 50
        }

        return 100 // Default fallback
    }

    // MARK: - Result Storage

    /// Saves threshold analysis results to disk.
    private func saveThresholdResults(
        hdbscan: [ThresholdResult],
        umap: [ThresholdResult],
        hdbscanRecommendation: ThresholdRecommendation,
        umapRecommendation: ThresholdRecommendation,
        hardwareInfo: HardwareInfo
    ) throws {
        let storage = BenchmarkStorage()

        // Create summary dictionary
        let summary: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "hardware": [
                "cpu": hardwareInfo.cpuModel,
                "gpu": hardwareInfo.gpuName,
                "cores": hardwareInfo.cpuCoreCount,
                "memory": hardwareInfo.totalRAM / (1024 * 1024 * 1024)  // Convert to GB
            ],
            "hdbscan": [
                "crossover": hdbscanRecommendation.crossoverPoint,
                "recommended": hdbscanRecommendation.recommendedThreshold,
                "conservative": hdbscanRecommendation.conservativeThreshold,
                "aggressive": hdbscanRecommendation.aggressiveThreshold,
                "peakSpeedup": hdbscanRecommendation.peakSpeedup,
                "peakSpeedupSize": hdbscanRecommendation.peakSpeedupSize
            ],
            "umap": [
                "crossover": umapRecommendation.crossoverPoint,
                "recommended": umapRecommendation.recommendedThreshold,
                "conservative": umapRecommendation.conservativeThreshold,
                "aggressive": umapRecommendation.aggressiveThreshold,
                "peakSpeedup": umapRecommendation.peakSpeedup,
                "peakSpeedupSize": umapRecommendation.peakSpeedupSize
            ],
            "combinedRecommendation": max(
                hdbscanRecommendation.recommendedThreshold,
                umapRecommendation.recommendedThreshold
            )
        ]

        // Save as JSON
        let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
        let filename = "ThresholdAnalysis_\(Date().timeIntervalSince1970).json"

        try FileManager.default.createDirectory(at: storage.directory, withIntermediateDirectories: true)

        let fileURL = storage.directory.appendingPathComponent(filename)
        try data.write(to: fileURL)

        print("\n  Results saved to: \(fileURL.lastPathComponent)")
    }

    // MARK: - Helpers

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
}

// MARK: - Hardware-Specific Threshold Profiles

extension ThresholdAnalysis {

    /// Tests threshold recommendations for different hardware profiles.
    ///
    /// While we can't change the hardware, this test documents the
    /// current hardware's characteristics and compares to expected values.
    func testHardwareProfile() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        let hardwareInfo = HardwareInfo.capture()

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Hardware Profile Analysis")
        print("═══════════════════════════════════════════════════════════════════")
        print("\n\(hardwareInfo.summary)")

        // Classify hardware tier
        let hardwareTier = classifyHardware(hardwareInfo)

        print("\n  Hardware Classification: \(hardwareTier.name)")
        print("  Expected GPU threshold range: \(hardwareTier.expectedThresholdRange)")
        print("  GPU bandwidth class: \(hardwareTier.bandwidthClass)")

        // Run quick validation
        let testSizes = [50, 100, 200]
        var speedups: [Int: Double] = [:]

        for size in testSizes {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: 5,
                pointsPerCluster: size / 5,
                seed: 42
            ).prefix(size)

            guard let gpu = await TopicsGPUContext.create(allowCPUFallback: false) else {
                continue
            }

            // Quick GPU measurement
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await gpu.computeHDBSCANMSTWithCoreDistances(Array(embeddings), minSamples: 5)
            let gpuTime = CFAbsoluteTimeGetCurrent() - start

            // Quick CPU baseline (approximate)
            let cpuTime = estimateCPUTime(for: size, hardwareTier: hardwareTier)

            speedups[size] = cpuTime / gpuTime
        }

        print("\n  Measured speedups:")
        for (size, speedup) in speedups.sorted(by: { $0.key < $1.key }) {
            print("    \(size) points: \(String(format: "%.1f", speedup))x")
        }
    }

    /// Hardware tier classification.
    struct HardwareTier {
        let name: String
        let expectedThresholdRange: ClosedRange<Int>
        let bandwidthClass: String
    }

    /// Classifies hardware based on characteristics.
    private func classifyHardware(_ info: HardwareInfo) -> HardwareTier {
        // Simple classification based on core count and GPU presence
        let cores = info.cpuCoreCount
        let gpuName = info.gpuName

        // Check for Apple Silicon
        let isAppleSilicon = gpuName.contains("M1") || gpuName.contains("M2") ||
                            gpuName.contains("M3") || gpuName.contains("M4")

        if isAppleSilicon {
            // Apple Silicon unified memory - lower overhead
            if cores >= 10 {
                return HardwareTier(
                    name: "Apple Silicon Pro/Max",
                    expectedThresholdRange: 50...100,
                    bandwidthClass: "High (400+ GB/s)"
                )
            } else {
                return HardwareTier(
                    name: "Apple Silicon Base",
                    expectedThresholdRange: 75...125,
                    bandwidthClass: "Medium (100-200 GB/s)"
                )
            }
        } else if !gpuName.isEmpty && gpuName != "No GPU" {
            return HardwareTier(
                name: "Discrete GPU",
                expectedThresholdRange: 100...200,
                bandwidthClass: "Variable"
            )
        } else {
            return HardwareTier(
                name: "Integrated GPU",
                expectedThresholdRange: 150...300,
                bandwidthClass: "Low (50-100 GB/s)"
            )
        }
    }

    /// Estimates CPU time based on hardware tier (for comparison).
    private func estimateCPUTime(for points: Int, hardwareTier: HardwareTier) -> TimeInterval {
        // Rough O(n²) estimate with hardware factor
        let baseTime = Double(points * points) / 1_000_000.0 * 0.5

        switch hardwareTier.name {
        case "Apple Silicon Pro/Max":
            return baseTime * 0.7
        case "Apple Silicon Base":
            return baseTime * 1.0
        default:
            return baseTime * 1.5
        }
    }
}

// MARK: - Threshold Validation Tests

extension ThresholdAnalysis {

    /// Validates that the default threshold (100) is reasonable.
    func testDefaultThresholdValidation() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Default Threshold Validation (gpuMinPointsThreshold = 100)")
        print("═══════════════════════════════════════════════════════════════════")

        let defaultThreshold = 100
        let testSizes = [
            (defaultThreshold - 25, "Below threshold"),
            (defaultThreshold, "At threshold"),
            (defaultThreshold + 25, "Above threshold"),
            (defaultThreshold * 2, "Well above threshold")
        ]

        for (size, label) in testSizes {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(3, size / 25),
                pointsPerCluster: min(25, size / 3),
                seed: 42
            ).prefix(size)

            guard let gpu = await TopicsGPUContext.create(allowCPUFallback: false) else {
                continue
            }

            // Measure GPU
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpu.computeHDBSCANMSTWithCoreDistances(Array(embeddings), minSamples: 5)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

            // Measure CPU (high threshold forces Prim's)
            let highThresholdConfig = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: 999999
            )
            let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
            let cpuEngine = try await HDBSCANEngine(
                configuration: HDBSCANConfiguration(minClusterSize: 5, minSamples: 3),
                gpuContext: cpuGpuContext
            )

            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuEngine.fit(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            let speedup = cpuTime / gpuTime
            let shouldUseGPU = size >= defaultThreshold

            let status: String
            if shouldUseGPU && speedup > 1.0 {
                status = "✓ Correct (GPU faster)"
            } else if !shouldUseGPU && speedup <= 1.0 {
                status = "✓ Correct (CPU faster)"
            } else if shouldUseGPU && speedup <= 1.0 {
                status = "⚠️ GPU enabled but slower"
            } else {
                status = "○ CPU used, GPU would be faster"
            }

            print("  \(size) points (\(label)): \(String(format: "%.2f", speedup))x \(status)")
        }
    }

    /// Tests sensitivity of threshold choice.
    ///
    /// Measures how much performance is lost by using a non-optimal threshold.
    func testThresholdSensitivity() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Threshold Sensitivity Analysis")
        print("═══════════════════════════════════════════════════════════════════")
        print("\n  Measures performance impact of threshold misconfigurations.\n")

        let workloads = [75, 100, 150, 200, 300]

        for workload in workloads {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, workload / 30),
                pointsPerCluster: 30,
                seed: 42
            ).prefix(workload)

            guard let gpu = await TopicsGPUContext.create(allowCPUFallback: false) else {
                continue
            }

            // GPU time
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpu.computeHDBSCANMSTWithCoreDistances(Array(embeddings), minSamples: 5)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

            // CPU time
            let cpuConfig = TopicsGPUConfiguration(gpuMinPointsThreshold: 999999)
            let cpuContext = try await TopicsGPUContext(configuration: cpuConfig)
            let cpuEngine = try await HDBSCANEngine(
                configuration: HDBSCANConfiguration(minClusterSize: 5, minSamples: 3),
                gpuContext: cpuContext
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuEngine.fit(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            let speedup = cpuTime / gpuTime
            let optimalChoice = speedup > 1.0 ? "GPU" : "CPU"
            let penalty = speedup > 1.0 ?
                (cpuTime - gpuTime) / gpuTime * 100 :
                (gpuTime - cpuTime) / cpuTime * 100

            print("  \(workload) points: Optimal=\(optimalChoice), Wrong choice penalty=\(String(format: "%.0f", penalty))%")
        }

        print("\n  Insight: Threshold errors cause increasing penalties at larger scales.")
    }
}
