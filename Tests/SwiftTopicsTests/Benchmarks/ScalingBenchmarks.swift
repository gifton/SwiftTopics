// ScalingBenchmarks.swift
// SwiftTopicsTests
//
// Scaling analysis benchmarks for understanding algorithmic complexity.
// Part of Phase 5: Scaling Analysis

import XCTest
@testable import SwiftTopics

// MARK: - Scaling Benchmarks

/// Scaling analysis benchmarks for understanding algorithmic complexity.
///
/// Generates performance curves to:
/// - Calculate scaling exponents (log-log regression)
/// - Identify CPU/GPU crossover points
/// - Guide threshold tuning decisions
///
/// ## Expected Results
///
/// - **HDBSCAN CPU**: ~O(N²) scaling (exponent ≈ 2.0)
/// - **HDBSCAN GPU**: Better scaling due to parallel MST (exponent < 2.0)
/// - **UMAP CPU**: Mixed (k-NN is ~O(N log N), optimization is ~O(E))
/// - **UMAP GPU**: Lower exponent for optimization phase
///
/// ## Output Formats
///
/// Results can be exported to:
/// - Console tables with box-drawing characters
/// - CSV format for spreadsheet analysis
/// - Markdown tables for documentation
final class ScalingBenchmarks: XCTestCase {

    // MARK: - Configuration

    /// Fine-grained scales for HDBSCAN curve fitting.
    static let hdbscanScales: [Int] = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

    /// Fine-grained scales for UMAP curve fitting.
    static let umapScales: [Int] = [50, 100, 200, 300, 500, 750, 1000, 1500]

    /// Fine-grained scales for Pipeline curve fitting.
    static let pipelineScales: [Int] = [100, 250, 500, 750, 1000]

    /// Quick scales for CI/development.
    static let quickScales: [Int] = [100, 250, 500]

    /// HDBSCAN configuration for benchmarks.
    let hdbscanConfig = HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3,
        logTiming: false
    )

    /// UMAP configuration for benchmarks.
    let umapConfig = UMAPConfiguration(
        nNeighbors: 15,
        minDist: 0.1,
        metric: .euclidean,
        nEpochs: 100,
        learningRate: 1.0
    )

    /// Topic model configuration for pipeline benchmarks.
    let modelConfig = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
        clustering: HDBSCANConfiguration(minClusterSize: 5, minSamples: 3),
        representation: CTFIDFConfiguration(keywordsPerTopic: 10),
        coherence: nil,
        seed: 42
    )

    // MARK: - HDBSCAN Scaling

    /// Full HDBSCAN scaling analysis with curve fitting.
    func testHDBSCANScalingCurves() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  HDBSCAN Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let hardwareInfo = HardwareInfo.capture()
        print("\n\(hardwareInfo.summary)\n")

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        // Run benchmarks at each scale
        for n in Self.hdbscanScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, n / 50),
                pointsPerCluster: min(50, n / 5),
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing (high threshold forces Prim's algorithm)
            let highThresholdConfig = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: 999999
            )
            let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
            let cpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: cpuGpuContext)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuEngine.fit(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: gpu)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuEngine.fit(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(String(format: "%5d", n)) points: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")
        }

        // Calculate scaling results
        let cpuResult = calculateScalingResult(algorithm: "HDBSCAN", path: "CPU", data: cpuTimes)
        let gpuResult = calculateScalingResult(algorithm: "HDBSCAN", path: "GPU", data: gpuTimes)

        print("\n  Scaling Analysis:")
        print("    CPU exponent: \(String(format: "%.2f", cpuResult.exponent)) (R² = \(String(format: "%.3f", cpuResult.rSquared))) → \(cpuResult.interpretation)")
        print("    GPU exponent: \(String(format: "%.2f", gpuResult.exponent)) (R² = \(String(format: "%.3f", gpuResult.rSquared))) → \(gpuResult.interpretation)")

        // Find crossover point
        if let crossover = findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "HDBSCAN") {
            print("\n  Crossover Point:")
            print("    GPU becomes faster at N ≈ \(crossover.crossoverN) points")
            print("    Recommendation: Set gpuMinPointsThreshold = \(crossover.crossoverN)-\(crossover.crossoverN + 25)")
        }

        // Export results
        let scalingData = ScalingData(
            algorithm: "HDBSCAN",
            cpuResults: cpuResult,
            gpuResults: gpuResult,
            crossover: findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "HDBSCAN")
        )
        try exportScalingData(scalingData)

        // Assertions
        XCTAssertGreaterThan(cpuResult.rSquared, 0.9, "CPU scaling should have good fit")
        XCTAssertGreaterThan(gpuResult.rSquared, 0.9, "GPU scaling should have good fit")
    }

    /// Quick HDBSCAN scaling check (fewer points).
    func testHDBSCANScalingQuick() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        print("\n  HDBSCAN Quick Scaling Check:")

        for n in Self.quickScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, n / 50),
                pointsPerCluster: 50,
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing
            let highThresholdConfig = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: 999999
            )
            let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
            let cpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: cpuGpuContext)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuEngine.fit(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: gpu)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuEngine.fit(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("    \(n) points: Speedup=\(String(format: "%.1f", speedup))x")
        }

        let cpuExponent = calculateScalingExponent(cpuTimes)
        let gpuExponent = calculateScalingExponent(gpuTimes)

        print("    CPU exponent: \(String(format: "%.2f", cpuExponent))")
        print("    GPU exponent: \(String(format: "%.2f", gpuExponent))")

        // GPU should scale at least as well as CPU
        XCTAssertLessThan(gpuExponent, cpuExponent + 0.5)
    }

    // MARK: - UMAP Scaling

    /// Full UMAP scaling analysis.
    func testUMAPScalingCurves() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        for n in Self.umapScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, n / 50),
                pointsPerCluster: min(50, n / 5),
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing
            let cpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: nil
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuReducer.fitTransform(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: gpu
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuReducer.fitTransform(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(String(format: "%5d", n)) points: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")
        }

        // Calculate scaling results
        let cpuResult = calculateScalingResult(algorithm: "UMAP", path: "CPU", data: cpuTimes)
        let gpuResult = calculateScalingResult(algorithm: "UMAP", path: "GPU", data: gpuTimes)

        print("\n  Scaling Analysis:")
        print("    CPU exponent: \(String(format: "%.2f", cpuResult.exponent)) (R² = \(String(format: "%.3f", cpuResult.rSquared))) → \(cpuResult.interpretation)")
        print("    GPU exponent: \(String(format: "%.2f", gpuResult.exponent)) (R² = \(String(format: "%.3f", gpuResult.rSquared))) → \(gpuResult.interpretation)")

        // Find crossover point
        if let crossover = findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "UMAP") {
            print("\n  Crossover Point:")
            print("    GPU becomes faster at N ≈ \(crossover.crossoverN) points")
        }

        // Export results
        let scalingData = ScalingData(
            algorithm: "UMAP",
            cpuResults: cpuResult,
            gpuResults: gpuResult,
            crossover: findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "UMAP")
        )
        try exportScalingData(scalingData)

        XCTAssertGreaterThan(cpuResult.rSquared, 0.85, "CPU scaling should have reasonable fit")
    }

    /// Quick UMAP scaling check.
    func testUMAPScalingQuick() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        print("\n  UMAP Quick Scaling Check:")

        for n in Self.quickScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, n / 50),
                pointsPerCluster: 50,
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing
            let cpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: nil
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuReducer.fitTransform(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: gpuContext
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuReducer.fitTransform(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("    \(n) points: Speedup=\(String(format: "%.1f", speedup))x")
        }

        let cpuExponent = calculateScalingExponent(cpuTimes)
        let gpuExponent = calculateScalingExponent(gpuTimes)

        print("    CPU exponent: \(String(format: "%.2f", cpuExponent))")
        print("    GPU exponent: \(String(format: "%.2f", gpuExponent))")
    }

    // MARK: - Pipeline Scaling

    /// Full pipeline scaling analysis.
    func testPipelineScalingCurves() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Pipeline Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        let config = modelConfig

        for n in Self.pipelineScales {
            let (documents, embeddings) = generateTestData(count: n)

            // CPU timing
            let cpuModel = TopicModel(configuration: config)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuModel = TopicModel(configuration: config)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(String(format: "%5d", n)) docs: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")
        }

        // Calculate scaling results
        let cpuResult = calculateScalingResult(algorithm: "Pipeline", path: "CPU", data: cpuTimes)
        let gpuResult = calculateScalingResult(algorithm: "Pipeline", path: "GPU", data: gpuTimes)

        print("\n  Scaling Analysis:")
        print("    CPU exponent: \(String(format: "%.2f", cpuResult.exponent)) (R² = \(String(format: "%.3f", cpuResult.rSquared)))")
        print("    GPU exponent: \(String(format: "%.2f", gpuResult.exponent)) (R² = \(String(format: "%.3f", gpuResult.rSquared)))")

        // Export results
        let scalingData = ScalingData(
            algorithm: "Pipeline",
            cpuResults: cpuResult,
            gpuResults: gpuResult,
            crossover: findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "Pipeline")
        )
        try exportScalingData(scalingData)
    }

    // MARK: - Crossover Analysis

    /// Finds the CPU/GPU crossover point for HDBSCAN.
    func testHDBSCANCrossoverPoint() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        // Fine-grained scales around expected crossover
        let crossoverScales = [25, 50, 75, 100, 125, 150, 200, 250, 300]

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  HDBSCAN Crossover Point Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        for n in crossoverScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(3, n / 25),
                pointsPerCluster: 25,
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing
            let highThresholdConfig = TopicsGPUConfiguration(
                preferHighPerformance: true,
                gpuMinPointsThreshold: 999999
            )
            let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
            let cpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: cpuGpuContext)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuEngine.fit(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: gpu)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuEngine.fit(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let diff = cpuTime - gpuTime
            let marker = diff > 0 ? "GPU faster" : "CPU faster"
            print("  \(String(format: "%3d", n)) points: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)) ← \(marker)")
        }

        if let crossover = findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "HDBSCAN") {
            print("\n  Crossover Analysis:")
            print("    Crossover N: \(crossover.crossoverN) points")
            print("    CPU time at crossover: \(formatTime(crossover.cpuTimeAtCrossover))")
            print("    GPU time at crossover: \(formatTime(crossover.gpuTimeAtCrossover))")
            print("    Confidence: \(crossover.confidenceLevel)")
            print("\n  Recommendation:")
            print("    Set gpuMinPointsThreshold = \(max(50, crossover.crossoverN - 25))-\(crossover.crossoverN + 25)")
        } else {
            print("\n  No crossover found - GPU is always faster or slower")
        }
    }

    /// Finds the CPU/GPU crossover point for UMAP.
    func testUMAPCrossoverPoint() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        // Fine-grained scales around expected crossover
        let crossoverScales = [25, 50, 75, 100, 150, 200, 250, 300]

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Crossover Point Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        for n in crossoverScales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(3, n / 25),
                pointsPerCluster: 25,
                dimension: 384,
                seed: UInt64(n)
            ).prefix(n)

            // CPU timing
            let cpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: nil
            )
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuReducer.fitTransform(Array(embeddings))
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((n, cpuTime))

            // GPU timing
            let gpuReducer = UMAPReducer(
                configuration: umapConfig,
                nComponents: 15,
                seed: 42,
                gpuContext: gpu
            )
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuReducer.fitTransform(Array(embeddings))
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((n, gpuTime))

            let diff = cpuTime - gpuTime
            let marker = diff > 0 ? "GPU faster" : "CPU faster"
            print("  \(String(format: "%3d", n)) points: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)) ← \(marker)")
        }

        if let crossover = findCrossoverPoint(cpuData: cpuTimes, gpuData: gpuTimes, algorithm: "UMAP") {
            print("\n  Crossover Analysis:")
            print("    Crossover N: \(crossover.crossoverN) points")
            print("    Confidence: \(crossover.confidenceLevel)")
        }
    }

    // MARK: - Combined Analysis

    /// Runs all scaling analyses and generates summary report.
    func testFullScalingAnalysis() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║              Full Scaling Analysis Report                         ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")

        let hardwareInfo = HardwareInfo.capture()
        print("\n\(hardwareInfo.summary)")

        var allResults: [ScalingResult] = []

        // Run HDBSCAN analysis (subset of scales for time)
        print("\n─── HDBSCAN ───────────────────────────────────────────────────────")
        let hdbscanResults = try await runScalingAnalysis(
            algorithm: "HDBSCAN",
            scales: [100, 250, 500, 1000],
            runner: runHDBSCANBenchmark
        )
        allResults.append(contentsOf: [hdbscanResults.cpu, hdbscanResults.gpu])

        // Run UMAP analysis
        print("\n─── UMAP ──────────────────────────────────────────────────────────")
        let umapResults = try await runScalingAnalysis(
            algorithm: "UMAP",
            scales: [100, 250, 500, 750],
            runner: runUMAPBenchmark
        )
        allResults.append(contentsOf: [umapResults.cpu, umapResults.gpu])

        // Run Pipeline analysis
        print("\n─── Pipeline ──────────────────────────────────────────────────────")
        let pipelineResults = try await runScalingAnalysis(
            algorithm: "Pipeline",
            scales: [100, 250, 500],
            runner: runPipelineBenchmark
        )
        allResults.append(contentsOf: [pipelineResults.cpu, pipelineResults.gpu])

        // Generate summary table
        print("\n")
        print(generateMarkdownTable(allResults))

        // Export combined CSV
        try exportCombinedCSV(
            hdbscan: hdbscanResults,
            umap: umapResults,
            pipeline: pipelineResults
        )

        print("\n  Results exported to BenchmarkResults/")
    }

    // MARK: - Helper Methods

    /// Calculates scaling exponent using log-log regression.
    private func calculateScalingExponent(_ data: [(n: Int, time: Double)]) -> Double {
        guard data.count >= 2 else { return 0 }

        let logData = data.map { (log(Double($0.n)), log($0.time)) }

        let count = Double(logData.count)
        let sumX = logData.reduce(0) { $0 + $1.0 }
        let sumY = logData.reduce(0) { $0 + $1.1 }
        let sumXY = logData.reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = logData.reduce(0) { $0 + $1.0 * $1.0 }

        let slope = (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX)
        return slope
    }

    /// Calculates R² (coefficient of determination) for log-log regression.
    private func calculateRSquared(_ data: [(n: Int, time: Double)]) -> Double {
        guard data.count >= 2 else { return 0 }

        let logData = data.map { (log(Double($0.n)), log($0.time)) }

        let count = Double(logData.count)
        let sumX = logData.reduce(0) { $0 + $1.0 }
        let sumY = logData.reduce(0) { $0 + $1.1 }
        let sumXY = logData.reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = logData.reduce(0) { $0 + $1.0 * $1.0 }

        let slope = (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX)
        let intercept = (sumY - slope * sumX) / count

        // Calculate predicted values and residuals
        var ssRes = 0.0
        var ssTot = 0.0
        let meanY = sumY / count

        for (x, y) in logData {
            let predicted = slope * x + intercept
            ssRes += (y - predicted) * (y - predicted)
            ssTot += (y - meanY) * (y - meanY)
        }

        return 1 - (ssRes / ssTot)
    }

    /// Calculates full scaling result with interpretation.
    private func calculateScalingResult(
        algorithm: String,
        path: String,
        data: [(n: Int, time: Double)]
    ) -> ScalingResult {
        let exponent = calculateScalingExponent(data)
        let rSquared = calculateRSquared(data)

        let interpretation: String
        if exponent < 1.2 {
            interpretation = "O(N) - linear"
        } else if exponent < 1.5 {
            interpretation = "O(N^1.3) - subquadratic"
        } else if exponent < 1.8 {
            interpretation = "O(N^1.5) - superlinear"
        } else if exponent < 2.2 {
            interpretation = "O(N²) - quadratic"
        } else {
            interpretation = "O(N^\(String(format: "%.1f", exponent))) - superquadratic"
        }

        return ScalingResult(
            algorithm: algorithm,
            path: path,
            exponent: exponent,
            rSquared: rSquared,
            interpretation: interpretation,
            dataPoints: data
        )
    }

    /// Finds crossover point using linear interpolation.
    private func findCrossoverPoint(
        cpuData: [(n: Int, time: Double)],
        gpuData: [(n: Int, time: Double)],
        algorithm: String
    ) -> CrossoverResult? {
        guard cpuData.count == gpuData.count, cpuData.count >= 2 else { return nil }

        // Find where CPU becomes slower than GPU
        var crossoverIndex: Int?
        for i in 0..<(cpuData.count - 1) {
            let cpuFasterNow = cpuData[i].time < gpuData[i].time
            let cpuSlowerNext = cpuData[i + 1].time > gpuData[i + 1].time

            if cpuFasterNow && cpuSlowerNext {
                crossoverIndex = i
                break
            }
        }

        // If no crossover found, check if GPU is always faster
        if crossoverIndex == nil {
            let gpuAlwaysFaster = zip(cpuData, gpuData).allSatisfy { $0.time > $1.time }
            if gpuAlwaysFaster {
                // GPU is faster even at smallest scale
                return CrossoverResult(
                    algorithm: algorithm,
                    crossoverN: cpuData[0].n,
                    cpuTimeAtCrossover: cpuData[0].time,
                    gpuTimeAtCrossover: gpuData[0].time,
                    confidenceLevel: "low (GPU always faster)"
                )
            }
            return nil
        }

        guard let idx = crossoverIndex else { return nil }

        // Linear interpolation to find exact crossover
        let n1 = Double(cpuData[idx].n)
        let n2 = Double(cpuData[idx + 1].n)
        let cpuDiff1 = cpuData[idx].time - gpuData[idx].time
        let cpuDiff2 = cpuData[idx + 1].time - gpuData[idx + 1].time

        // Solve for crossover: cpuDiff1 + t * (cpuDiff2 - cpuDiff1) = 0
        let t = -cpuDiff1 / (cpuDiff2 - cpuDiff1)
        let crossoverN = Int(n1 + t * (n2 - n1))

        // Interpolate times at crossover
        let cpuTime = cpuData[idx].time + t * (cpuData[idx + 1].time - cpuData[idx].time)
        let gpuTime = gpuData[idx].time + t * (gpuData[idx + 1].time - gpuData[idx].time)

        // Confidence based on how close data points are
        let confidence: String
        if n2 - n1 <= 50 {
            confidence = "high"
        } else if n2 - n1 <= 150 {
            confidence = "medium"
        } else {
            confidence = "low"
        }

        return CrossoverResult(
            algorithm: algorithm,
            crossoverN: crossoverN,
            cpuTimeAtCrossover: cpuTime,
            gpuTimeAtCrossover: gpuTime,
            confidenceLevel: confidence
        )
    }

    /// Exports scaling data to CSV format.
    private func exportScalingData(_ data: ScalingData) throws {
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("BenchmarkResults")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let filename = "\(data.algorithm)_scaling_\(dateString()).csv"
        let fileURL = directory.appendingPathComponent(filename)

        var csv = "algorithm,path,n,time_ms,speedup\n"

        // CPU data
        for point in data.cpuResults.dataPoints {
            csv += "\(data.algorithm),CPU,\(point.n),\(String(format: "%.2f", point.time * 1000)),\n"
        }

        // GPU data with speedup
        for (i, point) in data.gpuResults.dataPoints.enumerated() {
            let cpuTime = data.cpuResults.dataPoints[i].time
            let speedup = cpuTime / point.time
            csv += "\(data.algorithm),GPU,\(point.n),\(String(format: "%.2f", point.time * 1000)),\(String(format: "%.2f", speedup))\n"
        }

        try csv.write(to: fileURL, atomically: true, encoding: .utf8)
        print("  CSV exported: \(filename)")
    }

    /// Exports combined CSV for all algorithms.
    private func exportCombinedCSV(
        hdbscan: (cpu: ScalingResult, gpu: ScalingResult),
        umap: (cpu: ScalingResult, gpu: ScalingResult),
        pipeline: (cpu: ScalingResult, gpu: ScalingResult)
    ) throws {
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("BenchmarkResults")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let filename = "scaling_analysis_\(dateString()).csv"
        let fileURL = directory.appendingPathComponent(filename)

        var csv = "algorithm,path,n,time_ms\n"

        for result in [hdbscan.cpu, hdbscan.gpu, umap.cpu, umap.gpu, pipeline.cpu, pipeline.gpu] {
            for point in result.dataPoints {
                csv += "\(result.algorithm),\(result.path),\(point.n),\(String(format: "%.2f", point.time * 1000))\n"
            }
        }

        try csv.write(to: fileURL, atomically: true, encoding: .utf8)
        print("  Combined CSV exported: \(filename)")
    }

    /// Generates markdown table of scaling results.
    private func generateMarkdownTable(_ results: [ScalingResult]) -> String {
        var md = "═══════════════════════════════════════════════════════════════════\n"
        md += "  Scaling Analysis Summary\n"
        md += "═══════════════════════════════════════════════════════════════════\n\n"
        md += "| Algorithm | Path | Exponent | R² | Interpretation |\n"
        md += "|-----------|------|----------|-----|----------------|\n"

        for result in results {
            md += "| \(result.algorithm) | \(result.path) | "
            md += "\(String(format: "%.2f", result.exponent)) | "
            md += "\(String(format: "%.3f", result.rSquared)) | "
            md += "\(result.interpretation) |\n"
        }

        return md
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

    /// Generates date string for filenames.
    private func dateString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm"
        return formatter.string(from: Date())
    }

    // MARK: - Benchmark Runners

    /// Generic scaling analysis runner.
    private func runScalingAnalysis(
        algorithm: String,
        scales: [Int],
        runner: (Int) async throws -> (cpu: Double, gpu: Double)
    ) async throws -> (cpu: ScalingResult, gpu: ScalingResult) {
        var cpuTimes: [(n: Int, time: Double)] = []
        var gpuTimes: [(n: Int, time: Double)] = []

        for n in scales {
            let (cpuTime, gpuTime) = try await runner(n)
            cpuTimes.append((n, cpuTime))
            gpuTimes.append((n, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(String(format: "%5d", n)): CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), \(String(format: "%.1f", speedup))x")
        }

        let cpuResult = calculateScalingResult(algorithm: algorithm, path: "CPU", data: cpuTimes)
        let gpuResult = calculateScalingResult(algorithm: algorithm, path: "GPU", data: gpuTimes)

        print("  Exponents: CPU=\(String(format: "%.2f", cpuResult.exponent)), GPU=\(String(format: "%.2f", gpuResult.exponent))")

        return (cpuResult, gpuResult)
    }

    /// HDBSCAN benchmark runner.
    private func runHDBSCANBenchmark(n: Int) async throws -> (cpu: Double, gpu: Double) {
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: max(5, n / 50),
            pointsPerCluster: min(50, n / 5),
            dimension: 384,
            seed: UInt64(n)
        ).prefix(n)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)!

        // CPU timing
        let highThresholdConfig = TopicsGPUConfiguration(
            preferHighPerformance: true,
            gpuMinPointsThreshold: 999999
        )
        let cpuGpuContext = try await TopicsGPUContext(configuration: highThresholdConfig)
        let cpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: cpuGpuContext)
        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await cpuEngine.fit(Array(embeddings))
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        // GPU timing
        let gpuEngine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: gpuContext)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpuEngine.fit(Array(embeddings))
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        return (cpuTime, gpuTime)
    }

    /// UMAP benchmark runner.
    private func runUMAPBenchmark(n: Int) async throws -> (cpu: Double, gpu: Double) {
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: max(5, n / 50),
            pointsPerCluster: min(50, n / 5),
            dimension: 384,
            seed: UInt64(n)
        ).prefix(n)

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)!

        // CPU timing
        let cpuReducer = UMAPReducer(
            configuration: umapConfig,
            nComponents: 15,
            seed: 42,
            gpuContext: nil
        )
        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await cpuReducer.fitTransform(Array(embeddings))
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        // GPU timing
        let gpuReducer = UMAPReducer(
            configuration: umapConfig,
            nComponents: 15,
            seed: 42,
            gpuContext: gpuContext
        )
        let gpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpuReducer.fitTransform(Array(embeddings))
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        return (cpuTime, gpuTime)
    }

    /// Pipeline benchmark runner.
    private func runPipelineBenchmark(n: Int) async throws -> (cpu: Double, gpu: Double) {
        let (documents, embeddings) = generateTestData(count: n)
        let config = modelConfig

        // CPU timing
        let cpuModel = TopicModel(configuration: config)
        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        // GPU timing
        let gpuModel = TopicModel(configuration: config)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        return (cpuTime, gpuTime)
    }

    /// Generates test documents and embeddings.
    private func generateTestData(count: Int) -> ([Document], [Embedding]) {
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: max(5, count / 50),
            pointsPerCluster: min(50, count / 5),
            dimension: 384,
            seed: UInt64(count)
        ).prefix(count)

        let topics = [
            "machine learning algorithms neural networks deep learning training models optimization",
            "software development programming code testing debugging deployment architecture",
            "data analysis statistics visualization charts graphs insights patterns trends",
            "cloud computing infrastructure servers scaling deployment containers kubernetes",
            "user experience design interface usability accessibility interaction feedback"
        ]

        let documents = (0..<count).map { i in
            let clusterIndex = i / max(1, count / 5)
            let baseContent = topics[clusterIndex % topics.count]
            let content = "Document \(i) discusses \(baseContent) with additional context."
            return Document(
                id: DocumentID(),
                content: content,
                metadata: nil
            )
        }

        return (documents, Array(embeddings))
    }
}

// MARK: - Supporting Types

/// Result of scaling exponent calculation.
struct ScalingResult {
    /// Algorithm name (e.g., "HDBSCAN", "UMAP").
    let algorithm: String

    /// Path type (e.g., "CPU", "GPU").
    let path: String

    /// Calculated scaling exponent.
    let exponent: Double

    /// R² (coefficient of determination) - goodness of fit.
    let rSquared: Double

    /// Human-readable interpretation (e.g., "O(N²) - quadratic").
    let interpretation: String

    /// Raw data points used for calculation.
    let dataPoints: [(n: Int, time: Double)]
}

/// Result of crossover point analysis.
struct CrossoverResult {
    /// Algorithm name.
    let algorithm: String

    /// Approximate N where GPU becomes faster than CPU.
    let crossoverN: Int

    /// CPU time at the crossover point.
    let cpuTimeAtCrossover: Double

    /// GPU time at the crossover point.
    let gpuTimeAtCrossover: Double

    /// Confidence level of the estimate ("high", "medium", "low").
    let confidenceLevel: String
}

/// Combined scaling data for export.
struct ScalingData {
    /// Algorithm name.
    let algorithm: String

    /// CPU scaling results.
    let cpuResults: ScalingResult

    /// GPU scaling results.
    let gpuResults: ScalingResult

    /// Optional crossover point analysis.
    let crossover: CrossoverResult?
}
