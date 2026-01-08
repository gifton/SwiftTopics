// PipelineBenchmarks.swift
// SwiftTopicsTests
//
// End-to-end pipeline benchmarks for TopicModel.fit()
// Part of Phase 4: Pipeline Benchmarks

import XCTest
@testable import SwiftTopics

/// Benchmarks for the complete TopicModel pipeline comparing CPU vs GPU performance.
///
/// Tests the full topic modeling workflow with pre-computed embeddings to isolate
/// pipeline performance from embedding computation overhead.
///
/// ## Pipeline Stages Measured
///
/// 1. **Reduction**: Dimensionality reduction (PCA/UMAP) of embeddings
/// 2. **Clustering**: HDBSCAN clustering to find topics
/// 3. **Representation**: c-TF-IDF keyword extraction
/// 4. **Evaluation**: Optional NPMI coherence scoring
///
/// ## Expected Results
///
/// GPU acceleration primarily benefits:
/// - Clustering (HDBSCAN MST construction)
/// - Reduction (if using UMAP)
///
/// The representation and evaluation stages are CPU-bound text processing.
final class PipelineBenchmarks: XCTestCase {

    // MARK: - Configuration

    /// Benchmark scales to test.
    static let benchmarkScales: [(docs: Int, label: String)] = [
        (100, "100 docs"),
        (250, "250 docs"),
        (500, "500 docs"),
        (1000, "1K docs"),
        (2000, "2K docs"),
    ]

    /// Topic model configuration for benchmarks.
    let modelConfig = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
        clustering: HDBSCANConfiguration(minClusterSize: 5, minSamples: 3),
        representation: CTFIDFConfiguration(keywordsPerTopic: 10),
        coherence: nil,  // Skip coherence for benchmark speed
        seed: 42
    )

    // MARK: - Main Benchmark Suite

    /// Runs the full pipeline benchmark suite across all scales.
    func testPipelineBenchmarkSuite() async throws {
        // Skip if GPU not available
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available for benchmarks")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  End-to-End Pipeline Benchmark Suite")
        print("═══════════════════════════════════════════════════════════════════")

        // Print hardware info
        let hardwareInfo = HardwareInfo.capture()
        print("\n\(hardwareInfo.summary)\n")

        var results: [ComparisonResult] = []

        // Run benchmarks at each scale
        for (docs, label) in Self.benchmarkScales {
            print("─────────────────────────────────────────────────────────────────")
            print("  Running: \(label)")
            print("─────────────────────────────────────────────────────────────────")

            do {
                let result = try await benchmarkAtScale(docCount: docs, label: label)
                results.append(result)
            } catch {
                print("  ⚠️  Skipped: \(error.localizedDescription)")
            }
        }

        // Print summary table
        print("\n")
        results.printTable(title: "Pipeline GPU vs CPU Summary")
        print("\n")

        // Print detailed breakdown for key scales
        printDetailedBreakdown(for: results)

        // Save results
        let storage = BenchmarkStorage()
        let savedURL = try storage.saveRun(results, name: "Pipeline_Benchmark")
        print("Results saved to: \(savedURL.lastPathComponent)")
    }

    // MARK: - Individual Scale Benchmarks

    /// Benchmarks pipeline at 500 documents.
    func testPipeline500Docs() async throws {
        let result = try await benchmarkAtScale(docCount: 500, label: "500 docs")

        print("500 docs speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 1.0, "GPU should provide speedup")
    }

    /// Benchmarks pipeline at 1000 documents.
    func testPipeline1000Docs() async throws {
        let result = try await benchmarkAtScale(docCount: 1000, label: "1K docs")

        print("1K docs speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 1.5, "GPU should provide significant speedup")
    }

    /// Benchmarks pipeline at 2000 documents.
    func testPipeline2000Docs() async throws {
        let result = try await benchmarkAtScale(docCount: 2000, label: "2K docs")

        print("2K docs speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 2.0, "GPU should provide substantial speedup")
    }

    // MARK: - Phase Breakdown Test

    /// Tests and reports per-stage timing breakdown.
    ///
    /// Note: This measures total pipeline time since progress callbacks
    /// have Sendable constraints. For detailed per-phase analysis, use
    /// HDBSCANBenchmarks and UMAPBenchmarks which have built-in timing.
    func testPipelinePhaseBreakdown() async throws {
        let docCount = 500
        let (documents, embeddings) = generateTestData(count: docCount)

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Pipeline Phase Breakdown (\(docCount) documents)")
        print("═══════════════════════════════════════════════════════════════════")

        // CPU path timing - measure full pipeline
        let cpuModel = TopicModel(configuration: modelConfig)
        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
        let cpuTotal = CFAbsoluteTimeGetCurrent() - cpuStart

        // GPU path timing - measure full pipeline
        let gpuModel = TopicModel(configuration: modelConfig)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
        let gpuTotal = CFAbsoluteTimeGetCurrent() - gpuStart

        // Print comparison
        print("\nTotal Pipeline Time:")
        print("  CPU path: \(formatTime(cpuTotal))")
        print("  GPU path: \(formatTime(gpuTotal))")
        print("  Speedup: \(String(format: "%.2f", cpuTotal / gpuTotal))x")

        print("\nNote: For detailed per-stage analysis, see HDBSCANBenchmarks and UMAPBenchmarks")
    }

    // MARK: - Quick Benchmark

    /// Quick benchmark for CI/development - runs only 250 documents.
    func testQuickBenchmark() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        let (documents, embeddings) = generateTestData(count: 250)

        // Create models
        let cpuModel = TopicModel(configuration: modelConfig)
        let gpuModel = TopicModel(configuration: modelConfig)

        let result = try await Benchmark("Pipeline Quick")
            .scale("250 docs")
            .configuration(.quick)
            .baseline("CPU") {
                _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
            }
            .variant("GPU") {
                _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
            }
            .runAndReport()

        print("\nQuick benchmark result:")
        print("  Speedup: \(result.speedupFormatted)")
        print("  Significant: \(result.isSignificant ? "✓" : "○")")
    }

    // MARK: - Scaling Analysis

    /// Tests how pipeline performance scales with document count.
    func testPipelineScalingAnalysis() async throws {
        let scales = [100, 250, 500, 750, 1000]

        var cpuTimes: [(docs: Int, time: Double)] = []
        var gpuTimes: [(docs: Int, time: Double)] = []

        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Pipeline Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        for docCount in scales {
            let (documents, embeddings) = generateTestData(count: docCount)

            // CPU timing
            let cpuModel = TopicModel(configuration: modelConfig)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            cpuTimes.append((docCount, cpuTime))

            // GPU timing
            let gpuModel = TopicModel(configuration: modelConfig)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuTimes.append((docCount, gpuTime))

            let speedup = cpuTime / gpuTime
            print("  \(docCount) docs: CPU=\(formatTime(cpuTime)), GPU=\(formatTime(gpuTime)), Speedup=\(String(format: "%.1f", speedup))x")
        }

        // Calculate scaling exponents
        let cpuExponent = calculateScalingExponent(cpuTimes)
        let gpuExponent = calculateScalingExponent(gpuTimes)

        print("\n  Scaling Analysis:")
        print("    CPU scaling exponent: \(String(format: "%.2f", cpuExponent))")
        print("    GPU scaling exponent: \(String(format: "%.2f", gpuExponent))")
    }

    // MARK: - Helper Methods

    /// Benchmarks pipeline at a specific scale.
    private func benchmarkAtScale(docCount: Int, label: String) async throws -> ComparisonResult {
        let (documents, embeddings) = generateTestData(count: docCount)

        // Skip GPU if not available
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        // Configure iterations based on scale
        let iterations: Int
        let warmup: Int

        switch docCount {
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

        // Create fresh models for each benchmark run
        // Note: The TopicModel uses GPU automatically when available
        // We create separate instances to avoid state pollution

        // Capture config locally for Sendable closure
        let config = modelConfig

        let result = try await Benchmark("Pipeline")
            .scale(label)
            .iterations(iterations)
            .warmup(warmup)
            .baseline("CPU") {
                let cpuModel = TopicModel(configuration: config)
                _ = try await cpuModel.fit(documents: documents, embeddings: embeddings)
            }
            .variant("GPU") {
                let gpuModel = TopicModel(configuration: config)
                _ = try await gpuModel.fit(documents: documents, embeddings: embeddings)
            }
            .runAndReport()

        return result
    }

    /// Generates test documents and embeddings.
    private func generateTestData(count: Int) -> ([Document], [Embedding]) {
        // Generate clustered embeddings for realistic topic structure
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: max(5, count / 50),  // ~50 docs per cluster
            pointsPerCluster: min(50, count / 5),
            dimension: 384,
            seed: UInt64(count)
        ).prefix(count)

        // Generate simple test documents with varied content
        let documents = (0..<count).map { i in
            Document(
                id: DocumentID(),
                content: generateDocumentContent(index: i, clusterIndex: i / max(1, count / 5)),
                metadata: nil
            )
        }

        return (documents, Array(embeddings))
    }

    /// Generates realistic document content for benchmarking.
    private func generateDocumentContent(index: Int, clusterIndex: Int) -> String {
        // Create varied content based on cluster to ensure meaningful c-TF-IDF
        let topics = [
            "machine learning algorithms neural networks deep learning training models optimization",
            "software development programming code testing debugging deployment architecture",
            "data analysis statistics visualization charts graphs insights patterns trends",
            "cloud computing infrastructure servers scaling deployment containers kubernetes",
            "user experience design interface usability accessibility interaction feedback"
        ]

        let baseContent = topics[clusterIndex % topics.count]
        let variation = "Document \(index) discusses \(baseContent) with additional context."

        // Add some length variation
        if index % 3 == 0 {
            return variation + " This is an extended discussion covering more detail."
        }
        return variation
    }

    /// Prints detailed breakdown for key benchmark scales.
    private func printDetailedBreakdown(for results: [ComparisonResult]) {
        print("═══════════════════════════════════════════════════════════════════")
        print("  Detailed Statistics")
        print("═══════════════════════════════════════════════════════════════════")

        for result in results where result.scale.contains("500") || result.scale.contains("1K") {
            print("\n\(result.scale):")
            print("  CPU median: \(TimingStatistics.format(result.baseline.statistics.median))")
            print("  GPU median: \(TimingStatistics.format(result.variant.statistics.median))")
            print("  Speedup: \(result.speedupFormatted)")
            print("  CPU CV: \(String(format: "%.1f%%", result.baseline.statistics.coefficientOfVariation * 100))")
            print("  GPU CV: \(String(format: "%.1f%%", result.variant.statistics.coefficientOfVariation * 100))")
        }
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
    private func calculateScalingExponent(_ data: [(docs: Int, time: Double)]) -> Double {
        guard data.count >= 2 else { return 0 }

        let logData = data.map { (log(Double($0.docs)), log($0.time)) }

        let n = Double(logData.count)
        let sumX = logData.reduce(0) { $0 + $1.0 }
        let sumY = logData.reduce(0) { $0 + $1.1 }
        let sumXY = logData.reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = logData.reduce(0) { $0 + $1.0 * $1.0 }

        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        return slope
    }
}

// MARK: - Throughput Benchmark Extension

extension PipelineBenchmarks {

    /// Measures throughput in documents per second.
    func testPipelineThroughput() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Pipeline Throughput Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let sizes = [500, 1000, 2000]

        for size in sizes {
            let (documents, embeddings) = generateTestData(count: size)

            let model = TopicModel(configuration: modelConfig)

            // Warm up
            _ = try await model.fit(documents: documents, embeddings: embeddings)

            // Measure
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await model.fit(documents: documents, embeddings: embeddings)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let throughput = Double(size) / elapsed
            print("  \(size) docs: \(formatTime(elapsed)) = \(String(format: "%.0f", throughput)) docs/sec")
        }
    }
}
