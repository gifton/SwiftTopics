// BenchmarkInfrastructureTests.swift
// SwiftTopicsTests
//
// Tests for Day 1 benchmark infrastructure.

import XCTest
@testable import SwiftTopics

/// Tests for benchmark infrastructure (Day 1).
///
/// These tests verify the core measurement functionality works correctly
/// before building out the full benchmark suite.
final class BenchmarkInfrastructureTests: XCTestCase {

    // MARK: - Exit Criteria Test

    /// Verifies the Day 1 exit criteria works as specified.
    ///
    /// From BENCHMARK_SUITE_PLAN.md:
    /// ```swift
    /// let result = try await BenchmarkHarness.compare(
    ///     name: "Test",
    ///     scale: "100 points",
    ///     configuration: .default,
    ///     baseline: ("A", { try await Task.sleep(for: .milliseconds(10)) }),
    ///     variant: ("B", { try await Task.sleep(for: .milliseconds(5)) })
    /// )
    /// print(result.speedup)  // Should print ~2.0x
    /// ```
    func testExitCriteria() async throws {
        let result = try await BenchmarkHarness.compare(
            name: "Test",
            scale: "100 points",
            configuration: .default,
            baseline: ("A", { try await Task.sleep(for: .milliseconds(10)) }),
            variant: ("B", { try await Task.sleep(for: .milliseconds(5)) })
        )

        // Speedup should be approximately 2.0x (A is ~2x slower than B)
        print("Speedup: \(result.speedup)")

        XCTAssertGreaterThan(result.speedup, 1.5, "Variant should be significantly faster")
        XCTAssertLessThan(result.speedup, 3.0, "Speedup should be approximately 2x")

        // Verify basic result structure
        XCTAssertEqual(result.name, "Test")
        XCTAssertEqual(result.scale, "100 points")
        XCTAssertEqual(result.baseline.label, "A")
        XCTAssertEqual(result.variant.label, "B")

        // Verify statistics were computed
        XCTAssertEqual(result.baseline.statistics.sampleCount, 10) // default iterations
        XCTAssertEqual(result.variant.statistics.sampleCount, 10)
        XCTAssertEqual(result.baseline.statistics.failedCount, 0)
        XCTAssertEqual(result.variant.statistics.failedCount, 0)

        print("Result summary:")
        print(result.summary)
    }

    // MARK: - Statistics Calculator Tests

    /// Tests that statistics are computed correctly.
    func testStatisticsCalculator() {
        let samples: [Duration] = [
            .milliseconds(10),
            .milliseconds(12),
            .milliseconds(11),
            .milliseconds(9),
            .milliseconds(10),
            .milliseconds(13),
            .milliseconds(8),
            .milliseconds(10),
            .milliseconds(11),
            .milliseconds(10),
        ]

        let stats = StatisticsCalculator.compute(samples: samples, failedCount: 0)

        // Verify basic metrics
        XCTAssertEqual(stats.sampleCount, 10)
        XCTAssertEqual(stats.failedCount, 0)

        // Min/max
        XCTAssertEqual(stats.min, .milliseconds(8))
        XCTAssertEqual(stats.max, .milliseconds(13))

        // Mean should be ~10.4ms (104ms total / 10)
        let meanMs = stats.meanMilliseconds
        XCTAssertGreaterThan(meanMs, 10.0)
        XCTAssertLessThan(meanMs, 11.0)

        // Median should be 10ms (middle values are 10, 10)
        XCTAssertEqual(stats.medianMilliseconds, 10.0, accuracy: 0.5)

        // Verify percentiles are ordered correctly
        XCTAssertLessThanOrEqual(stats.p25.nanoseconds, stats.median.nanoseconds)
        XCTAssertLessThanOrEqual(stats.median.nanoseconds, stats.p75.nanoseconds)
        XCTAssertLessThanOrEqual(stats.p75.nanoseconds, stats.p95.nanoseconds)
        XCTAssertLessThanOrEqual(stats.p95.nanoseconds, stats.p99.nanoseconds)

        // CV should be relatively low for this consistent data
        XCTAssertLessThan(stats.coefficientOfVariation, 0.2)

        print("Statistics summary:")
        print(stats.summary)
    }

    // MARK: - Configuration Tests

    /// Tests benchmark configuration presets.
    func testConfigurationPresets() {
        // Default
        XCTAssertEqual(BenchmarkConfiguration.default.iterations, 10)
        XCTAssertEqual(BenchmarkConfiguration.default.warmupIterations, 3)
        XCTAssertTrue(BenchmarkConfiguration.default.interleaved)

        // Quick
        XCTAssertEqual(BenchmarkConfiguration.quick.iterations, 5)
        XCTAssertEqual(BenchmarkConfiguration.quick.warmupIterations, 2)

        // Thorough
        XCTAssertEqual(BenchmarkConfiguration.thorough.iterations, 25)
        XCTAssertEqual(BenchmarkConfiguration.thorough.warmupIterations, 5)
    }

    // MARK: - Single Measurement Test

    /// Tests measuring a single operation.
    func testSingleMeasurement() async throws {
        let result = try await BenchmarkHarness.measure(
            name: "Sleep Test",
            label: "Test",
            scale: "N/A",
            configuration: .quick,
            operation: {
                try await Task.sleep(for: .milliseconds(5))
            }
        )

        // Verify result
        XCTAssertEqual(result.name, "Sleep Test")
        XCTAssertEqual(result.label, "Test")
        XCTAssertEqual(result.statistics.sampleCount, 5) // quick config

        // Sleep should take at least 5ms
        XCTAssertGreaterThan(result.statistics.median.nanoseconds, 4_000_000)

        // But not too much more (allow 50% overhead for scheduler jitter)
        XCTAssertLessThan(result.statistics.median.nanoseconds, 15_000_000)
    }

    // MARK: - Interleaved vs Sequential Test

    /// Tests that interleaved mode produces similar results to sequential.
    func testInterleavedExecution() async throws {
        // Quick config with interleaved=true
        let interleavedConfig = BenchmarkConfiguration(
            iterations: 5,
            warmupIterations: 2,
            interleaved: true
        )

        let interleavedResult = try await BenchmarkHarness.compare(
            name: "Interleaved Test",
            scale: "test",
            configuration: interleavedConfig,
            baseline: ("Slow", { try await Task.sleep(for: .milliseconds(8)) }),
            variant: ("Fast", { try await Task.sleep(for: .milliseconds(4)) })
        )

        // Sequential config
        let sequentialConfig = BenchmarkConfiguration(
            iterations: 5,
            warmupIterations: 2,
            interleaved: false
        )

        let sequentialResult = try await BenchmarkHarness.compare(
            name: "Sequential Test",
            scale: "test",
            configuration: sequentialConfig,
            baseline: ("Slow", { try await Task.sleep(for: .milliseconds(8)) }),
            variant: ("Fast", { try await Task.sleep(for: .milliseconds(4)) })
        )

        // Both should show ~2x speedup
        XCTAssertGreaterThan(interleavedResult.speedup, 1.5)
        XCTAssertGreaterThan(sequentialResult.speedup, 1.5)

        print("Interleaved speedup: \(interleavedResult.speedupFormatted)")
        print("Sequential speedup: \(sequentialResult.speedupFormatted)")
    }

    // MARK: - Failure Handling Test

    /// Tests that failures are tracked correctly.
    func testFailureTracking() async throws {
        // Use actor for thread-safe counting in concurrent context
        let counter = FailCounter()

        let result = try await BenchmarkHarness.measure(
            name: "Failing Test",
            label: "Flaky",
            scale: "N/A",
            configuration: .quick,
            operation: {
                let count = await counter.increment()
                if count % 2 == 0 {
                    throw NSError(domain: "Test", code: 1, userInfo: nil)
                }
                try await Task.sleep(for: .milliseconds(1))
            }
        )

        // Should have some successful samples and some failures
        // With 5 iterations + 2 warmup = 7 calls, alternating fail pattern
        XCTAssertGreaterThan(result.statistics.sampleCount, 0)
        XCTAssertGreaterThan(result.statistics.failedCount, 0)
        XCTAssertEqual(result.statistics.sampleCount + result.statistics.failedCount, 5)

        print("Samples: \(result.statistics.sampleCount), Failed: \(result.statistics.failedCount)")
    }

    // MARK: - Speedup Formatting Test

    /// Tests speedup formatting.
    func testSpeedupFormatting() async throws {
        let result = try await BenchmarkHarness.compare(
            name: "Format Test",
            scale: "test",
            configuration: .quick,
            baseline: ("Slow", { try await Task.sleep(for: .milliseconds(10)) }),
            variant: ("Fast", { try await Task.sleep(for: .milliseconds(1)) })
        )

        // Should be roughly 10x
        let formatted = result.speedupFormatted
        XCTAssertTrue(formatted.hasSuffix("x"), "Should end with 'x'")
        print("Formatted speedup: \(formatted)")
    }

    // MARK: - Day 2: DSL Builder Tests

    /// Tests the Day 2 exit criteria - DSL builder with runAndReport().
    ///
    /// From BENCHMARK_SUITE_PLAN.md:
    /// ```swift
    /// let result = try await Benchmark("Test Benchmark")
    ///     .scale("100 points")
    ///     .iterations(10)
    ///     .baseline("Slow") { /* ... */ }
    ///     .variant("Fast") { /* ... */ }
    ///     .runAndReport()
    /// ```
    func testDSLExitCriteria() async throws {
        let result = try await Benchmark("Test Benchmark")
            .scale("100 points")
            .iterations(10)
            .baseline("Slow") { try await Task.sleep(for: .milliseconds(10)) }
            .variant("Fast") { try await Task.sleep(for: .milliseconds(5)) }
            .runAndReport()

        // Verify speedup is approximately 2x
        print("DSL Exit Criteria - Speedup: \(result.speedupFormatted)")
        XCTAssertGreaterThan(result.speedup, 1.5, "Variant should be significantly faster")
        XCTAssertLessThan(result.speedup, 3.0, "Speedup should be approximately 2x")

        // Verify result structure
        XCTAssertEqual(result.name, "Test Benchmark")
        XCTAssertEqual(result.scale, "100 points")
        XCTAssertEqual(result.baseline.label, "Slow")
        XCTAssertEqual(result.variant.label, "Fast")
    }

    /// Tests the DSL builder with all configuration options.
    func testDSLFullConfiguration() async throws {
        let result = try await Benchmark("Full Config Test")
            .scale("test scale")
            .iterations(5)
            .warmup(2)
            .seed(12345)
            .interleaved(true)
            .baseline("A") { try await Task.sleep(for: .milliseconds(8)) }
            .variant("B") { try await Task.sleep(for: .milliseconds(4)) }
            .run()

        // Verify configuration was applied
        XCTAssertEqual(result.baseline.statistics.sampleCount, 5)
        XCTAssertEqual(result.variant.statistics.sampleCount, 5)

        // Verify labels
        XCTAssertEqual(result.baseline.label, "A")
        XCTAssertEqual(result.variant.label, "B")

        // Verify ~2x speedup
        XCTAssertGreaterThan(result.speedup, 1.5)
        XCTAssertLessThan(result.speedup, 3.0)
    }

    /// Tests the DSL builder with preset configuration.
    func testDSLPresetConfiguration() async throws {
        let result = try await Benchmark("Preset Config")
            .scale("quick test")
            .configuration(.quick) // 5 iterations, 2 warmup
            .baseline("Base") { try await Task.sleep(for: .milliseconds(6)) }
            .variant("Fast") { try await Task.sleep(for: .milliseconds(2)) }
            .run()

        // Quick config should have 5 iterations
        XCTAssertEqual(result.baseline.statistics.sampleCount, 5)
        XCTAssertEqual(result.variant.statistics.sampleCount, 5)

        // Verify ~3x speedup
        XCTAssertGreaterThan(result.speedup, 2.0)
        XCTAssertLessThan(result.speedup, 5.0)
    }

    /// Tests that missing operations throw an error.
    func testDSLMissingOperationError() async throws {
        // Missing both operations
        do {
            _ = try await Benchmark("Missing Ops").run()
            XCTFail("Should throw missingOperation error")
        } catch let error as BenchmarkError {
            if case .missingOperation = error {
                // Expected
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }

        // Missing variant
        do {
            _ = try await Benchmark("Missing Variant")
                .baseline("A") { try await Task.sleep(for: .milliseconds(1)) }
                .run()
            XCTFail("Should throw missingOperation error")
        } catch let error as BenchmarkError {
            if case .missingOperation = error {
                // Expected
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }
    }

    // MARK: - Day 2: Reporter Tests

    /// Tests console formatting for a single result.
    func testConsoleReporterFormat() async throws {
        let result = try await Benchmark("Format Test")
            .scale("100 items")
            .configuration(.quick)
            .baseline("CPU") { try await Task.sleep(for: .milliseconds(10)) }
            .variant("GPU") { try await Task.sleep(for: .milliseconds(2)) }
            .run()

        let output = BenchmarkReporter.console.formatConsole(result)

        // Verify output contains expected elements
        XCTAssertTrue(output.contains("Format Test"), "Should contain benchmark name")
        XCTAssertTrue(output.contains("100 items"), "Should contain scale")
        XCTAssertTrue(output.contains("CPU"), "Should contain baseline label")
        XCTAssertTrue(output.contains("GPU"), "Should contain variant label")
        XCTAssertTrue(output.contains("Speedup"), "Should contain speedup header")
        XCTAssertTrue(output.contains("╔"), "Should use box-drawing characters")
        XCTAssertTrue(output.contains("║"), "Should use box-drawing characters")
        XCTAssertTrue(output.contains("╚"), "Should use box-drawing characters")

        print("Console output:\n\(output)")
    }

    /// Tests JSON formatting for a single result.
    func testJSONReporterFormat() async throws {
        let result = try await Benchmark("JSON Test")
            .scale("50 items")
            .configuration(.quick)
            .baseline("Old") { try await Task.sleep(for: .milliseconds(8)) }
            .variant("New") { try await Task.sleep(for: .milliseconds(4)) }
            .run()

        let json = BenchmarkReporter.json.formatJSON(result)

        // Verify JSON is valid and contains expected fields
        XCTAssertTrue(json.contains("\"name\""), "Should contain name field")
        XCTAssertTrue(json.contains("JSON Test"), "Should contain benchmark name")
        XCTAssertTrue(json.contains("\"scale\""), "Should contain scale field")
        XCTAssertTrue(json.contains("\"baseline\""), "Should contain baseline field")
        XCTAssertTrue(json.contains("\"variant\""), "Should contain variant field")
        XCTAssertTrue(json.contains("\"statistics\""), "Should contain statistics field")

        // Verify JSON is parseable (need to use ISO8601 date strategy)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        XCTAssertNoThrow(try decoder.decode(ComparisonResult.self, from: data))

        print("JSON output:\n\(json)")
    }

    /// Tests detailed statistics output.
    func testDetailedStatsOutput() async throws {
        let result = try await Benchmark("Detailed Test")
            .scale("test")
            .configuration(.quick)
            .baseline("Slow") { try await Task.sleep(for: .milliseconds(10)) }
            .variant("Fast") { try await Task.sleep(for: .milliseconds(5)) }
            .run()

        let detailedOutput = BenchmarkReporter.console.formatDetailedStats(result)

        // Verify detailed output contains expected statistics
        XCTAssertTrue(detailedOutput.contains("Samples:"), "Should show sample count")
        XCTAssertTrue(detailedOutput.contains("Median:"), "Should show median")
        XCTAssertTrue(detailedOutput.contains("Mean:"), "Should show mean")
        XCTAssertTrue(detailedOutput.contains("Stddev:"), "Should show standard deviation")
        XCTAssertTrue(detailedOutput.contains("P95:"), "Should show 95th percentile")
        XCTAssertTrue(detailedOutput.contains("CV:"), "Should show coefficient of variation")

        print("Detailed stats:\n\(detailedOutput)")
    }

    /// Tests multiple results table formatting.
    func testMultipleResultsTable() async throws {
        var results: [ComparisonResult] = []

        // Create results at different scales
        for (scale, baseMs, varMs) in [("Small", 5, 2), ("Medium", 10, 4), ("Large", 20, 8)] {
            let result = try await Benchmark("Scaling Test")
                .scale(scale)
                .configuration(.quick)
                .baseline("CPU") {
                    try await Task.sleep(for: .milliseconds(baseMs))
                }
                .variant("GPU") {
                    try await Task.sleep(for: .milliseconds(varMs))
                }
                .run()
            results.append(result)
        }

        let tableOutput = BenchmarkReporter.console.formatConsoleTable(results, title: "Scaling Analysis")

        // Verify table contains all scales
        XCTAssertTrue(tableOutput.contains("Scaling Analysis"), "Should contain title")
        XCTAssertTrue(tableOutput.contains("Small"), "Should contain first scale")
        XCTAssertTrue(tableOutput.contains("Medium"), "Should contain second scale")
        XCTAssertTrue(tableOutput.contains("Large"), "Should contain third scale")

        print("Table output:\n\(tableOutput)")
    }

    /// Tests the runAndExportJSON convenience method.
    func testRunAndExportJSON() async throws {
        let (result, json) = try await Benchmark("Export Test")
            .scale("test")
            .configuration(.quick)
            .baseline("A") { try await Task.sleep(for: .milliseconds(6)) }
            .variant("B") { try await Task.sleep(for: .milliseconds(3)) }
            .runAndExportJSON()

        // Verify result
        XCTAssertEqual(result.name, "Export Test")
        XCTAssertGreaterThan(result.speedup, 1.5)

        // Verify JSON (need to use ISO8601 date strategy)
        XCTAssertTrue(json.contains("Export Test"))
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        XCTAssertNoThrow(try decoder.decode(ComparisonResult.self, from: data))
    }

    /// Tests the printSummary and printDetailed extensions.
    func testPrintExtensions() async throws {
        let result = try await Benchmark("Extension Test")
            .scale("test")
            .configuration(.quick)
            .baseline("Base") { try await Task.sleep(for: .milliseconds(8)) }
            .variant("Fast") { try await Task.sleep(for: .milliseconds(4)) }
            .run()

        // These should print without errors
        result.printSummary()
        result.printDetailed()

        // Test array extension
        [result].printTable(title: "Single Result Table")
    }
}

// MARK: - Test Helpers

/// Thread-safe counter for testing failure patterns.
private actor FailCounter {
    private var count = 0

    func increment() -> Int {
        count += 1
        return count
    }
}

// MARK: - Day 3: Hardware, Test Data, and Storage Tests

extension BenchmarkInfrastructureTests {

    // MARK: - Hardware Info Tests

    /// Tests that hardware info can be captured.
    func testHardwareInfoCapture() {
        let info = HardwareInfo.capture()

        // Verify basic system info is populated
        XCTAssertFalse(info.osVersion.isEmpty, "OS version should be populated")
        XCTAssertFalse(info.cpuModel.isEmpty, "CPU model should be populated")
        XCTAssertFalse(info.architecture.isEmpty, "Architecture should be populated")

        // Verify CPU info
        XCTAssertGreaterThan(info.cpuCoreCount, 0, "Should have at least 1 CPU core")
        XCTAssertGreaterThanOrEqual(info.cpuPerformanceCores, 0)
        XCTAssertGreaterThanOrEqual(info.cpuEfficiencyCores, 0)

        // Verify memory info
        XCTAssertGreaterThan(info.totalRAM, 0, "Total RAM should be > 0")
        XCTAssertGreaterThan(info.availableRAM, 0, "Available RAM should be > 0")

        // Verify GPU info (assuming Metal is available)
        XCTAssertFalse(info.gpuName.isEmpty, "GPU name should be populated")
        XCTAssertGreaterThanOrEqual(info.gpuCoreCount, 0)

        print("Hardware Info:\n\(info.summary)")
    }

    /// Tests hardware info summary formatting.
    func testHardwareInfoSummary() {
        let info = HardwareInfo.capture()
        let summary = info.summary
        let shortSummary = info.shortSummary

        // Summary should contain key information
        XCTAssertTrue(summary.contains("Hardware:"), "Should contain Hardware label")
        XCTAssertTrue(summary.contains("GPU:"), "Should contain GPU label")
        XCTAssertTrue(summary.contains("GB RAM"), "Should contain RAM info")

        XCTAssertFalse(shortSummary.isEmpty, "Short summary should not be empty")

        print("Summary:\n\(summary)")
        print("Short: \(shortSummary)")
    }

    /// Tests hardware info is Codable.
    func testHardwareInfoCodable() throws {
        let info = HardwareInfo.capture()

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(info)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(HardwareInfo.self, from: data)

        XCTAssertEqual(info.cpuModel, decoded.cpuModel)
        XCTAssertEqual(info.cpuCoreCount, decoded.cpuCoreCount)
        XCTAssertEqual(info.totalRAM, decoded.totalRAM)
        XCTAssertEqual(info.gpuName, decoded.gpuName)
    }

    // MARK: - Test Data Generator Tests

    /// Tests random embedding generation.
    func testRandomEmbeddings() {
        let embeddings = TestDataGenerator.randomEmbeddings(count: 100, dimension: 384, seed: 42)

        XCTAssertEqual(embeddings.count, 100)
        XCTAssertTrue(embeddings.allSatisfy { $0.dimension == 384 })

        // Verify reproducibility with same seed
        let embeddings2 = TestDataGenerator.randomEmbeddings(count: 100, dimension: 384, seed: 42)
        XCTAssertEqual(embeddings[0].vector, embeddings2[0].vector, "Same seed should produce same embeddings")
        XCTAssertEqual(embeddings[99].vector, embeddings2[99].vector)

        // Different seed should produce different results
        let embeddings3 = TestDataGenerator.randomEmbeddings(count: 100, dimension: 384, seed: 123)
        XCTAssertNotEqual(embeddings[0].vector, embeddings3[0].vector, "Different seed should produce different embeddings")
    }

    /// Tests clustered embedding generation - the Day 3 exit criteria.
    func testClusteredEmbeddings() {
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: 100,
            seed: 42
        )

        XCTAssertEqual(embeddings.count, 500, "Should have 5 clusters × 100 points = 500 total")
        XCTAssertTrue(embeddings.allSatisfy { $0.dimension == 384 })

        // Verify reproducibility
        let embeddings2 = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: 100,
            seed: 42
        )
        XCTAssertEqual(embeddings[0].vector, embeddings2[0].vector)

        print("Day 3 Exit Criteria - Generated \(embeddings.count) clustered embeddings")
    }

    /// Tests clustered embeddings have cluster structure.
    func testClusteredEmbeddingsStructure() {
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 3,
            pointsPerCluster: 10,
            dimension: 8,  // Small dimension for easier verification
            clusterSpread: 0.01,
            seed: 42
        )

        // Points in same cluster should be close together
        // First 10 points are cluster 0, next 10 are cluster 1, etc.
        let cluster0Point0 = embeddings[0]
        let cluster0Point5 = embeddings[5]
        let cluster1Point0 = embeddings[10]

        let intraClusterDist = cluster0Point0.euclideanDistance(cluster0Point5)
        let interClusterDist = cluster0Point0.euclideanDistance(cluster1Point0)

        // Intra-cluster distance should be smaller than inter-cluster distance
        // (with high probability given the cluster spread)
        print("Intra-cluster distance: \(intraClusterDist)")
        print("Inter-cluster distance: \(interClusterDist)")

        // With low spread, intra should typically be much smaller
        // This is a probabilistic test but should pass reliably with spread=0.01
        XCTAssertLessThan(intraClusterDist, interClusterDist * 2,
                          "Points in same cluster should be relatively close")
    }

    // MARK: - Benchmark Storage Tests

    /// Tests saving and loading a result.
    func testStorageSaveAndLoad() async throws {
        // Create a temporary directory for this test
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("BenchmarkStorageTest_\(UUID().uuidString)")
        let storage = BenchmarkStorage(directory: tempDir)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // Create a benchmark result
        let result = try await Benchmark("Storage Test")
            .scale("100 items")
            .configuration(.quick)
            .baseline("Old") { try await Task.sleep(for: .milliseconds(8)) }
            .variant("New") { try await Task.sleep(for: .milliseconds(4)) }
            .run()

        // Save
        let savedURL = try storage.save(result)
        XCTAssertTrue(FileManager.default.fileExists(atPath: savedURL.path))

        // Load
        let loaded = try storage.loadPrevious(name: "Storage Test", count: 1)
        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded[0].name, result.name)
        XCTAssertEqual(loaded[0].scale, result.scale)

        print("Day 3 Exit Criteria - Saved to: \(savedURL.lastPathComponent)")
    }

    /// Tests saving multiple results as a run.
    func testStorageSaveRun() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("BenchmarkRunTest_\(UUID().uuidString)")
        let storage = BenchmarkStorage(directory: tempDir)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        var results: [ComparisonResult] = []
        for scale in ["Small", "Medium"] {
            let result = try await Benchmark("Run Test")
                .scale(scale)
                .configuration(.quick)
                .baseline("A") { try await Task.sleep(for: .milliseconds(4)) }
                .variant("B") { try await Task.sleep(for: .milliseconds(2)) }
                .run()
            results.append(result)
        }

        let savedURL = try storage.saveRun(results, name: "TestRun")
        XCTAssertTrue(FileManager.default.fileExists(atPath: savedURL.path))
    }

    /// Tests regression detection.
    func testRegressionDetection() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("RegressionTest_\(UUID().uuidString)")
        let storage = BenchmarkStorage(directory: tempDir)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // First result - no baseline
        let result1 = try await Benchmark("Regression Test")
            .scale("test")
            .configuration(.quick)
            .baseline("A") { try await Task.sleep(for: .milliseconds(10)) }
            .variant("B") { try await Task.sleep(for: .milliseconds(5)) }
            .run()

        let status1 = try storage.detectRegression(current: result1)
        XCTAssertEqual(status1, .noBaseline)

        // Save first result
        try storage.save(result1)

        // Second result - similar performance (stable)
        let result2 = try await Benchmark("Regression Test")
            .scale("test")
            .configuration(.quick)
            .baseline("A") { try await Task.sleep(for: .milliseconds(10)) }
            .variant("B") { try await Task.sleep(for: .milliseconds(5)) }
            .run()

        let status2 = try storage.detectRegression(current: result2)
        print("Regression status: \(status2.description)")

        // Should be either stable or slight improvement/regression
        switch status2 {
        case .noBaseline:
            XCTFail("Should have a baseline now")
        case .improved, .stable, .regressed:
            // All acceptable - timing can vary slightly
            break
        }
    }

    /// Tests regression status descriptions.
    func testRegressionStatusDescriptions() {
        XCTAssertEqual(RegressionStatus.noBaseline.description, "No baseline available")
        XCTAssertEqual(RegressionStatus.stable.description, "Stable (within threshold)")
        XCTAssertTrue(RegressionStatus.improved(percentage: 15.5).description.contains("15.5%"))
        XCTAssertTrue(RegressionStatus.regressed(percentage: 8.2).description.contains("8.2%"))
    }

    // MARK: - Day 3 Exit Criteria Integration Test

    /// Full Day 3 exit criteria test.
    ///
    /// From BENCHMARK_SUITE_PLAN.md:
    /// ```swift
    /// let embeddings = TestDataGenerator.clusteredEmbeddings(
    ///     clusterCount: 5, pointsPerCluster: 100, seed: 42
    /// )
    /// // Result includes hardware info, saves to JSON
    /// try BenchmarkStorage().save(result)
    /// ```
    func testDay3ExitCriteria() async throws {
        // Step 1: Generate clustered embeddings
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: 100,
            seed: 42
        )
        XCTAssertEqual(embeddings.count, 500)

        // Step 2: Create a benchmark using the embeddings
        let result = try await Benchmark("Day 3 Integration")
            .scale("500 points")
            .configuration(.quick)
            .baseline("Process") {
                // Simulate processing embeddings
                var sum: Float = 0
                for embedding in embeddings {
                    sum += embedding.l2Norm
                }
                _ = sum // Use the result to prevent optimization
            }
            .variant("FastProcess") {
                // Faster simulated processing
                _ = embeddings.count
            }
            .run()

        XCTAssertGreaterThan(result.speedup, 1.0)

        // Step 3: Capture hardware info
        let hardwareInfo = HardwareInfo.capture()
        XCTAssertFalse(hardwareInfo.cpuModel.isEmpty)

        // Step 4: Save to storage
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("Day3ExitCriteria_\(UUID().uuidString)")
        let storage = BenchmarkStorage(directory: tempDir)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let savedURL = try storage.save(result)
        XCTAssertTrue(FileManager.default.fileExists(atPath: savedURL.path))

        print("═══════════════════════════════════════════════════════════")
        print("Day 3 Exit Criteria - PASSED")
        print("═══════════════════════════════════════════════════════════")
        print("• Embeddings: \(embeddings.count) clustered points")
        print("• Hardware: \(hardwareInfo.shortSummary)")
        print("• Speedup: \(result.speedupFormatted)")
        print("• Saved to: \(savedURL.lastPathComponent)")
        print("═══════════════════════════════════════════════════════════")
    }
}
