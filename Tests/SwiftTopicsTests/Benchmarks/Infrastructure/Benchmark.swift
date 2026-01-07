// Benchmark.swift
// SwiftTopicsTests
//
// Fluent DSL builder for constructing and running benchmarks.
// Part of Day 2: Core Features

import Foundation

// MARK: - Benchmark Builder

/// Fluent builder for constructing and running benchmarks.
///
/// Provides a chainable API for configuring benchmark parameters and operations.
/// Delegates actual measurement to `BenchmarkHarness`.
///
/// ## Usage
/// ```swift
/// let result = try await Benchmark("Test")
///     .scale("1000 points")
///     .iterations(10)
///     .baseline("CPU") { try await cpuOperation() }
///     .variant("GPU") { try await gpuOperation() }
///     .runAndReport()
/// ```
///
/// ## Thread Safety
/// The builder is designed for sequential configuration followed by a single `run()` call.
/// All operations captured are `@Sendable` for safe async execution.
public final class Benchmark: @unchecked Sendable {

    // MARK: - Properties

    private let benchmarkName: String
    private var scaleDescription: String = ""
    private var config: BenchmarkConfiguration = .default
    private var baselineOperation: (@Sendable () async throws -> Void)?
    private var variantOperation: (@Sendable () async throws -> Void)?
    private var baselineLabel: String = "Baseline"
    private var variantLabel: String = "Variant"

    // MARK: - Initialization

    /// Creates a new benchmark builder with the given name.
    ///
    /// - Parameter name: Descriptive name for this benchmark (e.g., "HDBSCAN Clustering")
    public init(_ name: String) {
        self.benchmarkName = name
    }

    // MARK: - Scale Configuration

    /// Sets the scale descriptor for this benchmark.
    ///
    /// The scale describes the problem size being benchmarked.
    ///
    /// - Parameter scale: Scale descriptor (e.g., "1000 points", "500 documents")
    /// - Returns: Self for chaining
    @discardableResult
    public func scale(_ scale: String) -> Self {
        self.scaleDescription = scale
        return self
    }

    // MARK: - Iteration Configuration

    /// Sets the number of measured iterations.
    ///
    /// More iterations provide more statistical accuracy but take longer.
    /// Default is 10 iterations.
    ///
    /// - Parameter count: Number of measured iterations
    /// - Returns: Self for chaining
    @discardableResult
    public func iterations(_ count: Int) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: count,
            warmupIterations: config.warmupIterations,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: config.seed
        )
        return self
    }

    /// Sets the number of warmup iterations.
    ///
    /// Warmup iterations are run before measurement to prime caches
    /// and allow JIT compilation to stabilize.
    /// Default is 3 warmup iterations.
    ///
    /// - Parameter count: Number of warmup iterations
    /// - Returns: Self for chaining
    @discardableResult
    public func warmup(_ count: Int) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: count,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: config.seed
        )
        return self
    }

    /// Sets the random seed for reproducible test data.
    ///
    /// Using the same seed ensures consistent benchmark conditions
    /// across runs.
    ///
    /// - Parameter seed: Random seed value
    /// - Returns: Self for chaining
    @discardableResult
    public func seed(_ seed: UInt64) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: config.warmupIterations,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: seed
        )
        return self
    }

    /// Sets whether to use interleaved execution.
    ///
    /// Interleaved mode alternates baseline and variant operations (B-V-B-V...)
    /// to reduce thermal and cache bias. Default is true.
    ///
    /// - Parameter interleaved: Whether to interleave operations
    /// - Returns: Self for chaining
    @discardableResult
    public func interleaved(_ interleaved: Bool) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: config.warmupIterations,
            timeout: config.timeout,
            interleaved: interleaved,
            seed: config.seed
        )
        return self
    }

    /// Sets the complete benchmark configuration.
    ///
    /// Use this to apply a preset configuration or fully customize all options.
    ///
    /// - Parameter config: Complete benchmark configuration
    /// - Returns: Self for chaining
    @discardableResult
    public func configuration(_ config: BenchmarkConfiguration) -> Self {
        self.config = config
        return self
    }

    // MARK: - Operation Configuration

    /// Sets the baseline operation to measure.
    ///
    /// The baseline is typically the slower implementation (e.g., CPU)
    /// against which the variant is compared.
    ///
    /// - Parameters:
    ///   - label: Label for the baseline (default: "Baseline")
    ///   - operation: Async operation to measure
    /// - Returns: Self for chaining
    @discardableResult
    public func baseline(
        _ label: String = "Baseline",
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Self {
        self.baselineLabel = label
        self.baselineOperation = operation
        return self
    }

    /// Sets the variant operation to measure.
    ///
    /// The variant is typically the faster implementation (e.g., GPU)
    /// being compared to the baseline.
    ///
    /// - Parameters:
    ///   - label: Label for the variant (default: "Variant")
    ///   - operation: Async operation to measure
    /// - Returns: Self for chaining
    @discardableResult
    public func variant(
        _ label: String = "Variant",
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Self {
        self.variantLabel = label
        self.variantOperation = operation
        return self
    }

    // MARK: - Execution

    /// Runs the benchmark and returns the comparison result.
    ///
    /// - Returns: Comparison result with timing statistics and speedup metrics
    /// - Throws: `BenchmarkError.missingOperation` if baseline or variant not set
    public func run() async throws -> ComparisonResult {
        guard let baseline = baselineOperation,
              let variant = variantOperation else {
            throw BenchmarkError.missingOperation
        }

        return try await BenchmarkHarness.compare(
            name: benchmarkName,
            scale: scaleDescription,
            configuration: config,
            baseline: (baselineLabel, baseline),
            variant: (variantLabel, variant)
        )
    }

    /// Runs the benchmark and prints formatted results.
    ///
    /// This is the most common entry point for interactive benchmarking.
    /// After measurement, results are formatted and printed to the console.
    ///
    /// - Parameter reporter: Reporter to use for output (default: `.console`)
    /// - Returns: Comparison result with timing statistics and speedup metrics
    /// - Throws: `BenchmarkError.missingOperation` if baseline or variant not set
    @discardableResult
    public func runAndReport(to reporter: BenchmarkReporter = .console) async throws -> ComparisonResult {
        let result = try await run()
        reporter.report(result)
        return result
    }

    /// Runs the benchmark and returns JSON output.
    ///
    /// Convenience method for getting machine-readable results.
    ///
    /// - Returns: Tuple of (result, JSON string)
    /// - Throws: `BenchmarkError.missingOperation` if baseline or variant not set
    public func runAndExportJSON() async throws -> (result: ComparisonResult, json: String) {
        let result = try await run()
        let json = BenchmarkReporter.json.formatJSON(result)
        return (result, json)
    }
}
