// BenchmarkHarness.swift
// SwiftTopicsTests
//
// Core benchmark execution engine using ContinuousClock.
// Part of Day 1: Foundation

import Foundation

// MARK: - Benchmark Harness

/// Core benchmark execution engine.
///
/// Provides high-precision timing using `ContinuousClock` with support for:
/// - Warmup iterations (excluded from statistics)
/// - Interleaved execution (B-V-B-V pattern for paired comparison)
/// - Failure tracking and recovery
///
/// ## Interleaved Execution
/// For paired comparisons, the harness runs baseline and variant operations
/// in an interleaved pattern to reduce thermal and cache bias:
/// ```
/// Warmup:   B V B V B V (3 warmup each)
/// Measured: B V B V B V B V B V ... (10 each)
/// ```
///
/// ## Usage
/// ```swift
/// let result = try await BenchmarkHarness.compare(
///     name: "Test",
///     scale: "1000 points",
///     configuration: .default,
///     baseline: ("CPU", { try await cpuOperation() }),
///     variant: ("GPU", { try await gpuOperation() })
/// )
/// print("Speedup: \(result.speedup)x")
/// ```
public enum BenchmarkHarness {

    /// The clock used for all timing measurements.
    private static let clock = ContinuousClock()

    // MARK: - Single Measurement

    /// Measures a single operation.
    ///
    /// - Parameters:
    ///   - name: Name of the benchmark
    ///   - label: Label for this operation (e.g., "CPU", "GPU")
    ///   - scale: Scale descriptor (e.g., "1000 points")
    ///   - configuration: Benchmark configuration
    ///   - operation: The async operation to measure
    /// - Returns: Benchmark result with timing statistics
    /// - Throws: `BenchmarkError.allIterationsFailed` if all iterations fail
    public static func measure(
        name: String,
        label: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        operation: @escaping @Sendable () async throws -> Void
    ) async throws -> BenchmarkResult {

        var samples: [Duration] = []
        var failures: [BenchmarkFailure] = []

        // Warmup iterations (not measured)
        for _ in 0..<configuration.warmupIterations {
            do {
                try await operation()
            } catch {
                // Warmup failures are ignored - operation may need warmup
            }
        }

        // Measured iterations
        for iteration in 0..<configuration.iterations {
            do {
                let elapsed = try await clock.measure {
                    try await operation()
                }
                samples.append(elapsed)
            } catch {
                failures.append(BenchmarkFailure(
                    iteration: iteration,
                    error: error.localizedDescription
                ))
            }
        }

        // Check for complete failure
        if samples.isEmpty {
            throw BenchmarkError.allIterationsFailed(
                failures.map { $0.error }
            )
        }

        // Compute statistics
        let statistics = StatisticsCalculator.compute(
            samples: samples,
            failedCount: failures.count
        )

        return BenchmarkResult(
            name: name,
            label: label,
            scale: scale,
            statistics: statistics,
            timestamp: Date(),
            samples: samples,
            failures: failures
        )
    }

    // MARK: - Paired Comparison

    /// Compares baseline vs variant with paired/interleaved execution.
    ///
    /// This is the primary entry point for CPU vs GPU comparisons.
    /// Uses interleaved execution to reduce thermal and cache bias.
    ///
    /// - Parameters:
    ///   - name: Name of the benchmark
    ///   - scale: Scale descriptor (e.g., "1000 points")
    ///   - configuration: Benchmark configuration
    ///   - baseline: Tuple of (label, operation) for baseline
    ///   - variant: Tuple of (label, operation) for variant
    /// - Returns: Comparison result with speedup metrics
    /// - Throws: `BenchmarkError.allIterationsFailed` if all iterations of either fail
    public static func compare(
        name: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        baseline: (label: String, operation: @Sendable () async throws -> Void),
        variant: (label: String, operation: @Sendable () async throws -> Void)
    ) async throws -> ComparisonResult {

        if configuration.interleaved {
            return try await compareInterleaved(
                name: name,
                scale: scale,
                configuration: configuration,
                baseline: baseline,
                variant: variant
            )
        } else {
            return try await compareSequential(
                name: name,
                scale: scale,
                configuration: configuration,
                baseline: baseline,
                variant: variant
            )
        }
    }

    // MARK: - Interleaved Execution

    /// Executes baseline and variant in interleaved pattern (B-V-B-V...).
    private static func compareInterleaved(
        name: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        baseline: (label: String, operation: @Sendable () async throws -> Void),
        variant: (label: String, operation: @Sendable () async throws -> Void)
    ) async throws -> ComparisonResult {

        var baselineSamples: [Duration] = []
        var variantSamples: [Duration] = []
        var baselineFailures: [BenchmarkFailure] = []
        var variantFailures: [BenchmarkFailure] = []

        // Warmup phase (interleaved)
        for _ in 0..<configuration.warmupIterations {
            // Baseline warmup
            try? await baseline.operation()
            // Variant warmup
            try? await variant.operation()
        }

        // Measured phase (interleaved)
        for iteration in 0..<configuration.iterations {
            // Measure baseline
            do {
                let elapsed = try await clock.measure {
                    try await baseline.operation()
                }
                baselineSamples.append(elapsed)
            } catch {
                baselineFailures.append(BenchmarkFailure(
                    iteration: iteration,
                    error: error.localizedDescription
                ))
            }

            // Measure variant
            do {
                let elapsed = try await clock.measure {
                    try await variant.operation()
                }
                variantSamples.append(elapsed)
            } catch {
                variantFailures.append(BenchmarkFailure(
                    iteration: iteration,
                    error: error.localizedDescription
                ))
            }
        }

        // Check for complete failures
        if baselineSamples.isEmpty {
            throw BenchmarkError.allIterationsFailed(
                baselineFailures.map { "\(baseline.label): \($0.error)" }
            )
        }
        if variantSamples.isEmpty {
            throw BenchmarkError.allIterationsFailed(
                variantFailures.map { "\(variant.label): \($0.error)" }
            )
        }

        // Build results
        let baselineResult = BenchmarkResult(
            name: name,
            label: baseline.label,
            scale: scale,
            statistics: StatisticsCalculator.compute(
                samples: baselineSamples,
                failedCount: baselineFailures.count
            ),
            timestamp: Date(),
            samples: baselineSamples,
            failures: baselineFailures
        )

        let variantResult = BenchmarkResult(
            name: name,
            label: variant.label,
            scale: scale,
            statistics: StatisticsCalculator.compute(
                samples: variantSamples,
                failedCount: variantFailures.count
            ),
            timestamp: Date(),
            samples: variantSamples,
            failures: variantFailures
        )

        return ComparisonResult(
            name: name,
            scale: scale,
            baseline: baselineResult,
            variant: variantResult
        )
    }

    // MARK: - Sequential Execution

    /// Executes baseline fully, then variant fully.
    private static func compareSequential(
        name: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        baseline: (label: String, operation: @Sendable () async throws -> Void),
        variant: (label: String, operation: @Sendable () async throws -> Void)
    ) async throws -> ComparisonResult {

        // Run baseline
        let baselineResult = try await measure(
            name: name,
            label: baseline.label,
            scale: scale,
            configuration: configuration,
            operation: baseline.operation
        )

        // Run variant
        let variantResult = try await measure(
            name: name,
            label: variant.label,
            scale: scale,
            configuration: configuration,
            operation: variant.operation
        )

        return ComparisonResult(
            name: name,
            scale: scale,
            baseline: baselineResult,
            variant: variantResult
        )
    }

    // MARK: - Scaling Suite

    /// Runs multiple comparisons at different scales.
    ///
    /// Useful for generating scaling curves to understand algorithmic complexity.
    ///
    /// - Parameters:
    ///   - name: Name of the benchmark suite
    ///   - scales: Array of (label, setup) tuples where setup returns baseline/variant operations
    ///   - configuration: Benchmark configuration
    /// - Returns: Array of comparison results for each scale
    public static func scalingSuite(
        name: String,
        scales: [(
            label: String,
            setup: @Sendable () async throws -> (
                baseline: @Sendable () async throws -> Void,
                variant: @Sendable () async throws -> Void
            )
        )],
        configuration: BenchmarkConfiguration
    ) async throws -> [ComparisonResult] {

        var results: [ComparisonResult] = []

        for (scaleLabel, setup) in scales {
            let (baselineOp, variantOp) = try await setup()

            let result = try await compare(
                name: name,
                scale: scaleLabel,
                configuration: configuration,
                baseline: ("Baseline", baselineOp),
                variant: ("Variant", variantOp)
            )

            results.append(result)
        }

        return results
    }
}

// MARK: - ComparisonResult Helpers

extension ComparisonResult {

    /// Formatted speedup string (e.g., "42.3x").
    public var speedupFormatted: String {
        if speedup.isInfinite {
            return "∞"
        }
        return String(format: "%.1fx", speedup)
    }

    /// Quick summary for console output.
    public var summary: String {
        """
        \(name) [\(scale)]
        \(baseline.label): \(TimingStatistics.format(baseline.statistics.median))
        \(variant.label): \(TimingStatistics.format(variant.statistics.median))
        Speedup: \(speedupFormatted)\(isSignificant ? " ✓" : "")
        """
    }
}
