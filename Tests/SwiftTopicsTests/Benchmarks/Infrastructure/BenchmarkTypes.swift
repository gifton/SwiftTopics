// BenchmarkTypes.swift
// SwiftTopicsTests
//
// Core data structures for the benchmark suite.
// Part of Day 1: Foundation

import Foundation

// MARK: - Benchmark Configuration

/// Configuration for a benchmark run.
///
/// Controls the number of iterations, warmup behavior, and execution strategy.
///
/// ## Presets
/// - `.default`: 10 iterations, 3 warmup, interleaved
/// - `.quick`: 5 iterations, 2 warmup, interleaved
/// - `.thorough`: 25 iterations, 5 warmup, interleaved
public struct BenchmarkConfiguration: Sendable {

    /// Number of measured iterations (after warmup).
    public let iterations: Int

    /// Number of warmup iterations (excluded from stats).
    public let warmupIterations: Int

    /// Optional timeout per iteration (nil = no timeout).
    public let timeout: Duration?

    /// Whether to run baseline and variant interleaved (B-V-B-V pattern).
    ///
    /// Interleaved execution reduces thermal and cache bias between runs.
    public let interleaved: Bool

    /// Random seed for reproducible test data.
    public let seed: UInt64

    /// Creates a benchmark configuration.
    ///
    /// - Parameters:
    ///   - iterations: Number of measured iterations (default: 10)
    ///   - warmupIterations: Number of warmup iterations (default: 3)
    ///   - timeout: Optional timeout per iteration
    ///   - interleaved: Whether to interleave baseline/variant runs (default: true)
    ///   - seed: Random seed for reproducibility (default: 42)
    public init(
        iterations: Int = 10,
        warmupIterations: Int = 3,
        timeout: Duration? = nil,
        interleaved: Bool = true,
        seed: UInt64 = 42
    ) {
        precondition(iterations >= 1, "Must have at least 1 iteration")
        precondition(warmupIterations >= 0, "Warmup iterations must be non-negative")

        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.timeout = timeout
        self.interleaved = interleaved
        self.seed = seed
    }

    /// Default configuration: 10 iterations, 3 warmup, interleaved.
    public static let `default` = BenchmarkConfiguration(
        iterations: 10,
        warmupIterations: 3,
        timeout: nil,
        interleaved: true,
        seed: 42
    )

    /// Quick configuration: 5 iterations, 2 warmup.
    public static let quick = BenchmarkConfiguration(
        iterations: 5,
        warmupIterations: 2,
        timeout: nil,
        interleaved: true,
        seed: 42
    )

    /// Thorough configuration: 25 iterations, 5 warmup.
    public static let thorough = BenchmarkConfiguration(
        iterations: 25,
        warmupIterations: 5,
        timeout: nil,
        interleaved: true,
        seed: 42
    )
}

// MARK: - Timing Statistics

/// Comprehensive timing statistics for a benchmark.
///
/// Provides central tendency, spread, and percentile metrics for detailed
/// performance analysis.
///
/// ## Key Metrics
/// - `mean`, `median`: Central tendency measures
/// - `standardDeviation`, `variance`: Spread measures
/// - `p25`, `p75`, `p95`, `p99`: Percentile distribution
/// - `coefficientOfVariation`: Normalized variability (stddev/mean)
public struct TimingStatistics: Sendable, Codable {

    // MARK: - Core Metrics

    /// Number of successful samples.
    public let sampleCount: Int

    /// Number of failed iterations.
    public let failedCount: Int

    // MARK: - Central Tendency

    /// Arithmetic mean of all samples.
    public let mean: Duration

    /// Median (50th percentile) of all samples.
    public let median: Duration

    // MARK: - Spread

    /// Minimum sample value.
    public let min: Duration

    /// Maximum sample value.
    public let max: Duration

    /// Standard deviation of samples.
    public let standardDeviation: Duration

    /// Variance in nanoseconds squared.
    public let varianceNanosSquared: Double

    // MARK: - Percentiles

    /// 25th percentile (first quartile).
    public let p25: Duration

    /// 75th percentile (third quartile).
    public let p75: Duration

    /// 95th percentile.
    public let p95: Duration

    /// 99th percentile.
    public let p99: Duration

    // MARK: - Derived Metrics

    /// Interquartile range (p75 - p25).
    public var iqr: Duration {
        Duration.nanoseconds(p75.nanoseconds - p25.nanoseconds)
    }

    /// Coefficient of variation (stddev / mean).
    ///
    /// A normalized measure of dispersion. Lower is better.
    /// - < 0.1: Very consistent
    /// - 0.1-0.3: Acceptable
    /// - > 0.3: High variability
    public var coefficientOfVariation: Double {
        guard mean.nanoseconds > 0 else { return 0 }
        return Double(standardDeviation.nanoseconds) / Double(mean.nanoseconds)
    }

    // MARK: - Convenience Accessors

    /// Median in milliseconds.
    public var medianMilliseconds: Double {
        Double(median.nanoseconds) / 1_000_000.0
    }

    /// Mean in milliseconds.
    public var meanMilliseconds: Double {
        Double(mean.nanoseconds) / 1_000_000.0
    }

    /// Creates timing statistics from pre-computed values.
    public init(
        sampleCount: Int,
        failedCount: Int,
        mean: Duration,
        median: Duration,
        min: Duration,
        max: Duration,
        standardDeviation: Duration,
        varianceNanosSquared: Double,
        p25: Duration,
        p75: Duration,
        p95: Duration,
        p99: Duration
    ) {
        self.sampleCount = sampleCount
        self.failedCount = failedCount
        self.mean = mean
        self.median = median
        self.min = min
        self.max = max
        self.standardDeviation = standardDeviation
        self.varianceNanosSquared = varianceNanosSquared
        self.p25 = p25
        self.p75 = p75
        self.p95 = p95
        self.p99 = p99
    }
}

// MARK: - Duration Helpers

extension Duration {
    /// Total nanoseconds as Int64.
    var nanoseconds: Int64 {
        let (seconds, attoseconds) = self.components
        return seconds * 1_000_000_000 + attoseconds / 1_000_000_000
    }
}

// MARK: - Benchmark Failure

/// Records a failed iteration without stopping the benchmark.
public struct BenchmarkFailure: Sendable, Codable {

    /// Which iteration failed (0-based).
    public let iteration: Int

    /// Error description.
    public let error: String

    /// When the failure occurred.
    public let timestamp: Date

    /// Creates a benchmark failure record.
    public init(iteration: Int, error: String, timestamp: Date = Date()) {
        self.iteration = iteration
        self.error = error
        self.timestamp = timestamp
    }
}

// MARK: - Benchmark Result

/// Result of a single benchmark (one operation, one configuration).
public struct BenchmarkResult: Sendable, Codable {

    /// Name of the benchmark.
    public let name: String

    /// Label for this variant (e.g., "CPU", "GPU").
    public let label: String

    /// Scale descriptor (e.g., "1000 points").
    public let scale: String

    /// Computed statistics.
    public let statistics: TimingStatistics

    /// When the benchmark was run.
    public let timestamp: Date

    /// Individual timing samples (for detailed analysis).
    public let samples: [Duration]

    /// Any failures that occurred during measurement.
    public let failures: [BenchmarkFailure]

    /// Creates a benchmark result.
    public init(
        name: String,
        label: String,
        scale: String,
        statistics: TimingStatistics,
        timestamp: Date = Date(),
        samples: [Duration],
        failures: [BenchmarkFailure]
    ) {
        self.name = name
        self.label = label
        self.scale = scale
        self.statistics = statistics
        self.timestamp = timestamp
        self.samples = samples
        self.failures = failures
    }
}

// MARK: - Comparison Result

/// Result of comparing baseline vs variant (e.g., CPU vs GPU).
///
/// Provides speedup metrics and statistical significance assessment.
public struct ComparisonResult: Sendable, Codable {

    /// Name of the benchmark.
    public let name: String

    /// Scale descriptor (e.g., "1000 points").
    public let scale: String

    /// Baseline result (e.g., CPU).
    public let baseline: BenchmarkResult

    /// Variant result (e.g., GPU).
    public let variant: BenchmarkResult

    /// When the comparison was run.
    public let timestamp: Date

    // MARK: - Speedup Metrics

    /// Speedup ratio using median (baseline.median / variant.median).
    ///
    /// - Returns: > 1.0 means variant is faster
    public var speedup: Double {
        guard variant.statistics.median.nanoseconds > 0 else { return .infinity }
        return Double(baseline.statistics.median.nanoseconds) /
               Double(variant.statistics.median.nanoseconds)
    }

    /// Speedup ratio using mean instead of median.
    public var speedupMean: Double {
        guard variant.statistics.mean.nanoseconds > 0 else { return .infinity }
        return Double(baseline.statistics.mean.nanoseconds) /
               Double(variant.statistics.mean.nanoseconds)
    }

    /// Whether the variant is statistically significantly faster.
    ///
    /// Requires:
    /// - Speedup > 1.2x
    /// - Both baseline and variant have CV < 0.3
    public var isSignificant: Bool {
        speedup > 1.2 &&
        baseline.statistics.coefficientOfVariation < 0.3 &&
        variant.statistics.coefficientOfVariation < 0.3
    }

    /// Creates a comparison result.
    public init(
        name: String,
        scale: String,
        baseline: BenchmarkResult,
        variant: BenchmarkResult,
        timestamp: Date = Date()
    ) {
        self.name = name
        self.scale = scale
        self.baseline = baseline
        self.variant = variant
        self.timestamp = timestamp
    }
}

// MARK: - Benchmark Error

/// Errors that can occur during benchmarking.
public enum BenchmarkError: Error, Sendable {

    /// No operation provided to benchmark.
    case missingOperation

    /// GPU is not available (test should fail).
    case gpuUnavailable

    /// All iterations failed.
    case allIterationsFailed([String])

    /// Fixture file not found and cannot be generated.
    case fixtureNotFound(String)

    /// Configuration is invalid.
    case invalidConfiguration(String)
}

extension BenchmarkError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingOperation:
            return "No operation provided to benchmark"
        case .gpuUnavailable:
            return "GPU is not available for benchmarking"
        case .allIterationsFailed(let errors):
            return "All iterations failed: \(errors.joined(separator: ", "))"
        case .fixtureNotFound(let name):
            return "Fixture not found: \(name)"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        }
    }
}
