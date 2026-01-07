// StatisticsCalculator.swift
// SwiftTopicsTests
//
// Statistical computations for benchmark timing data.
// Part of Day 1: Foundation

import Foundation

// MARK: - Statistics Calculator

/// Computes comprehensive statistics from timing samples.
///
/// ## Computed Metrics
/// - Central tendency: mean, median
/// - Spread: stddev, variance, min, max
/// - Percentiles: p25, p75, p95, p99
/// - Derived: IQR, coefficient of variation
///
/// ## Usage
/// ```swift
/// let stats = StatisticsCalculator.compute(
///     samples: durations,
///     failedCount: 0
/// )
/// print("Median: \(stats.medianMilliseconds)ms")
/// ```
public enum StatisticsCalculator {

    /// Computes full statistics suite from timing samples.
    ///
    /// - Parameters:
    ///   - samples: Array of timing measurements (must not be empty)
    ///   - failedCount: Number of failed iterations to record
    /// - Returns: Computed timing statistics
    /// - Precondition: samples must not be empty
    public static func compute(
        samples: [Duration],
        failedCount: Int
    ) -> TimingStatistics {
        precondition(!samples.isEmpty, "Cannot compute statistics from empty samples")

        // Convert to nanoseconds for computation
        let nanoseconds = samples.map { $0.nanoseconds }
        let sorted = nanoseconds.sorted()
        let count = sorted.count

        // Central tendency
        let sum = nanoseconds.reduce(0, +)
        let mean = sum / Int64(count)
        let median = percentile(sorted: sorted, p: 0.50)

        // Spread
        let minVal = sorted.first!
        let maxVal = sorted.last!

        // Variance and standard deviation
        let squaredDiffs = nanoseconds.map { ns -> Double in
            let diff = Double(ns - mean)
            return diff * diff
        }
        let variance = squaredDiffs.reduce(0, +) / Double(count)
        let stddev = Int64(variance.squareRoot())

        // Percentiles
        let p25 = percentile(sorted: sorted, p: 0.25)
        let p75 = percentile(sorted: sorted, p: 0.75)
        let p95 = percentile(sorted: sorted, p: 0.95)
        let p99 = percentile(sorted: sorted, p: 0.99)

        return TimingStatistics(
            sampleCount: count,
            failedCount: failedCount,
            mean: .nanoseconds(mean),
            median: .nanoseconds(median),
            min: .nanoseconds(minVal),
            max: .nanoseconds(maxVal),
            standardDeviation: .nanoseconds(stddev),
            varianceNanosSquared: variance,
            p25: .nanoseconds(p25),
            p75: .nanoseconds(p75),
            p95: .nanoseconds(p95),
            p99: .nanoseconds(p99)
        )
    }

    // MARK: - Private Helpers

    /// Computes percentile value from sorted array using linear interpolation.
    ///
    /// Uses the "exclusive" percentile method (R-6 in R terminology),
    /// which is commonly used for sample data.
    ///
    /// - Parameters:
    ///   - sorted: Sorted array of values
    ///   - p: Percentile (0.0 to 1.0)
    /// - Returns: Interpolated percentile value
    private static func percentile(sorted: [Int64], p: Double) -> Int64 {
        let count = sorted.count

        // Edge cases
        if count == 1 {
            return sorted[0]
        }

        // Calculate position (0-indexed)
        let position = p * Double(count - 1)
        let lowerIndex = Int(position.rounded(.down))
        let upperIndex = min(lowerIndex + 1, count - 1)
        let fraction = position - Double(lowerIndex)

        // Linear interpolation
        let lower = sorted[lowerIndex]
        let upper = sorted[upperIndex]

        return lower + Int64(Double(upper - lower) * fraction)
    }
}

// MARK: - Formatting Helpers

extension TimingStatistics {

    /// Formats the duration in human-readable form.
    ///
    /// - Nanoseconds: "123 ns"
    /// - Microseconds: "123.4 µs"
    /// - Milliseconds: "123.4 ms"
    /// - Seconds: "1.234 s"
    public static func format(_ duration: Duration) -> String {
        let ns = duration.nanoseconds

        if ns < 1_000 {
            return "\(ns) ns"
        } else if ns < 1_000_000 {
            return String(format: "%.1f µs", Double(ns) / 1_000.0)
        } else if ns < 1_000_000_000 {
            return String(format: "%.1f ms", Double(ns) / 1_000_000.0)
        } else {
            return String(format: "%.3f s", Double(ns) / 1_000_000_000.0)
        }
    }

    /// Human-readable summary of statistics.
    public var summary: String {
        """
        Samples: \(sampleCount) (failed: \(failedCount))
        Median: \(Self.format(median))
        Mean: \(Self.format(mean))
        Stddev: \(Self.format(standardDeviation))
        Range: \(Self.format(min)) - \(Self.format(max))
        P95: \(Self.format(p95))
        CV: \(String(format: "%.1f%%", coefficientOfVariation * 100))
        """
    }
}
