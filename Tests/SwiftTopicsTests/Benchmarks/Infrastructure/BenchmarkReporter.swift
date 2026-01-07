// BenchmarkReporter.swift
// SwiftTopicsTests
//
// Console and JSON output formatting for benchmark results.
// Part of Day 2: Core Features

import Foundation

// MARK: - Benchmark Reporter

/// Formats and outputs benchmark results.
///
/// Supports two output formats:
/// - **Console**: Pretty-printed tables with box-drawing characters
/// - **JSON**: Machine-readable format for storage and analysis
///
/// ## Usage
/// ```swift
/// let result = try await Benchmark("Test").run()
/// BenchmarkReporter.console.report(result)
/// ```
public final class BenchmarkReporter: Sendable {

    // MARK: - Format

    /// Output format for benchmark results.
    public enum Format: Sendable {
        /// Pretty-printed console tables with box-drawing characters.
        case console
        /// Machine-readable JSON format.
        case json
    }

    // MARK: - Presets

    /// Console reporter with pretty tables.
    public static let console = BenchmarkReporter(format: .console)

    /// JSON reporter for machine-readable output.
    public static let json = BenchmarkReporter(format: .json)

    // MARK: - Properties

    private let format: Format

    // MARK: - Initialization

    /// Creates a reporter with the specified format.
    ///
    /// - Parameter format: Output format to use
    public init(format: Format) {
        self.format = format
    }

    // MARK: - Single Result Reporting

    /// Reports a single comparison result.
    ///
    /// - Parameter result: Comparison result to report
    public func report(_ result: ComparisonResult) {
        switch format {
        case .console:
            print(formatConsole(result))
        case .json:
            print(formatJSON(result))
        }
    }

    /// Reports a comparison result with detailed statistics.
    ///
    /// Shows full statistical breakdown including percentiles and variance.
    ///
    /// - Parameter result: Comparison result to report
    public func reportDetailed(_ result: ComparisonResult) {
        switch format {
        case .console:
            print(formatConsole(result))
            print(formatDetailedStats(result))
        case .json:
            print(formatJSON(result))
        }
    }

    // MARK: - Multiple Results Reporting

    /// Reports multiple comparison results as a table.
    ///
    /// Useful for scaling analysis across different problem sizes.
    ///
    /// - Parameters:
    ///   - results: Array of comparison results
    ///   - title: Title for the results table
    public func report(_ results: [ComparisonResult], title: String) {
        switch format {
        case .console:
            print(formatConsoleTable(results, title: title))
        case .json:
            print(formatJSONArray(results, title: title))
        }
    }

    // MARK: - Console Formatting

    /// Formats a single result for console output.
    ///
    /// - Parameter result: Comparison result to format
    /// - Returns: Formatted string with box-drawing table
    public func formatConsole(_ result: ComparisonResult) -> String {
        let titleWidth = 60
        let colWidth = 14

        var output = ""

        // Top border
        output += "╔" + String(repeating: "═", count: titleWidth) + "╗\n"

        // Title
        let title = "  \(result.name)"
        let paddedTitle = title.padding(toLength: titleWidth, withPad: " ", startingAt: 0)
        output += "║\(paddedTitle)║\n"

        // Scale subtitle
        if !result.scale.isEmpty {
            let subtitle = "  Scale: \(result.scale)"
            let paddedSubtitle = subtitle.padding(toLength: titleWidth, withPad: " ", startingAt: 0)
            output += "║\(paddedSubtitle)║\n"
        }

        // Header separator
        output += "╠" + String(repeating: "═", count: colWidth)
        output += "╦" + String(repeating: "═", count: colWidth)
        output += "╦" + String(repeating: "═", count: colWidth)
        output += "╦" + String(repeating: "═", count: titleWidth - 3 * colWidth - 3) + "╣\n"

        // Column headers
        let baselineHeader = padCenter(result.baseline.label, width: colWidth)
        let variantHeader = padCenter(result.variant.label, width: colWidth)
        let speedupHeader = padCenter("Speedup", width: colWidth)
        let statusHeader = padCenter("Status", width: titleWidth - 3 * colWidth - 3)
        output += "║\(baselineHeader)║\(variantHeader)║\(speedupHeader)║\(statusHeader)║\n"

        // Data separator
        output += "╠" + String(repeating: "═", count: colWidth)
        output += "╬" + String(repeating: "═", count: colWidth)
        output += "╬" + String(repeating: "═", count: colWidth)
        output += "╬" + String(repeating: "═", count: titleWidth - 3 * colWidth - 3) + "╣\n"

        // Data row
        let baselineTime = padRight(formatDuration(result.baseline.statistics.median), width: colWidth)
        let variantTime = padRight(formatDuration(result.variant.statistics.median), width: colWidth)
        let speedup = padRight(result.speedupFormatted, width: colWidth)
        let status = padRight(result.isSignificant ? "✓ Significant" : "○ Not significant", width: titleWidth - 3 * colWidth - 3)
        output += "║\(baselineTime)║\(variantTime)║\(speedup)║\(status)║\n"

        // Bottom border
        output += "╚" + String(repeating: "═", count: colWidth)
        output += "╩" + String(repeating: "═", count: colWidth)
        output += "╩" + String(repeating: "═", count: colWidth)
        output += "╩" + String(repeating: "═", count: titleWidth - 3 * colWidth - 3) + "╝"

        return output
    }

    /// Formats multiple results as a console table.
    ///
    /// - Parameters:
    ///   - results: Array of comparison results
    ///   - title: Title for the table
    /// - Returns: Formatted string with box-drawing table
    public func formatConsoleTable(_ results: [ComparisonResult], title: String) -> String {
        guard !results.isEmpty else { return "No results to display" }

        let titleWidth = 72
        let scaleWidth = 14
        let timeWidth = 14
        let speedupWidth = 12
        let statusWidth = titleWidth - scaleWidth - 2 * timeWidth - speedupWidth - 4

        var output = ""

        // Get labels from first result
        let baselineLabel = results[0].baseline.label
        let variantLabel = results[0].variant.label

        // Top border
        output += "╔" + String(repeating: "═", count: titleWidth) + "╗\n"

        // Title
        let paddedTitle = "  \(title)".padding(toLength: titleWidth, withPad: " ", startingAt: 0)
        output += "║\(paddedTitle)║\n"

        // Header separator
        output += "╠" + String(repeating: "═", count: scaleWidth)
        output += "╦" + String(repeating: "═", count: timeWidth)
        output += "╦" + String(repeating: "═", count: timeWidth)
        output += "╦" + String(repeating: "═", count: speedupWidth)
        output += "╦" + String(repeating: "═", count: statusWidth) + "╣\n"

        // Column headers
        output += "║\(padCenter("Scale", width: scaleWidth))"
        output += "║\(padCenter(baselineLabel, width: timeWidth))"
        output += "║\(padCenter(variantLabel, width: timeWidth))"
        output += "║\(padCenter("Speedup", width: speedupWidth))"
        output += "║\(padCenter("Status", width: statusWidth))║\n"

        // Data separator
        output += "╠" + String(repeating: "═", count: scaleWidth)
        output += "╬" + String(repeating: "═", count: timeWidth)
        output += "╬" + String(repeating: "═", count: timeWidth)
        output += "╬" + String(repeating: "═", count: speedupWidth)
        output += "╬" + String(repeating: "═", count: statusWidth) + "╣\n"

        // Data rows
        for result in results {
            let scale = padRight(result.scale, width: scaleWidth)
            let baselineTime = padRight(formatDuration(result.baseline.statistics.median), width: timeWidth)
            let variantTime = padRight(formatDuration(result.variant.statistics.median), width: timeWidth)
            let speedup = padRight(result.speedupFormatted, width: speedupWidth)
            let status = padRight(result.isSignificant ? "✓ PASS" : "○ N/S", width: statusWidth)
            output += "║\(scale)║\(baselineTime)║\(variantTime)║\(speedup)║\(status)║\n"
        }

        // Bottom border
        output += "╚" + String(repeating: "═", count: scaleWidth)
        output += "╩" + String(repeating: "═", count: timeWidth)
        output += "╩" + String(repeating: "═", count: timeWidth)
        output += "╩" + String(repeating: "═", count: speedupWidth)
        output += "╩" + String(repeating: "═", count: statusWidth) + "╝"

        return output
    }

    /// Formats detailed statistics for a result.
    ///
    /// - Parameter result: Comparison result
    /// - Returns: Formatted statistics breakdown
    public func formatDetailedStats(_ result: ComparisonResult) -> String {
        var output = "\nDetailed Statistics:\n"
        output += "─────────────────────────────────────────────────────────────\n"

        // Baseline stats
        output += "  \(result.baseline.label):\n"
        output += formatStatisticsBlock(result.baseline.statistics, indent: 4)

        output += "\n"

        // Variant stats
        output += "  \(result.variant.label):\n"
        output += formatStatisticsBlock(result.variant.statistics, indent: 4)

        output += "─────────────────────────────────────────────────────────────"

        return output
    }

    /// Formats a statistics block with indentation.
    private func formatStatisticsBlock(_ stats: TimingStatistics, indent: Int) -> String {
        let pad = String(repeating: " ", count: indent)
        var output = ""

        output += "\(pad)Samples: \(stats.sampleCount)"
        if stats.failedCount > 0 {
            output += " (failed: \(stats.failedCount))"
        }
        output += "\n"

        output += "\(pad)Median:  \(formatDuration(stats.median))\n"
        output += "\(pad)Mean:    \(formatDuration(stats.mean))\n"
        output += "\(pad)Stddev:  \(formatDuration(stats.standardDeviation))\n"
        output += "\(pad)Range:   \(formatDuration(stats.min)) - \(formatDuration(stats.max))\n"
        output += "\(pad)P25/P75: \(formatDuration(stats.p25)) / \(formatDuration(stats.p75))\n"
        output += "\(pad)P95:     \(formatDuration(stats.p95))\n"
        output += "\(pad)P99:     \(formatDuration(stats.p99))\n"
        output += "\(pad)CV:      \(String(format: "%.1f%%", stats.coefficientOfVariation * 100))\n"

        return output
    }

    // MARK: - JSON Formatting

    /// Formats a single result as JSON.
    ///
    /// - Parameter result: Comparison result to format
    /// - Returns: JSON string
    public func formatJSON(_ result: ComparisonResult) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        do {
            let data = try encoder.encode(result)
            return String(data: data, encoding: .utf8) ?? "{}"
        } catch {
            return "{ \"error\": \"\(error.localizedDescription)\" }"
        }
    }

    /// Formats multiple results as JSON array.
    ///
    /// - Parameters:
    ///   - results: Array of comparison results
    ///   - title: Title for the benchmark suite
    /// - Returns: JSON string
    public func formatJSONArray(_ results: [ComparisonResult], title: String) -> String {
        let wrapper = BenchmarkSuiteJSON(title: title, results: results, timestamp: Date())

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        do {
            let data = try encoder.encode(wrapper)
            return String(data: data, encoding: .utf8) ?? "{}"
        } catch {
            return "{ \"error\": \"\(error.localizedDescription)\" }"
        }
    }

    // MARK: - Helper Methods

    /// Formats a Duration for display.
    private func formatDuration(_ duration: Duration) -> String {
        TimingStatistics.format(duration)
    }

    /// Pads a string to center alignment.
    private func padCenter(_ string: String, width: Int) -> String {
        let totalPadding = width - string.count
        guard totalPadding > 0 else { return String(string.prefix(width)) }

        let leftPad = totalPadding / 2
        let rightPad = totalPadding - leftPad
        return String(repeating: " ", count: leftPad) + string + String(repeating: " ", count: rightPad)
    }

    /// Pads a string to right alignment (left-pads with spaces).
    private func padRight(_ string: String, width: Int) -> String {
        let padding = width - string.count - 1
        guard padding > 0 else { return " " + String(string.prefix(width - 1)) }
        return " " + string + String(repeating: " ", count: padding)
    }
}

// MARK: - JSON Wrapper Types

/// Wrapper for JSON serialization of benchmark suite results.
private struct BenchmarkSuiteJSON: Codable, Sendable {
    let title: String
    let results: [ComparisonResult]
    let timestamp: Date
}

// MARK: - Quick Print Extensions

extension ComparisonResult {

    /// Prints a quick summary to console.
    public func printSummary() {
        BenchmarkReporter.console.report(self)
    }

    /// Prints detailed statistics to console.
    public func printDetailed() {
        BenchmarkReporter.console.reportDetailed(self)
    }
}

extension Array where Element == ComparisonResult {

    /// Prints results table to console.
    ///
    /// - Parameter title: Title for the table
    public func printTable(title: String) {
        BenchmarkReporter.console.report(self, title: title)
    }
}
