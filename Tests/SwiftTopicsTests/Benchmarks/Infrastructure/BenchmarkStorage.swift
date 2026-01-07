// BenchmarkStorage.swift
// SwiftTopicsTests
//
// JSON persistence with timestamps and regression detection.
// Part of Day 3: Polish

import Foundation

// MARK: - Benchmark Storage

/// Persists benchmark results to JSON files.
///
/// Enables historical tracking of benchmark results for regression detection
/// and trend analysis. Results are stored with timestamps in a dedicated directory.
///
/// ## File Structure
/// ```
/// BenchmarkResults/
/// ├── HDBSCAN_2026-01-06T14-32-00.json
/// ├── HDBSCAN_2026-01-05T10-15-30.json
/// ├── UMAP_2026-01-06T14-35-00.json
/// └── FullPipeline_2026-01-06T15-00-00.json
/// ```
///
/// ## Usage
/// ```swift
/// let storage = BenchmarkStorage()
/// let url = try storage.save(result)
/// let previous = try storage.loadPrevious(name: "HDBSCAN", count: 5)
/// let status = try storage.detectRegression(current: result)
/// ```
public final class BenchmarkStorage: Sendable {

    // MARK: - Properties

    /// Directory for benchmark result files.
    public let directory: URL

    /// Date formatter for filenames.
    /// Using nonisolated(unsafe) since ISO8601DateFormatter is not Sendable
    /// but our usage is safe (only formatting, thread-safe in practice).
    nonisolated(unsafe) private static let filenameFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate, .withTime, .withDashSeparatorInDate]
        return formatter
    }()

    // MARK: - Initialization

    /// Creates a storage instance with the default directory.
    public init() {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        self.directory = cwd.appendingPathComponent("BenchmarkResults")
    }

    /// Creates a storage instance with a custom directory.
    public init(directory: URL) {
        self.directory = directory
    }

    // MARK: - Saving

    /// Saves a comparison result with timestamp.
    ///
    /// - Parameter result: The comparison result to save.
    /// - Returns: URL where the result was saved.
    /// - Throws: If serialization or file writing fails.
    @discardableResult
    public func save(_ result: ComparisonResult) throws -> URL {
        try ensureDirectoryExists()

        let filename = generateFilename(name: result.name, timestamp: result.timestamp)
        let fileURL = directory.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(result)
        try data.write(to: fileURL)

        return fileURL
    }

    /// Saves multiple results as a benchmark run.
    ///
    /// - Parameters:
    ///   - results: Array of comparison results.
    ///   - name: Name for the benchmark run.
    /// - Returns: URL where the run was saved.
    /// - Throws: If serialization or file writing fails.
    @discardableResult
    public func saveRun(_ results: [ComparisonResult], name: String) throws -> URL {
        try ensureDirectoryExists()

        let filename = generateFilename(name: name, timestamp: Date())
        let fileURL = directory.appendingPathComponent(filename)

        let run = BenchmarkRun(name: name, results: results, timestamp: Date())

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(run)
        try data.write(to: fileURL)

        return fileURL
    }

    // MARK: - Loading

    /// Loads previous results for a given benchmark name.
    ///
    /// - Parameters:
    ///   - name: Benchmark name to search for.
    ///   - count: Maximum number of results to load.
    /// - Returns: Array of previous results, most recent first.
    /// - Throws: If file reading or deserialization fails.
    public func loadPrevious(name: String, count: Int = 5) throws -> [ComparisonResult] {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            return []
        }

        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])

        // Sanitize name the same way as generateFilename
        let sanitizedName = name.replacingOccurrences(of: " ", with: "_")

        // Filter by sanitized name prefix and sort by date (most recent first)
        let matchingFiles = files
            .filter { $0.lastPathComponent.hasPrefix(sanitizedName) && $0.pathExtension == "json" }
            .sorted { file1, file2 in
                let date1 = (try? file1.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let date2 = (try? file2.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return date1 > date2
            }
            .prefix(count)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        var results: [ComparisonResult] = []
        for fileURL in matchingFiles {
            let data = try Data(contentsOf: fileURL)
            let result = try decoder.decode(ComparisonResult.self, from: data)
            results.append(result)
        }
        return results
    }

    // MARK: - Regression Detection

    /// Compares current result to previous baseline.
    ///
    /// - Parameters:
    ///   - current: The current benchmark result.
    ///   - threshold: Percentage threshold for regression (default: 10%).
    /// - Returns: Regression status.
    /// - Throws: If loading previous results fails.
    public func detectRegression(
        current: ComparisonResult,
        threshold: Double = 0.10
    ) throws -> RegressionStatus {
        let previous = try loadPrevious(name: current.name, count: 1)

        guard let baseline = previous.first else {
            return .noBaseline
        }

        let currentSpeedup = current.speedup
        let baselineSpeedup = baseline.speedup

        let change = (currentSpeedup - baselineSpeedup) / baselineSpeedup

        if change > threshold {
            return .improved(percentage: change * 100)
        } else if change < -threshold {
            return .regressed(percentage: abs(change) * 100)
        } else {
            return .stable
        }
    }

    // MARK: - Helpers

    private func ensureDirectoryExists() throws {
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    private func generateFilename(name: String, timestamp: Date) -> String {
        let sanitizedName = name.replacingOccurrences(of: " ", with: "_")
        let dateStr = Self.filenameFormatter.string(from: timestamp)
            .replacingOccurrences(of: ":", with: "-")
        return "\(sanitizedName)_\(dateStr).json"
    }
}

// MARK: - Regression Status

/// Status of performance regression detection.
public enum RegressionStatus: Sendable, Equatable {
    /// No previous baseline to compare against.
    case noBaseline

    /// Performance improved by the given percentage.
    case improved(percentage: Double)

    /// Performance is stable (within threshold).
    case stable

    /// Performance regressed by the given percentage.
    case regressed(percentage: Double)

    /// Human-readable description.
    public var description: String {
        switch self {
        case .noBaseline:
            return "No baseline available"
        case .improved(let pct):
            return String(format: "Improved by %.1f%%", pct)
        case .stable:
            return "Stable (within threshold)"
        case .regressed(let pct):
            return String(format: "Regressed by %.1f%%", pct)
        }
    }
}

// MARK: - Benchmark Run

/// A collection of benchmark results from a single run.
private struct BenchmarkRun: Codable, Sendable {
    let name: String
    let results: [ComparisonResult]
    let timestamp: Date
}
