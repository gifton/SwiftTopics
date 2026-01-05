// IncrementalUpdateConfiguration.swift
// SwiftTopics
//
// Configuration for incremental topic model updates

import Foundation

// MARK: - Incremental Update Configuration

/// Configuration for incremental topic model updates.
///
/// This configuration controls the behavior of `IncrementalTopicUpdater`,
/// including:
///
/// - **Buffer thresholds**: When to trigger micro-retrain or initial training
/// - **Full refresh triggers**: Growth ratio, time interval, and drift thresholds
/// - **Topic matching**: Similarity threshold for matching topics across retrains
///
/// ## Presets
///
/// Three presets are available:
/// - `default`: Balanced for typical journaling use (30 entries/retrain)
/// - `aggressive`: More frequent updates, lower thresholds
/// - `conservative`: Less frequent updates, higher thresholds
///
/// ## Example
///
/// ```swift
/// // Use default configuration
/// let updater = try IncrementalTopicUpdater(
///     storage: storage,
///     updateConfiguration: .default
/// )
///
/// // Custom configuration for low-frequency updates
/// let config = IncrementalUpdateConfiguration(
///     microRetrainThreshold: 50,
///     fullRefreshGrowthRatio: 2.0
/// )
/// ```
public struct IncrementalUpdateConfiguration: Sendable, Codable, Hashable {

    // MARK: - Buffer Thresholds

    /// Minimum entries before initial model creation.
    ///
    /// Before this threshold is reached, documents receive outlier assignments
    /// and are buffered. Once reached, initial training creates the first model.
    ///
    /// - Default: 30
    /// - Range: 10-100 (lower values may produce poor initial topics)
    public var coldStartThreshold: Int

    /// Entries to buffer before triggering micro-retrain.
    ///
    /// After each micro-retrain, the buffer is cleared. This threshold determines
    /// how many new documents accumulate before the next retrain.
    ///
    /// - Default: 30
    /// - Range: 10-100 (lower = more frequent updates, higher = less overhead)
    public var microRetrainThreshold: Int

    /// Maximum buffer size before forced retrain.
    ///
    /// Safety limit to prevent unbounded buffer growth. If reached, micro-retrain
    /// is triggered regardless of other conditions.
    ///
    /// - Default: 100
    public var maxBufferSize: Int

    // MARK: - Full Refresh Triggers

    /// Corpus growth ratio to trigger full refresh.
    ///
    /// When `totalDocuments / documentsAtLastRetrain >= fullRefreshGrowthRatio`,
    /// a full refresh is recommended.
    ///
    /// - Default: 1.5 (50% growth since last full retrain)
    /// - Range: 1.2-3.0
    public var fullRefreshGrowthRatio: Float

    /// Maximum time between full refreshes.
    ///
    /// Even without significant growth or drift, periodic full refresh
    /// ensures the model stays well-calibrated.
    ///
    /// - Default: 180 days
    /// - Set to `nil` to disable time-based refresh
    public var fullRefreshMaxInterval: TimeInterval?

    /// Outlier rate threshold for early refresh.
    ///
    /// When the recent outlier rate exceeds this threshold, it indicates
    /// the model is struggling to classify new documents.
    ///
    /// - Default: 0.20 (20%)
    /// - Range: 0.1-0.5
    public var outlierRateThreshold: Float

    /// Drift ratio threshold for refresh.
    ///
    /// When `recentAverageDistance / overallAverageDistance >= driftRatioThreshold`,
    /// it indicates new documents are increasingly distant from existing topics.
    ///
    /// - Default: 1.5 (recent distance 50% higher than overall)
    /// - Range: 1.2-2.0
    public var driftRatioThreshold: Float

    // MARK: - Topic Matching

    /// Minimum cosine similarity for topic matching after retrain.
    ///
    /// Topics with similarity below this threshold are treated as new topics
    /// rather than matches to existing topics.
    ///
    /// - Default: 0.7
    /// - Range: 0.5-0.9 (higher = stricter matching)
    public var topicMatchingSimilarityThreshold: Float

    // MARK: - Transform Assignment

    /// Minimum similarity for a document to be assigned to a topic.
    ///
    /// Documents with similarity below this threshold to all topics
    /// are marked as outliers during transform assignment.
    ///
    /// - Default: 0.3
    /// - Range: 0.2-0.5
    public var transformOutlierThreshold: Float

    // MARK: - Drift Statistics

    /// Window size for recent drift statistics.
    ///
    /// The exponential moving average for drift detection uses this window.
    /// Smaller values are more responsive, larger values are more stable.
    ///
    /// - Default: 100
    public var driftWindowSize: Int

    // MARK: - Initialization

    /// Creates an incremental update configuration.
    ///
    /// - Parameters:
    ///   - coldStartThreshold: Entries before initial training. Default: 30.
    ///   - microRetrainThreshold: Entries between micro-retrains. Default: 30.
    ///   - maxBufferSize: Maximum buffer before forced retrain. Default: 100.
    ///   - fullRefreshGrowthRatio: Growth ratio for refresh. Default: 1.5.
    ///   - fullRefreshMaxInterval: Max time between refreshes. Default: 180 days.
    ///   - outlierRateThreshold: Outlier rate for refresh. Default: 0.20.
    ///   - driftRatioThreshold: Drift ratio for refresh. Default: 1.5.
    ///   - topicMatchingSimilarityThreshold: Topic match threshold. Default: 0.7.
    ///   - transformOutlierThreshold: Transform outlier threshold. Default: 0.3.
    ///   - driftWindowSize: Window for drift statistics. Default: 100.
    public init(
        coldStartThreshold: Int = 30,
        microRetrainThreshold: Int = 30,
        maxBufferSize: Int = 100,
        fullRefreshGrowthRatio: Float = 1.5,
        fullRefreshMaxInterval: TimeInterval? = 180 * 24 * 60 * 60,
        outlierRateThreshold: Float = 0.20,
        driftRatioThreshold: Float = 1.5,
        topicMatchingSimilarityThreshold: Float = 0.7,
        transformOutlierThreshold: Float = 0.3,
        driftWindowSize: Int = 100
    ) {
        self.coldStartThreshold = coldStartThreshold
        self.microRetrainThreshold = microRetrainThreshold
        self.maxBufferSize = maxBufferSize
        self.fullRefreshGrowthRatio = fullRefreshGrowthRatio
        self.fullRefreshMaxInterval = fullRefreshMaxInterval
        self.outlierRateThreshold = outlierRateThreshold
        self.driftRatioThreshold = driftRatioThreshold
        self.topicMatchingSimilarityThreshold = topicMatchingSimilarityThreshold
        self.transformOutlierThreshold = transformOutlierThreshold
        self.driftWindowSize = driftWindowSize
    }

    // MARK: - Presets

    /// Default configuration balanced for typical journaling use.
    ///
    /// Triggers micro-retrain every 30 entries, full refresh at 50% growth
    /// or 180 days, whichever comes first.
    public static let `default` = IncrementalUpdateConfiguration()

    /// Aggressive configuration for more frequent updates.
    ///
    /// Lower thresholds mean more responsive updates but higher overhead.
    /// Use when topic freshness is critical.
    public static let aggressive = IncrementalUpdateConfiguration(
        coldStartThreshold: 20,
        microRetrainThreshold: 20,
        maxBufferSize: 50,
        fullRefreshGrowthRatio: 1.3,
        fullRefreshMaxInterval: 90 * 24 * 60 * 60,  // 90 days
        outlierRateThreshold: 0.15,
        driftRatioThreshold: 1.3
    )

    /// Conservative configuration for less frequent updates.
    ///
    /// Higher thresholds reduce overhead but may allow topics to become stale.
    /// Use when processing power is limited or updates are less critical.
    public static let conservative = IncrementalUpdateConfiguration(
        coldStartThreshold: 50,
        microRetrainThreshold: 50,
        maxBufferSize: 150,
        fullRefreshGrowthRatio: 2.0,
        fullRefreshMaxInterval: 365 * 24 * 60 * 60,  // 365 days
        outlierRateThreshold: 0.25,
        driftRatioThreshold: 2.0
    )

    /// Configuration optimized for testing.
    ///
    /// Very low thresholds for quick iteration in tests.
    public static let testing = IncrementalUpdateConfiguration(
        coldStartThreshold: 5,
        microRetrainThreshold: 3,
        maxBufferSize: 10,
        fullRefreshGrowthRatio: 1.2,
        fullRefreshMaxInterval: 60,  // 1 minute
        outlierRateThreshold: 0.10,
        driftRatioThreshold: 1.2
    )
}

// MARK: - Validation

extension IncrementalUpdateConfiguration {

    /// Validates the configuration and returns any issues.
    ///
    /// - Returns: Array of validation issue descriptions, empty if valid.
    public func validate() -> [String] {
        var issues = [String]()

        if coldStartThreshold < 5 {
            issues.append("coldStartThreshold too low (minimum 5)")
        }

        if microRetrainThreshold < 3 {
            issues.append("microRetrainThreshold too low (minimum 3)")
        }

        if maxBufferSize < microRetrainThreshold {
            issues.append("maxBufferSize should be >= microRetrainThreshold")
        }

        if fullRefreshGrowthRatio < 1.1 {
            issues.append("fullRefreshGrowthRatio too low (minimum 1.1)")
        }

        if outlierRateThreshold < 0.05 || outlierRateThreshold > 0.8 {
            issues.append("outlierRateThreshold should be between 0.05 and 0.8")
        }

        if driftRatioThreshold < 1.1 {
            issues.append("driftRatioThreshold too low (minimum 1.1)")
        }

        if topicMatchingSimilarityThreshold < 0.3 || topicMatchingSimilarityThreshold > 0.95 {
            issues.append("topicMatchingSimilarityThreshold should be between 0.3 and 0.95")
        }

        if transformOutlierThreshold < 0.1 || transformOutlierThreshold > 0.7 {
            issues.append("transformOutlierThreshold should be between 0.1 and 0.7")
        }

        return issues
    }

    /// Whether this configuration is valid.
    public var isValid: Bool {
        validate().isEmpty
    }
}
