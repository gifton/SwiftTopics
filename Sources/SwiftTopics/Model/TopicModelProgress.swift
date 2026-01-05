// TopicModelProgress.swift
// SwiftTopics
//
// Progress reporting for topic modeling operations

import Foundation

// MARK: - Topic Model Stage

/// Stages of the topic modeling pipeline.
public enum TopicModelStage: Sendable, Hashable {

    /// Computing embeddings for documents.
    case embedding(current: Int, total: Int)

    /// Reducing dimensionality of embeddings.
    case reduction

    /// Clustering embeddings into topics.
    case clustering

    /// Extracting topic representations (keywords).
    case representation

    /// Evaluating topic coherence.
    case evaluation

    /// Pipeline complete.
    case complete

    /// Description of the current stage.
    public var description: String {
        switch self {
        case .embedding(let current, let total):
            return "Embedding documents (\(current)/\(total))"
        case .reduction:
            return "Reducing dimensions"
        case .clustering:
            return "Clustering embeddings"
        case .representation:
            return "Extracting keywords"
        case .evaluation:
            return "Evaluating coherence"
        case .complete:
            return "Complete"
        }
    }

    /// Estimated weight of this stage for progress calculation.
    internal var weight: Float {
        switch self {
        case .embedding:
            return 0.30  // Often the slowest part
        case .reduction:
            return 0.15
        case .clustering:
            return 0.30
        case .representation:
            return 0.15
        case .evaluation:
            return 0.10
        case .complete:
            return 0.0
        }
    }
}

// MARK: - Topic Model Progress

/// Progress information for a topic modeling operation.
///
/// Use this to track the progress of long-running operations
/// on large corpora.
public struct TopicModelProgress: Sendable {

    /// The current pipeline stage.
    public let stage: TopicModelStage

    /// Overall progress from 0.0 to 1.0.
    public let overallProgress: Float

    /// Time elapsed since the operation started.
    public let elapsedTime: TimeInterval

    /// Estimated time remaining (nil if cannot estimate).
    public let estimatedTimeRemaining: TimeInterval?

    /// Creates a progress update.
    public init(
        stage: TopicModelStage,
        overallProgress: Float,
        elapsedTime: TimeInterval,
        estimatedTimeRemaining: TimeInterval? = nil
    ) {
        self.stage = stage
        self.overallProgress = max(0, min(1, overallProgress))
        self.elapsedTime = elapsedTime
        self.estimatedTimeRemaining = estimatedTimeRemaining
    }

    /// Human-readable progress string.
    public var description: String {
        let percent = Int(overallProgress * 100)
        var result = "\(stage.description) (\(percent)%)"

        if let remaining = estimatedTimeRemaining, remaining > 0 {
            let minutes = Int(remaining / 60)
            let seconds = Int(remaining.truncatingRemainder(dividingBy: 60))
            if minutes > 0 {
                result += " - ~\(minutes)m \(seconds)s remaining"
            } else {
                result += " - ~\(seconds)s remaining"
            }
        }

        return result
    }
}

// MARK: - Progress Handler

/// A handler for receiving progress updates.
public typealias TopicModelProgressHandler = @Sendable (TopicModelProgress) -> Void

// MARK: - Progress Tracker

/// Internal utility for tracking progress across pipeline stages.
internal actor ProgressTracker {

    private let handler: TopicModelProgressHandler?
    private let startTime: Date
    private var currentStage: TopicModelStage
    private var stageStartProgress: Float

    /// Stage weights for progress calculation.
    private static let stageOrder: [TopicModelStage] = [
        .embedding(current: 0, total: 1),
        .reduction,
        .clustering,
        .representation,
        .evaluation,
        .complete
    ]

    /// Cumulative progress at the start of each stage.
    private static let stageStartProgress: [Float] = {
        var cumulative: Float = 0
        var starts: [Float] = []
        for stage in stageOrder {
            starts.append(cumulative)
            cumulative += stage.weight
        }
        return starts
    }()

    /// Creates a progress tracker.
    ///
    /// - Parameter handler: Optional handler to receive progress updates.
    init(handler: TopicModelProgressHandler?) {
        self.handler = handler
        self.startTime = Date()
        self.currentStage = .embedding(current: 0, total: 1)
        self.stageStartProgress = 0
    }

    /// Reports entering a new stage.
    func enterStage(_ stage: TopicModelStage) {
        currentStage = stage
        stageStartProgress = progressForStage(stage)
        reportProgress(inStageProgress: 0)
    }

    /// Reports progress within the current stage.
    ///
    /// - Parameter fraction: Progress within the stage (0.0 to 1.0).
    func reportInStageProgress(_ fraction: Float) {
        reportProgress(inStageProgress: fraction)
    }

    /// Marks the pipeline as complete.
    func complete() {
        currentStage = .complete
        reportProgress(inStageProgress: 1.0)
    }

    /// Gets the elapsed time.
    var elapsedTime: TimeInterval {
        Date().timeIntervalSince(startTime)
    }

    // MARK: - Private

    private func progressForStage(_ stage: TopicModelStage) -> Float {
        switch stage {
        case .embedding:
            return 0
        case .reduction:
            return TopicModelStage.embedding(current: 0, total: 1).weight
        case .clustering:
            return TopicModelStage.embedding(current: 0, total: 1).weight
                + TopicModelStage.reduction.weight
        case .representation:
            return TopicModelStage.embedding(current: 0, total: 1).weight
                + TopicModelStage.reduction.weight
                + TopicModelStage.clustering.weight
        case .evaluation:
            return TopicModelStage.embedding(current: 0, total: 1).weight
                + TopicModelStage.reduction.weight
                + TopicModelStage.clustering.weight
                + TopicModelStage.representation.weight
        case .complete:
            return 1.0
        }
    }

    private func reportProgress(inStageProgress: Float) {
        guard let handler = handler else { return }

        let stageWeight = currentStage.weight
        let overallProgress = stageStartProgress + (inStageProgress * stageWeight)

        let elapsed = Date().timeIntervalSince(startTime)
        let estimatedRemaining: TimeInterval?

        if overallProgress > 0.01 {
            let totalEstimated = elapsed / TimeInterval(overallProgress)
            estimatedRemaining = totalEstimated - elapsed
        } else {
            estimatedRemaining = nil
        }

        let progress = TopicModelProgress(
            stage: currentStage,
            overallProgress: overallProgress,
            elapsedTime: elapsed,
            estimatedTimeRemaining: estimatedRemaining
        )

        handler(progress)
    }
}

// MARK: - Progress Stream

/// An async stream for receiving progress updates.
///
/// Use with `for await` to receive updates:
/// ```swift
/// for await progress in progressStream {
///     print(progress.description)
/// }
/// ```
public struct TopicModelProgressStream: Sendable {

    private let stream: AsyncStream<TopicModelProgress>

    internal init(stream: AsyncStream<TopicModelProgress>) {
        self.stream = stream
    }

    /// Creates an async iterator for the progress stream.
    public func makeAsyncIterator() -> AsyncStream<TopicModelProgress>.AsyncIterator {
        stream.makeAsyncIterator()
    }
}

extension TopicModelProgressStream: AsyncSequence {
    public typealias Element = TopicModelProgress
}
