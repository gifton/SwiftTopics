// TrainingCheckpoint.swift
// SwiftTopics
//
// Checkpoint state for interruptible training operations

import Foundation

// MARK: - Training Phase

/// Phases of the topic model training pipeline.
///
/// Each phase represents a distinct step that can be checkpointed.
/// Phases with `supportsPartialCheckpoint = true` can be interrupted
/// mid-execution and resumed from partial state.
public enum TrainingPhase: Int, Codable, Sendable, CaseIterable {

    /// k-NN graph construction for UMAP.
    case umapKNN = 1

    /// Fuzzy simplicial set construction.
    case umapFuzzySet = 2

    /// UMAP SGD optimization (iterative, supports partial checkpoint).
    case umapOptimization = 3

    /// Core distance computation for HDBSCAN.
    case hdbscanCoreDistance = 4

    /// Minimum spanning tree construction (supports partial checkpoint).
    case hdbscanMST = 5

    /// Cluster extraction from condensed tree.
    case clusterExtraction = 6

    /// c-TF-IDF representation computation.
    case representation = 7

    /// Topic matching to preserve IDs across retrains.
    case topicMatching = 8

    /// Training complete.
    case complete = 9

    // MARK: - Phase Properties

    /// Whether this phase supports mid-phase checkpointing.
    ///
    /// Phases that are iterative or long-running support saving
    /// partial progress. Other phases are fast enough to restart.
    public var supportsPartialCheckpoint: Bool {
        switch self {
        case .umapKNN, .hdbscanCoreDistance:
            // These are parallelizable and can checkpoint per-batch
            return true
        case .umapOptimization, .hdbscanMST:
            // These are iterative and can checkpoint per-iteration
            return true
        default:
            return false
        }
    }

    /// Human-readable name for this phase.
    public var displayName: String {
        switch self {
        case .umapKNN: return "Building neighbor graph"
        case .umapFuzzySet: return "Computing fuzzy sets"
        case .umapOptimization: return "Optimizing layout"
        case .hdbscanCoreDistance: return "Computing core distances"
        case .hdbscanMST: return "Building minimum spanning tree"
        case .clusterExtraction: return "Extracting clusters"
        case .representation: return "Computing topic keywords"
        case .topicMatching: return "Matching topics"
        case .complete: return "Complete"
        }
    }

    /// Estimated duration for 1000 documents (seconds).
    ///
    /// These are rough estimates for planning purposes. Actual
    /// performance varies based on hardware and document characteristics.
    public var estimatedDuration: TimeInterval {
        switch self {
        case .umapKNN: return 8.0
        case .umapFuzzySet: return 0.5
        case .umapOptimization: return 15.0
        case .hdbscanCoreDistance: return 4.0
        case .hdbscanMST: return 3.0
        case .clusterExtraction: return 0.1
        case .representation: return 0.5
        case .topicMatching: return 0.1
        case .complete: return 0
        }
    }

    /// The previous phase in the pipeline, or nil for the first phase.
    public var previous: TrainingPhase? {
        guard let index = Self.allCases.firstIndex(of: self), index > 0 else {
            return nil
        }
        return Self.allCases[index - 1]
    }

    /// The next phase in the pipeline, or nil for completion.
    public var next: TrainingPhase? {
        guard let index = Self.allCases.firstIndex(of: self),
              index < Self.allCases.count - 1 else {
            return nil
        }
        return Self.allCases[index + 1]
    }

    /// Progress weight for overall progress calculation.
    ///
    /// Weights are proportional to typical execution time.
    public var progressWeight: Float {
        Float(estimatedDuration)
    }
}

// MARK: - Training Type

/// The type of training operation being checkpointed.
public enum TrainingType: String, Codable, Sendable {

    /// Micro-retrain on a small batch (typically 30 documents).
    ///
    /// Fast (~200ms) and runs after buffered entries reach threshold.
    case microRetrain

    /// Full refresh of the entire model.
    ///
    /// Slow (15-60s for 1000 docs) and runs periodically or on demand.
    case fullRefresh
}

// MARK: - Training Checkpoint

/// Complete checkpoint state for training resumption.
///
/// When training is interrupted (user closes app, system kills process),
/// the checkpoint enables resuming from the last saved state rather than
/// starting over.
///
/// ## Checkpoint Strategy
///
/// Checkpoints are saved:
/// - After each complete phase
/// - Every 3 seconds during long-running phases (UMAP optimization, MST)
/// - Before any potentially-failing operation
///
/// ## Storage
///
/// Large intermediate data (k-NN graph, embeddings, MST state) is stored
/// in separate files referenced by URL. The checkpoint itself is a small
/// JSON file with metadata and file references.
///
/// ## Thread Safety
///
/// `TrainingCheckpoint` is `Sendable` and `Codable`.
public struct TrainingCheckpoint: Sendable, Codable {

    // MARK: - Identity

    /// Unique ID for this training run.
    ///
    /// Used to detect stale checkpoints from previous runs.
    public let runID: UUID

    /// Type of training operation.
    public let trainingType: TrainingType

    /// When training started.
    public let startedAt: Date

    /// Document IDs included in this training batch.
    ///
    /// For micro-retrain: just the buffered documents.
    /// For full refresh: all documents in the corpus.
    public let documentIDs: [DocumentID]

    // MARK: - Progress

    /// Last phase that completed successfully.
    public let lastCompletedPhase: TrainingPhase?

    /// Current phase (may be partially complete).
    public let currentPhase: TrainingPhase

    /// Progress within current phase (0.0 to 1.0).
    ///
    /// Meaningful only for phases with `supportsPartialCheckpoint = true`.
    public let currentPhaseProgress: Float

    // MARK: - Phase-Specific State Paths

    /// Path to saved k-NN graph (after Phase 1).
    public let knnGraphPath: String?

    /// Path to saved fuzzy set (after Phase 2).
    public let fuzzySetPath: String?

    /// Path to UMAP embedding state (during Phase 3).
    public let umapEmbeddingPath: String?

    /// Current UMAP epoch (during Phase 3).
    public let umapCurrentEpoch: Int?

    /// Total UMAP epochs (during Phase 3).
    public let umapTotalEpochs: Int?

    /// Path to core distances (after Phase 4).
    public let coreDistancesPath: String?

    /// Path to partial MST state (during Phase 5).
    public let mstStatePath: String?

    /// Edges completed in MST construction (during Phase 5).
    public let mstEdgesCompleted: Int?

    /// Total edges in MST (during Phase 5).
    public let mstTotalEdges: Int?

    /// Path to cluster hierarchy (after Phase 6).
    public let hierarchyPath: String?

    /// Path to unmatched topics (after Phase 7).
    public let unmatchedTopicsPath: String?

    // MARK: - Resumption Control

    /// Attempts remaining before giving up.
    ///
    /// Decremented on each resume attempt. If a checkpoint consistently
    /// fails to resume, eventually it's abandoned and training restarts.
    public let attemptsRemaining: Int

    /// When this checkpoint was last updated.
    public let lastUpdatedAt: Date

    // MARK: - Initialization

    /// Creates a training checkpoint.
    public init(
        runID: UUID,
        trainingType: TrainingType,
        startedAt: Date,
        documentIDs: [DocumentID],
        lastCompletedPhase: TrainingPhase?,
        currentPhase: TrainingPhase,
        currentPhaseProgress: Float,
        knnGraphPath: String? = nil,
        fuzzySetPath: String? = nil,
        umapEmbeddingPath: String? = nil,
        umapCurrentEpoch: Int? = nil,
        umapTotalEpochs: Int? = nil,
        coreDistancesPath: String? = nil,
        mstStatePath: String? = nil,
        mstEdgesCompleted: Int? = nil,
        mstTotalEdges: Int? = nil,
        hierarchyPath: String? = nil,
        unmatchedTopicsPath: String? = nil,
        attemptsRemaining: Int = 3,
        lastUpdatedAt: Date = Date()
    ) {
        self.runID = runID
        self.trainingType = trainingType
        self.startedAt = startedAt
        self.documentIDs = documentIDs
        self.lastCompletedPhase = lastCompletedPhase
        self.currentPhase = currentPhase
        self.currentPhaseProgress = currentPhaseProgress
        self.knnGraphPath = knnGraphPath
        self.fuzzySetPath = fuzzySetPath
        self.umapEmbeddingPath = umapEmbeddingPath
        self.umapCurrentEpoch = umapCurrentEpoch
        self.umapTotalEpochs = umapTotalEpochs
        self.coreDistancesPath = coreDistancesPath
        self.mstStatePath = mstStatePath
        self.mstEdgesCompleted = mstEdgesCompleted
        self.mstTotalEdges = mstTotalEdges
        self.hierarchyPath = hierarchyPath
        self.unmatchedTopicsPath = unmatchedTopicsPath
        self.attemptsRemaining = attemptsRemaining
        self.lastUpdatedAt = lastUpdatedAt
    }

    // MARK: - Factory Methods

    /// Creates an initial checkpoint at the start of training.
    public static func initial(
        trainingType: TrainingType,
        documentIDs: [DocumentID]
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: UUID(),
            trainingType: trainingType,
            startedAt: Date(),
            documentIDs: documentIDs,
            lastCompletedPhase: nil,
            currentPhase: .umapKNN,
            currentPhaseProgress: 0
        )
    }

    // MARK: - Progress Updates

    /// Creates a new checkpoint with updated phase progress.
    public func withProgress(
        phase: TrainingPhase,
        progress: Float
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: runID,
            trainingType: trainingType,
            startedAt: startedAt,
            documentIDs: documentIDs,
            lastCompletedPhase: phase == currentPhase ? lastCompletedPhase : currentPhase,
            currentPhase: phase,
            currentPhaseProgress: progress,
            knnGraphPath: knnGraphPath,
            fuzzySetPath: fuzzySetPath,
            umapEmbeddingPath: umapEmbeddingPath,
            umapCurrentEpoch: umapCurrentEpoch,
            umapTotalEpochs: umapTotalEpochs,
            coreDistancesPath: coreDistancesPath,
            mstStatePath: mstStatePath,
            mstEdgesCompleted: mstEdgesCompleted,
            mstTotalEdges: mstTotalEdges,
            hierarchyPath: hierarchyPath,
            unmatchedTopicsPath: unmatchedTopicsPath,
            attemptsRemaining: attemptsRemaining,
            lastUpdatedAt: Date()
        )
    }

    /// Creates a new checkpoint after completing a phase.
    public func withCompletedPhase(_ phase: TrainingPhase) -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: runID,
            trainingType: trainingType,
            startedAt: startedAt,
            documentIDs: documentIDs,
            lastCompletedPhase: phase,
            currentPhase: phase.next ?? .complete,
            currentPhaseProgress: 0,
            knnGraphPath: knnGraphPath,
            fuzzySetPath: fuzzySetPath,
            umapEmbeddingPath: umapEmbeddingPath,
            umapCurrentEpoch: umapCurrentEpoch,
            umapTotalEpochs: umapTotalEpochs,
            coreDistancesPath: coreDistancesPath,
            mstStatePath: mstStatePath,
            mstEdgesCompleted: mstEdgesCompleted,
            mstTotalEdges: mstTotalEdges,
            hierarchyPath: hierarchyPath,
            unmatchedTopicsPath: unmatchedTopicsPath,
            attemptsRemaining: attemptsRemaining,
            lastUpdatedAt: Date()
        )
    }

    /// Creates a new checkpoint with decremented retry count.
    public func withDecrementedAttempts() -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: runID,
            trainingType: trainingType,
            startedAt: startedAt,
            documentIDs: documentIDs,
            lastCompletedPhase: lastCompletedPhase,
            currentPhase: currentPhase,
            currentPhaseProgress: currentPhaseProgress,
            knnGraphPath: knnGraphPath,
            fuzzySetPath: fuzzySetPath,
            umapEmbeddingPath: umapEmbeddingPath,
            umapCurrentEpoch: umapCurrentEpoch,
            umapTotalEpochs: umapTotalEpochs,
            coreDistancesPath: coreDistancesPath,
            mstStatePath: mstStatePath,
            mstEdgesCompleted: mstEdgesCompleted,
            mstTotalEdges: mstTotalEdges,
            hierarchyPath: hierarchyPath,
            unmatchedTopicsPath: unmatchedTopicsPath,
            attemptsRemaining: attemptsRemaining - 1,
            lastUpdatedAt: Date()
        )
    }

    /// Creates a new checkpoint with UMAP state.
    public func withUMAPState(
        embeddingPath: String,
        currentEpoch: Int,
        totalEpochs: Int
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: runID,
            trainingType: trainingType,
            startedAt: startedAt,
            documentIDs: documentIDs,
            lastCompletedPhase: lastCompletedPhase,
            currentPhase: .umapOptimization,
            currentPhaseProgress: Float(currentEpoch) / Float(totalEpochs),
            knnGraphPath: knnGraphPath,
            fuzzySetPath: fuzzySetPath,
            umapEmbeddingPath: embeddingPath,
            umapCurrentEpoch: currentEpoch,
            umapTotalEpochs: totalEpochs,
            coreDistancesPath: coreDistancesPath,
            mstStatePath: mstStatePath,
            mstEdgesCompleted: mstEdgesCompleted,
            mstTotalEdges: mstTotalEdges,
            hierarchyPath: hierarchyPath,
            unmatchedTopicsPath: unmatchedTopicsPath,
            attemptsRemaining: attemptsRemaining,
            lastUpdatedAt: Date()
        )
    }

    /// Creates a new checkpoint with MST state.
    public func withMSTState(
        statePath: String,
        edgesCompleted: Int,
        totalEdges: Int
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            runID: runID,
            trainingType: trainingType,
            startedAt: startedAt,
            documentIDs: documentIDs,
            lastCompletedPhase: lastCompletedPhase,
            currentPhase: .hdbscanMST,
            currentPhaseProgress: Float(edgesCompleted) / Float(totalEdges),
            knnGraphPath: knnGraphPath,
            fuzzySetPath: fuzzySetPath,
            umapEmbeddingPath: umapEmbeddingPath,
            umapCurrentEpoch: umapCurrentEpoch,
            umapTotalEpochs: umapTotalEpochs,
            coreDistancesPath: coreDistancesPath,
            mstStatePath: statePath,
            mstEdgesCompleted: edgesCompleted,
            mstTotalEdges: totalEdges,
            hierarchyPath: hierarchyPath,
            unmatchedTopicsPath: unmatchedTopicsPath,
            attemptsRemaining: attemptsRemaining,
            lastUpdatedAt: Date()
        )
    }

    // MARK: - Computed Properties

    /// Overall training progress (0.0 to 1.0).
    public var overallProgress: Float {
        let allPhases = TrainingPhase.allCases
        let totalWeight = allPhases.reduce(0) { $0 + $1.progressWeight }

        var completedWeight: Float = 0

        for phase in allPhases {
            if let completed = lastCompletedPhase, phase.rawValue <= completed.rawValue {
                completedWeight += phase.progressWeight
            } else if phase == currentPhase {
                completedWeight += phase.progressWeight * currentPhaseProgress
                break
            }
        }

        return totalWeight > 0 ? completedWeight / totalWeight : 0
    }

    /// Estimated time remaining based on progress and elapsed time.
    public var estimatedTimeRemaining: TimeInterval? {
        let elapsed = Date().timeIntervalSince(startedAt)
        let progress = overallProgress

        guard progress > 0.01 else { return nil }

        let totalEstimate = elapsed / Double(progress)
        return totalEstimate - elapsed
    }

    /// Whether this checkpoint can be resumed.
    public var canResume: Bool {
        attemptsRemaining > 0 && currentPhase != .complete
    }

    /// Training duration so far.
    public var elapsedTime: TimeInterval {
        lastUpdatedAt.timeIntervalSince(startedAt)
    }
}

// MARK: - Training Progress

/// Progress information during training.
///
/// Sent to progress callbacks to update UI and enable cancellation.
public struct TrainingProgress: Sendable {

    /// Current training phase.
    public let phase: TrainingPhase

    /// Progress within current phase (0.0 to 1.0).
    public let phaseProgress: Float

    /// Overall training progress (0.0 to 1.0).
    public let overallProgress: Float

    /// Estimated time remaining in seconds.
    public let estimatedTimeRemaining: TimeInterval?

    /// Whether training can be interrupted at this point.
    public let canInterrupt: Bool

    /// Human-readable status message.
    public var statusMessage: String {
        let percent = Int(overallProgress * 100)
        return "\(phase.displayName) (\(percent)%)"
    }

    /// Creates training progress.
    public init(
        phase: TrainingPhase,
        phaseProgress: Float,
        overallProgress: Float,
        estimatedTimeRemaining: TimeInterval?,
        canInterrupt: Bool
    ) {
        self.phase = phase
        self.phaseProgress = phaseProgress
        self.overallProgress = overallProgress
        self.estimatedTimeRemaining = estimatedTimeRemaining
        self.canInterrupt = canInterrupt
    }
}
