# Handoff: Phase 4 - Incremental Topic Updater

## Overview

**Task**: Implement the main `IncrementalTopicUpdater` actor that orchestrates the complete incremental update flow
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Type**: Implementation
**Predecessors**: Phase 1 (Storage) ✅, Phase 2 (Interruptible Training) ✅, Phase 3 (Topic Matching) ✅

---

## Context

We are building an incremental update system for `TopicModel` that avoids full retraining when new documents are added. This is for a journaling app where entries arrive one-by-one (~365/year).

**Key UX Flow**:
1. User writes new journal entry
2. Entry gets **immediate** topic assignment via centroid distance (<50ms)
3. Entry is buffered for future micro-retrain
4. When buffer reaches 30 entries → background micro-retrain (~200ms)
5. Periodically → full refresh in background (when drift detected)

---

## Previous Phases Complete ✅

### Phase 1: Storage Foundation
```
Sources/SwiftTopics/Incremental/Storage/
├── TopicModelStorage.swift       # Protocol for persistence
├── BufferedEntry.swift           # Entry awaiting retrain
├── IncrementalTopicModelState.swift  # Extended state + Vocabulary + DriftStatistics
└── FileBasedTopicModelStorage.swift  # File-based implementation
```

### Phase 2: Interruptible Training
```
Sources/SwiftTopics/Incremental/
├── Checkpoint/
│   ├── TrainingCheckpoint.swift      # Checkpoint state + TrainingPhase enum
│   └── CheckpointSerializer.swift    # Binary serialization
└── Training/
    └── InterruptibleTrainingRunner.swift  # Orchestrates training pipeline
```

### Phase 3: Topic Matching & Merging
```
Sources/SwiftTopics/Incremental/Merging/
├── HungarianMatcher.swift        # O(n³) optimal bipartite matching
├── TopicMatcher.swift            # Matches new→old topics by centroid similarity
├── TopicIDGenerator.swift        # Stable ID generation
└── ModelMerger.swift             # Merges mini-model into main model
```

### Key Types from Previous Phases

```swift
// Storage protocol (TopicModelStorage.swift)
public protocol TopicModelStorage: Sendable {
    func saveModelState(_ state: IncrementalTopicModelState) async throws
    func loadModelState() async throws -> IncrementalTopicModelState?
    func appendEmbeddings(_ embeddings: [(DocumentID, Embedding)]) async throws
    func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)]
    func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws
    func drainPendingBuffer() async throws -> [BufferedEntry]
    func pendingBufferCount() async throws -> Int
    func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws
    func loadCheckpoint() async throws -> TrainingCheckpoint?
    func clearCheckpoint() async throws
}

// Training runner (InterruptibleTrainingRunner.swift)
public actor InterruptibleTrainingRunner {
    public func runTraining(
        documents: [Document],
        embeddings: [Embedding],
        configuration: TopicModelConfiguration,
        type: TrainingType,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws -> TrainingResult

    public func resumeTraining(
        from checkpoint: TrainingCheckpoint,
        ...
    ) async throws -> TrainingResult
}

// Model merger (ModelMerger.swift)
public struct ModelMerger: Sendable {
    public func matchAndMerge(
        miniModel: MiniModelResult,
        mainModel: IncrementalTopicModelState,
        matchConfig: TopicMatcher.Configuration,
        mergeConfig: Configuration
    ) -> MergeResult
}

// Drift statistics (IncrementalTopicModelState.swift)
public struct DriftStatistics: Sendable, Codable {
    public var recentAverageDistance: Float
    public var overallAverageDistance: Float
    public var recentOutlierRate: Float
    public func needsRefresh(driftThreshold: Float, outlierThreshold: Float) -> Bool
}
```

---

## Phase 4 Objective

Implement the main `IncrementalTopicUpdater` actor that:
1. Processes new documents with immediate topic assignment
2. Buffers documents for micro-retrain
3. Triggers micro-retrain when threshold reached
4. Detects drift and recommends full refresh
5. Handles app lifecycle (resume interrupted training)

---

## Files to Read

1. **Design doc** (Sections 4.3-4.4): `DESIGN_INCREMENTAL_UPDATER.md`
2. **Storage protocol**: `Sources/SwiftTopics/Incremental/Storage/TopicModelStorage.swift`
3. **State structure**: `Sources/SwiftTopics/Incremental/Storage/IncrementalTopicModelState.swift`
4. **Training runner**: `Sources/SwiftTopics/Incremental/Training/InterruptibleTrainingRunner.swift`
5. **Model merger**: `Sources/SwiftTopics/Incremental/Merging/ModelMerger.swift`
6. **Existing TopicModel**: `Sources/SwiftTopics/Model/TopicModel.swift`

---

## Implementation Tasks

### Task 1: IncrementalUpdateConfiguration (~100 LOC)

Configuration for the updater's behavior.

```swift
// New file: Sources/SwiftTopics/Incremental/IncrementalUpdateConfiguration.swift

/// Configuration for incremental topic model updates.
public struct IncrementalUpdateConfiguration: Sendable, Codable {

    // MARK: - Buffer Thresholds

    /// Minimum entries before initial model creation.
    /// Default: 30
    public var coldStartThreshold: Int = 30

    /// Entries to buffer before micro-retrain.
    /// Default: 30
    public var microRetrainThreshold: Int = 30

    /// Maximum buffer size before forced retrain.
    /// Default: 100
    public var maxBufferSize: Int = 100

    // MARK: - Full Refresh Triggers

    /// Corpus growth ratio to trigger full refresh.
    /// Default: 1.5 (50% growth since last full retrain)
    public var fullRefreshGrowthRatio: Float = 1.5

    /// Maximum time between full refreshes.
    /// Default: 180 days
    public var fullRefreshMaxInterval: TimeInterval = 180 * 24 * 60 * 60

    /// Outlier rate threshold for early refresh.
    /// Default: 0.20 (20%)
    public var outlierRateThreshold: Float = 0.20

    /// Drift ratio threshold for refresh.
    /// Default: 1.5 (recent distance 50% higher than overall)
    public var driftRatioThreshold: Float = 1.5

    // MARK: - Topic Matching

    /// Minimum similarity for topic matching after retrain.
    /// Default: 0.7
    public var topicMatchingSimilarityThreshold: Float = 0.7

    // MARK: - Presets

    public static let `default` = IncrementalUpdateConfiguration()

    public static let aggressive = IncrementalUpdateConfiguration(
        microRetrainThreshold: 20,
        fullRefreshGrowthRatio: 1.3,
        outlierRateThreshold: 0.15
    )

    public static let conservative = IncrementalUpdateConfiguration(
        microRetrainThreshold: 50,
        fullRefreshGrowthRatio: 2.0,
        fullRefreshMaxInterval: 365 * 24 * 60 * 60
    )
}
```

### Task 2: TopicAssignment Result Type (~50 LOC)

The result returned from processing a document.

```swift
// New file: Sources/SwiftTopics/Incremental/TopicAssignment.swift

/// Result of assigning a document to a topic.
public struct TopicAssignment: Sendable {

    /// The assigned topic ID (-1 for outlier).
    public let topicID: TopicID

    /// Confidence in the assignment (0-1).
    /// Based on distance to centroid relative to other centroids.
    public let confidence: Float

    /// Distance to the assigned topic's centroid.
    public let distanceToCentroid: Float

    /// Whether this was assigned via transform (true) or full training (false).
    public let isTransformAssignment: Bool

    /// The topic keywords (for display).
    public let topicKeywords: [String]

    /// Whether this document is an outlier.
    public var isOutlier: Bool { topicID.isOutlier }
}
```

### Task 3: IncrementalTopicUpdater Actor (~600 LOC)

The main actor that orchestrates everything.

```swift
// New file: Sources/SwiftTopics/Incremental/IncrementalTopicUpdater.swift

/// Main actor for incremental topic model updates.
///
/// Handles the complete lifecycle of incremental topic modeling:
/// - Immediate topic assignment for new documents
/// - Background micro-retraining when buffer threshold reached
/// - Full refresh when drift is detected
/// - Interruption and resumption across app launches
///
/// ## Usage
///
/// ```swift
/// let updater = try IncrementalTopicUpdater(
///     storage: FileBasedTopicModelStorage(directory: modelDir),
///     modelConfiguration: .default,
///     updateConfiguration: .default
/// )
///
/// // On app launch, resume any interrupted training
/// try await updater.resumeIfNeeded()
///
/// // Process new documents
/// let assignment = try await updater.processDocument(doc, embedding: embedding)
///
/// // Before app termination
/// try await updater.prepareForTermination()
/// ```
public actor IncrementalTopicUpdater {

    // MARK: - Properties

    /// Storage backend for persistence.
    public let storage: TopicModelStorage

    /// Configuration for topic model training.
    public let modelConfiguration: TopicModelConfiguration

    /// Configuration for incremental updates.
    public let updateConfiguration: IncrementalUpdateConfiguration

    /// Current model state (nil during cold start).
    public private(set) var modelState: IncrementalTopicModelState?

    /// Whether a training operation is in progress.
    public private(set) var isTraining: Bool = false

    // MARK: - Private State

    private let trainingRunner: InterruptibleTrainingRunner
    private let merger: ModelMerger
    private var shouldContinueTraining: Bool = true

    // MARK: - Initialization

    /// Creates an incremental updater.
    ///
    /// - Parameters:
    ///   - storage: Storage backend for persistence.
    ///   - modelConfiguration: Configuration for topic model training.
    ///   - updateConfiguration: Configuration for incremental updates.
    public init(
        storage: TopicModelStorage,
        modelConfiguration: TopicModelConfiguration = .default,
        updateConfiguration: IncrementalUpdateConfiguration = .default
    ) async throws

    // MARK: - Document Processing

    /// Processes a new document and returns immediate topic assignment.
    ///
    /// This method:
    /// 1. If no model exists and buffer < threshold: buffers and returns outlier
    /// 2. If no model exists and buffer >= threshold: triggers initial training
    /// 3. If model exists: assigns via centroid distance, buffers for micro-retrain
    /// 4. If buffer >= threshold: triggers background micro-retrain
    ///
    /// - Parameters:
    ///   - document: The document to process.
    ///   - embedding: Pre-computed embedding for the document.
    /// - Returns: Topic assignment (immediate result).
    public func processDocument(
        _ document: Document,
        embedding: Embedding
    ) async throws -> TopicAssignment

    /// Processes multiple documents in batch.
    ///
    /// More efficient than individual calls for bulk imports.
    public func processDocuments(
        _ documents: [Document],
        embeddings: [Embedding]
    ) async throws -> [TopicAssignment]

    // MARK: - Training Control

    /// Forces a micro-retrain with current buffer.
    ///
    /// Call this if you want to incorporate buffered documents
    /// before the automatic threshold is reached.
    public func triggerMicroRetrain() async throws

    /// Triggers a full model refresh.
    ///
    /// This is a long-running operation (~15-60s for 1000 docs).
    /// Should be called from background processing context.
    ///
    /// - Parameter progress: Optional progress callback.
    public func triggerFullRefresh(
        progress: (@Sendable (TrainingProgress) async -> Void)? = nil
    ) async throws

    /// Cancels any in-progress training.
    ///
    /// Training will checkpoint and stop at next opportunity.
    public func cancelTraining()

    // MARK: - Lifecycle

    /// Resumes interrupted training from checkpoint.
    ///
    /// Call this on app launch to continue any interrupted training.
    ///
    /// - Returns: True if training was resumed, false if no checkpoint.
    @discardableResult
    public func resumeIfNeeded() async throws -> Bool

    /// Prepares for app termination.
    ///
    /// Saves checkpoint if training is in progress.
    public func prepareForTermination() async throws

    // MARK: - Queries

    /// Checks if full refresh is recommended based on drift metrics.
    public func shouldTriggerFullRefresh() -> Bool

    /// Returns all topics in current model.
    public func getTopics() -> [Topic]?

    /// Returns drift statistics for monitoring.
    public func getDriftStatistics() -> DriftStatistics?

    /// Returns the current buffer count.
    public func getPendingBufferCount() async throws -> Int

    // MARK: - Private Methods

    /// Assigns document to nearest topic centroid.
    private func assignViaCentroid(
        embedding: Embedding,
        state: IncrementalTopicModelState
    ) -> TopicAssignment

    /// Runs micro-retrain on buffered documents.
    private func runMicroRetrain(
        bufferedEntries: [BufferedEntry]
    ) async throws

    /// Runs full refresh on all documents.
    private func runFullRefresh(
        progress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws

    /// Updates drift statistics after assignment.
    private func updateDriftStatistics(
        distance: Float,
        isOutlier: Bool
    )
}
```

### Task 4: Transform-Only Assignment (~100 LOC)

The fast path for immediate topic assignment.

```swift
// This goes inside IncrementalTopicUpdater

/// Assigns a document to the nearest topic centroid.
///
/// Uses cosine similarity to find the best matching topic.
/// Returns outlier if no topic is similar enough.
private func assignViaCentroid(
    embedding: Embedding,
    state: IncrementalTopicModelState
) -> TopicAssignment {
    guard !state.centroids.isEmpty else {
        return TopicAssignment(
            topicID: .outlier,
            confidence: 0,
            distanceToCentroid: Float.infinity,
            isTransformAssignment: true,
            topicKeywords: []
        )
    }

    // Compute similarity to each centroid
    var bestIndex = -1
    var bestSimilarity: Float = -Float.infinity
    var similarities = [Float]()

    for (index, centroid) in state.centroids.enumerated() {
        let similarity = embedding.cosineSimilarity(centroid)
        similarities.append(similarity)

        if similarity > bestSimilarity {
            bestSimilarity = similarity
            bestIndex = index
        }
    }

    // Check if best match is good enough
    let outlierThreshold: Float = 0.3  // Minimum similarity to not be outlier

    if bestSimilarity < outlierThreshold {
        return TopicAssignment(
            topicID: .outlier,
            confidence: 0,
            distanceToCentroid: 1 - bestSimilarity,
            isTransformAssignment: true,
            topicKeywords: []
        )
    }

    // Compute confidence based on margin over second-best
    let sortedSimilarities = similarities.sorted(by: >)
    let margin = sortedSimilarities.count > 1
        ? sortedSimilarities[0] - sortedSimilarities[1]
        : sortedSimilarities[0]
    let confidence = min(1.0, margin + bestSimilarity) / 2

    let topic = state.topics.first { $0.id.value == bestIndex }
    let keywords = topic?.keywords.prefix(5).map(\.term) ?? []

    return TopicAssignment(
        topicID: TopicID(value: bestIndex),
        confidence: confidence,
        distanceToCentroid: 1 - bestSimilarity,
        isTransformAssignment: true,
        topicKeywords: Array(keywords)
    )
}
```

---

## Directory Structure After Phase 4

```
Sources/SwiftTopics/Incremental/
├── Storage/
│   ├── TopicModelStorage.swift
│   ├── BufferedEntry.swift
│   ├── IncrementalTopicModelState.swift
│   └── FileBasedTopicModelStorage.swift
├── Checkpoint/
│   ├── TrainingCheckpoint.swift
│   └── CheckpointSerializer.swift
├── Training/
│   └── InterruptibleTrainingRunner.swift
├── Merging/
│   ├── TopicMatcher.swift
│   ├── ModelMerger.swift
│   ├── HungarianMatcher.swift
│   └── TopicIDGenerator.swift
├── IncrementalTopicUpdater.swift          # NEW - Main actor
├── IncrementalUpdateConfiguration.swift   # NEW - Configuration
└── TopicAssignment.swift                  # NEW - Result type
```

---

## Algorithm Details

### Document Processing Flow

```
processDocument(doc, embedding)
    │
    ├─► No model exists?
    │       │
    │       ├─► Buffer < coldStartThreshold
    │       │       → Buffer entry, return outlier assignment
    │       │
    │       └─► Buffer >= coldStartThreshold
    │               → Trigger initial training (blocking)
    │               → Return real assignment
    │
    └─► Model exists
            │
            ├─► Assign via centroid (immediate, <1ms)
            │
            ├─► Buffer entry for micro-retrain
            │
            ├─► Update drift statistics
            │
            └─► Buffer >= microRetrainThreshold?
                    → Trigger background micro-retrain
                    → Return assignment (don't wait)
```

### Drift Detection

```swift
func shouldTriggerFullRefresh() -> Bool {
    guard let state = modelState else { return false }

    // Check growth ratio
    let growthRatio = state.growthRatio
    if growthRatio >= updateConfiguration.fullRefreshGrowthRatio {
        return true
    }

    // Check time since last refresh
    if let elapsed = state.timeSinceLastRetrain,
       elapsed >= updateConfiguration.fullRefreshMaxInterval {
        return true
    }

    // Check drift statistics
    let drift = state.driftStatistics
    if drift.needsRefresh(
        driftThreshold: updateConfiguration.driftRatioThreshold,
        outlierThreshold: updateConfiguration.outlierRateThreshold
    ) {
        return true
    }

    return false
}
```

---

## Exit Criteria

- [ ] `IncrementalTopicUpdater` processes documents with <50ms latency
- [ ] Buffer triggers micro-retrain at threshold
- [ ] Micro-retrain merges topics with stable IDs
- [ ] Drift statistics update after each assignment
- [ ] `shouldTriggerFullRefresh()` correctly detects when refresh needed
- [ ] `resumeIfNeeded()` continues interrupted training
- [ ] Tests verify: cold start → buffer → retrain → assignment flow
- [ ] All existing tests still pass
- [ ] Build succeeds with no errors

---

## Test Scenarios to Implement

```swift
@Suite("Incremental Topic Updater")
struct IncrementalTopicUpdaterTests {

    @Test("Cold start buffers until threshold")
    func testColdStartBuffering() async throws {
        // Given: No model exists
        // When: Process 29 documents
        // Then: All return outlier, buffer count = 29
    }

    @Test("Cold start triggers initial training at threshold")
    func testColdStartTraining() async throws {
        // Given: No model exists
        // When: Process 30 documents
        // Then: Model is created, real assignments returned
    }

    @Test("Transform assignment is fast")
    func testTransformLatency() async throws {
        // Given: Model exists with 5 topics
        // When: Process 100 documents, measure time
        // Then: Average < 1ms per document
    }

    @Test("Micro-retrain triggers at threshold")
    func testMicroRetrainTrigger() async throws {
        // Given: Model exists, buffer is empty
        // When: Process 30 documents
        // Then: Micro-retrain is triggered, topics updated
    }

    @Test("Topic IDs remain stable after micro-retrain")
    func testTopicStability() async throws {
        // Given: Model with topics [Travel, Food, Work]
        // When: Micro-retrain with similar documents
        // Then: Same topic IDs exist after retrain
    }

    @Test("Drift statistics update correctly")
    func testDriftStatisticsUpdate() async throws {
        // Given: Model exists
        // When: Process documents with increasing distance
        // Then: recentAverageDistance increases
    }

    @Test("Full refresh triggered by growth")
    func testFullRefreshGrowthTrigger() async throws {
        // Given: Model trained on 100 docs, now has 160 docs
        // When: Check shouldTriggerFullRefresh()
        // Then: Returns true (60% growth > 50% threshold)
    }

    @Test("Resume continues interrupted training")
    func testResumeInterruptedTraining() async throws {
        // Given: Checkpoint exists from interrupted training
        // When: Call resumeIfNeeded()
        // Then: Training completes, checkpoint cleared
    }

    @Test("Cancel training saves checkpoint")
    func testCancelSavesCheckpoint() async throws {
        // Given: Training in progress
        // When: Call cancelTraining()
        // Then: Checkpoint is saved, training stops
    }
}
```

---

## Implementation Notes

### Concurrency Considerations

1. **Actor isolation**: All mutable state is in the actor
2. **Background training**: Use `Task { }` for non-blocking micro-retrain
3. **Cancellation**: Check `shouldContinueTraining` in training loops
4. **Progress callbacks**: Must be `@Sendable`

### Memory Management

1. **Don't load all embeddings**: Stream during full refresh
2. **Buffer limit**: Enforce `maxBufferSize` to prevent unbounded growth
3. **Checkpoint cleanup**: Clear after successful training

### Error Handling

```swift
public enum IncrementalUpdateError: Error, Sendable {
    case modelNotInitialized
    case embeddingDimensionMismatch(expected: Int, got: Int)
    case storageError(underlying: Error)
    case trainingInterrupted(phase: TrainingPhase, progress: Float)
    case insufficientDocuments(required: Int, provided: Int)
}
```

---

## Verification

```bash
# Build
swift build

# Run new tests
swift test --filter "IncrementalTopicUpdater"

# Run all tests to ensure no regressions
swift test

# Check LOC added
wc -l Sources/SwiftTopics/Incremental/*.swift
```

---

## References

- Design doc: `DESIGN_INCREMENTAL_UPDATER.md` (Sections 4.3, 4.6)
- Storage layer: `Sources/SwiftTopics/Incremental/Storage/`
- Training runner: `Sources/SwiftTopics/Incremental/Training/`
- Topic merging: `Sources/SwiftTopics/Incremental/Merging/`
- Existing TopicModel: `Sources/SwiftTopics/Model/TopicModel.swift`

---

*Created: January 2025*
*Depends on: Phase 1 (Storage), Phase 2 (Interruptible Training), Phase 3 (Topic Matching) - All Complete*
*Estimated LOC: ~850 (implementation) + ~400 (tests)*
