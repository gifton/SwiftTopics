# Handoff: Phase 2 - Interruptible Training Components

## Overview

**Task**: Implement interruptible training with checkpoint support for UMAP and HDBSCAN
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Type**: Implementation
**Predecessor**: Phase 1 (Storage Foundation) - COMPLETE

---

## Context

We are building an incremental update system for `TopicModel` that avoids full retraining when new documents are added. This is for a journaling app where entries arrive one-by-one (~365/year).

**Key UX requirement**: Users may open and immediately close the app. Training must be interruptible and resumable.

---

## Phase 1 Completed ✅

The storage foundation is complete. These files exist and are tested:

### New Files (read these first)
```
Sources/SwiftTopics/Incremental/
├── Storage/
│   ├── TopicModelStorage.swift       # Protocol for persistence
│   ├── BufferedEntry.swift           # Entry awaiting retrain
│   ├── IncrementalTopicModelState.swift  # Extended state + Vocabulary + DriftStatistics
│   └── FileBasedTopicModelStorage.swift  # File-based implementation
└── Checkpoint/
    └── TrainingCheckpoint.swift      # Checkpoint state + TrainingPhase enum

Tests/SwiftTopicsTests/Incremental/
└── StorageTests.swift                # 25 passing tests
```

### Key Types from Phase 1

```swift
// Training phases that can be checkpointed
public enum TrainingPhase: Int, Codable, Sendable, CaseIterable {
    case umapKNN = 1
    case umapFuzzySet = 2
    case umapOptimization = 3    // Supports partial checkpoint
    case hdbscanCoreDistance = 4
    case hdbscanMST = 5          // Supports partial checkpoint
    case clusterExtraction = 6
    case representation = 7
    case topicMatching = 8
    case complete = 9
}

// Checkpoint state
public struct TrainingCheckpoint: Sendable, Codable {
    let runID: UUID
    let trainingType: TrainingType  // .microRetrain or .fullRefresh
    let currentPhase: TrainingPhase
    let currentPhaseProgress: Float
    // ... phase-specific state paths
}

// Storage protocol method
func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws
func loadCheckpoint() async throws -> TrainingCheckpoint?
```

---

## Phase 2 Objective

Make UMAP optimization and HDBSCAN MST construction **interruptible** with checkpoint support.

### Why This Matters

- UMAP optimization runs 100-500 epochs, each taking ~30ms
- HDBSCAN MST construction is O(n²) and sequential
- These are the longest-running phases (15-20s for 1000 docs)
- Users closing the app mid-training must be able to resume

---

## Files to Read

1. **Design doc** (Section 4.4 Interruptibility): `DESIGN_INCREMENTAL_UPDATER.md`
2. **Phase 1 checkpoint structure**: `Sources/SwiftTopics/Incremental/Checkpoint/TrainingCheckpoint.swift`
3. **Current UMAP optimizer**: `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift`
4. **Current MST builder**: `Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift`
5. **Storage protocol**: `Sources/SwiftTopics/Incremental/Storage/TopicModelStorage.swift`

---

## Implementation Tasks

### Task 1: Interruptible UMAP Optimizer (~200 LOC)

Modify `UMAPOptimizer` to support:
- **Checkpoint callback**: Save embedding state every N epochs
- **Resume from checkpoint**: Accept initial embedding + starting epoch
- **Cancellation check**: `shouldContinue: () -> Bool` closure

```swift
// Target API
func optimize(
    fuzzySet: FuzzySimplicialSet,
    nEpochs: Int,
    learningRate: Float,
    negativeSampleRate: Int,
    startingEpoch: Int = 0,              // NEW: For resume
    initialEmbedding: [[Float]]? = nil,  // NEW: For resume
    shouldContinue: @escaping () -> Bool = { true },
    onCheckpoint: ((Int, [[Float]]) -> Void)? = nil  // NEW: epoch, embedding
) async -> [[Float]]
```

### Task 2: Interruptible MST Builder (~150 LOC)

Modify `PrimMSTBuilder` (or create wrapper) to support:
- **Partial MST state**: Edges completed so far
- **Resume from partial**: Continue from saved edges
- **Periodic checkpoint**: Save every N edges

```swift
// Target API
func buildMST(
    graph: MutualReachabilityGraph,
    startingEdges: [MSTEdge]? = nil,     // NEW: For resume
    shouldContinue: @escaping () -> Bool = { true },
    onCheckpoint: (([MSTEdge]) -> Void)? = nil
) -> [MSTEdge]
```

### Task 3: InterruptibleTrainingRunner (~400 LOC)

Create the orchestrator that:
- Runs the full training pipeline
- Checks `shouldContinue` between phases
- Saves checkpoints after each phase
- Saves mid-phase checkpoints every 3 seconds
- Handles resume from any checkpoint state

```swift
// New file: Sources/SwiftTopics/Incremental/Training/InterruptibleTrainingRunner.swift

actor InterruptibleTrainingRunner {

    func runTraining(
        documents: [Document],
        embeddings: [Embedding],
        existingState: IncrementalTopicModelState?,
        type: TrainingType,
        storage: TopicModelStorage,
        shouldContinue: @escaping () -> Bool,
        onProgress: @escaping (TrainingProgress) -> Void
    ) async throws -> IncrementalTopicModelState

    func resumeTraining(
        from checkpoint: TrainingCheckpoint,
        storage: TopicModelStorage,
        shouldContinue: @escaping () -> Bool,
        onProgress: @escaping (TrainingProgress) -> Void
    ) async throws -> IncrementalTopicModelState
}
```

### Task 4: Checkpoint Serialization Helpers (~100 LOC)

Create helpers to serialize/deserialize phase-specific state:
- UMAP embedding matrix (binary format)
- MST edge list (binary format)
- k-NN graph (already supported in Phase 1)

---

## Directory Structure After Phase 2

```
Sources/SwiftTopics/Incremental/
├── Storage/
│   ├── TopicModelStorage.swift
│   ├── BufferedEntry.swift
│   ├── IncrementalTopicModelState.swift
│   └── FileBasedTopicModelStorage.swift
├── Checkpoint/
│   ├── TrainingCheckpoint.swift
│   └── CheckpointSerializer.swift       # NEW
└── Training/
    └── InterruptibleTrainingRunner.swift  # NEW
```

---

## Exit Criteria

- [ ] `UMAPOptimizer.optimize()` accepts resume parameters and checkpoint callback
- [ ] MST builder accepts resume parameters and checkpoint callback
- [ ] `InterruptibleTrainingRunner` orchestrates full pipeline with interruption support
- [ ] Tests verify: start training → interrupt → resume → complete
- [ ] All existing tests still pass
- [ ] Build succeeds with no errors

---

## Test Scenarios to Implement

```swift
@Test("UMAP can resume from checkpoint")
func testUMAPResume() async throws {
    // Start optimization, interrupt at epoch 50
    // Resume from checkpoint
    // Verify final result matches uninterrupted run (approximately)
}

@Test("Training runner saves checkpoint on interrupt")
func testCheckpointOnInterrupt() async throws {
    // Start training with shouldContinue that returns false after 2s
    // Verify checkpoint was saved
    // Resume and verify completion
}

@Test("Full training pipeline is interruptible")
func testFullPipelineInterrupt() async throws {
    // Run training with periodic interrupts
    // Resume each time
    // Verify final model is valid
}
```

---

## Implementation Notes

### Checkpoint Timing Strategy
- After each complete phase: Always checkpoint
- During UMAP optimization: Every 50 epochs OR every 3 seconds
- During MST construction: Every 100 edges OR every 3 seconds

### Binary Format for UMAP Embedding
```
Header: nPoints (Int32), nDimensions (Int32)
Data: Float32 values in row-major order
```

### Binary Format for MST Edges
```
Header: edgeCount (Int32)
Per edge: source (Int32), target (Int32), weight (Float32)
```

---

## Verification

```bash
# Build
swift build

# Run new tests
swift test --filter "InterruptibleTraining"

# Run all tests to ensure no regressions
swift test

# Check LOC added
wc -l Sources/SwiftTopics/Incremental/Training/*.swift
```

---

## References

- Design doc: `DESIGN_INCREMENTAL_UPDATER.md` (Section 4.4, 5.3)
- Phase 1 storage: `Sources/SwiftTopics/Incremental/Storage/`
- Current UMAP: `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift`
- Current MST: `Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift`

---

*Created: January 2025*
*Depends on: Phase 1 (Storage Foundation) - Complete*
*Estimated LOC: ~850*
*Estimated Time: 3-4 days*
