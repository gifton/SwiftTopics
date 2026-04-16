# Handoff: Phase 3 - Topic Matching & Merging

## Overview

**Task**: Implement topic stability across retrains via matching and merging
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Type**: Implementation
**Predecessors**: Phase 1 (Storage Foundation) ✅, Phase 2 (Interruptible Training) ✅

---

## Context

We are building an incremental update system for `TopicModel` that avoids full retraining when new documents are added. This is for a journaling app where entries arrive one-by-one (~365/year).

**Key UX requirement**: When topics are retrained (micro-retrain on 30 new entries, or full refresh), topic IDs must remain stable. Users who tagged journal entries with "Travel" should still see those entries under "Travel" after a retrain.

---

## Previous Phases Complete ✅

### Phase 1: Storage Foundation
```
Sources/SwiftTopics/Incremental/
├── Storage/
│   ├── TopicModelStorage.swift       # Protocol for persistence
│   ├── BufferedEntry.swift           # Entry awaiting retrain
│   ├── IncrementalTopicModelState.swift  # Extended state + Vocabulary + DriftStatistics
│   └── FileBasedTopicModelStorage.swift  # File-based implementation
└── Checkpoint/
    └── TrainingCheckpoint.swift      # Checkpoint state + TrainingPhase enum
```

### Phase 2: Interruptible Training Components
```
Sources/SwiftTopics/Incremental/
├── Checkpoint/
│   └── CheckpointSerializer.swift    # Binary serialization for embeddings/MST
└── Training/
    └── InterruptibleTrainingRunner.swift  # Orchestrates training pipeline

Modified:
├── Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift
│   └── Added: optimizeInterruptible() with checkpoint callbacks
└── Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift
    └── Added: InterruptibleMSTBuilder with resumption support
```

### Key Types from Previous Phases

```swift
// Training phases (TrainingCheckpoint.swift)
public enum TrainingPhase: Int, Codable, Sendable {
    case umapKNN = 1
    case umapFuzzySet = 2
    case umapOptimization = 3
    case hdbscanCoreDistance = 4
    case hdbscanMST = 5
    case clusterExtraction = 6
    case representation = 7
    case topicMatching = 8      // <-- THIS PHASE
    case complete = 9
}

// Topic structure (Topic.swift)
public struct Topic: Sendable, Identifiable {
    public let id: TopicID
    public let keywords: [String]
    public let centroid: Embedding?
    // ...
}

// State after training (IncrementalTopicModelState.swift)
public struct IncrementalTopicModelState: Sendable, Codable {
    public let topics: [Topic]
    public let assignments: ClusterAssignment
    public let centroids: [Embedding]
    public let vocabulary: IncrementalVocabulary
    // ...
}
```

---

## Phase 3 Objective

Implement topic matching and merging to ensure **topic ID stability** when:
1. **Micro-retrain**: 30 new documents trigger mini-model training, which must map back to existing topics
2. **Full refresh**: Complete retraining must preserve topic IDs where possible

### Why This Matters

Without topic matching:
- User tags entry with "Topic 3: Travel"
- After retrain, same cluster might become "Topic 7"
- User's mental model and any external references break

With topic matching:
- Same cluster maintains "Topic 3: Travel" ID
- Only genuinely new clusters get new IDs
- Merged/split topics get sensible handling

---

## Files to Read

1. **Design doc** (Section 4.5 Topic Matching): `DESIGN_INCREMENTAL_UPDATER.md`
2. **Topic structure**: `Sources/SwiftTopics/Model/Topic.swift`
3. **Cluster assignment**: `Sources/SwiftTopics/Clustering/ClusterAssignment.swift`
4. **Embedding/distance utilities**: `Sources/SwiftTopics/Model/Embedding.swift`
5. **Current TopicModel**: `Sources/SwiftTopics/Model/TopicModel.swift`

---

## Implementation Tasks

### Task 1: TopicMatcher (~200 LOC)

Match topics from a new training run to existing topics based on centroid similarity.

```swift
// New file: Sources/SwiftTopics/Incremental/Merging/TopicMatcher.swift

/// Matches new topics to existing topics for ID stability.
public struct TopicMatcher: Sendable {

    /// Matching configuration.
    public struct Configuration: Sendable {
        /// Minimum similarity (cosine) to consider a match.
        public let similarityThreshold: Float  // Default: 0.7

        /// Whether to allow many-to-one matching (merges).
        public let allowMerges: Bool  // Default: true

        /// Whether to allow one-to-many matching (splits).
        public let allowSplits: Bool  // Default: true
    }

    /// Result of topic matching.
    public struct MatchResult: Sendable {
        /// Mapping from new topic index to old topic ID.
        /// nil = new topic (no match found)
        public let newToOld: [Int: TopicID?]

        /// Topics that were merged (multiple old → one new).
        public let merges: [(oldIDs: [TopicID], newIndex: Int)]

        /// Topics that were split (one old → multiple new).
        public let splits: [(oldID: TopicID, newIndices: [Int])]

        /// Genuinely new topics (no similar existing topic).
        public let newTopicIndices: [Int]

        /// Old topics that disappeared (no matching new topic).
        public let retiredTopicIDs: [TopicID]
    }

    /// Matches new topics to existing topics.
    ///
    /// Uses Hungarian algorithm for optimal bipartite matching
    /// based on centroid cosine similarity.
    ///
    /// - Parameters:
    ///   - newCentroids: Centroids from new training run.
    ///   - oldTopics: Existing topics with their IDs.
    ///   - configuration: Matching configuration.
    /// - Returns: Match result with ID mappings.
    public func match(
        newCentroids: [Embedding],
        oldTopics: [Topic],
        configuration: Configuration = .default
    ) -> MatchResult
}
```

### Task 2: ModelMerger (~300 LOC)

Merge a mini-model's topics into the main model, using match results.

```swift
// New file: Sources/SwiftTopics/Incremental/Merging/ModelMerger.swift

/// Merges a mini-model into the main model.
public struct ModelMerger: Sendable {

    /// Configuration for merging.
    public struct Configuration: Sendable {
        /// How to update centroids when merging.
        public let centroidMergeStrategy: CentroidMergeStrategy

        /// How to combine keywords when merging.
        public let keywordMergeStrategy: KeywordMergeStrategy

        /// Maximum number of keywords per topic.
        public let maxKeywordsPerTopic: Int  // Default: 10
    }

    public enum CentroidMergeStrategy: Sendable {
        /// Weighted average by document count.
        case weightedAverage

        /// Use the larger cluster's centroid.
        case largerCluster

        /// Recompute from all documents.
        case recompute
    }

    public enum KeywordMergeStrategy: Sendable {
        /// Union of keywords, re-ranked by c-TF-IDF.
        case unionReranked

        /// Keep existing keywords, add new ones.
        case existingFirst

        /// Recompute from merged vocabulary.
        case recompute
    }

    /// Result of merging.
    public struct MergeResult: Sendable {
        /// Updated topics with stable IDs.
        public let topics: [Topic]

        /// Updated centroids.
        public let centroids: [Embedding]

        /// Updated document assignments.
        public let assignments: ClusterAssignment

        /// ID of the next new topic (for future topics).
        public let nextTopicID: Int

        /// Summary of changes.
        public let summary: MergeSummary
    }

    public struct MergeSummary: Sendable {
        public let topicsUnchanged: Int
        public let topicsUpdated: Int
        public let topicsCreated: Int
        public let topicsMerged: Int
        public let topicsRetired: Int
    }

    /// Merges mini-model results into main model.
    ///
    /// - Parameters:
    ///   - miniModel: Topics and assignments from mini-model.
    ///   - mainModel: Current main model state.
    ///   - newDocuments: Documents in the mini-model.
    ///   - matchResult: Result from TopicMatcher.
    ///   - configuration: Merge configuration.
    /// - Returns: Merged result with stable topic IDs.
    public func merge(
        miniModel: MiniModelResult,
        mainModel: IncrementalTopicModelState,
        newDocuments: [Document],
        matchResult: TopicMatcher.MatchResult,
        configuration: Configuration = .default
    ) -> MergeResult
}

/// Result from training a mini-model on new documents.
public struct MiniModelResult: Sendable {
    public let topics: [Topic]
    public let centroids: [Embedding]
    public let assignments: ClusterAssignment
    public let vocabulary: IncrementalVocabulary
}
```

### Task 3: Hungarian Algorithm (~150 LOC)

Implement optimal bipartite matching for topic assignment.

```swift
// New file: Sources/SwiftTopics/Incremental/Merging/HungarianMatcher.swift

/// Optimal bipartite matching using the Hungarian algorithm.
///
/// Given an n×m cost matrix, finds the assignment that minimizes total cost.
/// Used for matching new topics to old topics based on distance (1 - similarity).
public struct HungarianMatcher: Sendable {

    /// Finds optimal assignment for a cost matrix.
    ///
    /// - Parameter costs: n×m cost matrix where costs[i][j] is the cost
    ///   of assigning row i to column j. Use Float.infinity for forbidden assignments.
    /// - Returns: Array of (row, col) pairs representing the optimal assignment.
    public func solve(costs: [[Float]]) -> [(row: Int, col: Int)]
}
```

### Task 4: Topic ID Management (~100 LOC)

Helper for generating and tracking topic IDs.

```swift
// Can be added to IncrementalTopicModelState.swift or separate file

/// Manages stable topic ID generation.
public struct TopicIDGenerator: Sendable, Codable {
    private var nextID: Int

    public init(startingFrom: Int = 0)

    /// Generates a new unique topic ID.
    public mutating func next() -> TopicID

    /// Updates to track the highest seen ID.
    public mutating func observe(_ id: TopicID)
}
```

---

## Directory Structure After Phase 3

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
└── Merging/                              # NEW
    ├── TopicMatcher.swift                # NEW
    ├── ModelMerger.swift                 # NEW
    └── HungarianMatcher.swift            # NEW
```

---

## Algorithm Details

### Hungarian Algorithm

The Hungarian algorithm solves the assignment problem in O(n³) time. For topic matching:

1. Create cost matrix: `cost[i][j] = 1 - cosineSimilarity(newCentroid[i], oldCentroid[j])`
2. Run Hungarian algorithm to find minimum-cost bipartite matching
3. Filter matches below similarity threshold
4. Handle unmatched topics (new topics, retired topics)

```
Example:
Old Topics: [Travel, Food, Work]
New Topics: [A, B, C, D]

Similarity Matrix:
         Travel  Food  Work
    A      0.9   0.2   0.3
    B      0.1   0.85  0.2
    C      0.2   0.3   0.8
    D      0.3   0.4   0.1   <- No good match

Hungarian Match:
  A → Travel (0.9)
  B → Food (0.85)
  C → Work (0.8)
  D → NEW TOPIC (below threshold)
```

### Centroid Merge Strategy

When a mini-model topic matches an existing topic:

```swift
// Weighted average (recommended)
let totalDocs = oldDocCount + newDocCount
let mergedCentroid = (oldCentroid * oldDocCount + newCentroid * newDocCount) / totalDocs
```

---

## Exit Criteria

- [ ] `TopicMatcher` matches new topics to old topics using Hungarian algorithm
- [ ] `ModelMerger` produces stable topic IDs after micro-retrain
- [ ] New topics get new IDs, matched topics keep existing IDs
- [ ] Tests verify: match → merge → assignments still valid
- [ ] All existing tests still pass
- [ ] Build succeeds with no errors

---

## Test Scenarios to Implement

```swift
@Test("Topics maintain IDs after micro-retrain with similar data")
func testTopicIDStability() async throws {
    // Given: model with 3 topics [Travel, Food, Work]
    // When: micro-retrain adds 30 new entries (similar to existing)
    // Then: same 3 topics with same IDs exist
}

@Test("New cluster gets new topic ID")
func testNewTopicCreation() async throws {
    // Given: model with 3 topics
    // When: micro-retrain adds entries about completely new subject
    // Then: 4th topic created with new ID, others unchanged
}

@Test("Merged topics handled correctly")
func testTopicMerge() async throws {
    // Given: model with [Travel-Europe, Travel-Asia]
    // When: retrain produces single [Travel] cluster
    // Then: merged topic keeps one ID, other retired
}

@Test("Hungarian matching finds optimal assignment")
func testHungarianOptimal() throws {
    // Test that Hungarian algorithm finds minimum cost assignment
}

@Test("Below-threshold matches become new topics")
func testSimilarityThreshold() throws {
    // Given: similarity threshold 0.7
    // When: best match is 0.5
    // Then: treated as new topic, not matched
}
```

---

## Implementation Notes

### Similarity Threshold

- Default: 0.7 cosine similarity
- Too low (0.5): False matches, unstable IDs
- Too high (0.9): Too many "new" topics, fragmentation
- Tunable per-application needs

### Handling Edge Cases

1. **Empty mini-model**: No new topics, just return existing state
2. **Empty main model**: All topics are "new", start ID sequence
3. **All topics merge into one**: Common after corpus refinement
4. **Topic splits**: One old topic maps to multiple new (usually not matched)

### Performance Considerations

- Hungarian: O(n³) but n = topic count (typically 5-20), so fast
- Centroid comparison: O(n × m × d) where d = embedding dimension
- For journaling app (small topic count), this is instant

---

## Verification

```bash
# Build
swift build

# Run new tests
swift test --filter "TopicMatcher\|ModelMerger\|Hungarian"

# Run all tests to ensure no regressions
swift test

# Check LOC added
wc -l Sources/SwiftTopics/Incremental/Merging/*.swift
```

---

## References

- Design doc: `DESIGN_INCREMENTAL_UPDATER.md` (Section 4.5)
- Phase 1 storage: `Sources/SwiftTopics/Incremental/Storage/`
- Phase 2 training: `Sources/SwiftTopics/Incremental/Training/`
- Topic model: `Sources/SwiftTopics/Model/TopicModel.swift`
- BERTopic merge_models: [GitHub](https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py#L1234)

---

*Created: January 2025*
*Depends on: Phase 1 (Storage), Phase 2 (Interruptible Training) - Complete*
*Estimated LOC: ~750*
*Estimated Time: 3-4 days*
