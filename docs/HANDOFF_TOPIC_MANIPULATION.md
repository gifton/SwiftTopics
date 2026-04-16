# Handoff: Topic Manipulation Methods

## Overview

**Task**: Implement topic merge and reduce functionality for TopicModel
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Estimated LOC**: ~140
**Estimated Time**: 1-2 hours

## Context

SwiftTopics is a BERTopic-inspired topic modeling library. The core pipeline and convenience methods (`findTopics`, `search`) are complete. Two topic manipulation methods remain: merging specific topics and reducing total topic count.

## Read First

1. `Sources/SwiftTopics/Model/TopicModel.swift` - Main orchestrator (see existing `findTopics`, `search` implementations)
2. `Sources/SwiftTopics/Model/TopicModelState.swift` - State structure with `FittedTopicModelState`
3. `Sources/SwiftTopics/Core/TopicModelResult.swift` - `Topic`, `TopicID`, `TopicAssignment` types
4. `Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift` - For recomputing keywords after merge

## Tasks

### 1. merge(topics: [Int]) (~60 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Combine multiple topics into a single topic.

**Interface**:
```swift
/// Merges multiple topics into a single topic.
///
/// All documents assigned to the specified topics are reassigned to the new merged topic.
/// Keywords are recomputed using c-TF-IDF over the combined documents.
///
/// - Parameter topicIds: IDs of topics to merge (must have at least 2).
/// - Returns: The newly created merged topic.
/// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
/// - Throws: `TopicModelError.invalidInput` if any ID doesn't exist or fewer than 2 IDs provided.
///
/// ## Example
/// ```swift
/// // Merge topics 0 and 2 into a single topic
/// let merged = try await model.merge(topics: [0, 2])
/// print("Merged topic has \(merged.size) documents")
/// ```
public func merge(topics topicIds: [Int]) async throws -> Topic
```

**Implementation Steps**:
1. Validate model is fitted
2. Validate at least 2 topic IDs provided
3. Validate all topic IDs exist in `fittedState.topics`
4. Collect all document indices assigned to these topics from `fittedState.assignment`
5. Choose new topic ID strategy: use the **lowest** of the merged IDs (maintains ordering intuition)
6. Create new merged topic:
   - Size = sum of merged topic sizes
   - Centroid = mean of merged document embeddings (use `fittedState.embeddings`)
   - Keywords = recompute using c-TF-IDF (see below)
7. Update `fittedState`:
   - Update `assignment` labels for affected documents
   - Remove old topics from `topics` array
   - Add new merged topic
   - Recalculate centroids array
8. Return the new `Topic`

**Keyword Recomputation**:
```swift
// Option A: Simple merge of existing keywords (fast but less accurate)
// Combine keywords from all merged topics, deduplicate, take top N by score

// Option B: Full c-TF-IDF recompute (more accurate)
// Use CTFIDFRepresenter on just the merged cluster's documents
// This requires access to original documents (stored in fittedState.documents)
```

Recommend **Option B** for accuracy since documents are available in `fittedState.documents`.

**Edge Cases**:
- Merging topics that include the outlier topic (-1): Should this be allowed? Recommend: NO, throw error
- Merging all topics into one: Valid operation
- Topic IDs that don't exist: Throw `invalidInput`

---

### 2. reduce(to count: Int) (~80 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Reduce total topic count by iteratively merging most similar pairs.

**Interface**:
```swift
/// Reduces the number of topics by merging similar ones.
///
/// Uses hierarchical agglomerative clustering to iteratively merge
/// the most similar topic pairs until the target count is reached.
/// Similarity is measured by centroid distance in embedding space.
///
/// - Parameter count: Target number of topics (must be >= 1 and < current count).
/// - Returns: Array of final topics after reduction.
/// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
/// - Throws: `TopicModelError.invalidInput` if count is invalid.
///
/// ## Example
/// ```swift
/// // Reduce from 10 topics to 5
/// let reducedTopics = try await model.reduce(to: 5)
/// print("Now have \(reducedTopics.count) topics")
/// ```
public func reduce(to count: Int) async throws -> [Topic]
```

**Implementation Steps**:
1. Validate model is fitted
2. Get current non-outlier topic count
3. Validate `count >= 1` and `count < currentTopicCount`
4. Compute pairwise topic similarity matrix:
   - Use **centroid Euclidean distance** (simpler) or cosine similarity
   - Matrix is symmetric, only need upper triangle
5. While current topic count > target:
   a. Find most similar pair (smallest distance)
   b. Call `merge(topics:)` for that pair
   c. Update similarity matrix (remove merged rows/cols, add new topic row/col)
6. Return `fittedState.topics`

**Similarity Computation**:
```swift
// For each pair of topics (i, j):
let similarity = euclideanDistance(centroids[i], centroids[j])
// Lower distance = more similar = merge first
```

**Optimization Note**:
- For small topic counts (<50), recomputing full matrix after each merge is fine
- For larger counts, could use incremental updates, but likely unnecessary for typical use cases

---

## Testing

Add tests to `Tests/SwiftTopicsTests/SwiftTopicsTests.swift`:

```swift
// MARK: - Topic Manipulation Tests

@Test("merge combines topics correctly")
func testMergeTopics() async throws {
    // Setup: Create model with 3+ clear clusters
    // Merge two of them
    // Verify: combined size, new topic exists, old topics removed
}

@Test("merge recomputes keywords")
func testMergeRecomputesKeywords() async throws {
    // Verify merged topic has keywords from combined documents
}

@Test("merge throws for invalid topic IDs")
func testMergeInvalidTopics() async throws {
    // Test: non-existent ID, only 1 ID, empty array
}

@Test("merge throws for outlier topic")
func testMergeOutlierTopic() async throws {
    // Attempting to merge outlier topic (-1) should throw
}

@Test("reduce decreases topic count")
func testReduceTopics() async throws {
    // Setup: Model with 5+ topics
    // Reduce to 3
    // Verify: exactly 3 topics remain
}

@Test("reduce merges most similar first")
func testReduceMergesSimilar() async throws {
    // Setup: Create clusters with known distances
    // Reduce by 1
    // Verify: the closest pair was merged
}

@Test("reduce throws for invalid count")
func testReduceInvalidCount() async throws {
    // Test: count >= current count, count < 1
}

@Test("reduce to 1 creates single topic")
func testReduceToOne() async throws {
    // Should be able to reduce everything to 1 topic
}
```

---

## Dependencies

The implementation relies on:
- `FittedTopicModelState.documents` - for keyword recomputation
- `FittedTopicModelState.embeddings` - for centroid calculation
- `FittedTopicModelState.centroids` - for similarity computation
- `CTFIDFRepresenter` - for keyword extraction (may need to call directly)

These were added in the previous implementation of `findTopics` and `search`.

---

## Exit Criteria

- [ ] `merge(topics:)` implemented with full keyword recomputation
- [ ] `reduce(to:)` implemented with centroid-based similarity
- [ ] All 8+ new tests pass
- [ ] Existing 146 tests still pass: `swift test`
- [ ] No new warnings: `swift build`
- [ ] Doc comments on both public methods

---

## Verification Commands

```bash
# Build
swift build

# Run all tests
swift test

# Run specific tests for new features
swift test --filter "testMerge"
swift test --filter "testReduce"

# Verify test count increased
swift test 2>&1 | grep "tests in"
```

---

*Created: January 2025*
*Depends on: findTopics and search implementation (completed)*
