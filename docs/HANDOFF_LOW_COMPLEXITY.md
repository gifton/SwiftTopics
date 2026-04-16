# Handoff: Low-Complexity Feature Additions

## Overview

**Task**: Complete missing TopicModel convenience methods
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Estimated LOC**: ~260
**Estimated Time**: 2-3 hours

## Context

SwiftTopics is a BERTopic-inspired topic modeling library. The core pipeline (embed → reduce → cluster → represent → evaluate) is complete and all 137 tests pass. Several convenience methods specified in SPEC.md are not yet implemented.

## Read First

1. `SPEC.md` (Sections 2.1, 2.6) - Feature specifications
2. `Sources/SwiftTopics/Model/TopicModel.swift` - Main orchestrator class
3. `Sources/SwiftTopics/Protocols/EmbeddingProvider.swift` - Provider protocol
4. `Sources/SwiftTopics/Core/TopicModelResult.swift` - Result types

## Tasks (In Order of Complexity)

### 1. PrecomputedEmbeddingProvider (~30 LOC)

**File**: `Sources/SwiftTopics/Embedding/PrecomputedEmbeddingProvider.swift` (new file)

**Purpose**: Simple provider that returns pre-computed embeddings from a dictionary. Useful when embeddings are generated externally or cached.

**Interface**:
```swift
public struct PrecomputedEmbeddingProvider: EmbeddingProvider {
    private let embeddings: [String: Embedding]
    public var dimension: Int

    /// Creates a provider with pre-computed embeddings.
    /// - Parameter embeddings: Dictionary mapping text to embedding.
    public init(embeddings: [String: Embedding])

    /// Returns the embedding for the given text.
    /// - Throws: EmbeddingError.textNotFound if text not in dictionary.
    public func embed(_ text: String) async throws -> Embedding

    /// Returns embeddings for multiple texts.
    public func embedBatch(_ texts: [String]) async throws -> [Embedding]
}
```

**Notes**:
- Add `EmbeddingError.textNotFound(String)` case if it doesn't exist
- Dimension should be inferred from first embedding in dictionary

---

### 2. findTopics(for: String) (~40 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Given arbitrary text, embed it and find which topics it belongs to.

**Interface**:
```swift
/// Finds topic assignments for arbitrary text.
///
/// - Parameter text: The text to classify.
/// - Returns: Topic assignments sorted by probability (highest first).
/// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
public func findTopics(for text: String) async throws -> [TopicAssignment]
```

**Implementation**:
1. Check model is fitted (has `embeddingProvider` and `fittedState`)
2. Embed the text using stored provider
3. Transform through fitted reducer
4. Compute distance to each cluster centroid
5. Convert distances to probabilities (softmax or normalized inverse distance)
6. Return `TopicAssignment` array sorted by probability

**Dependencies**:
- Requires cluster centroids to be stored in `TopicModelState`
- Check if centroids are already computed during `fit()`; if not, add that

---

### 3. search(query: String, topK: Int) (~50 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Semantic search - find documents most similar to query.

**Interface**:
```swift
/// Searches for documents similar to the query.
///
/// - Parameters:
///   - query: Search query text.
///   - topK: Maximum number of results (default: 10).
/// - Returns: Documents with similarity scores, sorted by relevance.
/// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
public func search(query: String, topK: Int = 10) async throws -> [(document: Document, score: Float)]
```

**Implementation**:
1. Embed query using stored provider
2. Compute cosine similarity to all stored document embeddings
3. Sort by similarity (descending)
4. Return top-K with scores

**Dependencies**:
- **Important**: Requires storing document embeddings in `TopicModelState`
- Currently `TopicModelState` may not store embeddings (check!)
- If not stored, either:
  - Add `documentEmbeddings: [Embedding]` to state
  - Or make this method require embeddings as parameter

---

### 4. merge(topics: [Int]) (~60 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Combine multiple topics into one.

**Interface**:
```swift
/// Merges multiple topics into a single topic.
///
/// - Parameter topicIds: IDs of topics to merge (must have at least 2).
/// - Returns: The newly created merged topic.
/// - Throws: `TopicModelError.invalidTopicIds` if any ID doesn't exist.
public func merge(topics topicIds: [Int]) async throws -> Topic
```

**Implementation**:
1. Validate all topic IDs exist
2. Collect all document IDs assigned to these topics
3. Create new topic ID (max existing + 1)
4. Reassign documents to new topic in `documentTopics`
5. Recompute c-TF-IDF keywords for merged cluster
6. Remove old topics from state
7. Invalidate coherence scores (need recomputation)
8. Return new `Topic`

**Notes**:
- Consider what happens to outliers (-1 topic)
- Merged topic ID strategy: use lowest of merged IDs? Or new ID?

---

### 5. reduce(to count: Int) (~80 LOC)

**File**: `Sources/SwiftTopics/Model/TopicModel.swift`

**Purpose**: Reduce total topics to N by merging most similar pairs.

**Interface**:
```swift
/// Reduces the number of topics by merging similar ones.
///
/// - Parameter count: Target number of topics.
/// - Returns: Array of final topics after reduction.
/// - Throws: `TopicModelError.invalidCount` if count >= current topic count.
public func reduce(to count: Int) async throws -> [Topic]
```

**Implementation**:
1. Validate count < current topic count
2. Compute topic similarity matrix:
   - Option A: Centroid distance in embedding space
   - Option B: Keyword overlap (Jaccard similarity)
   - Recommend: Centroid distance
3. While topic count > target:
   - Find most similar pair
   - Call `merge(topics:)`
   - Update similarity matrix
4. Return final topics

**Notes**:
- Hierarchical agglomerative approach (bottom-up)
- Could use different linkage methods (single, complete, average)
- Start with simple "merge closest centroids"

---

## Testing

Add tests to `Tests/SwiftTopicsTests/SwiftTopicsTests.swift`:

```swift
// MARK: - TopicModel Convenience Methods Tests

@Test("PrecomputedEmbeddingProvider returns stored embeddings")
func testPrecomputedProvider() async throws { ... }

@Test("PrecomputedEmbeddingProvider throws for unknown text")
func testPrecomputedProviderUnknownText() async throws { ... }

@Test("findTopics returns assignments for text")
func testFindTopics() async throws { ... }

@Test("findTopics throws when not fitted")
func testFindTopicsNotFitted() async throws { ... }

@Test("search returns similar documents")
func testSearch() async throws { ... }

@Test("merge combines topics correctly")
func testMergeTopics() async throws { ... }

@Test("merge throws for invalid topic IDs")
func testMergeInvalidTopics() async throws { ... }

@Test("reduce decreases topic count")
func testReduceTopics() async throws { ... }
```

---

## Exit Criteria

- [ ] All 5 features implemented
- [ ] Tests pass: `swift test`
- [ ] No new warnings: `swift build`
- [ ] Each method has doc comments
- [ ] Edge cases handled (unfitted model, empty input, invalid IDs)

---

## Verification Commands

```bash
# Build
swift build

# Run all tests
swift test

# Run specific tests for new features
swift test --filter "Precomputed"
swift test --filter "findTopics"
swift test --filter "search"
swift test --filter "merge"
swift test --filter "reduce"
```

---

*Created: January 2025*
