# SwiftTopics Phase 5 Kickoff: Topic Representation (c-TF-IDF)

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-4 are complete.

**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`

## Completed Work

### Phase 0: Core Types (~300 LOC)
- `Core/Document.swift` - Document, DocumentID, DocumentMetadata
- `Core/Embedding.swift` - Vector wrapper with VectorCore integration
- `Core/Topic.swift` - Topic with keywords and statistics
- `Core/ClusterAssignment.swift` - HDBSCAN output structure
- `Core/TopicModelResult.swift` - Complete pipeline result
- `Protocols/*.swift` - EmbeddingProvider, DimensionReducer, ClusteringEngine, TopicRepresenter

### Phase 1: GPU Integration (~200 LOC)
- `Acceleration/GPUContext.swift` - TopicsGPUContext actor wrapping Metal4Context
- `Utilities/Eigendecomposition.swift` - LAPACK ssyev/dsyev for eigendecomposition
- `Utilities/RandomState.swift` - xorshift128+ seedable RNG

### Phase 2: Spatial Indexing (~500 LOC)
- `Clustering/SpatialIndex/SpatialIndex.swift` - Protocol + DistanceMetric enum
- `Clustering/SpatialIndex/BallTree.swift` - CPU k-NN with branch pruning
- `Clustering/SpatialIndex/GPUBatchKNN.swift` - GPU batch k-NN via FusedL2TopKKernel

### Phase 3: HDBSCAN Clustering (~750 LOC)
- `Clustering/HDBSCAN/CoreDistance.swift` - k-th neighbor distance computation
- `Clustering/HDBSCAN/MutualReachability.swift` - Density-aware distance metric
- `Clustering/HDBSCAN/MinimumSpanningTree.swift` - Prim's algorithm + Union-Find
- `Clustering/HDBSCAN/ClusterHierarchyBuilder.swift` - Dendrogram with stability scores
- `Clustering/HDBSCAN/ClusterExtraction.swift` - EOM/leaf cluster selection
- `Clustering/HDBSCAN/HDBSCAN.swift` - Orchestrator implementing ClusteringEngine

### Phase 4: PCA Dimensionality Reduction (~390 LOC)
- `Reduction/PCA.swift` - PCAReducer implementing DimensionReducer protocol
  - Center data, compute covariance via BLAS, eigendecomposition, projection
  - Whitening support, variance ratio threshold selection
  - Builder pattern, convenience functions, array extensions

## Phase 5: Topic Representation - c-TF-IDF

**Duration**: 3-4 days | **LOC**: ~400

Extract interpretable keywords for each cluster using class-based TF-IDF.

### What is c-TF-IDF?

Class-based TF-IDF (c-TF-IDF) is a variant of TF-IDF that treats each cluster as a single "document". This produces keywords that distinguish one topic from others:

```
c-TF-IDF(term t, cluster c) = tf(t, c) × log(1 + A / tf(t, corpus))
```

Where:
- `tf(t, c)` = frequency of term t in all documents of cluster c
- `A` = average number of words per cluster
- `tf(t, corpus)` = total frequency of term t across all clusters

### Deliverables

#### 5.1 Tokenizer
```
Sources/SwiftTopics/Representation/Tokenizer.swift
```
- Whitespace/punctuation tokenization
- Lowercase normalization
- Stop word filtering (configurable list)
- Minimum token length filter
- N-gram support (optional, for phrases)

#### 5.2 Vocabulary Builder
```
Sources/SwiftTopics/Representation/Vocabulary.swift
```
- Build vocabulary from corpus
- Term frequency per document
- Document frequency per term
- Vocabulary pruning (min_df, max_df)

#### 5.3 c-TF-IDF Computation
```
Sources/SwiftTopics/Representation/cTFIDF.swift
```
- Aggregate documents by cluster
- Compute term frequency per cluster
- Compute c-TF-IDF scores
- Rank terms by score per cluster
- Return top-N keywords per topic

#### 5.4 Topic Representer Implementation
```
Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift
```
- Implement `TopicRepresenter` protocol
- Configuration for top-K keywords, min/max df, etc.

### TopicRepresenter Protocol (from Phase 0)

```swift
public protocol TopicRepresenter: Sendable {
    associatedtype Configuration: RepresentationConfiguration

    var configuration: Configuration { get }

    /// Extracts topic representations from clustered documents.
    func represent(
        documents: [Document],
        assignments: ClusterAssignment
    ) async throws -> [Topic]
}
```

### Algorithm Steps

1. **Tokenize**: Split each document into tokens
2. **Build vocabulary**: Create term -> index mapping, compute document frequencies
3. **Aggregate by cluster**: Concatenate all documents in each cluster
4. **Compute term frequency per cluster**: Count tokens in each cluster
5. **Compute c-TF-IDF scores**: Apply the formula
6. **Extract top-K keywords**: Sort by score, return top-K per cluster

### Key Considerations

#### Stop Words
Provide a default English stop word list but allow customization:
```swift
public struct TokenizerConfiguration: Sendable, Codable {
    public let stopWords: Set<String>
    public let minTokenLength: Int  // default: 2
    public let maxTokenLength: Int  // default: 50
    public let lowercase: Bool      // default: true
    public let removeNumbers: Bool  // default: false

    public static let `default`: TokenizerConfiguration
    public static let english: TokenizerConfiguration  // with English stop words
}
```

#### Vocabulary Pruning
Remove terms that appear too rarely or too frequently:
```swift
public struct VocabularyConfiguration: Sendable, Codable {
    public let minDocumentFrequency: Int     // default: 2 (term must appear in >= 2 docs)
    public let maxDocumentFrequency: Float?  // default: 0.95 (ignore terms in >95% of docs)
    public let maxVocabularySize: Int?       // default: nil (no limit)
}
```

#### Empty Clusters
Handle clusters with no documents gracefully (return empty keyword list).

### Exit Criteria
- [ ] Keywords are interpretable and distinctive per cluster
- [ ] Handles empty clusters gracefully
- [ ] Handles single-document clusters
- [ ] Performance: <100ms for 100 clusters
- [ ] swift build passes
- [ ] swift test passes

### Constraints
- Swift 6 strict concurrency (all types must be Sendable)
- Use existing `Document` and `Topic` types from Phase 0
- Implement the `TopicRepresenter` protocol
- Target: iOS/macOS/visionOS 26+

### Reference Materials
- SPEC.md Part 2.4 (Topic Representation Layer)
- `Protocols/TopicRepresenter.swift` for the protocol interface
- `Core/Document.swift` for Document type
- `Core/Topic.swift` for Topic type (output)
- `Core/ClusterAssignment.swift` for cluster labels

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading:
1. `Protocols/TopicRepresenter.swift` - understand the interface
2. `Core/Document.swift` - understand input document structure
3. `Core/Topic.swift` - understand output topic structure
4. `Core/ClusterAssignment.swift` - understand cluster labels

Then implement:
1. Create `Sources/SwiftTopics/Representation/` directory
2. Implement Tokenizer, Vocabulary, cTFIDF, CTFIDFRepresenter
3. Add tests for topic representation
4. Verify with `swift build && swift test`
