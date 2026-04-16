# SwiftTopics Phase 6 Kickoff: Coherence Evaluation

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-5 are complete.

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

### Phase 5: Topic Representation - c-TF-IDF (~400 LOC)
- `Representation/Tokenizer.swift` - Text tokenization with stop words, bigrams
- `Representation/Vocabulary.swift` - Term/document frequency computation
- `Representation/cTFIDF.swift` - Class-based TF-IDF scoring algorithm
- `Representation/CTFIDFRepresenter.swift` - TopicRepresenter implementation with MMR diversification

## Phase 6: Coherence Evaluation

**Duration**: 3-4 days | **LOC**: ~400

Implement topic quality metrics for evaluation and hyperparameter tuning.

### What is Coherence?

Coherence measures how semantically related the keywords in a topic are. High coherence indicates that the topic's keywords frequently co-occur in the corpus, suggesting they represent a meaningful concept.

### NPMI (Normalized Pointwise Mutual Information)

NPMI is the standard coherence metric. It measures the association between word pairs:

```
PMI(w1, w2) = log(P(w1, w2) / (P(w1) × P(w2)))
NPMI(w1, w2) = PMI(w1, w2) / -log(P(w1, w2))
```

NPMI normalizes PMI to the range [-1, 1]:
- **+1**: Words always co-occur (perfect association)
- **0**: Words are independent
- **-1**: Words never co-occur

Topic coherence averages NPMI over all word pairs in the topic's top keywords.

### Deliverables

#### 6.1 Co-occurrence Counter
```
Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift
```
- Sliding window co-occurrence (configurable window size)
- Document co-occurrence (boolean - did words appear together in any document?)
- Efficient sparse storage (dictionary-based)
- Handle word pairs symmetrically: P(w1, w2) = P(w2, w1)

#### 6.2 NPMI Scorer
```
Sources/SwiftTopics/Evaluation/NPMIScorer.swift
```
- Compute P(w1), P(w2), P(w1, w2) from co-occurrence counts
- NPMI formula with smoothing to avoid log(0)
- Average over all word pairs (w_i, w_j) for i < j
- Return per-word-pair scores for debugging

#### 6.3 Coherence Evaluator
```
Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift
```
- `evaluate(topics:corpus:) -> CoherenceResult`
- Per-topic coherence scores
- Aggregate statistics (mean, median, min, max)
- Configurable: window size, top-N words to consider, epsilon smoothing

#### 6.4 Diversity Metrics (Optional)
```
Sources/SwiftTopics/Evaluation/DiversityMetrics.swift
```
- Topic diversity: percentage of unique words across all topics
- Redundancy score: keyword overlap between topics

### Algorithm Steps

1. **Build co-occurrence matrix from corpus**:
   - For each document, extract tokens
   - Count how many times each word pair appears within a window
   - Track individual word frequencies

2. **For each topic**:
   - Take top-K keywords
   - For each pair (w_i, w_j) where i < j:
     - Look up P(w_i), P(w_j), P(w_i, w_j)
     - Compute NPMI(w_i, w_j)
   - Average all NPMI scores

3. **Aggregate across topics**:
   - Mean coherence
   - Median coherence (robust to outliers)
   - Identify low-coherence topics

### Key Considerations

#### Window Size
- **Small window (5-10)**: Captures syntactic co-occurrence (words in same phrase)
- **Large window (50-100)**: Captures semantic co-occurrence (words in same paragraph)
- **Document-level**: Boolean - did words appear in same document?

Recommendation: Start with sliding window of 10 (default) or document-level.

#### Smoothing
Add epsilon to probabilities to avoid log(0):
```swift
let epsilon: Float = 1e-12
let pmi = log((p_w1_w2 + epsilon) / ((p_w1 + epsilon) * (p_w2 + epsilon)))
let npmi = pmi / -log(p_w1_w2 + epsilon)
```

#### Performance
For a corpus of N documents with vocabulary V:
- Building co-occurrence: O(N × L × W) where L = avg doc length, W = window size
- Evaluating K topics with T keywords: O(K × T²) lookups

Use sparse storage (dictionaries) since most word pairs never co-occur.

### Expected Interface

```swift
public struct CoherenceConfiguration: Sendable, Codable {
    /// Window size for co-occurrence counting.
    public let windowSize: Int  // default: 10

    /// Whether to use document-level co-occurrence instead of sliding window.
    public let useDocumentCooccurrence: Bool  // default: false

    /// Number of top keywords to consider per topic.
    public let topKeywords: Int  // default: 10

    /// Smoothing epsilon for probability calculations.
    public let epsilon: Float  // default: 1e-12
}

public struct CoherenceResult: Sendable {
    /// Per-topic coherence scores.
    public let topicScores: [Float]

    /// Mean coherence across all topics.
    public let meanCoherence: Float

    /// Median coherence across all topics.
    public let medianCoherence: Float

    /// Number of topics evaluated.
    public let topicCount: Int
}

public protocol CoherenceEvaluator: Sendable {
    var configuration: CoherenceConfiguration { get }

    func evaluate(
        topics: [Topic],
        documents: [Document]
    ) async -> CoherenceResult
}
```

### Exit Criteria
- [ ] NPMI scores in expected range [-1, 1]
- [ ] Coherence correlates with topic quality (manual inspection)
- [ ] Can differentiate good vs. bad topic models
- [ ] Handles edge cases: empty topics, unknown words, single-word topics
- [ ] Performance: <1s for 100 topics on 1000 documents
- [ ] swift build passes
- [ ] swift test passes

### Constraints
- Swift 6 strict concurrency (all types must be Sendable)
- Use existing `Topic` type (has `keywords: [TopicKeyword]`)
- Use existing `Document` type
- Use existing `Tokenizer` from Phase 5 for consistency
- Target: iOS/macOS/visionOS 26+

### Reference Materials
- SPEC.md Part 2.5 (Evaluation Layer)
- `Core/Topic.swift` for Topic and TopicKeyword types
- `Core/Document.swift` for Document type
- `Representation/Tokenizer.swift` for tokenization

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading:
1. `Core/Topic.swift` - understand Topic and TopicKeyword structure
2. `Representation/Tokenizer.swift` - use consistent tokenization
3. `Core/Document.swift` - understand Document structure

Then implement:
1. Create `Sources/SwiftTopics/Evaluation/` directory
2. Implement CooccurrenceCounter, NPMIScorer, CoherenceEvaluator
3. Add tests for coherence evaluation
4. Verify with `swift build && swift test`
