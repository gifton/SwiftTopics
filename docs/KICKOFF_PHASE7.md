# SwiftTopics Phase 7 Kickoff: TopicModel Orchestrator

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-6 are complete.

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

### Phase 6: Coherence Evaluation (~400 LOC)
- `Evaluation/CooccurrenceCounter.swift` - Word pair co-occurrence counting (sliding window & document-level)
- `Evaluation/NPMIScorer.swift` - Normalized Pointwise Mutual Information scoring
- `Evaluation/CoherenceEvaluator.swift` - Topic coherence evaluation with aggregate statistics
- `Evaluation/DiversityMetrics.swift` - Topic diversity and redundancy metrics

## Phase 7: TopicModel Orchestrator

**Duration**: 4-5 days | **LOC**: ~600

Create the main API that coordinates the full pipeline: embeddings → reduction → clustering → representation → evaluation.

### What is the TopicModel?

The `TopicModel` is the public-facing API that users interact with. It:
1. Accepts documents (with pre-computed embeddings or via an EmbeddingProvider)
2. Reduces dimensionality (PCA or later UMAP)
3. Clusters embeddings (HDBSCAN)
4. Extracts topic keywords (c-TF-IDF)
5. Evaluates coherence (NPMI)
6. Returns a complete `TopicModelResult`

### Deliverables

#### 7.1 Configuration
```
Sources/SwiftTopics/Model/TopicModelConfiguration.swift
```
- `TopicModelConfiguration` with component-specific sub-configs
- Presets: `.default`, `.fast`, `.quality`
- Validation of parameter combinations
- `Codable` for serialization

```swift
public struct TopicModelConfiguration: Sendable, Codable {
    /// Dimensionality reduction configuration.
    public let reduction: ReductionConfiguration

    /// Clustering configuration.
    public let clustering: HDBSCANConfiguration

    /// Topic representation configuration.
    public let representation: CTFIDFConfiguration

    /// Coherence evaluation configuration (optional).
    public let coherence: CoherenceConfiguration?

    /// Random seed for reproducibility.
    public let seed: UInt64?

    /// Preset configurations
    public static let `default`: TopicModelConfiguration
    public static let fast: TopicModelConfiguration      // Fewer iterations, simpler reduction
    public static let quality: TopicModelConfiguration   // More keywords, stricter clustering
}
```

#### 7.2 TopicModel Actor
```
Sources/SwiftTopics/Model/TopicModel.swift
```
The main orchestrator, implemented as an actor for thread safety.

```swift
public actor TopicModel {
    /// Configuration used by this model.
    public let configuration: TopicModelConfiguration

    /// Whether the model has been fitted.
    public var isFitted: Bool { get }

    /// The topics discovered during fitting.
    public var topics: [Topic]? { get }

    // MARK: - Fitting

    /// Fits the model on documents with pre-computed embeddings.
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster into topics.
    ///   - embeddings: Pre-computed document embeddings.
    /// - Returns: Complete topic modeling result.
    public func fit(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> TopicModelResult

    /// Fits the model on documents using an embedding provider.
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster into topics.
    ///   - embeddingProvider: Provider to compute embeddings.
    /// - Returns: Complete topic modeling result.
    public func fit(
        documents: [Document],
        embeddingProvider: any EmbeddingProvider
    ) async throws -> TopicModelResult

    // MARK: - Transform

    /// Assigns new documents to existing topics.
    ///
    /// - Parameters:
    ///   - documents: New documents to assign.
    ///   - embeddings: Pre-computed embeddings.
    /// - Returns: Topic assignments for each document.
    public func transform(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> [TopicAssignment]

    // MARK: - Convenience

    /// Fits the model and returns assignments (combines fit + transform).
    public func fitTransform(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> TopicModelResult
}
```

#### 7.3 TopicAssignment
```
Sources/SwiftTopics/Model/TopicAssignment.swift
```
Result of assigning a document to a topic.

```swift
public struct TopicAssignment: Sendable, Codable {
    /// The assigned topic ID (-1 for outliers).
    public let topicID: TopicID

    /// Confidence/probability of the assignment.
    public let probability: Float

    /// Distance to topic centroid (lower = more representative).
    public let distanceToCentroid: Float?

    /// Whether this document is an outlier.
    public var isOutlier: Bool { topicID.isOutlier }
}
```

#### 7.4 Progress Reporting (Optional)
```
Sources/SwiftTopics/Model/TopicModelProgress.swift
```
For long-running operations on large corpora.

```swift
public enum TopicModelStage: Sendable {
    case embedding(current: Int, total: Int)
    case reduction
    case clustering
    case representation
    case evaluation
    case complete
}

public struct TopicModelProgress: Sendable {
    public let stage: TopicModelStage
    public let overallProgress: Float  // 0.0 to 1.0
    public let elapsedTime: TimeInterval
}
```

#### 7.5 Serialization
```
Sources/SwiftTopics/Model/TopicModelState.swift
```
Save and load fitted models.

```swift
public struct TopicModelState: Codable {
    /// Version for compatibility checking.
    public let version: Int

    /// Configuration used to train the model.
    public let configuration: TopicModelConfiguration

    /// Discovered topics.
    public let topics: [Topic]

    /// Cluster assignments for training documents.
    public let assignments: ClusterAssignment

    /// PCA transformation matrix (for projecting new data).
    public let pcaComponents: [Float]?

    /// Topic centroids (for assigning new documents).
    public let centroids: [Embedding]?

    /// Training timestamp.
    public let trainedAt: Date
}

extension TopicModel {
    /// Exports the current model state.
    public func exportState() throws -> TopicModelState

    /// Creates a fitted model from exported state.
    public init(state: TopicModelState) throws
}
```

### Algorithm Steps

The `fit` method orchestrates the full pipeline:

```
1. Validate inputs (document count matches embedding count)
2. Optional: Compute embeddings via EmbeddingProvider
3. Reduce dimensionality: embeddings → PCA → reduced embeddings
4. Cluster: reduced embeddings → HDBSCAN → ClusterAssignment
5. Extract topics: documents + assignments → c-TF-IDF → Topics with keywords
6. Evaluate: topics + documents → NPMI → coherence scores
7. Build result: TopicModelResult with all outputs
```

### Key Considerations

#### Thread Safety
- `TopicModel` is an actor, ensuring isolated state
- All methods are `async`
- Result types are `Sendable`

#### Error Handling
```swift
public enum TopicModelError: Error, Sendable {
    case notFitted
    case invalidInput(String)
    case embeddingDimensionMismatch(expected: Int, got: Int)
    case noTopicsDiscovered
    case serializationFailed(String)
}
```

#### Configuration Presets
```swift
extension TopicModelConfiguration {
    // Default: balanced quality and speed
    public static let `default` = TopicModelConfiguration(
        reduction: ReductionConfiguration(components: 50),
        clustering: HDBSCANConfiguration(minClusterSize: 10),
        representation: CTFIDFConfiguration(keywordsPerTopic: 10),
        coherence: .default,
        seed: nil
    )

    // Fast: prioritize speed over quality
    public static let fast = TopicModelConfiguration(
        reduction: ReductionConfiguration(components: 20),
        clustering: HDBSCANConfiguration(minClusterSize: 5, minSamples: 3),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,  // Skip coherence evaluation
        seed: nil
    )

    // Quality: prioritize topic quality
    public static let quality = TopicModelConfiguration(
        reduction: ReductionConfiguration(components: 100),
        clustering: HDBSCANConfiguration(minClusterSize: 15, clusterSelectionMethod: .eom),
        representation: CTFIDFConfiguration(keywordsPerTopic: 15, diversify: true),
        coherence: CoherenceConfiguration(topKeywords: 15),
        seed: nil
    )
}
```

### Expected Interface

```swift
// Basic usage
let model = TopicModel(configuration: .default)
let result = try await model.fit(documents: documents, embeddings: embeddings)

print("Found \(result.topics.count) topics")
for topic in result.topics {
    print("Topic \(topic.id): \(topic.keywordSummary())")
    print("  Coherence: \(topic.coherenceScore ?? 0)")
}

// Assign new documents
let newAssignments = try await model.transform(
    documents: newDocuments,
    embeddings: newEmbeddings
)

// Save and load
let state = try model.exportState()
let data = try JSONEncoder().encode(state)
// ... save to file ...

let loadedState = try JSONDecoder().decode(TopicModelState.self, from: data)
let loadedModel = try TopicModel(state: loadedState)
```

### Exit Criteria
- [ ] Full pipeline runs end-to-end with pre-computed embeddings
- [ ] Progress reporting works (if implemented)
- [ ] Model can be saved and loaded
- [ ] Thread-safe (actor isolation verified)
- [ ] Configuration presets work correctly
- [ ] swift build passes
- [ ] swift test passes

### Constraints
- Swift 6 strict concurrency (all types must be Sendable)
- Actor-based design for TopicModel
- Use existing components from Phases 1-6
- Target: iOS/macOS/visionOS 26+

### Reference Materials
- ROADMAP.md Phase 7 section
- `Core/TopicModelResult.swift` for result structure
- `Protocols/EmbeddingProvider.swift` for provider interface
- `Reduction/PCA.swift` for reduction interface
- `Clustering/HDBSCAN/HDBSCAN.swift` for clustering interface
- `Representation/CTFIDFRepresenter.swift` for representation interface
- `Evaluation/CoherenceEvaluator.swift` for coherence interface

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading:
1. `Core/TopicModelResult.swift` - understand the expected result structure
2. `Protocols/EmbeddingProvider.swift` - provider interface
3. Existing component APIs (PCA, HDBSCAN, CTFIDFRepresenter, NPMICoherenceEvaluator)

Then implement:
1. Create `Sources/SwiftTopics/Model/` directory
2. Implement TopicModelConfiguration, TopicModel, TopicAssignment
3. Add TopicModelState for serialization
4. Add tests for the orchestrator
5. Verify with `swift build && swift test`
