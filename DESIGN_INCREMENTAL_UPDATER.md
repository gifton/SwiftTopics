# Design: Incremental Topic Model Updater

## Executive Summary

**Problem**: SwiftTopics currently requires full retraining when new documents are added. For a journaling application where entries arrive one-by-one (365+/year), this creates an unacceptable UX—users expect immediate topic assignment without multi-second delays.

**Recommended Approach**: Implement a **Buffered Micro-Retrain with Model Merging** strategy (hybrid of periodic batch refresh + BERTopic-style model merging). New entries receive immediate topic assignment via transform-only, while batches of 30 entries trigger fast (~200ms) mini-model training that merges into the main model.

**Key Tradeoff**: We accept that transform-only assignment may be slightly less accurate than full retraining, but gain sub-50ms response times for new entries and seamless background model evolution. Full quality is restored through periodic micro-retrains that users never notice.

---

## 1. Literature Review Findings

### 1.1 BERTopic Incremental Approaches

BERTopic (the Python library that inspired SwiftTopics) offers two approaches for incremental learning:

| Approach | Mechanism | Quality | Recommended |
|----------|-----------|---------|-------------|
| **`.partial_fit()`** | Uses MiniBatchKMeans + IncrementalPCA | Lower | No |
| **`.merge_models()`** | Train separate model, merge via similarity | Higher | **Yes** |

**Key insight from BERTopic author** ([GitHub #683](https://github.com/MaartenGr/BERTopic/issues/683)): The `merge_models` approach is explicitly recommended because alternative algorithms (IncrementalPCA, MiniBatchKMeans) produce significantly worse results than UMAP + HDBSCAN.

**OnlineCountVectorizer** solves vocabulary drift with:
- **Decay parameter**: Weights recent terms higher (e.g., 0.1 = 10% decay per iteration)
- **delete_min_df**: Removes low-frequency terms to prevent vocabulary bloat

*Source: [BERTopic Online Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/online/online.html)*

### 1.2 HDBSCAN Prediction for New Points

HDBSCAN's `approximate_predict()` assigns new points to existing clusters by locating where they would fall in the condensed tree—**without modifying the tree**.

**Requirements**:
- Must set `prediction_data=True` during fit (stores extra state)
- Returns soft membership probabilities via `membership_vector()`

**Critical limitations**:
- Cannot discover new clusters
- Cannot split or merge existing clusters
- Recommendation: "Cache your data and retrain periodically to avoid drift"

*Source: [HDBSCAN Prediction Tutorial](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html)*

### 1.3 UMAP Transform for New Points

UMAP's `transform()` projects new points using k-NN interpolation:
1. Find k nearest neighbors in training data
2. Weighted average of their low-dimensional positions

**Quality considerations**:
- Transform produces lower-quality embeddings than fit_transform
- Known issue: [GitHub #755](https://github.com/lmcinnes/umap/issues/755) - shifted points
- Recommendation: "Periodically refit on updated reference dataset"

*Source: [UMAP Documentation](https://umap-learn.readthedocs.io/)*

### 1.4 Streaming Density-Based Clustering Research

Recent research (2024-2025) on incremental HDBSCAN alternatives:

| Method | Approach | Speedup | Quality |
|--------|----------|---------|---------|
| **Bubble-tree** (Nov 2024) | Summarizes data into fixed "bubbles" | 100x vs exact | Good with 10% compression |
| **FISHDBC** | Incremental density-based clustering | Comparable to HDBSCAN | Sometimes better |
| **IncGridDBC** (2025) | Grid-based cell connections | 60x vs competitors | Good |

**Key finding**: Exact incremental HDBSCAN maintenance is possible but complex. Summarization-based approaches offer practical tradeoffs.

*Sources: [Dynamic HDBSCAN Research](https://arxiv.org/html/2412.07789v1), [FISHDBC Paper](https://arxiv.org/pdf/1910.07283)*

### 1.5 Key Insights Informing Design

1. **HDBSCAN and UMAP are transductive** - they assume access to full dataset; incremental is always an approximation
2. **Model merging beats online algorithms** - BERTopic author's experience shows this clearly
3. **Periodic refresh is essential** - all approaches recommend retraining to avoid drift
4. **Transform-only degrades gracefully** - quality loss is gradual, not catastrophic

---

## 2. Architecture Options Comparison

### Option A: Transform-Only (Simplest)

New documents use fitted model's centroids for nearest-neighbor assignment. No retraining ever.

### Option B: Periodic Batch Refresh

Buffer new documents, trigger full retrain when threshold reached. Handle topic ID stability via matching.

### Option C: Hybrid Incremental

Transform for small updates, partial component retraining based on drift detection.

### Option D: Model Merging (BERTopic-style)

Train mini-model on new batch, merge topics via embedding similarity.

### Comparison Matrix

| Aspect | Option A | Option B | Option C | Option D |
|--------|----------|----------|----------|----------|
| **Computational complexity** | O(k) per doc | O(n²) every N docs | O(m²) + O(k) | O(m²) + O(k·k') |
| **Memory requirements** | ~4KB per topic | ~2KB × n docs | ~10KB × n docs | ~2KB × n docs |
| **Topic ID stability** | 100% stable | Requires matching | Mostly stable | Mostly stable |
| **Quality over time** | Degrades | Good (refreshed) | Best | Excellent |
| **New topic discovery** | Never | At refresh | With outlier buffer | At merge |
| **Implementation complexity** | ~100 LOC | ~400 LOC | ~800 LOC | ~600 LOC |
| **Latency (new doc)** | <50ms | <50ms + periodic delay | <50ms | <50ms |
| **Journal use case fit** | Poor | Good | Over-engineered | **Excellent** |

### Use Case Recommendations

- **Static corpus analysis**: Option A
- **Periodic batch ingestion**: Option B
- **High-frequency streaming**: Option C
- **Journal/note-taking apps**: **Option B + D hybrid**

---

## 3. Recommended Approach

### Buffered Micro-Retrain with Model Merging (B + D Hybrid)

This approach combines the simplicity of periodic refresh with the quality of model merging:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INCREMENTAL UPDATE FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NEW ENTRY ──► Quick Transform ──► Immediate Topic Assignment           │
│      │              (<50ms)              │                              │
│      │                                   ▼                              │
│      │                            User sees result                      │
│      │                                                                  │
│      ▼                                                                  │
│  Buffer Entry ──► Check threshold ──► If >= 30 entries:                │
│                         │                    │                          │
│                         │                    ▼                          │
│                         │            ┌──────────────────┐               │
│                         │            │  MICRO-RETRAIN   │               │
│                         │            │  (~200-500ms)    │               │
│                         │            │                  │               │
│                         │            │  1. Train mini-  │               │
│                         │            │     model on 30  │               │
│                         │            │     new entries  │               │
│                         │            │                  │               │
│                         │            │  2. Merge into   │               │
│                         │            │     main model   │               │
│                         │            │                  │               │
│                         │            │  3. Update       │               │
│                         │            │     centroids    │               │
│                         │            └──────────────────┘               │
│                         │                    │                          │
│                         ▼                    ▼                          │
│                    Buffer < 30        Topics refined                    │
│                    (wait for more)    (background, seamless)            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  PERIODIC FULL REFRESH (background, every ~500 entries or 6 months)    │
│  - Rebuilds entire model from scratch                                   │
│  - Runs via iOS Background Processing API                               │
│  - User can close app; resumes on next launch                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Fits Journal Use Cases

1. **Day 1-30**: Cold start—first 30 entries build initial model
2. **Day 31+**: Every entry gets instant topic assignment
3. **Every ~30 entries**: Invisible micro-retrain (~200ms)
4. **Every ~6 months**: Background full refresh for optimal quality

### Tradeoffs Accepted

| Tradeoff | Mitigation |
|----------|------------|
| Transform-only may miss ideal topic | Micro-retrain catches up within 30 entries |
| Topic IDs may change at merge | Hungarian matching preserves >90% stability |
| Mini-model quality < full model | Merge aggregates signal; quality sufficient |
| Requires storing embeddings | ~2KB/doc is acceptable for journal scale |

---

## 4. Detailed Design

### 4.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ New Journal │                                                            │
│  │   Entry     │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │  Embedding  │────►│              STORAGE LAYER                        │  │
│  │  (External) │     │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  └──────┬──────┘     │  │ Embeddings │  │  Pending   │  │ Checkpoint │  │  │
│         │            │  │   (.bin)   │  │  Buffer    │  │   State    │  │  │
│         │            │  └────────────┘  └────────────┘  └────────────┘  │  │
│         │            └──────────────────────────────────────────────────┘  │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    QUICK TRANSFORM PATH (<50ms)                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │   │
│  │  │ Load Model   │───►│   Compute    │───►│   Return     │           │   │
│  │  │   State      │    │  Centroid    │    │  Assignment  │           │   │
│  │  │              │    │  Distances   │    │              │           │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 BUFFER MANAGEMENT                                    │   │
│  │                                                                      │   │
│  │   pending_count >= 30?  ───YES───►  TRIGGER MICRO-RETRAIN           │   │
│  │         │                                                            │   │
│  │         NO                                                           │   │
│  │         │                                                            │   │
│  │         ▼                                                            │   │
│  │   Wait for more entries                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MICRO-RETRAIN PATH (~200ms)                       │   │
│  │                                                                      │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│  │  │   Drain    │──►│   Train    │──►│   Merge    │──►│   Save     │ │   │
│  │  │   Buffer   │   │ Mini-Model │   │   Topics   │   │   State    │ │   │
│  │  │            │   │  (30 docs) │   │            │   │            │ │   │
│  │  └────────────┘   └────────────┘   └────────────┘   └────────────┘ │   │
│  │                                                                      │   │
│  │  Mini-model pipeline:                                                │   │
│  │  1. UMAP fit (30 docs) ─► ~50ms                                     │   │
│  │  2. HDBSCAN fit ─► ~20ms                                            │   │
│  │  3. c-TF-IDF ─► ~10ms                                               │   │
│  │  4. Topic matching ─► ~20ms                                         │   │
│  │  5. Centroid update ─► ~10ms                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              FULL REFRESH PATH (background, ~15-60s)                 │   │
│  │                                                                      │   │
│  │  Triggered when: corpus_size > last_full_retrain_size × 1.5         │   │
│  │              OR: time_since_last_retrain > 6 months                  │   │
│  │              OR: outlier_rate > 20%                                  │   │
│  │                                                                      │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│  │  │   Load     │──►│   Full     │──►│   Match    │──►│   Save     │ │   │
│  │  │   All      │   │  Pipeline  │   │   Topic    │   │   New      │ │   │
│  │  │  Embeds    │   │            │   │    IDs     │   │   State    │ │   │
│  │  └────────────┘   └────────────┘   └────────────┘   └────────────┘ │   │
│  │                         │                                            │   │
│  │                         ▼                                            │   │
│  │  Interruptible with checkpointing (see Section 4.4)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 State Requirements

#### 4.2.1 Extended TopicModelState

The existing `TopicModelState` must be extended to support incremental updates:

```swift
/// Extended state for incremental topic modeling
public struct IncrementalTopicModelState: Codable, Sendable {

    // MARK: - Base State (existing)

    /// Version for migration support
    let version: Int

    /// Model configuration
    let configuration: TopicModelConfiguration

    /// Discovered topics with keywords
    let topics: [Topic]

    /// Document-to-topic assignments
    let assignments: ClusterAssignment

    /// Topic centroids in embedding space
    let centroids: [Embedding]

    // MARK: - Incremental State (new)

    /// Vocabulary with term frequencies per topic
    let vocabulary: Vocabulary

    /// IDF values for c-TF-IDF computation
    let idfValues: [String: Float]

    /// Total document count (including buffered)
    let totalDocumentCount: Int

    /// When the model was last fully retrained
    let lastFullRetrainDate: Date

    /// Document count at last full retrain
    let documentCountAtLastRetrain: Int

    /// Running statistics for drift detection
    let driftStatistics: DriftStatistics

    // MARK: - UMAP Transform State (new)

    /// k-NN graph for UMAP transform (optional, for high-quality transform)
    /// Stored separately due to size
    let hasKNNGraph: Bool

    /// UMAP embedding coordinates for training data
    /// Stored separately due to size
    let hasReducedEmbeddings: Bool
}

/// Statistics for detecting model drift
public struct DriftStatistics: Codable, Sendable {
    /// Average distance to assigned centroid (recent entries)
    var recentAverageDistance: Float

    /// Average distance to assigned centroid (all time)
    var overallAverageDistance: Float

    /// Percentage of recent entries marked as outliers
    var recentOutlierRate: Float

    /// Number of entries in recent window
    var recentWindowSize: Int

    /// Threshold for triggering refresh
    var driftThreshold: Float
}
```

#### 4.2.2 Storage Size Estimates

| Component | Formula | 1,000 Docs | 5,000 Docs | 10,000 Docs |
|-----------|---------|------------|------------|-------------|
| **Embeddings** | 512d × 4B × n | 2.0 MB | 10.0 MB | 20.0 MB |
| **Reduced Embeddings** | 15d × 4B × n | 60 KB | 300 KB | 600 KB |
| **k-NN Graph** | k × 8B × n (k=15) | 120 KB | 600 KB | 1.2 MB |
| **Vocabulary** | ~10K terms avg | 200 KB | 400 KB | 600 KB |
| **Model State** | Fixed + topics | 50 KB | 80 KB | 120 KB |
| **Pending Buffer** | ~2.5KB × buffer | 75 KB | 75 KB | 75 KB |
| **Checkpoint** | Variable | 0-5 MB | 0-10 MB | 0-20 MB |
| **TOTAL (typical)** | - | **~2.5 MB** | **~11.5 MB** | **~22.5 MB** |

### 4.3 API Specification

#### 4.3.1 Core Protocol

```swift
/// Protocol for incremental topic model updates.
/// Consumers implement storage; library provides update logic.
public protocol IncrementalTopicUpdating: Actor {

    /// Storage backend for persistence
    var storage: TopicModelStorage { get }

    /// Current model state (nil if not initialized)
    var modelState: IncrementalTopicModelState? { get }

    /// Processes a new document and returns topic assignment.
    /// This is the primary entry point for streaming updates.
    ///
    /// - Parameters:
    ///   - document: The new document
    ///   - embedding: Pre-computed embedding
    /// - Returns: Topic assignment (immediate, may be refined later)
    func processDocument(
        _ document: Document,
        embedding: Embedding
    ) async throws -> TopicAssignment

    /// Forces a micro-retrain with current buffer.
    /// Normally called automatically when buffer threshold reached.
    func triggerMicroRetrain() async throws

    /// Triggers a full model refresh.
    /// Should be called in background processing context.
    func triggerFullRefresh() async throws

    /// Resumes any interrupted training from checkpoint.
    /// Call on app launch.
    func resumeIfNeeded() async throws

    /// Prepares for app termination by saving checkpoint.
    func prepareForTermination() async throws
}
```

#### 4.3.2 Storage Protocol

```swift
/// Protocol for persisting topic model state.
/// Implement to integrate with your app's persistence layer.
public protocol TopicModelStorage: Sendable {

    // MARK: - Model State

    /// Saves the fitted model state
    func saveModelState(_ state: IncrementalTopicModelState) async throws

    /// Loads the fitted model state
    func loadModelState() async throws -> IncrementalTopicModelState?

    // MARK: - Embeddings (Large Data)

    /// Appends embeddings for new documents (append-only)
    func appendEmbeddings(_ embeddings: [(DocumentID, Embedding)]) async throws

    /// Loads embeddings for specific documents
    func loadEmbeddings(for documentIDs: [DocumentID]) async throws -> [Embedding]

    /// Loads all embeddings (for retraining)
    func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)]

    /// Returns count without loading data
    func embeddingCount() async throws -> Int

    // MARK: - Reduced Embeddings (Optional, for high-quality transform)

    /// Saves reduced embeddings after training
    func saveReducedEmbeddings(_ embeddings: [(DocumentID, [Float])]) async throws

    /// Loads reduced embeddings
    func loadReducedEmbeddings() async throws -> [(DocumentID, [Float])]?

    // MARK: - k-NN Graph (Optional, for UMAP transform)

    /// Saves k-NN graph
    func saveKNNGraph(_ graph: NearestNeighborGraph) async throws

    /// Loads k-NN graph
    func loadKNNGraph() async throws -> NearestNeighborGraph?

    // MARK: - Training Checkpoint

    /// Saves training checkpoint for resumption
    func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws

    /// Loads most recent checkpoint (nil if none or complete)
    func loadCheckpoint() async throws -> TrainingCheckpoint?

    /// Clears checkpoint after successful training
    func clearCheckpoint() async throws

    // MARK: - Pending Buffer

    /// Adds documents to pending buffer
    func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws

    /// Loads and clears pending buffer
    func drainPendingBuffer() async throws -> [BufferedEntry]

    /// Returns pending count without loading
    func pendingBufferCount() async throws -> Int
}

/// Entry waiting to be incorporated into model
public struct BufferedEntry: Codable, Sendable {
    public let documentID: DocumentID
    public let embedding: Embedding
    public let tokenizedContent: [String]
    public let addedAt: Date
}
```

#### 4.3.3 Configuration

```swift
/// Configuration for incremental updates
public struct IncrementalUpdateConfiguration: Codable, Sendable {

    // MARK: - Buffer Thresholds

    /// Minimum entries before initial model creation
    /// Default: 30
    public var coldStartThreshold: Int = 30

    /// Entries to buffer before micro-retrain
    /// Default: 30
    public var microRetrainThreshold: Int = 30

    /// Maximum buffer size before forced retrain
    /// Default: 100
    public var maxBufferSize: Int = 100

    // MARK: - Full Refresh Triggers

    /// Corpus growth ratio to trigger full refresh
    /// Default: 1.5 (50% growth)
    public var fullRefreshGrowthRatio: Float = 1.5

    /// Maximum time between full refreshes
    /// Default: 180 days
    public var fullRefreshMaxInterval: TimeInterval = 180 * 24 * 60 * 60

    /// Outlier rate threshold for early refresh
    /// Default: 0.20 (20%)
    public var outlierRateThreshold: Float = 0.20

    // MARK: - Checkpointing

    /// UMAP epochs between checkpoints
    /// Default: 50
    public var umapCheckpointInterval: Int = 50

    /// MST iterations between checkpoints
    /// Default: 100
    public var mstCheckpointInterval: Int = 100

    // MARK: - Topic Stability

    /// Minimum similarity for topic matching after retrain
    /// Default: 0.8
    public var topicMatchingSimilarityThreshold: Float = 0.8

    /// Whether to preserve topic IDs across retrains
    /// Default: true
    public var preserveTopicIDs: Bool = true

    // MARK: - Presets

    /// Default configuration balanced for journal use
    public static let `default` = IncrementalUpdateConfiguration()

    /// Aggressive refresh for rapidly evolving content
    public static let aggressive = IncrementalUpdateConfiguration(
        microRetrainThreshold: 20,
        fullRefreshGrowthRatio: 1.3,
        outlierRateThreshold: 0.15
    )

    /// Conservative refresh for stable content
    public static let conservative = IncrementalUpdateConfiguration(
        microRetrainThreshold: 50,
        fullRefreshGrowthRatio: 2.0,
        fullRefreshMaxInterval: 365 * 24 * 60 * 60
    )
}
```

#### 4.3.4 Public API

```swift
/// Main actor for incremental topic model updates.
public actor IncrementalTopicUpdater: IncrementalTopicUpdating {

    // MARK: - Properties

    public let storage: TopicModelStorage
    public let configuration: IncrementalUpdateConfiguration
    public private(set) var modelState: IncrementalTopicModelState?

    // MARK: - Initialization

    /// Creates an incremental updater with storage and configuration.
    ///
    /// - Parameters:
    ///   - storage: Storage backend (file-based default available)
    ///   - configuration: Update behavior configuration
    public init(
        storage: TopicModelStorage,
        configuration: IncrementalUpdateConfiguration = .default
    )

    /// Creates an incremental updater with file-based storage.
    ///
    /// - Parameters:
    ///   - directory: Directory for storing model data
    ///   - configuration: Update behavior configuration
    public convenience init(
        directory: URL,
        configuration: IncrementalUpdateConfiguration = .default
    ) throws

    // MARK: - Document Processing

    /// Processes a new document and returns immediate topic assignment.
    ///
    /// This method:
    /// 1. Assigns document to nearest topic centroid (<50ms)
    /// 2. Buffers document for future micro-retrain
    /// 3. Triggers micro-retrain if buffer threshold reached
    ///
    /// - Parameters:
    ///   - document: The document to process
    ///   - embedding: Pre-computed embedding for the document
    /// - Returns: Topic assignment (immediate result)
    public func processDocument(
        _ document: Document,
        embedding: Embedding
    ) async throws -> TopicAssignment

    /// Processes multiple documents in batch.
    ///
    /// More efficient than individual calls for bulk imports.
    ///
    /// - Parameters:
    ///   - documents: Documents to process
    ///   - embeddings: Corresponding embeddings
    /// - Returns: Topic assignments for each document
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
    /// Supports interruption and resumption.
    ///
    /// - Parameter progress: Optional progress callback
    public func triggerFullRefresh(
        progress: ((TrainingProgress) -> Void)? = nil
    ) async throws

    // MARK: - Lifecycle

    /// Resumes interrupted training from checkpoint.
    ///
    /// Call this on app launch to continue any interrupted training.
    ///
    /// - Returns: True if training was resumed, false if no checkpoint
    @discardableResult
    public func resumeIfNeeded() async throws -> Bool

    /// Prepares for app termination.
    ///
    /// Saves checkpoint if training is in progress.
    /// Call this when app is about to terminate.
    public func prepareForTermination() async throws

    /// Checks if full refresh is recommended based on drift metrics.
    public func shouldTriggerFullRefresh() async -> Bool

    // MARK: - Queries

    /// Returns current topic assignments for a document.
    ///
    /// Uses the most up-to-date model state.
    public func getTopicAssignment(
        for embedding: Embedding
    ) async throws -> TopicAssignment

    /// Returns all topics in current model.
    public func getTopics() async -> [Topic]?

    /// Returns drift statistics for monitoring.
    public func getDriftStatistics() async -> DriftStatistics?
}

/// Progress information during training
public struct TrainingProgress: Sendable {
    public let phase: TrainingPhase
    public let phaseProgress: Float  // 0.0-1.0
    public let overallProgress: Float  // 0.0-1.0
    public let estimatedTimeRemaining: TimeInterval?
    public let canInterrupt: Bool
}
```

### 4.4 Interruptibility Design

#### 4.4.1 Training Phases and Checkpoints

```swift
/// Training phase enumeration with checkpoint support
public enum TrainingPhase: Int, Codable, Sendable, CaseIterable {
    case embedding = 0           // External, not our concern
    case umapKNN = 1             // k-NN graph construction
    case umapFuzzySet = 2        // Fuzzy simplicial set
    case umapOptimization = 3    // SGD optimization (iterative)
    case hdbscanCoreDistance = 4 // Core distance computation
    case hdbscanMST = 5          // MST construction (sequential)
    case clusterExtraction = 6   // EOM/Leaf selection
    case representation = 7      // c-TF-IDF keywords
    case topicMatching = 8       // Match to existing topics
    case complete = 9

    /// Whether this phase supports mid-phase checkpointing
    var supportsPartialCheckpoint: Bool {
        switch self {
        case .umapOptimization, .hdbscanMST, .umapKNN, .hdbscanCoreDistance:
            return true
        default:
            return false
        }
    }

    /// Estimated duration for 1000 documents (seconds)
    var estimatedDuration: TimeInterval {
        switch self {
        case .embedding: return 0
        case .umapKNN: return 8
        case .umapFuzzySet: return 0.5
        case .umapOptimization: return 15
        case .hdbscanCoreDistance: return 4
        case .hdbscanMST: return 3
        case .clusterExtraction: return 0.1
        case .representation: return 0.5
        case .topicMatching: return 0.1
        case .complete: return 0
        }
    }
}
```

#### 4.4.2 Checkpoint State Structure

```swift
/// Complete checkpoint state for training resumption
public struct TrainingCheckpoint: Codable, Sendable {

    // MARK: - Identity

    /// Unique ID for this training run
    public let runID: UUID

    /// Type of training (micro-retrain vs full refresh)
    public let trainingType: TrainingType

    /// When training started
    public let startedAt: Date

    /// Document IDs in this training batch
    public let documentIDs: [DocumentID]

    // MARK: - Progress

    /// Last completed phase
    public let lastCompletedPhase: TrainingPhase

    /// Current phase (may be partially complete)
    public let currentPhase: TrainingPhase

    /// Progress within current phase (0.0-1.0)
    public let currentPhaseProgress: Float

    // MARK: - Phase-Specific State (file paths)

    /// Path to k-NN graph (after Phase 1)
    public let knnGraphPath: URL?

    /// Path to fuzzy set (after Phase 2)
    public let fuzzySetPath: URL?

    /// UMAP state: embedding + current epoch (during Phase 3)
    public let umapEmbeddingPath: URL?
    public let umapCurrentEpoch: Int?
    public let umapTotalEpochs: Int?

    /// Core distances (after Phase 4)
    public let coreDistancesPath: URL?

    /// Partial MST state (during Phase 5)
    public let mstStatePath: URL?
    public let mstEdgesCompleted: Int?
    public let mstTotalEdges: Int?

    /// Cluster hierarchy (after Phase 6)
    public let hierarchyPath: URL?

    /// Topics before matching (after Phase 7)
    public let unmatchedTopicsPath: URL?

    // MARK: - Resumption

    /// Attempts remaining before giving up
    public let attemptsRemaining: Int

    public enum TrainingType: String, Codable, Sendable {
        case microRetrain
        case fullRefresh
    }
}
```

#### 4.4.3 Interruptible Training Runner

```swift
/// Runs training with interruption and checkpoint support
actor InterruptibleTrainingRunner {

    private let storage: TopicModelStorage
    private let checkpointInterval: TimeInterval = 3.0  // Checkpoint every 3 seconds
    private var lastCheckpointTime: Date = .distantPast
    private var currentCheckpoint: TrainingCheckpoint?

    /// Runs full training pipeline with interruption support
    func runTraining(
        documents: [Document],
        embeddings: [Embedding],
        existingState: IncrementalTopicModelState?,
        type: TrainingCheckpoint.TrainingType,
        shouldContinue: @escaping () -> Bool,
        onProgress: @escaping (TrainingProgress) -> Void
    ) async throws -> IncrementalTopicModelState {

        let runID = UUID()
        let documentIDs = documents.map(\.id)

        // Check for existing checkpoint to resume
        if let checkpoint = try await storage.loadCheckpoint(),
           checkpoint.runID == runID || checkpoint.documentIDs == documentIDs {
            return try await resumeTraining(
                from: checkpoint,
                documents: documents,
                embeddings: embeddings,
                existingState: existingState,
                shouldContinue: shouldContinue,
                onProgress: onProgress
            )
        }

        // Start fresh training
        var phase: TrainingPhase = .umapKNN

        // Phase 1: UMAP k-NN Graph
        onProgress(makeProgress(phase: .umapKNN, phaseProgress: 0))
        let knnGraph = try await buildKNNGraph(
            embeddings: embeddings,
            shouldContinue: shouldContinue,
            onProgress: { p in onProgress(self.makeProgress(phase: .umapKNN, phaseProgress: p)) }
        )
        try await checkpointAfterPhase(.umapKNN, knnGraph: knnGraph, runID: runID, documentIDs: documentIDs, type: type)

        // Phase 2: Fuzzy Simplicial Set
        onProgress(makeProgress(phase: .umapFuzzySet, phaseProgress: 0))
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)
        try await checkpointAfterPhase(.umapFuzzySet, fuzzySet: fuzzySet, runID: runID, documentIDs: documentIDs, type: type)

        // Phase 3: UMAP Optimization (interruptible)
        onProgress(makeProgress(phase: .umapOptimization, phaseProgress: 0))
        let reducedEmbeddings = try await optimizeUMAP(
            fuzzySet: fuzzySet,
            shouldContinue: shouldContinue,
            onProgress: { p in onProgress(self.makeProgress(phase: .umapOptimization, phaseProgress: p)) },
            runID: runID,
            documentIDs: documentIDs,
            type: type
        )

        // Phase 4: HDBSCAN Core Distances
        onProgress(makeProgress(phase: .hdbscanCoreDistance, phaseProgress: 0))
        let coreDistances = try await computeCoreDistances(
            embeddings: reducedEmbeddings,
            shouldContinue: shouldContinue,
            onProgress: { p in onProgress(self.makeProgress(phase: .hdbscanCoreDistance, phaseProgress: p)) }
        )
        try await checkpointAfterPhase(.hdbscanCoreDistance, coreDistances: coreDistances, runID: runID, documentIDs: documentIDs, type: type)

        // Phase 5: MST Construction (interruptible)
        onProgress(makeProgress(phase: .hdbscanMST, phaseProgress: 0))
        let mst = try await buildMST(
            embeddings: reducedEmbeddings,
            coreDistances: coreDistances,
            shouldContinue: shouldContinue,
            onProgress: { p in onProgress(self.makeProgress(phase: .hdbscanMST, phaseProgress: p)) },
            runID: runID,
            documentIDs: documentIDs,
            type: type
        )

        // Phase 6: Cluster Extraction (fast, no checkpoint needed)
        onProgress(makeProgress(phase: .clusterExtraction, phaseProgress: 0))
        let (clusters, hierarchy) = extractClusters(from: mst, coreDistances: coreDistances)

        // Phase 7: c-TF-IDF Representation
        onProgress(makeProgress(phase: .representation, phaseProgress: 0))
        let topics = try await extractTopicRepresentations(
            documents: documents,
            clusters: clusters
        )

        // Phase 8: Topic Matching (for micro-retrain)
        if let existing = existingState, type == .microRetrain {
            onProgress(makeProgress(phase: .topicMatching, phaseProgress: 0))
            let matchedTopics = matchTopics(new: topics, existing: existing.topics)
            return buildFinalState(topics: matchedTopics, /* ... */)
        }

        // Clear checkpoint on success
        try await storage.clearCheckpoint()

        onProgress(makeProgress(phase: .complete, phaseProgress: 1.0))
        return buildFinalState(topics: topics, /* ... */)
    }

    /// Saves checkpoint if enough time has passed
    private func maybeCheckpoint(
        phase: TrainingPhase,
        progress: Float,
        state: Any,
        runID: UUID,
        documentIDs: [DocumentID],
        type: TrainingCheckpoint.TrainingType
    ) async throws {
        let now = Date()
        guard now.timeIntervalSince(lastCheckpointTime) >= checkpointInterval else { return }

        // Build and save checkpoint
        let checkpoint = TrainingCheckpoint(
            runID: runID,
            trainingType: type,
            startedAt: currentCheckpoint?.startedAt ?? now,
            documentIDs: documentIDs,
            lastCompletedPhase: phase.previous,
            currentPhase: phase,
            currentPhaseProgress: progress,
            // ... phase-specific paths ...
            attemptsRemaining: 3
        )

        try await storage.saveCheckpoint(checkpoint)
        lastCheckpointTime = now
        currentCheckpoint = checkpoint
    }
}
```

### 4.5 Error Handling Strategy

```swift
/// Errors specific to incremental updates
public enum IncrementalUpdateError: Error, Sendable {

    // MARK: - Configuration Errors

    /// Model not initialized (call processDocument with initial batch first)
    case modelNotInitialized

    /// Embedding dimension doesn't match existing model
    case embeddingDimensionMismatch(expected: Int, got: Int)

    /// Storage backend error
    case storageError(underlying: Error)

    // MARK: - Training Errors

    /// Training was interrupted (checkpoint saved)
    case trainingInterrupted(phase: TrainingPhase, progress: Float)

    /// Checkpoint is corrupted and cannot be resumed
    case checkpointCorrupted(reason: String)

    /// Training failed after max retry attempts
    case trainingFailed(phase: TrainingPhase, underlying: Error)

    /// Insufficient documents for requested operation
    case insufficientDocuments(required: Int, provided: Int)

    // MARK: - Recovery Suggestions

    var recoverySuggestion: String {
        switch self {
        case .trainingInterrupted:
            return "Call resumeIfNeeded() on next app launch to continue."
        case .checkpointCorrupted:
            return "Clear checkpoint and retry with clearCheckpointAndRetry()."
        case .trainingFailed:
            return "Check logs for details. May need to reduce batch size."
        default:
            return ""
        }
    }
}
```

### 4.6 Thread Safety Considerations

1. **Actor isolation**: `IncrementalTopicUpdater` is an actor; all mutable state is isolated
2. **Storage protocol**: Requires `Sendable` conformance; implementations must be thread-safe
3. **Checkpoint atomicity**: File writes use atomic operations (write to temp, rename)
4. **Concurrent access**: Multiple `processDocument` calls are serialized by actor
5. **Background training**: Runs on actor; UI updates via progress callback

```swift
// Safe usage pattern from UI
Task {
    let assignment = try await updater.processDocument(doc, embedding: emb)
    // Back on main actor for UI update
    await MainActor.run {
        entry.topics = assignment
    }
}
```

---

## 5. Implementation Phases

### Phase 1: Storage Foundation (Week 1)

**Goal**: Implement storage protocol and file-based default

**Files to create**:
- `Sources/SwiftTopics/Incremental/Storage/TopicModelStorage.swift` (~150 LOC)
- `Sources/SwiftTopics/Incremental/Storage/FileBasedTopicModelStorage.swift` (~400 LOC)
- `Sources/SwiftTopics/Incremental/Storage/BufferedEntry.swift` (~50 LOC)

**Testable milestone**: Can append/load embeddings and buffer entries

**Estimated LOC**: ~600

### Phase 2: Checkpoint Infrastructure (Week 1-2)

**Goal**: Implement checkpoint state and serialization

**Files to create**:
- `Sources/SwiftTopics/Incremental/Checkpoint/TrainingCheckpoint.swift` (~200 LOC)
- `Sources/SwiftTopics/Incremental/Checkpoint/TrainingPhase.swift` (~100 LOC)
- `Sources/SwiftTopics/Incremental/Checkpoint/CheckpointSerializer.swift` (~150 LOC)

**Testable milestone**: Can save/load checkpoints, binary phase data

**Estimated LOC**: ~450

### Phase 3: Interruptible Training Components (Week 2-3)

**Goal**: Make UMAP and HDBSCAN interruptible

**Files to modify**:
- `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift` - Add checkpoint support (~100 LOC added)
- `Sources/SwiftTopics/Clustering/HDBSCAN/PrimMSTBuilder.swift` - Add checkpoint support (~80 LOC added)

**Files to create**:
- `Sources/SwiftTopics/Incremental/Training/InterruptibleTrainingRunner.swift` (~500 LOC)

**Testable milestone**: Can interrupt and resume UMAP optimization mid-epoch

**Estimated LOC**: ~680

### Phase 4: Topic Matching & Merging (Week 3)

**Goal**: Implement topic stability across retrains

**Files to create**:
- `Sources/SwiftTopics/Incremental/Merging/TopicMatcher.swift` (~200 LOC)
- `Sources/SwiftTopics/Incremental/Merging/ModelMerger.swift` (~300 LOC)

**Testable milestone**: Topics maintain stable IDs after micro-retrain

**Estimated LOC**: ~500

### Phase 5: Drift Detection (Week 3-4)

**Goal**: Implement drift statistics and refresh triggers

**Files to create**:
- `Sources/SwiftTopics/Incremental/Drift/DriftStatistics.swift` (~100 LOC)
- `Sources/SwiftTopics/Incremental/Drift/DriftDetector.swift` (~150 LOC)

**Testable milestone**: Can detect when full refresh is needed

**Estimated LOC**: ~250

### Phase 6: Main Actor Integration (Week 4)

**Goal**: Implement `IncrementalTopicUpdater` actor

**Files to create**:
- `Sources/SwiftTopics/Incremental/IncrementalTopicUpdater.swift` (~600 LOC)
- `Sources/SwiftTopics/Incremental/IncrementalUpdateConfiguration.swift` (~150 LOC)
- `Sources/SwiftTopics/Incremental/IncrementalTopicModelState.swift` (~200 LOC)

**Testable milestone**: End-to-end incremental update flow works

**Estimated LOC**: ~950

### Phase 7: Vocabulary Updates (Week 4-5)

**Goal**: Implement c-TF-IDF vocabulary management

**Files to create**:
- `Sources/SwiftTopics/Incremental/Vocabulary/Vocabulary.swift` (~150 LOC)
- `Sources/SwiftTopics/Incremental/Vocabulary/VocabularyUpdater.swift` (~200 LOC)

**Testable milestone**: Topic keywords update as new documents arrive

**Estimated LOC**: ~350

### Phase 8: Testing & Documentation (Week 5)

**Goal**: Comprehensive tests and documentation

**Files to create**:
- `Tests/SwiftTopicsTests/Incremental/IncrementalUpdaterTests.swift` (~500 LOC)
- `Tests/SwiftTopicsTests/Incremental/CheckpointTests.swift` (~300 LOC)
- `Tests/SwiftTopicsTests/Incremental/StorageTests.swift` (~300 LOC)

**Documentation updates**:
- Update `SPEC.md` with incremental update section
- Add usage examples to README

**Estimated LOC**: ~1100

---

## 6. Open Questions & Risks

### 6.1 Open Questions

| Question | Options | Recommendation |
|----------|---------|----------------|
| **Cold start threshold** | 20 / 30 / 50 entries | 30 (enough for meaningful clusters, not too long to wait) |
| **Mini-model UMAP epochs** | 100 / 200 / 500 | 200 (quality/speed balance for small batch) |
| **Topic matching algorithm** | Centroid similarity / Keyword overlap / Hybrid | Centroid similarity (faster, embedding-based) |
| **Background processing API** | BGAppRefreshTask / BGProcessingTask | BGProcessingTask (for >30s operations) |

### 6.2 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Topic instability after merge** | Medium | High | Hungarian matching + similarity threshold |
| **Checkpoint corruption** | Low | Medium | Atomic writes + validation on load |
| **Memory pressure during retrain** | Medium | Medium | Stream embeddings, don't load all at once |
| **UMAP transform quality degradation** | Medium | Low | Full refresh catches up periodically |
| **iOS background task killed** | High | Low | Checkpoint + resume design handles this |

### 6.3 Unknowns Requiring Prototyping

1. **Mini-model quality**: Need to validate that 30-doc UMAP/HDBSCAN produces usable topics
2. **Merge quality**: Need to test topic matching across different corpus evolution patterns
3. **Checkpoint size**: Need to measure actual checkpoint file sizes for different corpus sizes
4. **iOS background limits**: Need to test BGProcessingTask behavior for different training durations

---

## 7. Effort Estimate

### 7.1 Lines of Code by Phase

| Phase | New LOC | Modified LOC | Test LOC | Total |
|-------|---------|--------------|----------|-------|
| 1. Storage Foundation | 600 | 0 | 200 | 800 |
| 2. Checkpoint Infrastructure | 450 | 0 | 150 | 600 |
| 3. Interruptible Training | 500 | 180 | 200 | 880 |
| 4. Topic Matching | 500 | 0 | 150 | 650 |
| 5. Drift Detection | 250 | 0 | 100 | 350 |
| 6. Main Actor | 950 | 0 | 200 | 1150 |
| 7. Vocabulary Updates | 350 | 0 | 100 | 450 |
| 8. Testing & Docs | 0 | 0 | 600 | 600 |
| **TOTAL** | **3,600** | **180** | **1,700** | **5,480** |

### 7.2 Time Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 3-4 days | None |
| Phase 2 | 2-3 days | Phase 1 |
| Phase 3 | 4-5 days | Phase 2 |
| Phase 4 | 3-4 days | Phase 3 |
| Phase 5 | 2-3 days | Phase 4 |
| Phase 6 | 4-5 days | Phases 1-5 |
| Phase 7 | 2-3 days | Phase 6 |
| Phase 8 | 3-4 days | Phase 7 |
| **TOTAL** | **~5 weeks** | |

### 7.3 Suggested Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| **M1: Storage** | End of Week 1 | File-based storage with tests |
| **M2: Checkpoint** | End of Week 2 | Interruptible training POC |
| **M3: Core Flow** | End of Week 3 | `processDocument` works end-to-end |
| **M4: Production Ready** | End of Week 4 | Topic stability, drift detection |
| **M5: Release** | End of Week 5 | Full test coverage, documentation |

---

## 8. References

### Literature
- [BERTopic Online Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/online/online.html)
- [HDBSCAN Prediction Tutorial](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html)
- [HDBSCAN Soft Clustering](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Dynamic HDBSCAN with Bubble-tree](https://arxiv.org/html/2412.07789v1) (November 2024)
- [FISHDBC: Flexible Incremental HDBSCAN](https://arxiv.org/pdf/1910.07283)

### GitHub Issues
- [BERTopic #683: Incremental Learning](https://github.com/MaartenGr/BERTopic/issues/683)
- [BERTopic #2119: Dynamic Data Handling](https://github.com/MaartenGr/BERTopic/discussions/2119)
- [UMAP #755: Transform Point Shift](https://github.com/lmcinnes/umap/issues/755)

### SwiftTopics Internal
- `SPEC.md` Section 2.6: TopicModel Orchestrator
- `SPEC.md` Section 4.2: Optimization Strategies
- `Sources/SwiftTopics/Model/TopicModel.swift`: Current transform implementation
- `Sources/SwiftTopics/Model/TopicModelState.swift`: Current state structure

---

*Design Version: 1.0*
*Created: January 2025*
*Author: SwiftTopics Design Team*
*Status: Ready for Implementation*
