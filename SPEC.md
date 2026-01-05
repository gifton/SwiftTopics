# SwiftTopics Library Specification

## Executive Summary

**SwiftTopics** is a pure-Swift library for high-fidelity, on-device topic extraction. It implements a production-grade topic modeling pipeline inspired by BERTopic, optimized for Apple platforms with GPU acceleration via VectorAccelerate.

### Design Principles

1. **On-Device First** - All computation runs locally; no network calls required
2. **Embedding Agnostic** - Works with any embedding source (Apple NL, custom models, pre-computed)
3. **Mathematically Rigorous** - Implements academically-validated algorithms (HDBSCAN, UMAP, c-TF-IDF, NPMI)
4. **GPU-First Architecture** - Leverages VectorAccelerate's Metal 4 kernels for parallel operations
5. **Modular Architecture** - Each pipeline stage is independently usable and replaceable

### Platform Requirements

| Platform | Minimum Version | Reason |
|----------|-----------------|--------|
| **iOS** | 26.0+ | Metal 4, VectorAccelerate compatibility |
| **macOS** | 26.0+ | Metal 4, VectorAccelerate compatibility |
| **visionOS** | 26.0+ | Metal 4, VectorAccelerate compatibility |

> **Note**: This library targets cutting-edge platforms to leverage GPU acceleration via VectorAccelerate. It serves as a real-world testing ground for VectorAccelerate's Metal 4 kernels.

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **VectorAccelerate** | 0.3.1+ | GPU-accelerated distance computation, matrix operations, top-K selection |
| **VectorCore** | 0.1.6+ | (Transitive) CPU vector types, distance metrics, batch operations |

VectorAccelerate provides 25+ hand-tuned Metal kernels that eliminate the need to implement low-level math operations. VectorCore provides the foundational vector types and CPU fallbacks.

---

## Part 1: Architectural Overview

### 1.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SwiftTopics Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Embedding   │───▶│  Dimension   │───▶│  Clustering  │───▶│  Topic    │ │
│  │  Provider    │    │  Reduction   │    │   Engine     │    │  Repr.    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                    │                   │                   │       │
│        ▼                    ▼                   ▼                   ▼       │
│   EmbeddingProvider    DimensionReducer    ClusteringEngine    TopicRepresenter
│   (Protocol)           (Protocol)          (Protocol)          (Protocol)  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        TopicModel (Orchestrator)                      │  │
│  │  - Manages pipeline execution                                         │  │
│  │  - Handles document ↔ topic assignment                                │  │
│  │  - Exposes query/search APIs                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     CoherenceEvaluator (Quality)                      │  │
│  │  - NPMI scoring                                                       │  │
│  │  - C_V coherence                                                      │  │
│  │  - Topic diversity metrics                                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 GPU/CPU Hybrid Architecture

SwiftTopics uses a hybrid execution model: GPU for embarrassingly parallel operations, CPU for sequential graph algorithms.

#### GPU-Accelerated Operations (via VectorAccelerate)

| Operation | Kernel | Complexity | GPU Benefit |
|-----------|--------|------------|-------------|
| Pairwise distances | `L2DistanceKernel`, `CosineSimilarityKernel` | O(n²) | 🔥🔥🔥 Critical |
| Core distances (k-NN) | `FusedL2TopKKernel` | O(n × k) | 🔥🔥🔥 Critical |
| Covariance matrix | `MatrixMultiplyKernel` | O(n × d²) | 🔥🔥🔥 Critical |
| PCA projection | `MatrixMultiplyKernel` | O(n × d × k) | 🔥🔥 High |
| Data centering | `StatisticsKernel` | O(n × d) | 🔥🔥 High |
| Batch normalization | `L2NormalizationKernel` | O(n × d) | 🔥🔥 High |

#### CPU Operations (Sequential Algorithms)

| Operation | Reason | Complexity |
|-----------|--------|------------|
| MST construction | Sequential edge selection (Prim's) | O(n² log n) on edges |
| Cluster hierarchy | Must process edges in order | O(n) |
| Cluster extraction (EOM) | Tree traversal with dependencies | O(n) |
| Ball Tree construction | Recursive partitioning | O(n log n) |

#### New Kernels to Contribute to VectorAccelerate

These operations are SwiftTopics-specific but generally useful:

| Kernel | Purpose | Formula |
|--------|---------|---------|
| `MutualReachabilityKernel` | HDBSCAN distance metric | `max(core_a, core_b, dist_a_b)` |
| `CoreDistanceKernel` | k-th nearest neighbor distance | Per-point k-NN distance |
| `CTFIDFKernel` | Topic keyword scoring | `tf(t,c) × log(1 + A/tf(t,corpus))` |
| `NPMIKernel` | Coherence co-occurrence | Parallel counting with atomics |

### 1.3 Layer Responsibilities

| Layer | Responsibility | Key Types |
|-------|----------------|-----------|
| **Embedding** | Convert text → dense vectors | `EmbeddingProvider`, `Embedding` |
| **Reduction** | Compress high-dim → low-dim | `DimensionReducer`, `UMAP`, `PCA` |
| **Clustering** | Group similar vectors | `ClusteringEngine`, `HDBSCAN` |
| **Representation** | Extract topic keywords | `TopicRepresenter`, `cTFIDF` |
| **Evaluation** | Score topic quality | `CoherenceEvaluator`, `NPMIScorer` |
| **Orchestration** | Coordinate pipeline | `TopicModel`, `TopicModelConfiguration` |

### 1.3 Data Flow

```
Input: [Document]
           │
           ▼
    ┌─────────────────┐
    │ Tokenization    │  → [TokenizedDocument] (vocabulary, term frequencies)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Embedding       │  → [DocumentEmbedding] (document_id, vector: [Float])
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Dim Reduction   │  → [ReducedEmbedding] (document_id, vector: [Float], dims: 15-50)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Clustering      │  → ClusterAssignment (document_id → cluster_id, outlier flag)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Representation  │  → [Topic] (id, keywords, c-TF-IDF scores, coherence)
    └────────┬────────┘
             │
             ▼
Output: TopicModelResult
        - topics: [Topic]
        - documentTopics: [DocumentID: TopicID]
        - outliers: [DocumentID]
        - coherenceScore: Float
```

---

## Part 2: Core Components

### 2.1 Embedding Layer

#### Purpose
Provide a unified interface for obtaining document embeddings from various sources.

#### Design Decision: Protocol-Based Abstraction
The library does NOT include its own embedding model. Instead, it defines a protocol that consumers implement:

```
EmbeddingProvider (Protocol)
├── AppleNLEmbeddingProvider      // Uses NLEmbedding.sentenceEmbedding
├── PrecomputedEmbeddingProvider  // Uses pre-computed vectors (e.g., from GournalCore)
├── ONNXEmbeddingProvider         // Uses ONNX Runtime (optional, separate target)
└── CustomEmbeddingProvider       // Consumer-provided implementation
```

#### Rationale
- **GournalCore already has EmbeddingService** - No need to duplicate
- **Flexibility** - Different apps may use different embedding sources
- **Size** - Keeps library small; embedding models are optional dependencies

#### Contract
- Input: Text string or batch of strings
- Output: Dense vector(s) of consistent dimensionality
- Requirement: Must be deterministic (same text → same embedding)

#### Downstream Expectation
**GournalCore** will implement `EmbeddingProvider` as a thin wrapper around its existing `EmbeddingService`, passing through to the AppleNLContextualModel or custom model.

---

### 2.2 Dimension Reduction Layer

#### Purpose
Reduce embedding dimensionality (e.g., 384/512 → 15-50) to improve clustering quality and performance.

#### Design Decision: UMAP as Primary, PCA as Fallback

| Algorithm | Quality | Speed | Implementation Complexity |
|-----------|---------|-------|---------------------------|
| **UMAP** | High (preserves local + global structure) | Medium | High (~800 lines) |
| **PCA** | Medium (linear, loses non-linear structure) | Fast | Low (~150 lines) |
| **t-SNE** | High | Slow | Medium |

**Choice**: Implement both UMAP and PCA. UMAP default, PCA for speed-constrained scenarios.

#### UMAP Implementation Notes
UMAP (Uniform Manifold Approximation and Projection) requires:
1. **k-NN graph construction** - Find nearest neighbors in high-dim space
2. **Fuzzy simplicial set** - Build weighted graph from distances
3. **Low-dim optimization** - Iteratively minimize cross-entropy between high/low-dim graphs

Key parameters:
- `n_neighbors` (default: 15) - Local vs global structure tradeoff
- `min_dist` (default: 0.1) - How tightly points cluster
- `n_components` (default: 15) - Output dimensions

#### PCA Implementation Notes
Use Accelerate's `vDSP` for eigendecomposition:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Eigendecomposition via `ssyev` (LAPACK)
4. Project onto top-k eigenvectors

#### Downstream Expectation
- Input: `[[Float]]` matrix of shape (n_documents, embedding_dim)
- Output: `[[Float]]` matrix of shape (n_documents, reduced_dim)
- Must handle edge cases: single document, identical embeddings

---

### 2.3 Clustering Layer

#### Purpose
Group documents into coherent clusters representing topics, with automatic cluster count discovery and outlier detection.

#### Design Decision: HDBSCAN as Primary Algorithm

| Algorithm | Discovers K? | Handles Outliers? | Cluster Shapes | Implementation |
|-----------|--------------|-------------------|----------------|----------------|
| **HDBSCAN** | Yes | Yes | Arbitrary | Complex |
| **k-Means** | No | No | Spherical | Simple |
| **DBSCAN** | Yes | Yes | Arbitrary | Medium |
| **Label Propagation** | Yes | No | Graph-based | Simple |

**Choice**: HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

#### Rationale
- **No predefined K** - Topic count emerges from data
- **Outlier detection** - Documents that don't fit any topic are marked, not forced
- **Robust to noise** - Real journal data has noise; HDBSCAN handles it
- **Arbitrary shapes** - Topics aren't spherical in embedding space

#### HDBSCAN Algorithm Overview

```
Phase 1: Core Distance Computation
─────────────────────────────────
For each point, compute core_distance = distance to k-th nearest neighbor
This measures local density around each point

Phase 2: Mutual Reachability Distance
─────────────────────────────────────
mutual_reach(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
Creates a distance metric robust to varying densities

Phase 3: Minimum Spanning Tree
──────────────────────────────
Build MST on mutual reachability graph using Prim's algorithm
O(n² log n) with naive, O(n log n) with spatial indexing

Phase 4: Cluster Hierarchy
──────────────────────────
Convert MST to dendrogram by sorting edges and merging clusters
Track cluster births, deaths, and persistence

Phase 5: Cluster Extraction
───────────────────────────
Extract flat clustering using Excess of Mass (EOM) method
Select clusters that maximize stability (persistence × size)
Points not in stable clusters → outliers
```

#### Key Parameters
- `min_cluster_size` (default: 5) - Minimum documents to form a topic
- `min_samples` (default: nil, uses min_cluster_size) - Core point threshold
- `cluster_selection_epsilon` (default: 0.0) - Distance threshold for merging
- `cluster_selection_method` (default: .eom) - EOM vs leaf clustering

#### Downstream Expectation
- Input: `[[Float]]` reduced embeddings
- Output: `ClusterAssignment`
  - `labels: [Int]` - Cluster ID per document (-1 = outlier)
  - `probabilities: [Float]` - Cluster membership confidence
  - `outlierScores: [Float]` - How "outlier-like" each point is

---

### 2.4 Topic Representation Layer

#### Purpose
Extract interpretable keywords for each cluster using class-based TF-IDF (c-TF-IDF).

#### Design Decision: c-TF-IDF + Optional MMR Diversification

#### c-TF-IDF Algorithm

Traditional TF-IDF compares terms across documents. c-TF-IDF treats each cluster as a single "mega-document":

```
c-TF-IDF(term, cluster) = tf(term, cluster) × log(1 + A / tf(term, corpus))

Where:
- tf(term, cluster) = frequency of term in all documents of cluster
- A = average number of words per cluster
- tf(term, corpus) = frequency of term across entire corpus
```

This surfaces terms that are:
- Frequent within the cluster (high tf in cluster)
- Rare across other clusters (low tf in corpus)

#### MMR Diversification (Optional)

Maximal Marginal Relevance reduces keyword redundancy:

```
MMR(term) = λ × relevance(term) - (1-λ) × max_similarity(term, selected_terms)
```

Iteratively select terms that are relevant but dissimilar to already-selected terms.

#### Output Structure
For each cluster/topic:
- `keywords: [(term: String, score: Float)]` - Top-N terms by c-TF-IDF
- `representativeDocuments: [DocumentID]` - Documents closest to cluster centroid

#### Downstream Expectation
- Input: Cluster assignments + original documents + tokenization
- Output: `[Topic]` with keywords, scores, and representative docs
- Must handle: empty clusters, single-document clusters, very large clusters

---

### 2.5 Coherence Evaluation Layer

#### Purpose
Quantitatively measure topic quality using metrics that correlate with human judgment.

#### Design Decision: Implement NPMI and C_V

| Metric | Correlation with Human Judgment | Complexity | Reference Corpus |
|--------|--------------------------------|------------|------------------|
| **NPMI** | High | Low | Required |
| **C_V** | Highest | Medium | Required |
| **C_UMass** | Medium | Low | Uses training corpus |

#### NPMI (Normalized Pointwise Mutual Information)

```
NPMI(w1, w2) = (log P(w1, w2) / (P(w1) × P(w2))) / (-log P(w1, w2))

Topic coherence = average NPMI over all word pairs in topic keywords
```

Requires co-occurrence counts from a reference corpus (can be training corpus or external).

#### Implementation Notes
1. **Sliding window** - Count co-occurrences within N-word windows (default: 10)
2. **Smoothing** - Add epsilon to avoid log(0)
3. **Normalization** - NPMI ranges from -1 to +1

#### Downstream Expectation
- Input: `[Topic]` with keywords
- Output: Per-topic coherence scores + aggregate model coherence
- GournalCore can use coherence to:
  - Filter low-quality topics
  - Tune hyperparameters
  - Track topic quality over time

---

### 2.6 TopicModel Orchestrator

#### Purpose
Coordinate pipeline execution, manage state, and expose high-level APIs.

#### Responsibilities
1. **Pipeline execution** - Run embedding → reduction → clustering → representation
2. **State management** - Cache intermediate results for incremental updates
3. **Query interface** - Find topics for new documents, search by keyword
4. **Serialization** - Save/load trained models

#### Key Operations

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `fit(documents:)` | Train full pipeline | Initial model creation |
| `transform(documents:)` | Assign topics to new docs | New journal entries |
| `fitTransform(documents:)` | Train + assign | Convenience method |
| `getTopics()` | Return all topics | Display topic overview |
| `getDocumentTopics(id:)` | Topics for specific doc | Entry detail view |
| `findTopics(for: text)` | Topics for arbitrary text | Search/preview |
| `search(query:)` | Semantic topic search | Discovery features |
| `merge(topics:)` | Combine similar topics | Manual curation |
| `reduce(to: count)` | Reduce to N topics | Simplification |

#### Incremental Update Strategy
Full retraining is expensive. Support incremental updates:
1. **New document** - Embed, reduce, find nearest cluster, update c-TF-IDF
2. **Many new documents** - Batch update, recompute cluster stability
3. **Periodic refresh** - Full retrain when drift exceeds threshold

---

## Part 3: Integration Architecture

### 3.1 GournalCore Integration

#### Current State
GournalCore has `TopicClusteringService` with:
- TF-weighted candidate extraction
- Embedding via `EmbeddingService`
- Label propagation clustering
- Used by `IRGenerationService.extractConcepts()`

#### Migration Path

```
Phase 1: SwiftTopics as Dependency
──────────────────────────────────
GournalCore adds SwiftTopics as SPM dependency
Create GournalCoreEmbeddingProvider implementing EmbeddingProvider protocol
TopicClusteringService delegates to SwiftTopics

Phase 2: Enhanced Integration
─────────────────────────────
IRGenerationService uses SwiftTopics.TopicModel for concept extraction
Entry-level topic assignment stored alongside IR
Topic coherence tracked for quality monitoring

Phase 3: Full Topic Model
─────────────────────────
Corpus-wide TopicModel trained on all entries
Document ↔ topic relationships persisted
Topic browser/explorer UI powered by SwiftTopics queries
```

#### Adapter Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                      GournalCore                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            TopicClusteringService                    │   │
│  │  (Facade - delegates to SwiftTopics)                 │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         GournalCoreEmbeddingProvider                 │   │
│  │  implements: SwiftTopics.EmbeddingProvider           │   │
│  │  wraps: GournalCore.EmbeddingService                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       SwiftTopics                            │
├─────────────────────────────────────────────────────────────┤
│  TopicModel                                                  │
│  ├── DimensionReducer (UMAP/PCA)                            │
│  ├── ClusteringEngine (HDBSCAN)                             │
│  ├── TopicRepresenter (c-TF-IDF)                            │
│  └── CoherenceEvaluator (NPMI)                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Model Alignment

#### SwiftTopics Types → GournalCore Mapping

| SwiftTopics | GournalCore | Notes |
|-------------|-------------|-------|
| `Topic` | `ExtractedConcept` | Topic.keywords → concept.label + relatedTerms |
| `TopicModelResult` | `IntermediateRepresentation.concepts` | Aggregate IR stores topics |
| `DocumentTopicAssignment` | `JournalEntry.topics` | New field on entry |
| `CoherenceScore` | Quality metric | Store for monitoring |

#### Schema Considerations
GournalCore may need:
- `JournalEntry.topicIds: [UUID]?` - Assigned topic identifiers
- `TopicEntity` - Persisted topic model (separate from per-entry IR)
- `TopicModelMetadata` - Model version, training date, coherence

---

## Part 4: Performance Architecture

### 4.1 Computational Complexity

| Operation | Complexity | Dominant Factor |
|-----------|------------|-----------------|
| Embedding | O(n × d) | Depends on model |
| PCA | O(n × d²) | Covariance matrix |
| UMAP | O(n × k × log(n)) | k-NN graph |
| HDBSCAN | O(n² × log(n)) | Pairwise distances |
| c-TF-IDF | O(n × v) | v = vocabulary size |
| NPMI | O(t × k²) | t = topics, k = keywords |

**Bottleneck**: HDBSCAN's pairwise distance computation (O(n²))

### 4.2 Optimization Strategies

#### SIMD Acceleration (Accelerate/vDSP)
- Distance matrix computation
- Matrix multiplication (PCA, UMAP gradients)
- Vector operations (centroid computation)

#### Spatial Indexing (for HDBSCAN)
- **Ball tree** or **KD-tree** for k-NN queries
- Reduces O(n²) to O(n log n) for distance computation
- Implementation target: `SpatialIndex` module

#### Batch Processing
- Process documents in batches (e.g., 100 at a time)
- Amortize setup costs
- Enable progress reporting

#### Caching
- Cache k-NN graph (expensive to compute)
- Cache reduced embeddings
- Invalidate on corpus change

### 4.3 Memory Management

| Component | Memory Pattern | Strategy |
|-----------|----------------|----------|
| Embeddings | O(n × d) floats | Stream, don't hold all |
| Distance matrix | O(n²) floats | Compute on-demand or use spatial index |
| UMAP optimization | O(n × k) floats | Fixed working set |
| c-TF-IDF | O(v × t) floats | Sparse representation |

For 10,000 documents with 512-dim embeddings:
- Embeddings: ~20 MB
- Full distance matrix: ~400 MB (too large!)
- With spatial index: ~5 MB

**Decision**: Use spatial indexing, never materialize full distance matrix.

### 4.4 Performance Targets

| Operation | Target | Constraint |
|-----------|--------|------------|
| Single document transform | <50ms | Real-time UX |
| Batch (100 docs) | <2s | Background processing |
| Full retrain (1000 docs) | <30s | Acceptable for periodic refresh |
| Full retrain (10000 docs) | <5min | Background job |

---

## Part 5: Module Structure

### 5.1 Package Targets

```
SwiftTopics/
├── Package.swift
├── Sources/
│   ├── SwiftTopics/              # Main library
│   │   ├── Core/                 # Foundational types
│   │   │   ├── Document.swift
│   │   │   ├── Embedding.swift
│   │   │   ├── Topic.swift
│   │   │   └── TopicModelResult.swift
│   │   │
│   │   ├── Embedding/            # Embedding abstraction
│   │   │   ├── EmbeddingProvider.swift
│   │   │   └── PrecomputedProvider.swift
│   │   │
│   │   ├── Reduction/            # Dimensionality reduction
│   │   │   ├── DimensionReducer.swift
│   │   │   ├── PCA.swift
│   │   │   └── UMAP.swift
│   │   │
│   │   ├── Clustering/           # Clustering algorithms
│   │   │   ├── ClusteringEngine.swift
│   │   │   ├── HDBSCAN/
│   │   │   │   ├── HDBSCAN.swift
│   │   │   │   ├── CoreDistance.swift
│   │   │   │   ├── MinimumSpanningTree.swift
│   │   │   │   ├── ClusterHierarchy.swift
│   │   │   │   └── ClusterExtraction.swift
│   │   │   └── SpatialIndex/
│   │   │       ├── SpatialIndex.swift
│   │   │       ├── BallTree.swift
│   │   │       └── KDTree.swift
│   │   │
│   │   ├── Representation/       # Topic representation
│   │   │   ├── TopicRepresenter.swift
│   │   │   ├── cTFIDF.swift
│   │   │   └── Tokenizer.swift
│   │   │
│   │   ├── Evaluation/           # Quality metrics
│   │   │   ├── CoherenceEvaluator.swift
│   │   │   ├── NPMIScorer.swift
│   │   │   └── DiversityMetrics.swift
│   │   │
│   │   ├── Model/                # Orchestration
│   │   │   ├── TopicModel.swift
│   │   │   ├── TopicModelConfiguration.swift
│   │   │   └── IncrementalUpdater.swift
│   │   │
│   │   └── Utilities/            # Shared utilities
│   │       ├── LinearAlgebra.swift
│   │       ├── DistanceMetrics.swift
│   │       └── RandomState.swift
│   │
│   └── SwiftTopicsApple/         # Apple-specific providers (optional target)
│       └── AppleNLEmbeddingProvider.swift
│
└── Tests/
    └── SwiftTopicsTests/
        ├── HDBSCANTests.swift
        ├── UMAPTests.swift
        ├── cTFIDFTests.swift
        ├── CoherenceTests.swift
        └── IntegrationTests.swift
```

### 5.2 Dependency Graph

```
SwiftTopics (main target)
├── Dependencies: VectorAccelerate (→ VectorCore)
└── Imports: Foundation, Accelerate, Metal

SwiftTopicsApple (optional target)
├── Dependencies: SwiftTopics
└── Imports: NaturalLanguage
```

### 5.3 SPM Configuration

```swift
// Package.swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwiftTopics",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
        .visionOS(.v26)
    ],
    products: [
        .library(name: "SwiftTopics", targets: ["SwiftTopics"]),
        .library(name: "SwiftTopicsApple", targets: ["SwiftTopicsApple"]),
    ],
    dependencies: [
        .package(url: "https://github.com/gifton/VectorAccelerate.git", from: "0.3.1"),
    ],
    targets: [
        .target(
            name: "SwiftTopics",
            dependencies: [
                .product(name: "VectorAccelerate", package: "VectorAccelerate"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "SwiftTopicsApple",
            dependencies: ["SwiftTopics"]
        ),
        .testTarget(
            name: "SwiftTopicsTests",
            dependencies: ["SwiftTopics"]
        ),
    ]
)
```

### 5.4 What VectorAccelerate Provides (No Reimplementation Needed)

| Category | VectorAccelerate Component | SwiftTopics Usage |
|----------|---------------------------|-------------------|
| **Distance Metrics** | `L2DistanceKernel`, `CosineSimilarityKernel` | HDBSCAN pairwise distances |
| **k-NN Search** | `FusedL2TopKKernel`, `TopKSelectionKernel` | Core distance computation |
| **Matrix Operations** | `MatrixMultiplyKernel`, `MatrixTransposeKernel` | PCA covariance & projection |
| **Normalization** | `L2NormalizationKernel` | Embedding preprocessing |
| **Statistics** | `StatisticsKernel` | Mean centering for PCA |
| **Reduction** | `ParallelReductionKernel` | Sum, min, max operations |

### 5.5 What SwiftTopics Implements

| Category | Component | Reason |
|----------|-----------|--------|
| **Eigendecomposition** | LAPACK `ssyev` wrapper | Not in VectorAccelerate |
| **Ball Tree** | Spatial index structure | Algorithmic, not just math |
| **HDBSCAN** | Full algorithm | Core clustering logic |
| **c-TF-IDF** | Topic representation | Domain-specific |
| **NPMI** | Coherence scoring | Domain-specific |
| **TopicModel** | Orchestration | Pipeline coordination |

---

## Part 6: API Design Principles

### 6.1 Concurrency Model

All public APIs are `async` and designed for Swift Concurrency:
- `TopicModel` is an `actor` (thread-safe mutable state)
- Stateless algorithms (HDBSCAN, PCA) are pure functions
- Progress reporting via `AsyncStream`

### 6.2 Error Handling

Structured error types:
- `TopicModelError` - High-level pipeline errors
- `ClusteringError` - HDBSCAN-specific errors
- `ReductionError` - UMAP/PCA errors
- `CoherenceError` - Evaluation errors

All errors include context (document count, parameters, etc.) for debugging.

### 6.3 Configuration Philosophy

**Opinionated defaults, full customization available:**

```
TopicModelConfiguration
├── .default          → Balanced for most use cases
├── .fast             → Speed over quality
├── .quality          → Quality over speed
└── .custom(...)      → Full parameter control
```

Parameters exposed but not required. Sensible defaults based on BERTopic research.

### 6.4 Observability

- **Logging** - Structured logging via `os_log` / unified logging
- **Metrics** - Timing for each pipeline stage
- **Diagnostics** - Intermediate results for debugging (cluster hierarchy, etc.)

---

## Part 7: Testing Strategy

### 7.1 Unit Tests

| Component | Test Focus |
|-----------|------------|
| HDBSCAN | Known datasets (blobs, moons), edge cases |
| UMAP | Dimension preservation, determinism with seed |
| c-TF-IDF | Score correctness, empty cluster handling |
| NPMI | Against known coherence values |

### 7.2 Integration Tests

- Full pipeline on synthetic data with known topics
- Comparison against Python BERTopic on same data
- Performance benchmarks

### 7.3 Property-Based Tests

- Cluster count ≤ document count
- All documents assigned (cluster or outlier)
- Coherence in valid range [-1, 1]
- Deterministic with fixed seed

---

## Part 8: Future Considerations

### 8.1 Potential Extensions

1. **Hierarchical Topics** - Tree structure of topics (subtopics)
2. **Dynamic Topics** - Topics that evolve over time (temporal modeling)
3. **Guided Topics** - Semi-supervised with seed words
4. **Multi-modal** - Topics from text + images

### 8.2 Metal Acceleration

For very large corpora (>50k documents):
- Distance matrix computation on GPU
- UMAP optimization on GPU
- Requires separate `SwiftTopicsMetal` target

### 8.3 Model Persistence

- Serialization format for trained TopicModel
- Versioning for backward compatibility
- Core Data / SwiftData integration patterns

---

## Appendix A: Reference Materials

### Academic Papers
- HDBSCAN: Campello et al., "Density-Based Clustering Based on Hierarchical Density Estimates" (2013)
- UMAP: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018)
- BERTopic: Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure" (2022)
- Topic Coherence: Röder et al., "Exploring the Space of Topic Coherence Measures" (2015)

### Implementation References
- scikit-learn HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan
- umap-learn: https://github.com/lmcinnes/umap
- BERTopic: https://github.com/MaartenGr/BERTopic
- Gensim CoherenceModel: https://radimrehurek.com/gensim/models/coherencemodel.html

### Apple Frameworks
- Accelerate: https://developer.apple.com/documentation/accelerate
- NaturalLanguage: https://developer.apple.com/documentation/naturallanguage
- vDSP: https://developer.apple.com/documentation/accelerate/vdsp

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **c-TF-IDF** | Class-based TF-IDF; measures term importance within a cluster relative to corpus |
| **Coherence** | Measure of topic interpretability based on word co-occurrence |
| **Core distance** | Distance to k-th nearest neighbor; measures local density |
| **HDBSCAN** | Hierarchical Density-Based Spatial Clustering of Applications with Noise |
| **Mutual reachability** | Distance metric that accounts for varying densities |
| **NPMI** | Normalized Pointwise Mutual Information; coherence metric |
| **Outlier** | Document that doesn't belong to any cluster |
| **UMAP** | Uniform Manifold Approximation and Projection; dimensionality reduction |

---

*Spec Version: 1.0*
*Library Version: 0.1.0-beta.1*
*Created: January 2025*
*Last Updated: January 2026*
*Author: SwiftTopics Design Team*
