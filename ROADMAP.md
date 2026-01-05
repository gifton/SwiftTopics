# SwiftTopics Implementation Roadmap

## Overview

**Total Estimated Effort**: 4-5 weeks (part-time) / 2-3 weeks (focused)
**Lines of Code Estimate**: ~3,000-4,000 lines Swift (reduced due to VectorAccelerate)
**Test Coverage Target**: >85%

### Platform Requirements

- **iOS**: 26.0+
- **macOS**: 26.0+
- **visionOS**: 26.0+

### Dependencies

- **VectorAccelerate** (0.3.1+) - GPU-accelerated kernels for distance, matrix ops, top-K
- **VectorCore** (0.1.6+) - Transitive dependency, CPU vector types and operations

### Scope Reduction from VectorAccelerate

VectorAccelerate eliminates the need to implement:
- ~~Distance metrics (euclidean, cosine, etc.)~~ → Use `L2DistanceKernel`, `CosineSimilarityKernel`
- ~~Basic linear algebra (dot product, normalization)~~ → Use `L2NormalizationKernel`
- ~~Matrix multiplication~~ → Use `MatrixMultiplyKernel`
- ~~Top-K selection~~ → Use `FusedL2TopKKernel`, `TopKSelectionKernel`
- ~~Statistics (mean, variance)~~ → Use `StatisticsKernel`

This reduces Phase 1 significantly and provides GPU acceleration for free.

---

## Phase 0: Foundation
**Duration**: 2-3 days
**LOC**: ~300

### Objective
Establish project structure, core types, and development infrastructure.

### Deliverables

#### 0.1 Project Setup
- [ ] Configure `Package.swift` with proper targets and Swift 6 settings
- [ ] Set up CI (GitHub Actions or local script)
- [ ] Configure SwiftLint/SwiftFormat rules
- [ ] Create CONTRIBUTING.md with code style guide

#### 0.2 Core Types
```
Sources/SwiftTopics/Core/
├── Document.swift          # Document abstraction (id, content, metadata)
├── Embedding.swift         # Vector wrapper with dimension validation
├── Topic.swift             # Topic with keywords, scores, coherence
├── ClusterAssignment.swift # Document → cluster mapping
└── TopicModelResult.swift  # Complete pipeline output
```

#### 0.3 Protocol Definitions
```
Sources/SwiftTopics/Protocols/
├── EmbeddingProvider.swift   # Input: text → Output: vector
├── DimensionReducer.swift    # Input: high-dim → Output: low-dim
├── ClusteringEngine.swift    # Input: vectors → Output: assignments
└── TopicRepresenter.swift    # Input: clusters + docs → Output: topics
```

### Exit Criteria
- [ ] `swift build` succeeds
- [ ] All protocols defined with documentation
- [ ] Core types are `Sendable` and `Codable`
- [ ] Basic unit tests for core types

### Dependencies
None

---

## Phase 1: GPU Integration & Linear Algebra
**Duration**: 1-2 days
**LOC**: ~200 (significantly reduced)

### Objective
Integrate VectorAccelerate kernels and implement remaining math utilities not provided by VectorAccelerate.

### What VectorAccelerate Provides (No Implementation Needed)

| Capability | VectorAccelerate Kernel |
|------------|------------------------|
| Euclidean distance | `L2DistanceKernel` |
| Cosine similarity | `CosineSimilarityKernel` |
| Pairwise distances | `L2DistanceKernel.computePairwise()` |
| Top-K selection | `FusedL2TopKKernel`, `TopKSelectionKernel` |
| Matrix multiply | `MatrixMultiplyKernel` |
| Matrix transpose | `MatrixTransposeKernel` |
| L2 normalization | `L2NormalizationKernel` |
| Mean/variance | `StatisticsKernel` |

### Deliverables

#### 1.1 VectorAccelerate Integration Layer
```
Sources/SwiftTopics/Acceleration/GPUContext.swift
```
- [ ] `TopicsGPUContext` actor wrapping `Metal4Context`
- [ ] Lazy initialization of required kernels
- [ ] Graceful error handling for GPU unavailability

#### 1.2 Eigendecomposition (Not in VectorAccelerate)
```
Sources/SwiftTopics/Utilities/Eigendecomposition.swift
```
- [ ] Symmetric eigendecomposition via LAPACK `ssyev`
- [ ] Returns eigenvalues (sorted) and eigenvectors
- [ ] Used by PCA for principal components

#### 1.3 Random Number Generation
```
Sources/SwiftTopics/Utilities/RandomState.swift
```
- [ ] Seedable random number generator
- [ ] Random sampling (with/without replacement)
- [ ] Shuffling with deterministic seed

### Exit Criteria
- [ ] GPU context initializes successfully
- [ ] Eigendecomposition matches known test cases
- [ ] Random state produces deterministic sequences with same seed

### Dependencies
- Phase 0 (Core types)
- VectorAccelerate package

---

## Phase 2: Spatial Indexing
**Duration**: 3-4 days
**LOC**: ~500

### Objective
Implement efficient k-nearest neighbor queries for HDBSCAN core distance computation.

### GPU vs CPU Trade-off

VectorAccelerate provides `FusedL2TopKKernel` which computes distances + top-K in one GPU pass. This is optimal for:
- **Batch queries**: Many queries against same dataset
- **Moderate k**: k < 100

Ball Tree is better for:
- **Single queries**: Individual k-NN lookups
- **Large k**: k > 100
- **Radius queries**: Find all points within distance threshold

**Decision**: Implement Ball Tree for flexibility, use GPU top-K for batch core distance computation.

### Deliverables

#### 2.1 Spatial Index Protocol
```
Sources/SwiftTopics/Clustering/SpatialIndex/SpatialIndex.swift
```
- [ ] `query(point:k:) -> [(index: Int, distance: Float)]`
- [ ] `queryRadius(point:radius:) -> [(index: Int, distance: Float)]`
- [ ] `build(points:)` factory method

#### 2.2 Ball Tree Implementation
```
Sources/SwiftTopics/Clustering/SpatialIndex/BallTree.swift
```
- [ ] Recursive ball tree construction
- [ ] k-NN query with branch pruning
- [ ] Radius query
- [ ] Leaf size tuning parameter
- [ ] Uses VectorAccelerate for distance computations within nodes

**Algorithm Notes**:
- Partition data recursively into hyperspheres
- Each node stores: center, radius, left/right children
- Query prunes branches where `distance(query, center) - radius > current_kth_distance`

#### 2.3 GPU Batch k-NN (Alternative Path)
```
Sources/SwiftTopics/Clustering/SpatialIndex/GPUBatchKNN.swift
```
- [ ] Wrapper around `FusedL2TopKKernel` for batch queries
- [ ] Used for core distance computation (all points at once)

### Exit Criteria
- [ ] k-NN results match brute-force (correctness)
- [ ] O(n log n) build time verified
- [ ] O(log n) average query time verified for Ball Tree
- [ ] GPU batch k-NN matches Ball Tree results

### Dependencies
- Phase 1 (GPU Integration)

### Milestone: M1 - Spatial Foundation ✓
**Checkpoint**: Core math and spatial indexing complete. Can efficiently find nearest neighbors in high-dimensional space.

---

## Phase 3: HDBSCAN Clustering
**Duration**: 7-10 days
**LOC**: ~800

### Objective
Implement the full HDBSCAN algorithm with cluster extraction.

### Deliverables

#### 3.1 Core Distance Computation
```
Sources/SwiftTopics/Clustering/HDBSCAN/CoreDistance.swift
```
- [ ] Compute core distance for each point (distance to k-th neighbor)
- [ ] Use spatial index for efficiency
- [ ] Handle edge case: k > number of points

#### 3.2 Mutual Reachability Graph
```
Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift
```
- [ ] Compute mutual reachability distance: `max(core_dist(a), core_dist(b), dist(a,b))`
- [ ] Sparse representation (don't materialize full matrix)

#### 3.3 Minimum Spanning Tree
```
Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift
```
- [ ] Prim's algorithm on mutual reachability graph
- [ ] Use spatial index + core distances for edge weights
- [ ] Output: sorted edges by weight

**Algorithm**:
1. Start with arbitrary point
2. Repeatedly add cheapest edge to unvisited point
3. Use priority queue (heap) for efficiency

#### 3.4 Cluster Hierarchy (Dendrogram)
```
Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchy.swift
```
- [ ] Build single-linkage hierarchy from MST
- [ ] Track cluster births (when cluster forms)
- [ ] Track cluster deaths (when cluster merges)
- [ ] Compute cluster stability (persistence × size)

**Data Structure**:
```
struct ClusterNode {
    let id: Int
    let parent: Int?
    let children: [Int]
    let birthLevel: Float  // Distance at which cluster formed
    let deathLevel: Float  // Distance at which cluster merged
    let size: Int
    var stability: Float   // Computed from persistence
}
```

#### 3.5 Cluster Extraction
```
Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift
```
- [ ] Excess of Mass (EOM) method - select clusters maximizing total stability
- [ ] Leaf clustering method - alternative, takes all leaf clusters
- [ ] Label assignment - map points to selected clusters
- [ ] Outlier detection - points not in any stable cluster
- [ ] Membership probability computation

#### 3.6 HDBSCAN Orchestrator
```
Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift
```
- [ ] `HDBSCANConfiguration` with parameters:
  - `minClusterSize: Int` (default: 5)
  - `minSamples: Int?` (default: nil, uses minClusterSize)
  - `clusterSelectionEpsilon: Float` (default: 0.0)
  - `clusterSelectionMethod: ClusterSelectionMethod` (default: .eom)
- [ ] `fit(embeddings:) -> ClusterAssignment`
- [ ] Deterministic with seed

### Exit Criteria
- [ ] Matches scikit-learn HDBSCAN output on test datasets (blobs, moons, circles)
- [ ] Correctly identifies outliers
- [ ] Handles edge cases: single point, all identical, very sparse
- [ ] Performance: 1000 points in <1s, 10000 points in <30s

### Dependencies
- Phase 2 (Spatial indexing)

### Milestone: M2 - Clustering Engine ✓
**Checkpoint**: Can cluster embeddings into topics with automatic cluster count discovery and outlier detection.

---

## Phase 4: Dimensionality Reduction - PCA
**Duration**: 2-3 days
**LOC**: ~200

### Objective
Implement PCA as the initial (faster) dimensionality reduction option.

### Deliverables

#### 4.1 PCA Implementation
```
Sources/SwiftTopics/Reduction/PCA.swift
```
- [ ] Center data (subtract mean)
- [ ] Compute covariance matrix
- [ ] Eigendecomposition via LAPACK (`ssyev`)
- [ ] Project onto top-k eigenvectors
- [ ] Store transformation for `transform()` on new data

#### 4.2 PCA Configuration
- [ ] `components: Int` - Number of output dimensions
- [ ] `whiten: Bool` - Scale by eigenvalues (optional)

### Exit Criteria
- [ ] Variance preserved matches expected (e.g., 95% with k components)
- [ ] Reconstruction error within tolerance
- [ ] Matches sklearn PCA output on test data

### Dependencies
- Phase 1 (Linear algebra)

---

## Phase 5: Topic Representation - c-TF-IDF
**Duration**: 3-4 days
**LOC**: ~400

### Objective
Extract interpretable keywords for each cluster using class-based TF-IDF.

### Deliverables

#### 5.1 Tokenizer
```
Sources/SwiftTopics/Representation/Tokenizer.swift
```
- [ ] Whitespace/punctuation tokenization
- [ ] Lowercase normalization
- [ ] Stop word filtering (configurable list)
- [ ] Minimum token length filter
- [ ] N-gram support (optional, for phrases)

#### 5.2 Vocabulary Builder
```
Sources/SwiftTopics/Representation/Vocabulary.swift
```
- [ ] Build vocabulary from corpus
- [ ] Term frequency per document
- [ ] Document frequency per term
- [ ] Vocabulary pruning (min_df, max_df)

#### 5.3 c-TF-IDF Computation
```
Sources/SwiftTopics/Representation/cTFIDF.swift
```
- [ ] Aggregate documents by cluster
- [ ] Compute term frequency per cluster
- [ ] Compute c-TF-IDF scores:
  ```
  c-TF-IDF(t, c) = tf(t, c) × log(1 + A / tf(t, corpus))
  ```
- [ ] Rank terms by score per cluster
- [ ] Return top-N keywords per topic

#### 5.4 MMR Diversification (Optional)
```
Sources/SwiftTopics/Representation/MMR.swift
```
- [ ] Maximal Marginal Relevance for keyword diversity
- [ ] Requires word embeddings (optional dependency)

### Exit Criteria
- [ ] Keywords are interpretable and distinctive per cluster
- [ ] Handles empty clusters gracefully
- [ ] Handles single-document clusters
- [ ] Performance: <100ms for 100 clusters

### Dependencies
- Phase 3 (Cluster assignments)

### Milestone: M3 - Full Pipeline (Basic) ✓
**Checkpoint**: Can run complete pipeline: embeddings → PCA → HDBSCAN → c-TF-IDF → topics with keywords.

---

## Phase 6: Coherence Evaluation
**Duration**: 3-4 days
**LOC**: ~400

### Objective
Implement topic quality metrics for evaluation and hyperparameter tuning.

### Deliverables

#### 6.1 Co-occurrence Counter
```
Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift
```
- [ ] Sliding window co-occurrence (configurable window size)
- [ ] Document co-occurrence (boolean)
- [ ] Efficient sparse storage

#### 6.2 NPMI Scorer
```
Sources/SwiftTopics/Evaluation/NPMIScorer.swift
```
- [ ] Compute P(w1), P(w2), P(w1, w2) from co-occurrence counts
- [ ] NPMI formula:
  ```
  NPMI(w1, w2) = log(P(w1,w2) / (P(w1)×P(w2))) / -log(P(w1,w2))
  ```
- [ ] Smoothing to avoid log(0)
- [ ] Average over word pairs in topic
- [ ] Aggregate across topics

#### 6.3 Coherence Evaluator
```
Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift
```
- [ ] `evaluate(topics:corpus:) -> CoherenceResult`
- [ ] Per-topic coherence scores
- [ ] Aggregate coherence (mean, median)
- [ ] Configurable: window size, top-N words to consider

#### 6.4 Diversity Metrics (Optional)
```
Sources/SwiftTopics/Evaluation/DiversityMetrics.swift
```
- [ ] Topic diversity: % unique words across topics
- [ ] Redundancy score: overlap between topics

### Exit Criteria
- [ ] NPMI scores correlate with human judgment (manual verification)
- [ ] Scores in expected range [-1, 1]
- [ ] Can differentiate good vs. bad topic models

### Dependencies
- Phase 5 (Topic representation with keywords)

---

## Phase 7: TopicModel Orchestrator
**Duration**: 4-5 days
**LOC**: ~600

### Objective
Create the main API that coordinates the full pipeline.

### Deliverables

#### 7.1 Configuration
```
Sources/SwiftTopics/Model/TopicModelConfiguration.swift
```
- [ ] Presets: `.default`, `.fast`, `.quality`
- [ ] Component-specific configs nested
- [ ] Validation of parameter combinations

#### 7.2 TopicModel Actor
```
Sources/SwiftTopics/Model/TopicModel.swift
```
- [ ] `fit(documents:embeddings:) async throws -> TopicModelResult`
- [ ] `transform(documents:embeddings:) async throws -> [TopicAssignment]`
- [ ] `fitTransform(documents:embeddings:) async throws -> TopicModelResult`
- [ ] `getTopics() -> [Topic]`
- [ ] `getDocumentTopic(id:) -> TopicAssignment?`
- [ ] `findTopic(for text:embedding:) async throws -> TopicAssignment`

#### 7.3 Progress Reporting
- [ ] `AsyncStream<TopicModelProgress>` for long-running operations
- [ ] Progress stages: embedding, reduction, clustering, representation, evaluation

#### 7.4 Serialization
```
Sources/SwiftTopics/Model/TopicModelState.swift
```
- [ ] `Codable` representation of trained model
- [ ] Save/load to file
- [ ] Version compatibility

### Exit Criteria
- [ ] Full pipeline runs end-to-end
- [ ] Progress reporting works
- [ ] Model can be saved and loaded
- [ ] Thread-safe (actor isolation)

### Dependencies
- All previous phases

### Milestone: M4 - Production API ✓
**Checkpoint**: Complete, usable API with configuration, progress, and serialization.

---

## Phase 8: UMAP (Enhancement)
**Duration**: 5-7 days
**LOC**: ~800

### Objective
Implement UMAP for higher-quality dimensionality reduction (optional but recommended).

### Deliverables

#### 8.1 k-NN Graph Construction
```
Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift
```
- [ ] Use spatial index for k-NN
- [ ] Compute distances to k nearest neighbors
- [ ] Handle disconnected components

#### 8.2 Fuzzy Simplicial Set
```
Sources/SwiftTopics/Reduction/UMAP/FuzzySimplicialSet.swift
```
- [ ] Compute membership strengths from distances
- [ ] Smooth k-nn distances: `ρ` (distance to nearest) and `σ` (bandwidth)
- [ ] Symmetrize the graph

#### 8.3 Spectral Embedding Initialization
```
Sources/SwiftTopics/Reduction/UMAP/SpectralEmbedding.swift
```
- [ ] Laplacian eigenmaps for initial low-dim positions
- [ ] Fallback to random initialization

#### 8.4 Optimization (SGD)
```
Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift
```
- [ ] Negative sampling for efficiency
- [ ] Attractive forces (connected points)
- [ ] Repulsive forces (sampled non-neighbors)
- [ ] Learning rate schedule
- [ ] Convergence detection

#### 8.5 UMAP Orchestrator
```
Sources/SwiftTopics/Reduction/UMAP/UMAP.swift
```
- [ ] `UMAPConfiguration`:
  - `nNeighbors: Int` (default: 15)
  - `minDist: Float` (default: 0.1)
  - `nComponents: Int` (default: 15)
  - `metric: DistanceMetric` (default: .euclidean)
  - `nEpochs: Int?` (default: nil, auto)
- [ ] `fit(embeddings:) -> [[Float]]`
- [ ] `transform(embeddings:) -> [[Float]]` (approximate)

### Exit Criteria
- [ ] Preserves local structure (k-NN accuracy in low-dim)
- [ ] Produces separable clusters
- [ ] Matches umap-learn output qualitatively
- [ ] Performance: 1000 points in <5s

### Dependencies
- Phase 2 (Spatial indexing)
- Phase 1 (Linear algebra)

### Milestone: M5 - Full Quality Pipeline ✓
**Checkpoint**: Complete pipeline with UMAP for highest quality results.

---

## Phase 9: Apple Integration Target
**Duration**: 2-3 days
**LOC**: ~200

### Objective
Create optional target with Apple-specific embedding providers.

### Deliverables

#### 9.1 Apple NL Embedding Provider
```
Sources/SwiftTopicsApple/AppleNLEmbeddingProvider.swift
```
- [ ] Implement `EmbeddingProvider` protocol
- [ ] Use `NLEmbedding.sentenceEmbedding(for:)`
- [ ] Batch processing support
- [ ] Language detection fallback

#### 9.2 Apple NL Word Embedding Provider
```
Sources/SwiftTopicsApple/AppleNLWordEmbeddingProvider.swift
```
- [ ] Word-level embeddings via `NLEmbedding.wordEmbedding(for:)`
- [ ] Aggregation strategies (mean, max pooling)

### Exit Criteria
- [ ] Works with Apple's built-in models
- [ ] No external dependencies
- [ ] Handles unsupported languages gracefully

### Dependencies
- Phase 0 (Protocols)

---

## Phase 10: GournalCore Integration
**Duration**: 3-4 days
**LOC**: ~300 (in GournalCore)

### Objective
Integrate SwiftTopics into GournalCore, replacing/enhancing TopicClusteringService.

### Deliverables

#### 10.1 GournalCore Embedding Provider
```
GournalCore/Sources/GournalCore/ML/GournalCoreEmbeddingProvider.swift
```
- [ ] Implement `SwiftTopics.EmbeddingProvider`
- [ ] Wrap `EmbeddingService`
- [ ] Handle batch embedding

#### 10.2 TopicClusteringService Update
```
GournalCore/Sources/GournalCore/ML/TopicClusteringService.swift
```
- [ ] Add SwiftTopics.TopicModel as optional backend
- [ ] Feature flag for migration
- [ ] Fallback to existing implementation

#### 10.3 IR Integration
- [ ] Map `SwiftTopics.Topic` to `ExtractedConcept`
- [ ] Store coherence scores in IR metadata
- [ ] Update `IRGenerationService`

### Exit Criteria
- [ ] GournalCore builds with SwiftTopics dependency
- [ ] Topic quality improved (measure via coherence)
- [ ] No regression in performance
- [ ] Feature flag allows rollback

### Dependencies
- Phase 7 (TopicModel API)

### Milestone: M6 - Integration Complete ✓
**Checkpoint**: SwiftTopics fully integrated into GournalCore with improved topic extraction.

---

## Timeline Summary

```
Week 1: Foundation + GPU Integration
├── Phase 0: Foundation (2-3 days)
└── Phase 1: GPU Integration (1-2 days) ← Reduced from 3-4 days

Week 2: Spatial + Clustering Start
├── Phase 2: Spatial Indexing (3-4 days)
└── Phase 3: HDBSCAN Start (3 days)

Week 3: Clustering + Reduction
├── Phase 3: HDBSCAN Complete (4-5 days)
└── Phase 4: PCA (1-2 days) ← Reduced (GPU matrix ops)

Week 4: Representation + Evaluation + Orchestration
├── Phase 5: c-TF-IDF (3-4 days)
├── Phase 6: Coherence (3-4 days)
└── Phase 7: TopicModel (3-4 days)

Week 5: Enhancement + Integration
├── Phase 8: UMAP (5-7 days) [Optional]
├── Phase 9: Apple Integration (2-3 days)
└── Phase 10: GournalCore Integration (3-4 days)
```

**Note**: Timeline reduced by ~1 week due to VectorAccelerate providing GPU-accelerated math operations.

---

## Milestone Checklist

| Milestone | Phase | Description | Target |
|-----------|-------|-------------|--------|
| **M1** | 0-2 | Spatial Foundation | End of Week 2 |
| **M2** | 3 | Clustering Engine | End of Week 3 |
| **M3** | 4-5 | Full Pipeline (Basic) | End of Week 4 |
| **M4** | 6-7 | Production API | End of Week 5 |
| **M5** | 8 | Full Quality Pipeline | End of Week 6 |
| **M6** | 9-10 | Integration Complete | End of Week 7 |

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| HDBSCAN complexity | High | Medium | Start early, follow scikit-learn closely |
| UMAP optimization instability | Medium | Medium | Extensive testing, fallback to PCA |
| Performance not meeting targets | High | Low | Profile early, optimize bottlenecks |
| Accelerate API limitations | Medium | Low | Research upfront, have fallback implementations |
| Integration breaks GournalCore | High | Low | Feature flags, comprehensive testing |

---

## Success Metrics

### Quality Metrics
- [ ] NPMI coherence ≥ 0.1 (positive coherence indicates interpretable topics)
- [ ] Topic diversity ≥ 0.7 (70% unique words across topics)
- [ ] Outlier rate ≤ 20% (most documents assigned to topics)

### Performance Metrics
- [ ] Single document transform: <50ms
- [ ] 100 documents: <2s
- [ ] 1000 documents: <30s
- [ ] Memory: <100MB for 10k documents

### Integration Metrics
- [ ] GournalCore build time: No significant increase
- [ ] Topic extraction quality: Measurably better than current
- [ ] No regressions in existing tests

---

## Appendix: Effort Estimates by Component

| Component | Lines | Complexity | Days | Notes |
|-----------|-------|------------|------|-------|
| Core Types | 300 | Low | 2 | Unchanged |
| GPU Integration | 150 | Low | 1 | Wrapper around VectorAccelerate |
| Eigendecomposition | 100 | Medium | 1 | LAPACK wrapper only |
| ~~Distance/LinAlg~~ | ~~500~~ | ~~Medium~~ | ~~4~~ | **Eliminated** - Use VectorAccelerate |
| Ball Tree | 400 | Medium | 3 | Uses GPU for distance |
| GPU Batch k-NN | 100 | Low | 0.5 | Wrapper around `FusedL2TopKKernel` |
| HDBSCAN | 700 | High | 6 | Reduced - GPU handles distances |
| PCA | 150 | Low | 1 | Reduced - GPU matrix ops |
| UMAP | 700 | High | 5 | Optional, reduced - GPU helps |
| Tokenizer | 150 | Low | 1 | Unchanged |
| c-TF-IDF | 250 | Medium | 2 | May add GPU kernel later |
| NPMI | 300 | Medium | 3 | May add GPU kernel later |
| TopicModel | 500 | Medium | 3 | Unchanged |
| Apple Provider | 200 | Low | 2 | Unchanged |
| **Total** | **~3,500** | - | **~28** | **Reduced from ~39 days** |

### What VectorAccelerate Saved

| Eliminated Work | Estimated LOC | Estimated Days |
|-----------------|---------------|----------------|
| Distance metrics | 300 | 2 |
| Linear algebra utilities | 200 | 2 |
| Matrix operations | 150 | 1 |
| Batch operations | 150 | 1 |
| **Total Saved** | **~800** | **~6 days** |

*Note: Days are working days with focused effort. Part-time schedule roughly doubles the calendar time.*

---

*Roadmap Version: 1.1*
*Created: January 2025*
*Updated: January 2025 - Added VectorAccelerate integration, reduced scope*
