# SwiftTopics Phase 8 Kickoff: UMAP Dimensionality Reduction

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-7 are complete.

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
- `Evaluation/CooccurrenceCounter.swift` - Word pair co-occurrence counting
- `Evaluation/NPMIScorer.swift` - Normalized Pointwise Mutual Information scoring
- `Evaluation/CoherenceEvaluator.swift` - Topic coherence evaluation
- `Evaluation/DiversityMetrics.swift` - Topic diversity and redundancy metrics

### Phase 7: TopicModel Orchestrator (~600 LOC)
- `Model/TopicModelConfiguration.swift` - Configuration with presets (.default, .fast, .quality)
- `Model/TopicModel.swift` - Main actor orchestrating the pipeline
- `Model/TopicModelProgress.swift` - Progress reporting with stages
- `Model/TopicModelState.swift` - Serialization for save/load

## Phase 8: UMAP Dimensionality Reduction

**Duration**: 5-7 days | **LOC**: ~800

Implement UMAP (Uniform Manifold Approximation and Projection) for higher-quality dimensionality reduction that preserves local structure better than PCA.

### What is UMAP?

UMAP is a non-linear dimensionality reduction technique based on manifold learning and topological data analysis. Key advantages over PCA:

1. **Preserves local structure**: Points that are close in high-dim stay close in low-dim
2. **Handles non-linear manifolds**: Can "unroll" complex structures
3. **Better cluster separation**: Produces more distinct clusters for HDBSCAN

### Algorithm Overview

```
High-dimensional embeddings (e.g., 768-dim)
                ↓
[1] k-NN Graph Construction
    - Find k nearest neighbors for each point
    - Store distances to neighbors
                ↓
[2] Fuzzy Simplicial Set
    - Convert distances to membership strengths
    - Compute ρ (distance to nearest) and σ (bandwidth)
    - Symmetrize: membership(a,b) = membership(a→b) + membership(b→a) - membership(a→b)×membership(b→a)
                ↓
[3] Spectral Embedding Initialization
    - Compute graph Laplacian
    - Find eigenvectors for initial positions
    - Fallback to random if spectral fails
                ↓
[4] SGD Optimization
    - For each epoch:
      - Attractive forces: Pull connected points together
      - Repulsive forces: Push non-neighbors apart (negative sampling)
    - Learning rate decay
                ↓
Low-dimensional embeddings (e.g., 15-dim)
```

### Deliverables

#### 8.1 k-NN Graph Construction
```
Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift
```
- Use existing `BallTree` or `GPUBatchKNN` for k-NN queries
- Store sparse graph: for each point, k neighbors + distances
- Handle disconnected components (add edges if needed)

```swift
public struct NearestNeighborGraph: Sendable {
    /// Neighbor indices for each point [n × k]
    public let indices: [[Int]]

    /// Distances to neighbors [n × k]
    public let distances: [[Float]]

    /// Number of points
    public let pointCount: Int

    /// Number of neighbors per point
    public let k: Int

    /// Builds k-NN graph from embeddings
    public static func build(
        embeddings: [Embedding],
        k: Int,
        metric: DistanceMetricType
    ) async throws -> NearestNeighborGraph
}
```

#### 8.2 Fuzzy Simplicial Set
```
Sources/SwiftTopics/Reduction/UMAP/FuzzySimplicialSet.swift
```
Converts distances to fuzzy membership strengths.

**Key formulas:**
```
ρ_i = distance to nearest neighbor of point i
σ_i = bandwidth such that Σ_j exp(-(d_ij - ρ_i) / σ_i) = log2(k)

membership(i → j) = exp(-(d_ij - ρ_i) / σ_i)  if d_ij > ρ_i
                  = 1.0                         otherwise

# Symmetrization (fuzzy union)
membership(i, j) = μ(i→j) + μ(j→i) - μ(i→j) × μ(j→i)
```

```swift
public struct FuzzySimplicialSet: Sendable {
    /// Sparse matrix of membership strengths
    public let memberships: SparseMatrix<Float>

    /// Per-point ρ values (distance to nearest)
    public let rho: [Float]

    /// Per-point σ values (bandwidth)
    public let sigma: [Float]

    /// Builds fuzzy set from k-NN graph
    public static func build(
        from graph: NearestNeighborGraph,
        localConnectivity: Float = 1.0
    ) -> FuzzySimplicialSet
}
```

#### 8.3 Spectral Embedding Initialization
```
Sources/SwiftTopics/Reduction/UMAP/SpectralEmbedding.swift
```
Provides good initial positions for optimization.

**Algorithm:**
1. Build normalized graph Laplacian: L = D - W (D = degree matrix, W = adjacency)
2. Compute smallest eigenvectors of L (excluding constant eigenvector)
3. Use first `nComponents` eigenvectors as initial coordinates

```swift
public struct SpectralEmbedding: Sendable {
    /// Computes spectral embedding for initialization
    /// - Returns: Initial coordinates [n × nComponents]
    public static func compute(
        adjacency: SparseMatrix<Float>,
        nComponents: Int
    ) throws -> [[Float]]

    /// Random initialization fallback
    public static func randomInitialization(
        pointCount: Int,
        nComponents: Int,
        seed: UInt64?
    ) -> [[Float]]
}
```

#### 8.4 UMAP Optimizer (SGD)
```
Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift
```
Optimizes low-dim positions using stochastic gradient descent.

**Forces:**
- **Attractive**: For edge (i,j) with membership w_ij:
  ```
  grad = -2 × a × b × w_ij × (d_ij^(2(b-1))) / (1 + a × d_ij^(2b)) × (y_i - y_j)
  ```
- **Repulsive** (negative sampling): For non-neighbors:
  ```
  grad = 2 × b / ((0.001 + d_ij^2) × (1 + a × d_ij^(2b))) × (y_i - y_j)
  ```

Where a ≈ 1.929 and b ≈ 0.7915 for minDist = 0.1 (derived from curve fitting).

```swift
public actor UMAPOptimizer {
    /// UMAP curve parameters (depend on minDist)
    private let a: Float
    private let b: Float

    /// Optimizes embedding positions
    public func optimize(
        initialEmbedding: [[Float]],
        fuzzySet: FuzzySimplicialSet,
        nEpochs: Int,
        learningRate: Float,
        negativeSampleRate: Int,
        seed: UInt64?
    ) async -> [[Float]]
}
```

**Optimization details:**
- Epochs: ~200-500 (auto-determined based on dataset size)
- Learning rate: Start at 1.0, decay to 0.0001
- Negative samples: 5 per positive sample
- Clipping: Limit gradient magnitude to prevent instability

#### 8.5 UMAP Orchestrator
```
Sources/SwiftTopics/Reduction/UMAP/UMAP.swift
```
Main entry point implementing `DimensionReducer` protocol.

```swift
public struct UMAPConfiguration: Sendable, Codable {
    /// Number of neighbors for manifold approximation.
    /// Higher = more global structure, lower = more local structure.
    public let nNeighbors: Int  // default: 15

    /// Minimum distance between points in low-dim space.
    /// Lower = tighter clusters, higher = more spread out.
    public let minDist: Float  // default: 0.1

    /// Output dimensionality.
    public let nComponents: Int  // default: 15

    /// Distance metric for high-dim space.
    public let metric: DistanceMetricType  // default: .euclidean

    /// Number of optimization epochs (nil = auto).
    public let nEpochs: Int?  // default: nil

    /// Learning rate for optimization.
    public let learningRate: Float  // default: 1.0

    /// Negative samples per positive sample.
    public let negativeSampleRate: Int  // default: 5

    /// Random seed for reproducibility.
    public let seed: UInt64?

    public static let `default` = UMAPConfiguration()
    public static let fast = UMAPConfiguration(nNeighbors: 10, nEpochs: 100)
    public static let quality = UMAPConfiguration(nNeighbors: 30, minDist: 0.05, nEpochs: 500)
}

public struct UMAPReducer: DimensionReducer {
    public let configuration: UMAPConfiguration

    public var outputDimension: Int { configuration.nComponents }
    public var isFitted: Bool { fittedGraph != nil }

    /// Fits and transforms in one pass
    public func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding]

    /// Transforms new data (approximate - uses nearest neighbor in training set)
    public func transform(_ embeddings: [Embedding]) async throws -> [Embedding]

    /// Fits without returning transformed data
    public mutating func fit(_ embeddings: [Embedding]) async throws
}
```

#### 8.6 Sparse Matrix Utility
```
Sources/SwiftTopics/Utilities/SparseMatrix.swift
```
Efficient sparse matrix for fuzzy set storage.

```swift
public struct SparseMatrix<T: Numeric & Sendable>: Sendable {
    /// CSR format: row pointers
    public let rowPointers: [Int]

    /// CSR format: column indices
    public let columnIndices: [Int]

    /// CSR format: values
    public let values: [T]

    /// Matrix dimensions
    public let rows: Int
    public let cols: Int

    /// Access element
    public subscript(row: Int, col: Int) -> T { get }

    /// Iterate non-zero elements in row
    public func nonZeroElements(inRow row: Int) -> [(col: Int, value: T)]

    /// Build from COO format (coordinate list)
    public static func fromCOO(
        rows: Int,
        cols: Int,
        entries: [(row: Int, col: Int, value: T)]
    ) -> SparseMatrix<T>
}
```

### Integration with TopicModel

Update `ReductionConfiguration` and `TopicModel` to support UMAP:

```swift
// In DimensionReducer.swift - already exists
public enum ReductionMethod: String, Sendable, Codable {
    case pca
    case umap  // Add this
    case none
}

// In TopicModel.swift - update reduceEmbeddings()
private func reduceEmbeddings(_ embeddings: [Embedding]) async throws -> ... {
    switch configuration.reduction.method {
    case .pca:
        // existing PCA code
    case .umap:
        let umapConfig = configuration.reduction.umapConfig ?? .default
        var umap = UMAPReducer(configuration: umapConfig)
        return try await umap.fitTransform(embeddings)
    case .none:
        return (embeddings, nil)
    }
}
```

### Mathematical Background

#### Fuzzy Set Theory
UMAP represents neighborhood relationships as fuzzy sets where membership is continuous [0,1] rather than binary. This allows smooth transitions between "neighbor" and "not neighbor".

#### Riemannian Geometry
The algorithm assumes data lies on a Riemannian manifold with locally varying metric. The ρ and σ parameters capture this local metric.

#### Cross-Entropy Optimization
The optimization minimizes cross-entropy between:
- High-dim fuzzy set (what we want to preserve)
- Low-dim fuzzy set (what we're learning)

### Performance Considerations

1. **k-NN is the bottleneck**: Use GPU batch k-NN for large datasets
2. **Sparse operations**: Fuzzy set is sparse (~k entries per row out of n)
3. **Parallel SGD**: Each point's update is independent within an epoch
4. **Memory**: Store only sparse fuzzy set, not full n×n matrix

### Exit Criteria

- [ ] Preserves local structure (k-NN preservation metric > 0.8)
- [ ] Produces visually separable clusters
- [ ] Matches umap-learn output qualitatively on test datasets
- [ ] Performance: 1000 points in <5s, 10000 points in <60s
- [ ] `swift build` passes
- [ ] `swift test` passes

### Test Cases

1. **Swiss Roll**: Classic manifold - should unroll
2. **Concentric Circles**: Should separate into rings
3. **Blobs**: Should maintain cluster separation
4. **High-dimensional**: 768-dim sentence embeddings → 15-dim

### Reference Materials

- [UMAP Paper](https://arxiv.org/abs/1802.03426) - McInnes, Healy, Melville
- [umap-learn source](https://github.com/lmcinnes/umap) - Reference implementation
- `Reduction/PCA.swift` - Existing DimensionReducer implementation
- `Clustering/SpatialIndex/BallTree.swift` - For k-NN queries
- `Protocols/DimensionReducer.swift` - Protocol to implement

### Constraints

- Swift 6 strict concurrency (all types must be Sendable)
- Target: iOS/macOS/visionOS 26+
- Use existing spatial index infrastructure
- Implement `DimensionReducer` protocol

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading:
1. `Protocols/DimensionReducer.swift` - Protocol to implement
2. `Reduction/PCA.swift` - Example DimensionReducer implementation
3. `Clustering/SpatialIndex/BallTree.swift` - k-NN infrastructure

Then implement in order:
1. `Sources/SwiftTopics/Utilities/SparseMatrix.swift` - Utility first
2. `Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift`
3. `Sources/SwiftTopics/Reduction/UMAP/FuzzySimplicialSet.swift`
4. `Sources/SwiftTopics/Reduction/UMAP/SpectralEmbedding.swift`
5. `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift`
6. `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift`
7. Update `TopicModel.swift` to support UMAP
8. Add tests
9. Verify with `swift build && swift test`
