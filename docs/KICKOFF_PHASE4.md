# SwiftTopics Phase 4 Kickoff: PCA Dimensionality Reduction

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-3 are complete.

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

## Phase 4: PCA Dimensionality Reduction

**Duration**: 2-3 days | **LOC**: ~200

Implement Principal Component Analysis (PCA) for reducing high-dimensional embeddings to lower dimensions before clustering.

### Why PCA?

PCA is used as a preprocessing step before HDBSCAN because:
1. **Curse of Dimensionality**: Distance metrics become less meaningful in high dimensions
2. **Performance**: Reduces computational cost of k-NN and pairwise distances
3. **Noise Reduction**: Projects out low-variance (noisy) dimensions
4. **Visualization**: Can reduce to 2-3D for plotting

Typical usage: 384-dimensional embeddings → 15-50 dimensions → HDBSCAN

### Deliverables

#### 4.1 PCA Implementation
```
Sources/SwiftTopics/Reduction/PCA.swift
```
- Implement `DimensionReducer` protocol
- Center data (subtract mean)
- Compute covariance matrix using GPU (`MatrixMultiplyKernel`)
- Eigendecomposition via LAPACK (`ssyev` - already in Phase 1)
- Project onto top-k eigenvectors
- Store transformation for `transform()` on new data

#### 4.2 Configuration
```swift
public struct PCAConfiguration: ReductionConfiguration {
    /// Number of output dimensions
    public let components: Int  // default: 50

    /// Whether to whiten (scale by eigenvalues)
    public let whiten: Bool  // default: false

    /// Minimum explained variance ratio to keep a component
    public let minVarianceRatio: Float?  // optional

    /// Random seed for reproducibility
    public let seed: UInt64?
}
```

### Key APIs Already Available

```swift
// GPU covariance matrix (Phase 1)
let gpu = try await TopicsGPUContext()
let covariance = try await gpu.computeCovarianceMatrix(embeddings)

// Eigendecomposition (Phase 1)
let eigenResult = try EigendecompositionF32.decompose(
    matrix: covariance,
    computeEigenvectors: true
)
// eigenResult.eigenvalues are sorted ascending, eigenvectors in columns

// Matrix projection (Phase 1)
let projected = try await gpu.projectEmbeddings(
    centered,
    transformation: eigenvectors
)
```

### DimensionReducer Protocol (from Phase 0)

```swift
public protocol DimensionReducer: Sendable {
    associatedtype Configuration: ReductionConfiguration

    var configuration: Configuration { get }

    /// Fit the reducer to data and transform
    func fitTransform(_ embeddings: [Embedding]) async throws -> [Embedding]

    /// Transform new data using fitted model
    func transform(_ embeddings: [Embedding]) async throws -> [Embedding]
}
```

### Algorithm Steps

1. **Center the data**:
   ```
   X_centered = X - mean(X)
   ```

2. **Compute covariance matrix** (GPU accelerated):
   ```
   C = (X_centered)ᵀ × X_centered / (n - 1)
   ```

3. **Eigendecomposition**:
   ```
   C × V = V × Λ
   ```
   Where V are eigenvectors (principal components) and Λ are eigenvalues

4. **Select top-k components**:
   - Sort eigenvalues descending
   - Take eigenvectors corresponding to top-k eigenvalues

5. **Project data**:
   ```
   X_reduced = X_centered × V_k
   ```

6. **Optional whitening**:
   ```
   X_whitened = X_reduced / sqrt(λ)
   ```

### Exit Criteria
- [ ] Matches sklearn PCA output on test data (within tolerance)
- [ ] Explained variance ratio computed correctly
- [ ] Reconstruction error = ||X - X_reconstructed||² is minimized
- [ ] `transform()` works on new data after `fit()`
- [ ] Handles edge cases: n < d, all zeros, single dimension
- [ ] swift build passes
- [ ] swift test passes

### Constraints
- Swift 6 strict concurrency (all types must be Sendable)
- Use VectorAccelerate for GPU operations (don't reimplement)
- Use existing Eigendecomposition.swift from Phase 1
- Actor-based concurrency for thread safety
- Target: iOS/macOS/visionOS 26+

### Reference Materials
- SPEC.md Part 2.2 (Dimension Reduction Layer)
- `Utilities/Eigendecomposition.swift` for eigenvalue computation
- `Acceleration/GPUContext.swift` for GPU operations
- `Protocols/DimensionReducer.swift` for the protocol interface

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading:
1. `Protocols/DimensionReducer.swift` - understand the interface
2. `Utilities/Eigendecomposition.swift` - understand available eigendecomposition API
3. `Acceleration/GPUContext.swift` - understand GPU matrix operations

Then implement:
1. Create `Sources/SwiftTopics/Reduction/PCA.swift`
2. Add tests for PCA in the test file
3. Verify with `swift build && swift test`
