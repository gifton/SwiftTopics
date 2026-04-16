# SwiftTopics Phase 3 Kickoff: HDBSCAN Clustering

## Session Context

You are continuing implementation of **SwiftTopics**, a topic modeling library for Apple platforms. Phases 0-2 are complete.

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
- `Utilities/Eigendecomposition.swift` - LAPACK ssyev/dsyev for PCA
- `Utilities/RandomState.swift` - xorshift128+ seedable RNG

### Phase 2: Spatial Indexing (~500 LOC)
- `Clustering/SpatialIndex/SpatialIndex.swift` - Protocol + DistanceMetric enum
- `Clustering/SpatialIndex/BallTree.swift` - CPU k-NN with branch pruning
- `Clustering/SpatialIndex/GPUBatchKNN.swift` - GPU batch k-NN via FusedL2TopKKernel

## Phase 3: HDBSCAN Clustering

**Duration**: 7-10 days | **LOC**: ~800

Implement the full HDBSCAN algorithm with cluster extraction.

### Deliverables

#### 3.1 Core Distance Computation
```
Sources/SwiftTopics/Clustering/HDBSCAN/CoreDistance.swift
```
- Compute core distance for each point (distance to k-th neighbor)
- Use GPUBatchKNN.computeCoreDistances(k:) for efficiency
- Handle edge case: k > number of points

#### 3.2 Mutual Reachability Graph
```
Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift
```
- Compute: `mutual_reach(a, b) = max(core_dist(a), core_dist(b), dist(a,b))`
- Sparse representation (don't materialize full n² matrix)
- Consider edge iterator pattern for MST construction

#### 3.3 Minimum Spanning Tree
```
Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift
```
- Prim's algorithm on mutual reachability graph
- Use priority queue (heap) for efficiency
- Output: sorted edges by weight for dendrogram construction

#### 3.4 Cluster Hierarchy (Dendrogram)
```
Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchy.swift
```
- Build single-linkage hierarchy from MST
- Track cluster births (when cluster forms) and deaths (when merges)
- Compute cluster stability: `stability = Σ (1/λ_death - 1/λ_birth) × |cluster|`

Data structure suggestion:
```swift
struct ClusterNode {
    let id: Int
    let parent: Int?
    let children: [Int]
    let birthLevel: Float  // Distance at which cluster formed
    let deathLevel: Float  // Distance at which cluster merged
    let size: Int
    var stability: Float
}
```

#### 3.5 Cluster Extraction
```
Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift
```
- **Excess of Mass (EOM)**: Select clusters maximizing total stability
- **Leaf clustering**: Alternative that takes all leaf clusters
- Label assignment: Map points to selected clusters
- Outlier detection: Points not in any stable cluster get label -1
- Membership probability computation

#### 3.6 HDBSCAN Orchestrator
```
Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift
```
- Implement `ClusteringEngine` protocol
- `HDBSCANConfiguration` with parameters:
  - `minClusterSize: Int` (default: 5)
  - `minSamples: Int?` (default: nil, uses minClusterSize)
  - `clusterSelectionEpsilon: Float` (default: 0.0)
  - `clusterSelectionMethod: ClusterSelectionMethod` (.eom or .leaf)
- `cluster(embeddings:) async throws -> ClusterAssignment`
- Deterministic with seed

### Key APIs Already Available

```swift
// GPU batch k-NN (use for core distances)
let gpuContext = try await TopicsGPUContext()
let knn = try GPUBatchKNN(context: gpuContext, embeddings: embeddings)
let coreDistances = try await knn.computeCoreDistances(k: minSamples)

// Or use the unified interface
let unified = try await UnifiedKNN(embeddings: embeddings, gpuContext: gpuContext)
let coreDistances = try await unified.computeCoreDistances(k: minSamples)

// Ball tree for single-point queries
let tree = try BallTree.build(points: points)
let neighbors = tree.query(point: query, k: 10)

// Random state for reproducibility
var rng = RandomState(seed: 42)
let shuffled = rng.shuffled(array)
```

### Existing ClusterAssignment Structure (from Phase 0)

```swift
public struct ClusterAssignment: Sendable, Codable {
    public let labels: [Int]           // -1 = outlier
    public let probabilities: [Float]  // Membership confidence
    public let outlierScores: [Float]  // How "outlier-like" each point is
    public let clusterCount: Int
    public let hierarchy: ClusterHierarchy?
}
```

### Exit Criteria
- [ ] Matches scikit-learn HDBSCAN output on test datasets (blobs, moons, circles)
- [ ] Correctly identifies outliers
- [ ] Handles edge cases: single point, all identical, very sparse
- [ ] Performance: 1000 points in <1s, 10000 points in <30s
- [ ] swift build passes
- [ ] swift test passes

### Constraints
- Swift 6 strict concurrency (all types must be Sendable)
- Use VectorAccelerate for GPU operations (don't reimplement)
- Actor-based concurrency for thread safety
- Target: iOS/macOS/visionOS 26+

### Reference Materials
- HDBSCAN Paper: Campello et al., "Density-Based Clustering Based on Hierarchical Density Estimates" (2013)
- scikit-learn implementation: https://github.com/scikit-learn-contrib/hdbscan
- Spec document: `/Users/goftin/dev/real/GournalV2/SwiftTopics/SPEC.md`
- Roadmap: `/Users/goftin/dev/real/GournalV2/SwiftTopics/ROADMAP.md`

---

## Quick Start Command

```bash
cd /Users/goftin/dev/real/GournalV2/SwiftTopics
swift build  # Verify current state compiles
```

Begin by reading the SPEC.md (Part 2.3 Clustering Layer) and existing ClusterAssignment.swift to understand the expected output structure, then implement the HDBSCAN components in order: CoreDistance → MutualReachability → MST → Hierarchy → Extraction → Orchestrator.
