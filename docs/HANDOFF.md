# Handoff Prompt for New Session

## SwiftTopics Library Implementation

### Context

SwiftTopics is a new pure-Swift library for high-fidelity, on-device topic extraction. It implements a production-grade topic modeling pipeline inspired by BERTopic, optimized for Apple platforms with GPU acceleration via VectorAccelerate.

**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`

**Key Documents**:
- `SPEC.md` - Full architectural specification, algorithm details, integration patterns
- `ROADMAP.md` - Implementation phases, milestones, effort estimates

---

### Platform Requirements

| Platform | Minimum Version |
|----------|-----------------|
| **iOS** | 26.0+ |
| **macOS** | 26.0+ |
| **visionOS** | 26.0+ |

> This library targets cutting-edge platforms to leverage Metal 4 GPU acceleration via VectorAccelerate.

---

### Dependencies

| Package | Version | Repository |
|---------|---------|------------|
| **VectorAccelerate** | 0.4.4+ | `github.com/gifton/VectorAccelerate` |
| **VectorCore** | 0.1.6+ | `github.com/gifton/VectorCore` (transitive) |

VectorAccelerate provides GPU-accelerated kernels that eliminate the need to implement:
- Distance metrics (L2, cosine) → `L2DistanceKernel`, `CosineSimilarityKernel`
- Matrix operations → `MatrixMultiplyKernel`, `MatrixTransposeKernel`
- Top-K selection → `FusedL2TopKKernel`
- Statistics → `StatisticsKernel`

---

### Why This Library Exists

GournalCore (a journaling app backend) has a `TopicClusteringService` that extracts topics from journal entries. The current implementation uses:
- TF-weighted candidate extraction
- Label propagation clustering
- No automatic cluster count discovery
- No outlier detection

SwiftTopics will provide:
- **HDBSCAN clustering** - Auto-discovers topic count, handles outliers
- **UMAP/PCA reduction** - Better clustering in lower dimensions
- **c-TF-IDF representation** - More discriminative keywords
- **NPMI coherence** - Quantitative quality metrics
- **GPU acceleration** - VectorAccelerate for heavy math operations

---

### Architecture: GPU/CPU Hybrid

| Operation | Execution | Kernel/Method |
|-----------|-----------|---------------|
| Pairwise distances | GPU | `L2DistanceKernel` |
| Core distances (k-NN) | GPU | `FusedL2TopKKernel` |
| Covariance matrix | GPU | `MatrixMultiplyKernel` |
| PCA projection | GPU | `MatrixMultiplyKernel` |
| MST construction | CPU | Prim's algorithm |
| Cluster hierarchy | CPU | Sequential tree building |
| Cluster extraction | CPU | EOM tree traversal |

---

### Project Structure (Target)

```
SwiftTopics/
├── Package.swift
├── SPEC.md
├── ROADMAP.md
├── Sources/
│   └── SwiftTopics/
│       ├── Core/                 # Document, Embedding, Topic, etc.
│       ├── Acceleration/         # GPU context wrapper
│       ├── Clustering/           # HDBSCAN, SpatialIndex
│       ├── Reduction/            # PCA (uses GPU matrix ops)
│       ├── Representation/       # c-TF-IDF, Tokenizer
│       ├── Evaluation/           # NPMI coherence
│       ├── Model/                # TopicModel orchestrator
│       └── Utilities/            # Eigendecomposition, RandomState
└── Tests/
    └── SwiftTopicsTests/
```

---

### Implementation Order

Per ROADMAP.md, implement in this order:

#### Phase 0: Foundation (2-3 days)
1. Core types: `Document`, `Embedding`, `Topic`, `ClusterAssignment`, `TopicModelResult`
2. Protocols: `EmbeddingProvider`, `DimensionReducer`, `ClusteringEngine`, `TopicRepresenter`

#### Phase 1: GPU Integration (1-2 days) ← Reduced scope!
1. `TopicsGPUContext` - Wrapper around VectorAccelerate's `Metal4Context`
2. `Eigendecomposition.swift` - LAPACK wrapper (not in VectorAccelerate)
3. `RandomState.swift` - Seedable RNG

#### Phase 2: Spatial Indexing (3-4 days)
1. `SpatialIndex` protocol
2. `BallTree` implementation for k-NN queries
3. `GPUBatchKNN` wrapper around `FusedL2TopKKernel`

#### Phase 3: HDBSCAN (6-7 days) - **Most Complex**
1. Core distance computation (uses GPU k-NN)
2. Mutual reachability graph
3. Minimum spanning tree (Prim's algorithm)
4. Cluster hierarchy (dendrogram)
5. Cluster extraction (Excess of Mass method)

See SPEC.md Part 2.3 for detailed algorithm breakdown.

---

### Key Technical Decisions (from SPEC.md)

1. **Embedding-agnostic**: Library doesn't include embedding models. Consumers implement `EmbeddingProvider` protocol.

2. **GPU-first for math**: VectorAccelerate kernels for distance, matrix ops, top-K.

3. **CPU for graph algorithms**: MST, hierarchy, extraction are inherently sequential.

4. **iOS 26+ only**: Required for Metal 4 and VectorAccelerate compatibility.

5. **Actor-based concurrency**: `TopicModel` is an actor for thread safety.

---

### Performance Targets

| Operation | Target |
|-----------|--------|
| Single document transform | <50ms |
| 100 documents | <2s |
| 1000 documents | <30s |

---

### Begin Implementation

**Recommended start**: Phase 0 + Phase 1 in parallel

1. Read `SPEC.md` sections:
   - Part 1: Architectural Overview (especially 1.2 GPU/CPU Hybrid)
   - Part 2.1: Embedding Layer (protocol design)
   - Part 2.3: Clustering Layer (HDBSCAN algorithm)

2. Update `Package.swift` with platforms and VectorAccelerate dependency

3. Create directory structure per SPEC.md Part 5.1

4. Implement core types with `Sendable` and `Codable` conformance

5. Implement GPU context wrapper

---

### Reference Materials

**HDBSCAN**:
- Paper: Campello et al., "Density-Based Clustering Based on Hierarchical Density Estimates"
- Reference: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

**BERTopic Pipeline**:
- https://maartengr.github.io/BERTopic/algorithm/algorithm.html

**VectorAccelerate**:
- Source: https://github.com/gifton/VectorAccelerate

---

### Success Criteria for First Milestone (M1)

By end of Phase 2:
- [ ] All core types defined and documented
- [ ] GPU context initializes successfully
- [ ] Eigendecomposition works
- [ ] Ball tree can find k-nearest neighbors
- [ ] GPU batch k-NN matches Ball Tree results
- [ ] Unit tests pass for all components
- [ ] `swift build` succeeds

---

*Handoff Version: 1.1*
*Updated: January 2025 - Added VectorAccelerate integration*
