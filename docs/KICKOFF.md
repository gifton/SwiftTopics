# SwiftTopics Implementation Kickoff

## Copy-Paste Prompt for New Session

```
I'm starting implementation of SwiftTopics, a topic modeling library.

**Location**: /Users/goftin/dev/real/GournalV2/SwiftTopics

**Read these first**:
1. SPEC.md - Full architecture (especially Part 1.2 GPU/CPU Hybrid, Part 2)
2. ROADMAP.md - Implementation phases
3. HANDOFF.md - Quick context

**Current state**:
- Package.swift configured (iOS/macOS/visionOS 26+)
- VectorAccelerate 0.3.2 dependency resolved
- `swift build` passes
- Empty source files ready for implementation

**Start with Phase 0 + Phase 1 in parallel**:

Phase 0 - Core Types (~300 LOC):
- Sources/SwiftTopics/Core/Document.swift
- Sources/SwiftTopics/Core/Embedding.swift
- Sources/SwiftTopics/Core/Topic.swift
- Sources/SwiftTopics/Core/ClusterAssignment.swift
- Sources/SwiftTopics/Core/TopicModelResult.swift
- Sources/SwiftTopics/Protocols/EmbeddingProvider.swift
- Sources/SwiftTopics/Protocols/DimensionReducer.swift
- Sources/SwiftTopics/Protocols/ClusteringEngine.swift
- Sources/SwiftTopics/Protocols/TopicRepresenter.swift

Phase 1 - GPU Integration (~200 LOC):
- Sources/SwiftTopics/Acceleration/GPUContext.swift (wrap Metal4Context)
- Sources/SwiftTopics/Utilities/Eigendecomposition.swift (LAPACK ssyev)
- Sources/SwiftTopics/Utilities/RandomState.swift (seedable RNG)

**Key constraints**:
- All types must be Sendable and Codable
- Use VectorAccelerate for distance/matrix ops (don't reimplement)
- Actor-based concurrency for thread safety
- Swift 6 strict concurrency

Begin by creating the directory structure and implementing core types.
```

---

## Quick Reference

| Resource | Path |
|----------|------|
| Full Spec | `/Users/goftin/dev/real/GournalV2/SwiftTopics/SPEC.md` |
| Roadmap | `/Users/goftin/dev/real/GournalV2/SwiftTopics/ROADMAP.md` |
| Handoff | `/Users/goftin/dev/real/GournalV2/SwiftTopics/HANDOFF.md` |
| Package | `/Users/goftin/dev/real/GournalV2/SwiftTopics/Package.swift` |

## Dependencies Available

From VectorAccelerate (no need to implement):
- `L2DistanceKernel` - Euclidean distance
- `CosineSimilarityKernel` - Cosine similarity
- `FusedL2TopKKernel` - k-NN in one GPU pass
- `MatrixMultiplyKernel` - Matrix multiplication
- `StatisticsKernel` - Mean, variance
- `L2NormalizationKernel` - Vector normalization

## Success Criteria (M1)

- [ ] All core types compile with Sendable + Codable
- [ ] GPU context initializes
- [ ] Eigendecomposition works on test matrices
- [ ] `swift build` passes
- [ ] `swift test` passes
