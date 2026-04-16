# VectorAccelerate Integration Handoff

## Overview

SwiftTopics needs to integrate new GPU-accelerated kernels from VectorAccelerate to eliminate CPU bottlenecks in the topic modeling training pipeline. VectorAccelerate has implemented all requested kernels per the specification in `/Users/goftin/dev/gsuite/VSK/future/VectorAccelerate/docs/SWIFTTOPICS_IMPLEMENTATION_PLAN.md`.

**Expected speedups:** 25-125x for HDBSCAN, 10-50x for UMAP optimization.

---

## Project Locations

- **SwiftTopics**: `/Users/goftin/dev/real/GournalV2/SwiftTopics/`
- **VectorAccelerate**: `/Users/goftin/dev/gsuite/VSK/future/VectorAccelerate/`

---

## Current Architecture (CPU-bound)

### HDBSCAN Pipeline
```
CoreDistanceComputer (GPU ✅) → MutualReachabilityGraph (CPU ❌) → PrimMSTBuilder (CPU ❌)
```

**Key files:**
- `Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift` — Main engine, `fitWithDetails()` at line 111
- `Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift` — CPU mutual reachability
- `Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift` — CPU Prim's algorithm
- `Sources/SwiftTopics/Acceleration/GPUContext.swift` — VectorAccelerate wrapper

### UMAP Pipeline
```
FuzzySimplicialSet → UMAPOptimizer.optimize() (CPU ❌, sequential edge processing)
```

**Key files:**
- `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift` — CPU gradient descent
- `Sources/SwiftTopics/Reduction/UMAP/FuzzySimplicialSet.swift` — Edge graph structure

---

## VectorAccelerate New Kernels (Validated ✅)

### HDBSCANDistanceModule
- **Location**: `Sources/VectorAccelerate/Modules/HDBSCANDistanceModule.swift`
- **API**: `computeMST(embeddings:minSamples:) async throws -> MSTResult`
- **Combines**: Core distances + MutualReachability + BoruvkaMST in one GPU pipeline
- **Output**: `MSTResult` with edges `[(source: Int, target: Int, weight: Float)]`

### UMAPGradientKernel
- **Location**: `Sources/VectorAccelerate/Kernels/Metal4/UMAPGradientKernel.swift`
- **API**: `optimizeEpoch(embedding:edges:...) async throws -> [[Float]]`
- **Features**: Segmented reduction (no atomic contention), negative sampling, target gradient accumulation

### Supporting Kernels
- `MutualReachabilityKernel` — Dense/sparse modes, dimension-optimized (384/512/768/1536)
- `BoruvkaMSTKernel` — GPU-parallel MST via Borůvka's algorithm, O(N) space

---

## Phase 1: HDBSCAN Integration (START HERE)

### Goal
Replace CPU MutualReachability + PrimMST with GPU HDBSCANDistanceModule.

### Tasks

#### 1.1 Add GPU MST method to TopicsGPUContext

**File**: `Sources/SwiftTopics/Acceleration/GPUContext.swift`

Add import and new method:

```swift
import VectorAccelerate  // Already present

// Add after line ~217 (after computeCoreDistances)

/// Computes HDBSCAN MST entirely on GPU using VectorAccelerate.
///
/// This fuses core distance computation, mutual reachability distance,
/// and MST construction into an efficient GPU pipeline using Borůvka's algorithm.
///
/// - Parameters:
///   - embeddings: The embeddings to cluster.
///   - minSamples: The k value for core distance (typically HDBSCAN's minSamples).
/// - Returns: Minimum spanning tree for cluster hierarchy construction.
public func computeHDBSCANMST(
    _ embeddings: [Embedding],
    minSamples: Int
) async throws -> MinimumSpanningTree {
    guard let context = metal4Context else {
        throw TopicsGPUError.gpuUnavailable
    }

    let vectors = embeddings.map { $0.vector }

    let module = try await HDBSCANDistanceModule(context: context)
    let result = try await module.computeMST(
        embeddings: vectors,
        minSamples: minSamples
    )

    // Convert VectorAccelerate MSTResult to SwiftTopics MinimumSpanningTree
    let edges = result.edges.map { edge in
        MSTEdge(
            source: edge.source,
            target: edge.target,
            weight: edge.weight
        )
    }

    return MinimumSpanningTree(edges: edges, pointCount: embeddings.count)
}
```

#### 1.2 Modify HDBSCANEngine to use GPU MST

**File**: `Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift`

Replace lines ~152-163 (Steps 2 & 3) with:

```swift
// Step 2 & 3: Build MST (GPU or CPU)
let mst: MinimumSpanningTree
if let gpu = gpuContext, embeddings.count >= 100 {
    // GPU path: Fused MutualReachability + Borůvka MST
    mst = try await gpu.computeHDBSCANMST(
        embeddings,
        minSamples: configuration.effectiveMinSamples
    )
} else {
    // CPU fallback for small datasets or no GPU
    let mrGraph = MutualReachabilityGraph(
        embeddings: embeddings,
        coreDistances: coreDistances
    )
    let mstBuilder = PrimMSTBuilder()
    mst = mstBuilder.build(from: mrGraph)
}
```

**Note**: The GPU path computes core distances internally, so for GPU we can skip the separate `computeCoreDistances()` call. Refactor to:

```swift
// Step 1-3: Core distances + MST (GPU path does both)
let coreDistances: [Float]
let mst: MinimumSpanningTree

if let gpu = gpuContext, embeddings.count >= 100 {
    // GPU path: All-in-one
    let result = try await gpu.computeHDBSCANMSTWithCoreDistances(
        embeddings,
        minSamples: configuration.effectiveMinSamples
    )
    coreDistances = result.coreDistances
    mst = result.mst
} else {
    // CPU path
    coreDistances = try await computeCoreDistances(embeddings)
    let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)
    mst = PrimMSTBuilder().build(from: mrGraph)
}
```

This requires adding a combined method to GPUContext that returns both.

#### 1.3 Add combined GPU method (optional optimization)

**File**: `Sources/SwiftTopics/Acceleration/GPUContext.swift`

```swift
/// Result of GPU HDBSCAN distance computation.
public struct HDBSCANDistanceResult: Sendable {
    public let coreDistances: [Float]
    public let mst: MinimumSpanningTree
}

/// Computes core distances and MST in one GPU pipeline.
public func computeHDBSCANMSTWithCoreDistances(
    _ embeddings: [Embedding],
    minSamples: Int
) async throws -> HDBSCANDistanceResult {
    guard let context = metal4Context else {
        throw TopicsGPUError.gpuUnavailable
    }

    let vectors = embeddings.map { $0.vector }

    let module = try await HDBSCANDistanceModule(context: context)
    let result = try await module.computeMST(
        embeddings: vectors,
        minSamples: minSamples
    )

    let edges = result.edges.map { edge in
        MSTEdge(source: edge.source, target: edge.target, weight: edge.weight)
    }
    let mst = MinimumSpanningTree(edges: edges, pointCount: embeddings.count)

    return HDBSCANDistanceResult(
        coreDistances: result.coreDistances,
        mst: mst
    )
}
```

#### 1.4 Testing

Create test file: `Tests/SwiftTopicsTests/Clustering/GPUHDBSCANTests.swift`

```swift
import XCTest
@testable import SwiftTopics

final class GPUHDBSCANTests: XCTestCase {

    func testGPUMSTMatchesCPU() async throws {
        // Generate test embeddings
        let embeddings = (0..<200).map { _ in
            Embedding(vector: (0..<384).map { _ in Float.random(in: -1...1) })
        }

        // CPU path
        let cpuEngine = try await HDBSCANEngine(configuration: .default, gpuContext: nil)
        let cpuResult = try await cpuEngine.fitWithDetails(embeddings)

        // GPU path
        let gpuContext = try await TopicsGPUContext()
        let gpuEngine = try await HDBSCANEngine(configuration: .default, gpuContext: gpuContext)
        let gpuResult = try await gpuEngine.fitWithDetails(embeddings)

        // Compare cluster assignments
        XCTAssertEqual(cpuResult.assignment.clusterCount, gpuResult.assignment.clusterCount)
        // Note: Exact label values may differ, but cluster structure should match
    }

    func testGPUMSTPerformance() async throws {
        let embeddings = (0..<1000).map { _ in
            Embedding(vector: (0..<384).map { _ in Float.random(in: -1...1) })
        }

        let gpuContext = try await TopicsGPUContext()

        let start = CFAbsoluteTimeGetCurrent()
        _ = try await gpuContext.computeHDBSCANMST(embeddings, minSamples: 5)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Should complete in < 0.5s for 1K points (vs ~2.5s CPU)
        XCTAssertLessThan(elapsed, 0.5, "GPU MST too slow: \(elapsed)s")
    }
}
```

---

## Phase 2: UMAP Integration

### Goal
Replace CPU gradient computation with GPU UMAPGradientKernel.

### Tasks

#### 2.1 Add UMAP kernel wrapper to TopicsGPUContext

**File**: `Sources/SwiftTopics/Acceleration/GPUContext.swift`

```swift
/// Cached UMAP gradient kernel.
private var umapGradientKernel: UMAPGradientKernel?

/// Gets or creates the UMAP gradient kernel.
public func getUMAPGradientKernel() async throws -> UMAPGradientKernel {
    if let kernel = umapGradientKernel {
        return kernel
    }

    guard let context = metal4Context else {
        throw TopicsGPUError.gpuUnavailable
    }

    let kernel = try await UMAPGradientKernel(context: context)
    umapGradientKernel = kernel
    return kernel
}

/// Runs one epoch of UMAP optimization on GPU.
public func optimizeUMAPEpoch(
    embedding: inout [[Float]],
    edges: [(source: Int, target: Int, weight: Float)],
    learningRate: Float,
    negativeSampleRate: Int,
    a: Float,
    b: Float
) async throws {
    let kernel = try await getUMAPGradientKernel()

    // Convert edges to UMAPEdge format
    let umapEdges = edges.map { edge in
        UMAPEdge(
            source: UInt32(edge.source),
            target: UInt32(edge.target),
            weight: edge.weight
        )
    }

    embedding = try await kernel.optimizeEpoch(
        embedding: embedding,
        edges: umapEdges,
        learningRate: learningRate,
        negativeSampleRate: negativeSampleRate,
        a: a,
        b: b
    )
}
```

#### 2.2 Add GPU optimization path to UMAPOptimizer

**File**: `Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift`

Add new method:

```swift
/// Optimizes using GPU acceleration.
///
/// This method uses VectorAccelerate's UMAPGradientKernel for parallel
/// gradient computation, providing 10-50x speedup over CPU.
///
/// - Parameters:
///   - fuzzySet: The high-dimensional fuzzy simplicial set.
///   - nEpochs: Number of optimization epochs.
///   - learningRate: Initial learning rate.
///   - negativeSampleRate: Number of negative samples per positive.
///   - gpuContext: GPU context for acceleration.
///   - progressHandler: Optional progress callback.
/// - Returns: Optimized embedding.
public func optimizeGPU(
    fuzzySet: FuzzySimplicialSet,
    nEpochs: Int,
    learningRate: Float = 1.0,
    negativeSampleRate: Int = 5,
    gpuContext: TopicsGPUContext,
    progressHandler: ((Float) -> Void)? = nil
) async throws -> [[Float]] {
    let edges = fuzzySet.toEdgeList()
    guard !edges.isEmpty else { return embedding }

    // Convert to tuple format for GPU
    let edgeTuples = edges.map { (source: $0.source, target: $0.target, weight: $0.weight) }

    for epoch in 0..<nEpochs {
        let alpha = learningRate * (1.0 - Float(epoch) / Float(nEpochs))

        try await gpuContext.optimizeUMAPEpoch(
            embedding: &embedding,
            edges: edgeTuples,
            learningRate: alpha,
            negativeSampleRate: negativeSampleRate,
            a: a,
            b: b
        )

        progressHandler?(Float(epoch + 1) / Float(nEpochs))
    }

    return embedding
}
```

#### 2.3 Update UMAP.swift to use GPU when available

**File**: `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift`

Modify the optimization call to prefer GPU:

```swift
// In the reduce() method, after creating the optimizer:
let result: [[Float]]
if let gpu = gpuContext {
    result = try await optimizer.optimizeGPU(
        fuzzySet: fuzzySet,
        nEpochs: configuration.nEpochs,
        learningRate: configuration.learningRate,
        negativeSampleRate: configuration.negativeSampleRate,
        gpuContext: gpu
    )
} else {
    result = await optimizer.optimize(
        fuzzySet: fuzzySet,
        nEpochs: configuration.nEpochs,
        learningRate: configuration.learningRate,
        negativeSampleRate: configuration.negativeSampleRate
    )
}
```

---

## Phase 3: Polish

### Tasks

1. **Automatic GPU/CPU selection**: Add threshold-based selection (GPU for n >= 100)
2. **Progress callbacks**: Wire GPU operation progress to TopicModelProgress
3. **Checkpoint support**: Update InterruptibleMSTBuilder and UMAPOptimizer.optimizeInterruptible for GPU paths
4. **Performance logging**: Add timing instrumentation
5. **Documentation**: Update README and docstrings

---

## Validation Checklist

- [ ] Phase 1: GPU HDBSCAN MST produces same cluster structure as CPU
- [ ] Phase 1: GPU HDBSCAN MST is 25x+ faster for 1K+ points
- [ ] Phase 2: GPU UMAP produces visually similar embeddings to CPU
- [ ] Phase 2: GPU UMAP is 10x+ faster for 1K+ points
- [ ] Phase 3: Graceful CPU fallback when GPU unavailable
- [ ] Phase 3: Progress reporting works with GPU paths

---

## Quick Reference: VectorAccelerate APIs

### HDBSCANDistanceModule
```swift
let module = try await HDBSCANDistanceModule(context: metal4Context)
let result = try await module.computeMST(embeddings: [[Float]], minSamples: Int)
// result.edges: [(source: Int, target: Int, weight: Float)]
// result.coreDistances: [Float]
```

### UMAPGradientKernel
```swift
let kernel = try await UMAPGradientKernel(context: metal4Context)
let newEmbedding = try await kernel.optimizeEpoch(
    embedding: [[Float]],
    edges: [UMAPEdge],
    learningRate: Float,
    negativeSampleRate: Int,
    a: Float,
    b: Float
)
```

### UMAPEdge Structure
```swift
struct UMAPEdge {
    let source: UInt32
    let target: UInt32
    let weight: Float
}
```

---

## Notes

- VectorAccelerate requires macOS 26.0+ / iOS 26.0+ (Metal 4)
- All kernels have dimension-optimized variants for 384, 512, 768, 1536
- BoruvkaMST uses O(N) space (no N² distance matrix materialization)
- UMAPGradientKernel uses segmented reduction to avoid atomic contention
