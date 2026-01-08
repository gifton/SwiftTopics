# UMAP Acceleration Plan

**Status**: Implemented
**Target Improvement**: 87s → ~5s (17x speedup at 1000 points)
**Priority**: High
**Date**: 2026-01-07

## Executive Summary

UMAP's full pipeline shows only 1.2x GPU speedup despite the optimization phase achieving 12.8x. Root cause analysis identified two bottlenecks consuming 97% of GPU path time:

| Bottleneck | Current Time | Target Time | Savings |
|------------|--------------|-------------|---------|
| Spectral Initialization | 77.5s | <0.1s (random) / ~3s (PCA) | ~75s |
| k-NN Graph (BallTree) | 8.1s | ~0.3s (GPU k-NN) | ~8s |
| **Total Savings** | | | **~83s** |

---

## Phase 1: Initialization Strategy

### Goal
Add configurable initialization method to bypass O(n³) spectral eigendecomposition.

### Status: Not Started

### Work Items

- [ ] **1.1** Add `UMAPInitialization` enum to `UMAPConfiguration.swift`
  - [ ] Define cases: `.spectral`, `.pca`, `.random`
  - [ ] Add documentation explaining trade-offs
  - Estimated: ~15 LOC

- [ ] **1.2** Add `initialization` property to `UMAPConfiguration`
  - [ ] Add property with default value `.spectral` (backward compatible)
  - [ ] Update initializers
  - [ ] Update presets (`.default`, `.fast`, etc.)
  - Estimated: ~20 LOC

- [ ] **1.3** Update `UMAPReducer.fit()` to use initialization config
  - [ ] Add switch statement for initialization method
  - [ ] Wire up `SpectralEmbedding.pcaInitialization()` (already exists)
  - [ ] Wire up `SpectralEmbedding.randomInitialization()` (already exists)
  - [ ] Pass original embeddings for PCA path
  - Estimated: ~25 LOC

- [ ] **1.4** Update `UMAPBuilder` with initialization setter
  - [ ] Add `.initialization(_ init: UMAPInitialization)` method
  - Estimated: ~10 LOC

- [ ] **1.5** Add preset for fast GPU-optimized UMAP
  - [ ] Create `UMAPConfiguration.gpuOptimized` with `.pca` init
  - [ ] Create `UMAPConfiguration.gpuFast` with `.random` init
  - Estimated: ~15 LOC

### Files to Modify

| File | Changes |
|------|---------|
| `Sources/SwiftTopics/Model/UMAPConfiguration.swift` | Add enum, property |
| `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift` | Switch on init method |

### Validation

- [ ] Build passes: `swift build`
- [ ] Existing tests pass: `swift test --filter UMAP`
- [ ] New benchmark shows expected speedup

---

## Phase 2: GPU k-NN Integration

### Goal
Use existing `FusedL2TopKKernel` from VectorAccelerate instead of CPU BallTree for k-NN graph construction.

### Status: Not Started

### Work Items

- [ ] **2.1** Add `gpuContext` parameter to `NearestNeighborGraph.build()`
  - [ ] Make parameter optional with default `nil`
  - [ ] Maintain backward compatibility
  - Estimated: ~10 LOC

- [ ] **2.2** Implement GPU k-NN path in `NearestNeighborGraph`
  - [ ] Check if gpuContext provided and point count >= threshold
  - [ ] Call `gpuContext.computeBatchKNN()` for GPU path
  - [ ] Convert result format to match existing structure
  - [ ] Fall back to BallTree if GPU fails
  - Estimated: ~40 LOC

- [ ] **2.3** Update `UMAPReducer.fit()` to pass gpuContext to k-NN
  - [ ] Pass `self.gpuContext` to `NearestNeighborGraph.build()`
  - Estimated: ~5 LOC

- [ ] **2.4** Update `InterruptibleTrainingRunner` k-NN call (if applicable)
  - [ ] Check if this path also needs GPU k-NN
  - Estimated: ~5 LOC

### Files to Modify

| File | Changes |
|------|---------|
| `Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift` | Add GPU path |
| `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift` | Pass gpuContext |
| `Sources/SwiftTopics/Incremental/Training/InterruptibleTrainingRunner.swift` | Optional |

### Validation

- [ ] Build passes: `swift build`
- [ ] Existing tests pass: `swift test --filter UMAP`
- [ ] k-NN results match between CPU and GPU paths (within tolerance)
- [ ] Benchmark shows ~8s savings

---

## Phase 3: Benchmark Updates

### Goal
Update benchmark expectations and add new tests for accelerated paths.

### Status: Not Started

### Work Items

- [ ] **3.1** Update `UMAPBenchmarks.swift` speedup assertions
  - [ ] Adjust expectations based on new architecture
  - [ ] Add tests for different initialization strategies
  - Estimated: ~30 LOC

- [ ] **3.2** Add initialization strategy comparison benchmark
  - [ ] `testInitializationStrategyComparison()` - compare spectral/PCA/random
  - [ ] Measure quality metrics (not just speed)
  - Estimated: ~60 LOC

- [ ] **3.3** Add GPU k-NN benchmark
  - [ ] `testGPUvsCPUKNN()` - compare BallTree vs GPU k-NN
  - Estimated: ~40 LOC

### Files to Modify

| File | Changes |
|------|---------|
| `Tests/SwiftTopicsTests/Benchmarks/UMAPBenchmarks.swift` | Update assertions, add tests |

---

## Implementation Details

### UMAPInitialization Enum

```swift
/// Initialization strategy for UMAP embedding coordinates.
///
/// The initialization method significantly impacts both performance and quality:
/// - **spectral**: Best quality, O(n³) complexity - use for small datasets (<500 points)
/// - **pca**: Good quality, O(n×d²) complexity - recommended for GPU acceleration
/// - **random**: Acceptable quality, O(n) complexity - fastest, may need more epochs
public enum UMAPInitialization: String, Sendable, Codable, CaseIterable {

    /// Spectral embedding using graph Laplacian eigenvectors.
    ///
    /// Provides the best initialization quality by preserving global graph structure.
    /// However, requires full eigendecomposition which is O(n³).
    ///
    /// **Recommended for**: Datasets with <500 points where quality is critical.
    case spectral

    /// PCA projection of the original embeddings.
    ///
    /// Fast initialization that preserves the main directions of variance.
    /// Good balance of quality and speed.
    ///
    /// **Recommended for**: Most use cases with GPU acceleration.
    case pca

    /// Random uniform initialization with appropriate scaling.
    ///
    /// Fastest initialization but may require more optimization epochs.
    /// The optimizer will converge to similar quality given enough epochs.
    ///
    /// **Recommended for**: Very large datasets (>5000 points) or when speed is critical.
    case random
}
```

### GPU k-NN Integration Pattern

```swift
// In NearestNeighborGraph.swift
public static func build(
    embeddings: [Embedding],
    k: Int,
    metric: DistanceMetric = .euclidean,
    gpuContext: TopicsGPUContext? = nil  // NEW PARAMETER
) async throws -> NearestNeighborGraph {

    let n = embeddings.count
    let gpuThreshold = gpuContext?.configuration.gpuMinPointsThreshold ?? 100

    // Use GPU k-NN if available and beneficial
    if let gpu = gpuContext, n >= gpuThreshold, metric == .euclidean {
        do {
            return try await buildWithGPU(embeddings: embeddings, k: k, gpuContext: gpu)
        } catch {
            // Log warning and fall back to BallTree
            // ...
        }
    }

    // Existing BallTree path
    // ...
}

private static func buildWithGPU(
    embeddings: [Embedding],
    k: Int,
    gpuContext: TopicsGPUContext
) async throws -> NearestNeighborGraph {
    // Use existing computeBatchKNN which wraps FusedL2TopKKernel
    let knnResult = try await gpuContext.computeBatchKNN(embeddings, k: k + 1)

    // Convert to NearestNeighborGraph format (exclude self-neighbors)
    var allNeighbors = [[Int]](repeating: [], count: embeddings.count)
    var allDistances = [[Float]](repeating: [], count: embeddings.count)

    for i in 0..<embeddings.count {
        let filtered = knnResult[i].filter { $0.index != i }.prefix(k)
        allNeighbors[i] = filtered.map { $0.index }
        allDistances[i] = filtered.map { $0.distance }
    }

    return NearestNeighborGraph(
        neighbors: allNeighbors,
        distances: allDistances,
        k: k,
        metric: .euclidean
    )
}
```

---

## Expected Results

### Performance Targets (1000 points)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| k-NN Graph | 8.11s | ~0.3s | **27x** |
| Spectral Init | 77.5s | ~0.01s (random) | **7750x** |
| Optimization | 1.53s | 1.53s | 1x (already GPU) |
| **Total** | **87.2s** | **~2-5s** | **17-40x** |

### Quality Expectations

| Init Method | Trustworthiness | Continuity | Notes |
|-------------|-----------------|------------|-------|
| Spectral | ~0.98 | ~0.97 | Best quality |
| PCA | ~0.96 | ~0.95 | Near-spectral quality |
| Random | ~0.94 | ~0.93 | May need 1.5x epochs |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU k-NN results differ from BallTree | Medium | Low | Tolerance-based comparison in tests |
| Random init degrades clustering quality | Low | Medium | Recommend PCA as default for GPU |
| API breaking change | Low | High | All new params are optional with defaults |

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Init Strategy | 2-3 hours | None |
| Phase 2: GPU k-NN | 2-3 hours | Phase 1 (for testing) |
| Phase 3: Benchmarks | 1-2 hours | Phases 1 & 2 |
| **Total** | **5-8 hours** | |

---

## References

- `Sources/SwiftTopics/Reduction/UMAP/SpectralEmbedding.swift` - Existing init methods
- `Sources/SwiftTopics/Acceleration/GPUContext.swift:173-198` - GPU k-NN API
- `Tests/SwiftTopicsTests/Benchmarks/UMAPBenchmarks.swift` - Performance baselines
- VectorAccelerate `FusedL2TopKKernel` - GPU k-NN implementation

---

## Appendix: Benchmark Evidence

### Phase Breakdown (1000 points, from testUMAPPhaseBreakdown)

```
CPU Path (no GPU context):
  k-NN Graph:       8.11s
  Fuzzy Set:        85.6ms
  Spectral Init:    77.54s
  Optimization:     19.61s
  Total:            105.35s

GPU Path:
  k-NN Graph:       8.11s    <- NOT using GPU k-NN!
  Fuzzy Set:        87.8ms
  Spectral Init:    77.49s   <- NOT accelerated!
  Optimization:     1.53s    <- 12.8x speedup
  Total:            87.22s

Per-phase speedups:
  k-NN Graph       1.0x  (both CPU)
  Spectral Init    1.0x  (both CPU)
  Optimization     12.8x (GPU working!)
  Total            1.2x  (bottlenecked)
```

### Amdahl's Law Calculation

With spectral init at 74% and optimization at 18% of CPU time:
```
Current: 1 / (0.74 + 0.08 + 0.18/12.8) = 1.2x

With random init + GPU k-NN:
  - k-NN: 8.1s → 0.3s (27x)
  - Init: 77.5s → 0.01s (7750x)
  - Optim: stays at 1.5s

New total: 0.3 + 0.01 + 0.1 + 1.5 ≈ 2s
Improvement: 87s / 2s ≈ 43x (theoretical max)
Conservative estimate: 17x (accounting for overhead)
```
