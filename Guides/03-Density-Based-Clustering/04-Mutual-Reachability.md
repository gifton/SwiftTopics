# 3.4 Mutual Reachability

> **The density-aware distance that makes HDBSCAN work.**

---

## The Concept

**Mutual reachability distance** is HDBSCAN's key innovation. It transforms raw distances to be **density-aware**.

The insight: two points in a sparse region should be considered "farther apart" than two points the same Euclidean distance apart in a dense region.

```
Euclidean distance (raw):

    Dense region:        Sparse region:
    A ●───── B           C ●───── D
      (0.5)                (0.5)

    Same distance! But...

Mutual reachability distance:

    Dense region:        Sparse region:
    A ●───── B           C ●───── D
      (0.5)                (2.0)

    Different! Sparse points are "farther apart"
    because there's no density "path" between them.
```

---

## Why It Matters

Mutual reachability solves DBSCAN's variable-density problem:

```
Without mutual reachability (raw distances):

    Dense cluster ●●●●●●●●●   and   Sparse cluster ●    ●    ●
    (internal distances ~0.1)       (internal distances ~1.0)

    Building MST: Dense cluster connects tightly,
    then reaches out to sparse points at distance ~5.0.
    Sparse cluster forms LAST, at very low density.

With mutual reachability:

    Dense cluster: distances stay ~0.1
    (Core distances are small, max is still the raw distance)

    Sparse cluster: distances ALSO ~0.3
    (Core distances are ~0.3, which becomes the mutual reachability)

    Both clusters form at similar "effective" density levels!
```

---

## The Mathematics

### Core Distance

First, compute the **core distance** for each point:

```
core_distance(p) = distance to k-th nearest neighbor of p

Where k = minSamples (typically 5-15)

Example with k = 3:

    Point A's neighbors (sorted by distance):
      B: 0.1
      C: 0.15
      D: 0.2   ← 3rd nearest
      E: 0.5
      F: 0.8

    core_distance(A) = 0.2

Intuition:
  - Small core distance = point is in a dense region
  - Large core distance = point is in a sparse region
```

### Mutual Reachability Formula

```
mutual_reachability(a, b) = max(core_dist(a), core_dist(b), dist(a, b))

Three-way maximum:
  1. Core distance of point a
  2. Core distance of point b
  3. Raw Euclidean distance between a and b
```

### Why This Works

```
Case 1: Both points in dense region
─────────────────────────────────────
  core_dist(a) = 0.1 (dense)
  core_dist(b) = 0.1 (dense)
  dist(a,b) = 0.5

  mutual_reach(a,b) = max(0.1, 0.1, 0.5) = 0.5

  → Raw distance dominates. Dense-to-dense stays normal.

Case 2: Both points in sparse region
─────────────────────────────────────
  core_dist(a) = 2.0 (sparse)
  core_dist(b) = 1.5 (sparse)
  dist(a,b) = 0.5

  mutual_reach(a,b) = max(2.0, 1.5, 0.5) = 2.0

  → Core distance dominates. Even nearby sparse points
    are "far apart" in mutual reachability space.

Case 3: Dense point to sparse point
─────────────────────────────────────
  core_dist(a) = 0.1 (dense)
  core_dist(b) = 2.0 (sparse)
  dist(a,b) = 0.3

  mutual_reach(a,b) = max(0.1, 2.0, 0.3) = 2.0

  → Sparse point's core distance dominates.
    Dense regions don't "absorb" sparse points.
```

---

## The Technique: Building the Mutual Reachability Graph

### Step 1: Compute All Core Distances

```
For each point, find the k-th nearest neighbor distance:

Points: A, B, C, D, E (k = 3)

Point   3rd Nearest Neighbor   Core Distance
────────────────────────────────────────────
  A     D at distance 0.3         0.3
  B     A at distance 0.2         0.2
  C     E at distance 0.4         0.4
  D     A at distance 0.3         0.3
  E     C at distance 0.4         0.4
```

### Step 2: Transform All Pairwise Distances

```
Original distance matrix:
        A     B     C     D     E
    A   0    0.15  0.5   0.3   0.6
    B  0.15   0    0.4   0.25  0.55
    C  0.5   0.4    0    0.35  0.3
    D  0.3   0.25  0.35   0    0.45
    E  0.6   0.55  0.3   0.45   0

Core distances: A=0.3, B=0.2, C=0.4, D=0.3, E=0.4

Mutual reachability matrix:
        A     B     C     D     E
    A   0    0.3   0.5   0.3   0.6
    B  0.3    0    0.4   0.3   0.55
    C  0.5   0.4    0    0.4   0.4
    D  0.3   0.3   0.4    0    0.45
    E  0.6   0.55  0.4   0.45   0

Calculations:
  mutual(A,B) = max(0.3, 0.2, 0.15) = 0.3  (was 0.15)
  mutual(A,C) = max(0.3, 0.4, 0.5) = 0.5   (unchanged)
  mutual(C,E) = max(0.4, 0.4, 0.3) = 0.4   (was 0.3)
```

### Step 3: Build MST on Mutual Reachability

```
Using Prim's algorithm on the mutual reachability graph:

1. Start at A
2. Add cheapest edge from A: A──B (0.3)
3. Add cheapest edge from {A,B}: A──D (0.3) or B──D (0.3)
4. Continue until all points connected

Resulting MST (edges sorted):
  A──B: 0.3
  A──D: 0.3  (or B──D)
  C──E: 0.4
  C──D: 0.4  (or D──E, connects the two groups)

This MST forms the basis for the cluster hierarchy.
```

---

## In SwiftTopics

### Core Distance Computation

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/CoreDistance.swift

/// Computes core distances for HDBSCAN clustering.
///
/// The core distance of a point is the distance to its k-th nearest neighbor,
/// which serves as a measure of local density.
public struct CoreDistanceComputer: Sendable {

    /// The number of neighbors to consider (k value).
    public let minSamples: Int

    /// Computes core distances for all points.
    public func compute(
        embeddings: [Embedding],
        gpuContext: TopicsGPUContext?
    ) async throws -> [Float] {
        let k = min(minSamples, embeddings.count - 1)

        // GPU path (preferred for n > 100)
        if let context = gpuContext {
            let gpuKNN = try GPUBatchKNN(context: context, dataset: points)
            return try await gpuKNN.computeCoreDistances(k: k)
        }

        // CPU fallback: O(n² log k) using heap-based selection
        return computeWithCPU(embeddings: embeddings, k: k)
    }
}
```

### GPU-Accelerated k-NN

```swift
// 📍 See: Sources/SwiftTopics/Clustering/SpatialIndex/GPUBatchKNN.swift

// Core distances require k-nearest-neighbor queries for all points.
// For n points, this is O(n²) distance computations.
// GPU acceleration makes this practical for large datasets.

// The GPU kernel (from VectorAccelerate):
// 1. Computes all pairwise L2 distances in parallel
// 2. Uses parallel top-k selection to find k nearest
// 3. Returns the k-th distance (core distance)
```

### Mutual Reachability Graph

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift

/// A sparse representation of the mutual reachability graph.
public struct MutualReachabilityGraph: Sendable {

    /// Core distances for all points.
    public let coreDistances: [Float]

    /// The original embeddings (for distance computation).
    private let embeddings: [Embedding]

    /// Computes the mutual reachability distance between two points.
    public func distance(from i: Int, to j: Int) -> Float {
        guard i != j else { return 0 }

        let euclidean = embeddings[i].euclideanDistance(embeddings[j])
        let coreI = coreDistances[i]
        let coreJ = coreDistances[j]

        return max(coreI, coreJ, euclidean)  // The key formula!
    }
}
```

### MST Construction

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift

/// Builds a minimum spanning tree using Prim's algorithm.
public struct PrimMSTBuilder: Sendable {

    /// Builds the MST from a mutual reachability graph.
    public func build(from graph: MutualReachabilityGraph) -> MinimumSpanningTree {
        let n = graph.count

        var mstEdges = [MSTEdge]()
        var inMST = [Bool](repeating: false, count: n)
        var heap = MinHeap<PrimHeapEntry>()

        // Start from point 0
        inMST[0] = true
        addEdgesToHeap(from: 0, graph: graph, inMST: inMST, heap: &heap)

        // Grow MST until all points included
        while let entry = heap.removeMin() {
            guard !inMST[entry.target] else { continue }

            mstEdges.append(MSTEdge(
                source: entry.source,
                target: entry.target,
                weight: entry.weight  // This is mutual reachability distance
            ))

            inMST[entry.target] = true
            addEdgesToHeap(from: entry.target, graph: graph, inMST: inMST, heap: &heap)
        }

        return MinimumSpanningTree(edges: mstEdges, pointCount: n)
    }
}
```

---

## Visualizing the Effect

```
Raw distance space:

    Dense cluster           Gap           Sparse cluster
    ●●●●●●●●●●                            ●     ●     ●
    ●●●●●●●●●●●                              ●
    ●●●●●●●●●●                            ●     ●     ●

    Dense internal: 0.1              Sparse internal: 0.5
    Dense to sparse: 3.0

MST on raw distances:
    Forms tight tree in dense region first,
    then long edges reach out to sparse region.
    Sparse cluster forms at very low λ (late).

Mutual reachability space:

    Dense cluster                         Sparse cluster
    (core dist ~0.1)                      (core dist ~0.5)

    Dense internal: max(0.1, 0.1, 0.1) = 0.1  (unchanged)
    Sparse internal: max(0.5, 0.5, 0.3) = 0.5 (inflated!)
    Dense to sparse: max(0.1, 0.5, 3.0) = 3.0 (unchanged)

MST on mutual reachability:
    Dense cluster: forms at λ = 10 (1/0.1)
    Sparse cluster: forms at λ = 2 (1/0.5)
    Connection: happens at λ = 0.33 (1/3.0)

    Both clusters are visible and stable!
```

---

## The minSamples Parameter

The `k` in core distance (called `minSamples` in HDBSCAN) is crucial:

```
Small k (e.g., 3):
─────────────────────────────────────────────────
  - Core distances are small (just 3rd neighbor)
  - More sensitive to local structure
  - May create many tiny clusters
  - Good for: Fine-grained topics

Large k (e.g., 15):
─────────────────────────────────────────────────
  - Core distances are larger (15th neighbor)
  - Smooths over local variations
  - Creates fewer, broader clusters
  - Good for: Coarse topics, noisy data

Default in SwiftTopics: minSamples = minClusterSize
─────────────────────────────────────────────────
  - If minClusterSize = 5, minSamples defaults to 5
  - This creates a natural relationship:
    "A cluster needs 5 points, and a core point needs 5 neighbors"
```

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

public struct HDBSCANConfiguration {
    /// Minimum cluster size.
    public let minClusterSize: Int  // Default: 5

    /// Minimum samples for core distance.
    /// If nil, defaults to minClusterSize.
    public let minSamples: Int?

    /// Effective minSamples value.
    public var effectiveMinSamples: Int {
        minSamples ?? minClusterSize
    }
}
```

---

## Memory and Performance

### Naive Approach: O(n²) Space

```
Full mutual reachability matrix:
  n = 10,000 points
  Matrix: 10,000 × 10,000 = 100,000,000 entries
  Memory: 400 MB (Float32)

For n = 100,000:
  Matrix: 10,000,000,000 entries
  Memory: 40 GB ❌
```

### SwiftTopics Approach: O(n) Space

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift

/// Instead of storing the full n×n distance matrix, this stores only the
/// edges needed for MST construction. For HDBSCAN, we don't need all n(n-1)/2
/// edges—we can compute them on-demand during Prim's algorithm.

// We store:
//   - Core distances: O(n)
//   - Embeddings: O(n × d)
//   - MST edges: O(n-1)

// We compute mutual reachability ON DEMAND:
public func distance(from i: Int, to j: Int) -> Float {
    let euclidean = embeddings[i].euclideanDistance(embeddings[j])
    return max(coreDistances[i], coreDistances[j], euclidean)
}
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Confusing Core Distance with Epsilon

```swift
// ❌ WRONG: Thinking core distance is like DBSCAN's epsilon
// Core distance is per-point, not global.

// In DBSCAN: ε is a fixed radius for all points
// In HDBSCAN: each point has its own "effective radius" = core distance
```

### Pitfall 2: Setting minSamples Too Low

```swift
// ⚠️ PROBLEMATIC: minSamples = 1
let config = HDBSCANConfiguration(minClusterSize: 5, minSamples: 1)

// Core distance = distance to 1st neighbor = very small
// Mutual reachability ≈ raw distance
// Loses density-awareness!

// ✅ BETTER: minSamples ≥ 3
// Rule of thumb: minSamples = minClusterSize is sensible
```

### Pitfall 3: Ignoring the Max Operation

```swift
// ⚠️ COMMON MISTAKE: Thinking mutual reachability is average

// Wrong: (core_a + core_b + dist) / 3
// Right: max(core_a, core_b, dist)

// The MAX makes it robust:
// - If one point is sparse, the edge is "long"
// - Dense-to-dense stays tight
// - Sparse never "hides" behind dense
```

### Pitfall 4: Expecting Symmetry to Matter

```swift
// Note: Mutual reachability IS symmetric
// mutual_reach(a, b) == mutual_reach(b, a)

// But the core distances are independent:
// core_dist(a) may differ greatly from core_dist(b)

// The max operation combines them symmetrically.
```

---

## Key Takeaways

1. **Core distance = local density**: Distance to k-th neighbor.

2. **Mutual reachability = density-aware distance**: max(core_a, core_b, dist).

3. **Sparse regions inflate**: Points in sparse regions become "farther apart."

4. **Dense regions preserve**: Raw distance dominates when both points are dense.

5. **MST on mutual reachability**: Creates hierarchy that respects varying densities.

6. **minSamples controls smoothing**: Higher k = smoother, lower k = more local.

---

## 💡 Key Insight

Mutual reachability is an **equalizer**. It asks: "How significant is this distance given the local density?"

```
In dense regions:
  - Small raw distances are significant
  - Mutual reachability ≈ raw distance

In sparse regions:
  - The same raw distance is less significant
  - Mutual reachability inflates to reflect "it takes more to be close here"

This lets dense and sparse clusters coexist in the same hierarchy,
each forming at their natural density level.
```

---

## Next Up

We've built the hierarchy on mutual reachability distances. Now let's learn how to **extract** the final clusters from that hierarchy.

**[→ 3.5 Cluster Extraction](./05-Cluster-Extraction.md)**

---

*Guide 3.4 of 3.5 • Chapter 3: Density-Based Clustering*
