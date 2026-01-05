# 2.3 UMAP Intuition

> **Manifold learning—preserving local neighborhoods in the reduced space.**

---

## The Concept

**UMAP** (Uniform Manifold Approximation and Projection) takes a fundamentally different approach from PCA. Instead of maximizing variance, UMAP asks:

> "Who are each point's neighbors? Keep those relationships in the reduced space."

```
High-D space:                      Low-D projection (UMAP):

    A ─── B                            A ─── B
    │     │       neighbors            │     │
    C ─── D       preserved    →       C ─── D

    E ─── F                            E ─── F
    │     │                            │     │
    G ─── H                            G ─── H

Local structure (who's near whom) is preserved.
Global structure (how far apart are A-E) may change.
```

---

## Why It Matters

### PCA's Limitation

PCA finds linear combinations of features. If your data lies on a curved surface (manifold), PCA can't capture that:

```
Data on a Swiss roll (3D curved surface):

    PCA projection:              UMAP projection:

      ████████                      ██
      ████████                     ████
      ████████         vs.        ██████
      ████████                   ████████
                                ██████████

    PCA "flattens" the roll.   UMAP "unrolls" it.
    Neighbors become mixed.    Neighbors stay neighbors.
```

### When UMAP Helps

For topic modeling:
- Embeddings often lie on non-linear manifolds
- UMAP can better separate clusters that PCA merges
- Result: More distinct topics with better boundaries

### When PCA Is Enough

- You need speed (PCA is 10-100× faster)
- Determinism matters (same input → same output)
- Your reduced dimension is low enough (15D) that HDBSCAN handles non-linearity

---

## The Mathematics (Intuition)

UMAP is based on topology and Riemannian geometry. Here's the intuitive version:

### Step 1: Build a Fuzzy Neighborhood Graph

For each point, find its k nearest neighbors and assign edge weights:

```
Point A's neighbors (k=3):

     0.9        0.7
  A ────── B ────── C
   ╲      ╱
    ╲0.6 ╱ 0.4
     ╲  ╱
      D

Edge weight = how "close" the neighbor is (exponential decay).
Closer neighbors get stronger connections.
```

The key insight: **distances are relative to each point's local density**.

```
Dense region:               Sparse region:
Only very close points      Even somewhat distant points
get high weights.           get moderate weights.

   ●●●●                         ●     ●
  ●●●●●●    vs.                   ●       ●
   ●●●●                         ●     ●

This handles varying-density data.
```

### Step 2: Build the Low-D Layout

Start with random positions in the target dimension (e.g., 15D). Then optimize:

```
Objective: Make the low-D neighborhood graph
           match the high-D neighborhood graph.

If A-B are neighbors in high-D:
  → Pull A-B closer in low-D

If A-C are NOT neighbors in high-D:
  → Push A-C apart in low-D
```

This is done via **stochastic gradient descent** over many iterations.

### Step 3: The Result

```
After optimization:

High-D neighborhood:          Low-D result:

    A ─ B                        A ─ B
    │   │                        │   │
    C ─ D                        C ─ D

    E ─ F                        E ─ F
    │   │                        │   │
    G ─ H                        G ─ H

Same local structure, different space.
```

---

## The Technique: UMAP Parameters

### `n_neighbors` (k)

How many neighbors to consider for each point.

```
Small k (5-10):              Large k (50-100):
- Very local structure       - More global structure
- May miss cluster shape     - Smoother embeddings
- Good for fine detail       - Good for coarse structure

For topic modeling: k = 15-30 is typical.
```

### `min_dist`

Minimum distance between points in the output.

```
Small min_dist (0.0-0.1):    Large min_dist (0.5-1.0):
- Points can be very close   - Points spread out
- Tight clusters             - Looser clusters
- Good for HDBSCAN           - Good for visualization

For topic modeling: min_dist = 0.0-0.2 works well.
```

### `n_epochs`

Number of optimization iterations.

```
Fewer epochs (50-100):       More epochs (200-500):
- Faster                     - Better convergence
- May not converge           - Slower
- Good for exploration       - Good for final results

For topic modeling: 200-300 is typical.
```

---

## In SwiftTopics

SwiftTopics implements UMAP in `UMAPReducer`:

```swift
// 📍 See: Sources/SwiftTopics/Reduction/UMAP/UMAP.swift

public struct UMAPReducer {
    /// Number of neighbors for graph construction.
    public let nNeighbors: Int

    /// Minimum distance in output space.
    public let minDist: Float

    /// Target dimensionality.
    public let nComponents: Int

    /// Distance metric for input space.
    public let metric: DistanceMetric

    /// Number of optimization epochs.
    public let nEpochs: Int

    /// Random seed for reproducibility.
    public let seed: UInt64?
}
```

### UMAP Pipeline

```swift
// 📍 See: Sources/SwiftTopics/Reduction/UMAP/UMAP.swift

public mutating func fit(_ embeddings: [Embedding]) async throws {
    // Step 1: Build k-NN graph
    let knnGraph = try await buildKNNGraph(embeddings, k: nNeighbors)

    // Step 2: Compute fuzzy simplicial set (neighborhood weights)
    let fuzzyGraph = try computeFuzzySimplicialSet(knnGraph)

    // Step 3: Initialize low-D positions (spectral or random)
    var positions = try await initializePositions(
        embeddings.count,
        dimensions: nComponents
    )

    // Step 4: Optimize via SGD
    positions = try await optimize(
        positions,
        graph: fuzzyGraph,
        epochs: nEpochs
    )

    self.embedding = positions
}
```

### The k-NN Graph

```swift
// 📍 See: Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift

private func buildKNNGraph(
    _ embeddings: [Embedding],
    k: Int
) async throws -> KNNGraph {
    // Use GPU-accelerated k-NN via VectorAccelerate
    let gpuKNN = try GPUBatchKNN(context: context, dataset: embeddings)
    let neighbors = try await gpuKNN.query(k: k)

    return KNNGraph(neighbors: neighbors)
}
```

### The Fuzzy Set

```swift
// 📍 See: Sources/SwiftTopics/Reduction/UMAP/FuzzySimplicialSet.swift

private func computeFuzzySimplicialSet(_ knn: KNNGraph) -> FuzzyGraph {
    // Convert distances to probabilities
    // Using smooth approximation to local connectivity

    for i in 0..<n {
        // Find sigma_i such that:
        // Σⱼ exp(-d(i,j) / sigma_i) = log2(k)
        let sigma = findSigma(distances: knn.distances[i])

        for (j, distance) in knn.neighbors[i] {
            // Probability that j is a neighbor of i
            let weight = exp(-max(0, distance - rho[i]) / sigma)
            graph.setWeight(i, j, weight)
        }
    }

    // Symmetrize: edge exists if either endpoint claims it
    return graph.symmetrized()
}
```

### The Optimization

```swift
// 📍 See: Sources/SwiftTopics/Reduction/UMAP/UMAPOptimizer.swift

private func optimize(
    _ positions: [[Float]],
    graph: FuzzyGraph,
    epochs: Int
) async throws -> [[Float]] {
    var pos = positions

    for epoch in 0..<epochs {
        // Sample positive edges (neighbors)
        for (i, j, weight) in graph.positiveEdges() {
            // Attractive force: pull neighbors together
            let gradient = attractiveGradient(pos[i], pos[j], weight)
            pos[i] = pos[i] - learningRate * gradient
            pos[j] = pos[j] + learningRate * gradient
        }

        // Sample negative edges (non-neighbors)
        for (i, k) in sampleNegatives(graph) {
            // Repulsive force: push non-neighbors apart
            let gradient = repulsiveGradient(pos[i], pos[k])
            pos[i] = pos[i] - learningRate * gradient
        }

        // Decay learning rate
        learningRate *= 0.99
    }

    return pos
}
```

---

## Comparing PCA and UMAP Output

```
Same data, different projections:

PCA (15D):                      UMAP (15D):

  ●●●●●●                         ●●●     ●●●
  ●●●●●●●                       ●●●●●   ●●●●●
  ●●●●●●                         ●●●     ●●●
  ●●●●●●●
  ●●●●●●                              ●●●●●
                                     ●●●●●●●
                                      ●●●●●

Linear projection mixes        Manifold learning separates
overlapping clusters.          clusters by neighborhoods.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Expecting Determinism

```swift
// ⚠️ UMAP is stochastic
let result1 = try await umap.fit(embeddings)
let result2 = try await umap.fit(embeddings)
// result1 ≠ result2 (different random initialization)

// ✅ Use seed for reproducibility
let umap = UMAPReducer(nComponents: 15, seed: 42)
```

### Pitfall 2: Too Few Neighbors

```swift
// ⚠️ k too small
let umap = UMAPReducer(nNeighbors: 3, ...)
// Very local structure; may create spurious clusters
```

### Pitfall 3: Using UMAP for Speed

```swift
// ⚠️ UMAP is slow for large N
// PCA: O(N·D²)
// UMAP: O(N·k·epochs) with significant constants

// For 10,000 documents:
// PCA: ~2 seconds
// UMAP: ~30 seconds
```

---

## Key Takeaways

1. **UMAP preserves neighborhoods**: Points that are neighbors in high-D stay neighbors in low-D.

2. **Non-linear manifolds**: UMAP can "unfold" curved structures that PCA flattens.

3. **Key parameters**: `n_neighbors` (locality), `min_dist` (spread), `n_epochs` (quality).

4. **Stochastic**: Results vary with random seed; use fixed seed for reproducibility.

5. **Slower than PCA**: 10-100× slower; use when quality matters more than speed.

---

## 💡 Key Insight

UMAP's power comes from its **local-to-global** approach:

```
1. Define what "neighbor" means locally (fuzzy graph)
2. Find a layout that preserves local relationships globally (optimization)

This captures non-linear structure that PCA misses,
at the cost of computation and determinism.
```

---

## Next Up

Now that we understand both methods, let's compare them for topic modeling:

**[→ 2.4 PCA vs UMAP](./04-PCA-vs-UMAP.md)**

---

*Guide 2.3 of 2.4 • Chapter 2: Dimensionality Reduction*
