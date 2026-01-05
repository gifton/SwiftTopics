# 2.1 Why Reduce Dimensions

> **The curse of dimensionality—why clustering fails in high-dimensional spaces.**

---

## The Concept

The **curse of dimensionality** is a collection of phenomena that occur in high-dimensional spaces, making many algorithms fail or behave unexpectedly.

For topic modeling, the key curse is **distance concentration**: as dimensions increase, all pairwise distances converge to the same value.

```
Imagine measuring distances between points:

In 2D:                          In 768D:
Distances vary widely           Distances barely vary

d₁ = 0.5                        d₁ = 22.31
d₂ = 3.2    (6× difference!)    d₂ = 22.47    (< 1% difference!)
d₃ = 1.8                        d₃ = 22.38
d₄ = 4.1                        d₄ = 22.52

Clear nearest neighbors         All points are "equally near"
```

---

## Why It Matters

Clustering algorithms rely on the concept of **neighborhood**—points that are closer should cluster together. When all distances are similar, there are no meaningful neighborhoods.

### HDBSCAN's Perspective

HDBSCAN finds **dense regions** separated by **sparse regions**. In high dimensions:

```
Low dimensions (works):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    Dense ●●●●●●                    Dense ●●●●●                  │
│    region  ●●●                     region ●●●●                  │
│                  Sparse                                         │
│                  gap                                            │
│                        Dense ●●●●                               │
│                        region●●●●                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

High dimensions (fails):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  │
│    ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●    │
│  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  │
│                                                                 │
│  Uniform density everywhere. No dense regions. No clusters.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Mathematics

### Distance Concentration Theorem

For random points uniformly distributed in a D-dimensional hypercube:

```
As D → ∞:
    max_distance - min_distance
    ────────────────────────── → 0
         min_distance

Translation: The relative difference between the farthest
and nearest points vanishes as dimension increases.
```

### Concrete Numbers

For N random points in [0,1]ᴰ:

| Dimension D | Mean Distance | Std Dev | Coefficient of Variation |
|-------------|---------------|---------|--------------------------|
| 2 | 0.52 | 0.24 | 0.46 |
| 10 | 1.29 | 0.22 | 0.17 |
| 100 | 4.08 | 0.22 | 0.05 |
| 768 | 11.31 | 0.22 | **0.02** |

At D=768, the coefficient of variation (CV = std/mean) is only 2%. **All distances are within ±2% of the mean.**

### Volume Concentration

Another curse manifestation: in high dimensions, most of the volume is near the surface of the hypercube.

```
Volume within distance r of center (normalized so total = 1):

D = 2:   At r = 0.5, about 25% of volume is "close to center"
D = 10:  At r = 0.5, about 0.1% is "close to center"
D = 768: At r = 0.5, essentially 0% is "close to center"

Result: Points are pushed to the "corners" of high-D space.
The interior is essentially empty.
```

---

## The Technique: Why Reduction Helps

Embeddings aren't random—they have **learned structure**. This structure lies on a **lower-dimensional manifold** within the high-D space.

```
768D space contains ~50D worth of "real" information:

768D embedding space:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    ╭───────────────────────╮                            │
│                   ╱                         ╲                           │
│                  ╱   Data lives on this      ╲                          │
│                 │    ~50D manifold            │                         │
│                  ╲                           ╱                          │
│                   ╲                         ╱                           │
│                    ╰───────────────────────╯                            │
│                                                                         │
│         Most of the 768 dimensions are "noise" or redundancy            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Dimensionality reduction finds this manifold and projects onto it:

```
768D → Reduction → 15D

Before:
- Distances concentrated
- No visible structure
- Clustering fails

After:
- Distances vary meaningfully
- Structure revealed
- Clustering succeeds
```

### Why Not Just Use Lower-D Embeddings?

You might ask: "Why not train a 15D embedding model?"

1. **Capacity**: 768D can encode more nuance than 15D
2. **Transfer learning**: Pretrained models are 384D-3072D
3. **Task-specific**: Optimal dimension for search ≠ optimal for clustering
4. **Information preservation**: Reduction keeps most information; low-D training loses it

The strategy is: embed at high-D for quality, reduce for clustering.

---

## In SwiftTopics

The reduction stage is configured via `ReductionConfiguration`:

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

public struct ReductionConfiguration: Sendable, Codable {
    /// Target output dimension.
    public let outputDimension: Int

    /// Reduction method (PCA, UMAP, or none).
    public let method: ReductionMethod

    /// Optional PCA-specific configuration.
    public let pcaConfig: PCAConfiguration?

    /// Optional UMAP-specific configuration.
    public let umapConfig: UMAPConfiguration?

    /// Random seed for reproducibility.
    public let seed: UInt64?
}

public enum ReductionMethod: String, Sendable, Codable {
    case pca
    case umap
    case none
}
```

### How Reduction Fits in the Pipeline

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModel.swift:825-877

private func reduceEmbeddings(
    _ embeddings: [Embedding]
) async throws -> (reduced: [Embedding], components: [Float]?) {
    let config = configuration.reduction

    // Skip reduction if method is none
    guard config.method != .none else {
        return (embeddings, nil)
    }

    // Don't reduce if already low-dimensional
    let inputDim = embeddings.first?.dimension ?? 0
    if inputDim <= config.outputDimension {
        return (embeddings, nil)
    }

    switch config.method {
    case .pca:
        var pca = PCAReducer(
            components: config.outputDimension,
            whiten: config.pcaConfig?.whiten ?? false,
            seed: config.seed
        )
        try await pca.fit(embeddings)
        let reduced = try await pca.transform(embeddings)
        return (reduced, pca.principalComponents)

    case .umap:
        var umap = UMAPReducer(
            nNeighbors: config.umapConfig?.nNeighbors ?? 15,
            minDist: config.umapConfig?.minDist ?? 0.1,
            nComponents: config.outputDimension,
            nEpochs: config.umapConfig?.nEpochs ?? 200,
            seed: config.seed
        )
        try await umap.fit(embeddings)
        let reduced = try await umap.transform(embeddings)
        return (reduced, nil)

    case .none:
        return (embeddings, nil)
    }
}
```

---

## Demonstrating the Curse

```swift
import Foundation

func demonstrateCurse() {
    let n = 1000  // Number of random points

    for dimension in [2, 10, 100, 768] {
        // Generate random points
        var points: [[Float]] = []
        for _ in 0..<n {
            let point = (0..<dimension).map { _ in Float.random(in: 0...1) }
            points.append(point)
        }

        // Compute all pairwise distances
        var distances: [Float] = []
        for i in 0..<n {
            for j in (i+1)..<n {
                var sumSq: Float = 0
                for d in 0..<dimension {
                    let diff = points[i][d] - points[j][d]
                    sumSq += diff * diff
                }
                distances.append(sqrt(sumSq))
            }
        }

        // Statistics
        let mean = distances.reduce(0, +) / Float(distances.count)
        let variance = distances.map { ($0 - mean) * ($0 - mean) }
                               .reduce(0, +) / Float(distances.count)
        let stdDev = sqrt(variance)
        let cv = stdDev / mean

        print("D=\(dimension): mean=\(mean), stdDev=\(stdDev), CV=\(cv)")
    }
}

// Output:
// D=2:   mean=0.52, stdDev=0.24, CV=0.46
// D=10:  mean=1.29, stdDev=0.22, CV=0.17
// D=100: mean=4.08, stdDev=0.22, CV=0.05
// D=768: mean=11.3, stdDev=0.22, CV=0.02  ← Everything is equidistant!
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Skipping Reduction

```swift
// ⚠️ Clustering raw 768D embeddings
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .none),
    clustering: HDBSCANConfiguration(minClusterSize: 5),
    // ...
)

// Result: Poor clusters or everything marked as noise
```

### Pitfall 2: Reducing Too Much

```swift
// ⚠️ Reducing to 2D (good for visualization, bad for clustering)
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 2,  // Too low!
        method: .pca
    ),
    // ...
)

// Result: Lost too much structure; topics are coarse
```

### Pitfall 3: Ignoring Input Dimension

```swift
// ⚠️ Reducing 50D embeddings to 100D
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 100,  // Higher than input!
        method: .pca
    ),
    // ...
)

// SwiftTopics handles this by skipping reduction,
// but it's a sign of misconfiguration
```

---

## Key Takeaways

1. **The curse of dimensionality** causes distances to concentrate—all points become equidistant in high-D.

2. **Clustering requires distance variation**: HDBSCAN needs dense and sparse regions to find clusters.

3. **Embeddings have lower-D structure**: The 768 dimensions contain ~50D of "real" information on a manifold.

4. **Reduction reveals structure**: Projecting onto the manifold restores meaningful distance variation.

5. **Don't skip reduction**: Always reduce before clustering unless your embeddings are already low-D.

---

## 💡 Key Insight

Dimensionality reduction doesn't discard information—it discards **noise**. The high-dimensional embedding space contains the signal (semantic structure) plus noise (redundant dimensions). Reduction extracts the signal.

```
768D = ~50D signal + ~718D noise
Reduction: 768D → 15D ≈ extracting most of the signal
```

---

## Next Up

Now that we understand *why* reduction is necessary, let's learn *how* PCA does it:

**[→ 2.2 PCA Explained](./02-PCA-Explained.md)**

---

*Guide 2.1 of 2.4 • Chapter 2: Dimensionality Reduction*
