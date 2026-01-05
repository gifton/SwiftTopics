# 2.2 PCA Explained

> **Principal Component Analysis—finding the directions of maximum variance.**

---

## The Concept

**Principal Component Analysis (PCA)** finds the directions in which your data varies the most, then projects onto those directions.

Imagine a cloud of points stretched like a cigar:

```
Original 2D data:                    After PCA rotation:

    │         ····                        │
    │       ·····                    PC2  │    ····
    │     ······                          │  ······
    │   ·······                           │·········
    │  ········                      ─────┼─────────── PC1
    │   ·······                           │·········
    │     ······                          │  ······
    │       ·····                         │    ····
    │         ····                        │
    └───────────────                      └───────────────

The cigar is tilted.               PCA aligns with the cigar.
                                   PC1 = direction of most spread.
                                   PC2 = perpendicular, less spread.
```

If we keep only PC1 (dropping PC2), we reduce from 2D to 1D while preserving most of the structure.

---

## Why It Matters

PCA is the **default reduction method** in SwiftTopics because:

1. **Fast**: O(N·D²) where N = documents, D = dimension
2. **Deterministic**: Same input → same output (given seed)
3. **Well-understood**: Mathematical properties are thoroughly studied
4. **GPU-accelerated**: Matrix operations map well to GPU

For topic modeling, PCA's linear assumption is often sufficient because:
- We're reducing 768D → 15D (significant compression)
- HDBSCAN handles non-linear clusters in the reduced space
- Speed matters for real-time applications

---

## The Mathematics

### Step 1: Center the Data

```
Given N embeddings of dimension D:
  X = [x₁, x₂, ..., xₙ]  where each xᵢ ∈ ℝᴰ

Compute the mean:
  μ = (1/N) Σᵢ xᵢ

Center the data:
  X̃ᵢ = xᵢ - μ

Centering ensures the first PC passes through the data center.
```

### Step 2: Compute the Covariance Matrix

```
Covariance matrix C ∈ ℝᴰˣᴰ:

  C = (1/N) X̃ᵀX̃

  Cᵢⱼ = covariance between dimension i and dimension j

The diagonal Cᵢᵢ = variance of dimension i.
Off-diagonal Cᵢⱼ = how dimensions i and j vary together.
```

### Step 3: Eigendecomposition

```
Find eigenvectors v and eigenvalues λ of C:

  Cv = λv

Eigenvector v: A direction in the data.
Eigenvalue λ: The variance along that direction.

Sort by eigenvalue (largest first):
  λ₁ ≥ λ₂ ≥ ... ≥ λᴰ

The eigenvector v₁ (with largest λ₁) is the
first principal component—the direction of maximum variance.
```

### Step 4: Project

```
Keep top k eigenvectors: V = [v₁, v₂, ..., vₖ]  (D × k matrix)

Project each centered point:
  yᵢ = Vᵀ · x̃ᵢ

Result: yᵢ ∈ ℝᵏ (reduced from D dimensions to k dimensions)
```

### Variance Explained

```
Total variance = Σᵢ λᵢ

Variance explained by first k components = (Σᵢ₌₁ᵏ λᵢ) / (Σᵢ λᵢ)

Example for 768D embeddings:
  - First 15 components might explain 70% of variance
  - First 50 components might explain 95% of variance

We're keeping the "important" directions.
```

---

## The Technique: PCA Step by Step

### Visual Example

```
100 points in 3D, elongated along one axis:

Original 3D view:                   Top 2 PCs (3D → 2D):

     z │    ╱╱╱                           │
       │   ╱╱╱╱                      PC2  │  ╱╱╱
       │  ╱╱╱╱╱                           │ ╱╱╱╱
       │ ╱╱╱╱╱╱                           │╱╱╱╱╱
───────┼──────── y                   ─────┼─────── PC1
       │╲                                 │
      ╱│                                  │
     x │                                  │

The cigar in 3D becomes         Most variance preserved.
a 2D ellipse when we drop       Only z-variation lost.
the least-varying direction.
```

### Why Eigenvalues = Variance

Intuitively:
- An eigenvector is a direction where the data is "stretched"
- The eigenvalue measures how much stretching (variance)
- Large eigenvalue → data is spread out in this direction → important
- Small eigenvalue → data is compact in this direction → can ignore

---

## In SwiftTopics

SwiftTopics implements PCA in `PCAReducer`:

```swift
// 📍 See: Sources/SwiftTopics/Reduction/PCA.swift

public struct PCAReducer {
    /// Number of components to keep.
    public let components: Int

    /// Whether to whiten (normalize variance).
    public let whiten: Bool

    /// Variance ratio to achieve (alternative to fixed components).
    public let varianceRatio: Float?

    /// Principal components (after fitting).
    public private(set) var principalComponents: [Float]?

    /// Mean vector (for centering new data).
    public private(set) var mean: [Float]?

    /// Eigenvalues (variance explained by each component).
    public private(set) var explainedVariance: [Float]?
}
```

### Fitting PCA

```swift
// 📍 See: Sources/SwiftTopics/Reduction/PCA.swift (fit method)

public mutating func fit(_ embeddings: [Embedding]) async throws {
    guard !embeddings.isEmpty else {
        throw PCAError.emptyInput
    }

    let n = embeddings.count
    let d = embeddings[0].dimension

    // Step 1: Compute mean
    var mean = [Float](repeating: 0, count: d)
    for emb in embeddings {
        for i in 0..<d {
            mean[i] += emb.vector[i]
        }
    }
    for i in 0..<d {
        mean[i] /= Float(n)
    }
    self.mean = mean

    // Step 2: Center data and compute covariance
    // (Uses GPU-accelerated matrix operations via VectorAccelerate)
    let covariance = try await computeCovariance(embeddings, mean: mean)

    // Step 3: Eigendecomposition
    let (eigenvalues, eigenvectors) = try eigendecompose(covariance, d: d)

    // Step 4: Select top k components
    let k = determineComponents(eigenvalues: eigenvalues)
    self.principalComponents = Array(eigenvectors.prefix(k * d))
    self.explainedVariance = Array(eigenvalues.prefix(k))
}
```

### Transforming Data

```swift
// 📍 See: Sources/SwiftTopics/Reduction/PCA.swift (transform method)

public func transform(_ embeddings: [Embedding]) async throws -> [Embedding] {
    guard let components = principalComponents,
          let mean = mean else {
        throw PCAError.notFitted
    }

    let k = self.components
    let d = embeddings[0].dimension

    var results = [Embedding]()
    results.reserveCapacity(embeddings.count)

    for emb in embeddings {
        // Center
        var centered = [Float](repeating: 0, count: d)
        for i in 0..<d {
            centered[i] = emb.vector[i] - mean[i]
        }

        // Project: y = Vᵀ · x
        var projected = [Float](repeating: 0, count: k)
        for i in 0..<k {
            var dot: Float = 0
            for j in 0..<d {
                dot += components[i * d + j] * centered[j]
            }
            projected[i] = dot
        }

        results.append(Embedding(vector: projected))
    }

    return results
}
```

### GPU Acceleration

The covariance computation uses VectorAccelerate:

```swift
// Heavy matrix operations delegated to GPU
let covariance = try await context.matrixMultiply(
    centered,      // N × D matrix
    centeredT,     // D × N matrix (transposed)
    alpha: 1.0 / Float(n)
)
// Result: D × D covariance matrix
```

---

## Configuration Options

### Number of Components

```swift
// Fixed number of components
let pca = PCAReducer(components: 15)

// By variance ratio (keep enough to explain 95% variance)
let pca = PCAReducer(components: 100, varianceRatio: 0.95)
// Will keep fewer than 100 if 95% is reached earlier
```

### Whitening

Whitening scales each component to have unit variance:

```swift
// Without whitening: PC1 might have variance 10, PC2 variance 2
// With whitening: All PCs have variance 1

let pca = PCAReducer(components: 15, whiten: true)
```

Whitening can help when downstream algorithms expect uniform scale, but often isn't necessary for topic modeling.

---

## Interpreting Results

### Explained Variance

```swift
let pca = PCAReducer(components: 50)
try await pca.fit(embeddings)

// How much variance does each component capture?
for (i, variance) in pca.explainedVariance!.enumerated() {
    print("PC\(i): \(variance) (\(variance / totalVariance * 100)%)")
}

// Typical output for 768D embeddings:
// PC0: 12.3 (8.2%)
// PC1: 8.7 (5.8%)
// PC2: 6.1 (4.1%)
// ...
// PC14: 1.2 (0.8%)
// Total for first 15: ~45% variance explained
```

### Cumulative Variance

```swift
var cumulative: Float = 0
for (i, variance) in pca.explainedVariance!.enumerated() {
    cumulative += variance / totalVariance
    print("First \(i+1) PCs: \(cumulative * 100)% variance")
}

// Typical:
// First 15 PCs: 45% variance
// First 50 PCs: 75% variance
// First 100 PCs: 90% variance
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Forgetting to Center

```swift
// ❌ WRONG: Projecting without centering
let projected = matrixMultiply(components, embeddings)
// Results shifted incorrectly!

// ✅ CORRECT: Center first
let centered = embeddings.map { $0 - mean }
let projected = matrixMultiply(components, centered)
```

SwiftTopics handles this automatically.

### Pitfall 2: Assuming PCA Captures "Meaning"

```swift
// ❌ WRONG: Interpreting PCs as features
// "PC1 is sentiment, PC2 is topic complexity"

// ✓ CORRECT: PCs are mathematical constructs
// They capture variance, not human-interpretable features
```

### Pitfall 3: Too Few Components

```swift
// ⚠️ Only 5 components for 768D embeddings
let pca = PCAReducer(components: 5)
// May explain only 20% of variance—too much loss!
```

---

## Key Takeaways

1. **PCA finds maximum variance directions**: Eigenvectors of covariance matrix.

2. **Eigenvalues measure importance**: Larger eigenvalue = more variance = more important.

3. **Projection preserves structure**: Top k PCs capture most of the signal.

4. **Fast and deterministic**: O(N·D²), same input → same output.

5. **Linear assumption**: PCA assumes linear structure (often sufficient for pre-clustering).

6. **SwiftTopics default**: PCA with 15 components works well for most topic modeling.

---

## 💡 Key Insight

PCA doesn't find "the best" representation—it finds the representation that **maximizes variance**. For many datasets, maximum variance ≈ maximum information, but this isn't always true.

```
If your data has structure that isn't aligned with variance:
  - PCA might miss it
  - UMAP (next guide) can capture non-linear structure

For topic modeling, PCA usually works because:
  - Embedding models already encode semantic structure
  - We just need to compress, not discover new patterns
```

---

## Next Up

UMAP takes a different approach—preserving local neighborhoods instead of maximizing variance:

**[→ 2.3 UMAP Intuition](./03-UMAP-Intuition.md)**

---

*Guide 2.2 of 2.4 • Chapter 2: Dimensionality Reduction*
