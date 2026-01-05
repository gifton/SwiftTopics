# 1.3 Distance Metrics

> **Choosing and understanding how to measure similarity between embeddings.**

---

## The Concept

When we say two embeddings are "similar," we need a precise mathematical definition. **Distance metrics** (or similarity measures) quantify how close two vectors are.

The two most common metrics for embeddings:

```
Euclidean Distance (L2):          Cosine Similarity:
Measures "straight-line" distance  Measures directional alignment

    b                                   b
    │\                                 /
    │ \  distance                     / angle θ
    │  \                             /
    │   \                           /
    a────                          a────

d(a,b) = ||a - b||               cos(θ) = (a·b)/(||a||·||b||)

Range: [0, ∞)                    Range: [-1, 1]
0 = identical                    1 = same direction
larger = more different          0 = orthogonal
                                 -1 = opposite direction
```

---

## Why It Matters

Different metrics can produce different nearest neighbors:

```
Consider three embeddings:

A = [1, 0]      (short vector pointing right)
B = [2, 0]      (longer vector pointing right)
C = [1, 0.5]   (short vector pointing slightly up)

Question: Which is more similar to A?

Euclidean distance:           Cosine similarity:
d(A,B) = |2-1| = 1.0         cos(A,B) = 1.0 (same direction!)
d(A,C) = √(0 + 0.25) = 0.5   cos(A,C) = 0.89 (slightly different)

Euclidean says: C is closer    Cosine says: B is more similar
```

The right choice depends on what you're measuring:
- **Euclidean**: Absolute position in space
- **Cosine**: Directional alignment (ignores magnitude)

---

## The Mathematics

### Euclidean Distance (L2)

```
d(a, b) = ||a - b||₂ = √(Σᵢ (aᵢ - bᵢ)²)
```

Properties:
- Measures the straight-line distance between points
- Sensitive to vector magnitude (length)
- Always non-negative: d(a,b) ≥ 0
- Identity: d(a,a) = 0
- Symmetric: d(a,b) = d(b,a)
- Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)

### Squared Euclidean Distance

```
d²(a, b) = Σᵢ (aᵢ - bᵢ)²
```

Often used because:
- Avoids expensive square root
- Same ranking as L2 (if d(a,b) < d(a,c), then d²(a,b) < d²(a,c))
- Faster computation

### Cosine Similarity

```
cos(a, b) = (a · b) / (||a||₂ · ||b||₂)
         = Σᵢ(aᵢ · bᵢ) / (√Σᵢaᵢ² · √Σᵢbᵢ²)
```

Properties:
- Measures the cosine of the angle between vectors
- Ignores magnitude (only considers direction)
- Range: [-1, 1]
  - 1: Vectors point in the same direction
  - 0: Vectors are orthogonal (perpendicular)
  - -1: Vectors point in opposite directions

### Cosine Distance (for clustering)

Since clustering algorithms expect a *distance* (lower = more similar), we convert:

```
cosine_distance(a, b) = 1 - cos(a, b)

Range: [0, 2]
0: Identical direction
1: Orthogonal
2: Opposite direction
```

### Dot Product

```
a · b = Σᵢ(aᵢ · bᵢ)
```

If vectors are **normalized** (||a|| = ||b|| = 1):
```
a · b = cos(a, b)

Dot product on normalized vectors = cosine similarity
```

This is why many embedding models produce normalized vectors.

---

## The Technique: Choosing a Metric

### When to Use Euclidean Distance

✅ **Use Euclidean when**:
- Magnitude matters (vector length carries information)
- Computing absolute positions
- Using algorithms that assume Euclidean space (k-means, some HDBSCAN variants)

### When to Use Cosine Similarity

✅ **Use Cosine when**:
- Only direction matters (common for text embeddings)
- Embeddings have varying norms
- Documents of different lengths should be comparable

### For Text Embeddings

Most text embedding models are trained with **cosine similarity** as the training objective. This means:

```
Model training objective:
  maximize cos(embed(text1), embed(text2)) for similar texts
  minimize cos(embed(text1), embed(text2)) for different texts
```

**Recommendation**: Use cosine similarity (or cosine distance) for text embeddings unless you have a specific reason for Euclidean.

### The Normalization Trick

If you normalize embeddings to unit length, Euclidean and cosine become related:

```
For normalized vectors (||a|| = ||b|| = 1):

d²(a, b) = ||a - b||²
         = ||a||² + ||b||² - 2(a · b)
         = 1 + 1 - 2·cos(a, b)
         = 2(1 - cos(a, b))
         = 2 · cosine_distance(a, b)

Euclidean² ∝ Cosine distance on normalized vectors!
```

This means you can use Euclidean-based algorithms on normalized embeddings and get cosine-like behavior.

---

## In SwiftTopics

SwiftTopics provides both distance metrics on `Embedding`:

```swift
// 📍 See: Sources/SwiftTopics/Core/Embedding.swift:144-172

extension Embedding {

    /// Computes the cosine similarity with another embedding.
    ///
    /// cos(θ) = (v · w) / (||v||₂ × ||w||₂)
    ///
    /// - Returns: Cosine similarity in range [-1, 1].
    public func cosineSimilarity(_ other: Embedding) -> Float {
        let dotProduct = dot(other)
        let normProduct = l2Norm * other.l2Norm
        guard normProduct > Float.ulpOfOne else { return 0 }
        return dotProduct / normProduct
    }

    /// Computes the Euclidean distance to another embedding.
    ///
    /// d(v, w) = ||v - w||₂ = √(Σ (vᵢ - wᵢ)²)
    public func euclideanDistance(_ other: Embedding) -> Float {
        precondition(dimension == other.dimension)
        var sumSquares: Float = 0
        for i in 0..<dimension {
            let diff = vector[i] - other.vector[i]
            sumSquares += diff * diff
        }
        return sumSquares.squareRoot()
    }

    /// Computes the dot product with another embedding.
    ///
    /// v · w = Σ vᵢ × wᵢ
    @inlinable
    public func dot(_ other: Embedding) -> Float {
        precondition(dimension == other.dimension)
        var result: Float = 0
        for i in 0..<dimension {
            result += vector[i] * other.vector[i]
        }
        return result
    }
}
```

### Normalization Support

```swift
// 📍 See: Sources/SwiftTopics/Core/Embedding.swift:119-124

extension Embedding {
    /// Returns a normalized copy of this embedding.
    ///
    /// v_normalized = v / ||v||₂
    public func normalized() -> Embedding {
        let norm = l2Norm
        guard norm > Float.ulpOfOne else { return self }
        let normalizedVector = vector.map { $0 / norm }
        return Embedding(vector: normalizedVector)
    }
}
```

### In HDBSCAN

SwiftTopics uses **Euclidean distance** for HDBSCAN clustering on reduced embeddings:

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift:100-108

public func distance(from i: Int, to j: Int) -> Float {
    guard i != j else { return 0 }
    // Euclidean distance between reduced embeddings
    let euclidean = embeddings[i].euclideanDistance(embeddings[j])
    let coreI = coreDistances[i]
    let coreJ = coreDistances[j]
    return max(coreI, coreJ, euclidean)  // Mutual reachability
}
```

### In Incremental Assignment

Topic assignment uses **cosine similarity** for matching new documents to topic centroids:

```swift
// 📍 See: Sources/SwiftTopics/Incremental/IncrementalTopicUpdater.swift:500-508

// Compute similarity to each centroid
for (index, centroid) in state.centroids.enumerated() {
    let similarity = embedding.cosineSimilarity(centroid)
    similarities.append(similarity)

    if similarity > bestSimilarity {
        bestSimilarity = similarity
        bestIndex = index
    }
}
```

---

## Comparing Metrics Visually

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EUCLIDEAN vs COSINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Euclidean distance:              Cosine similarity:                    │
│                                                                         │
│        ●B                              ●B                               │
│       /                               /                                 │
│      / d = 2.0                       / θ = 0°                           │
│     /                               /                                   │
│    ●A───────●C                     ●A─────────●C                        │
│       d = 1.5                           θ = 30°                         │
│                                                                         │
│  A is closer to C                  A is more similar to B               │
│  (physically nearer)               (same direction)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Comparison

```swift
let a = Embedding(vector: [1.0, 0.0, 0.0])
let b = Embedding(vector: [2.0, 0.0, 0.0])  // Same direction, 2x length
let c = Embedding(vector: [1.0, 0.5, 0.0])  // Slightly different direction

// Euclidean distances
print("Euclidean A→B: \(a.euclideanDistance(b))")  // 1.0
print("Euclidean A→C: \(a.euclideanDistance(c))")  // 0.5
// C is closer!

// Cosine similarities
print("Cosine A→B: \(a.cosineSimilarity(b))")      // 1.0 (perfect!)
print("Cosine A→C: \(a.cosineSimilarity(c))")      // 0.894
// B is more similar!

// After normalizing
let aNorm = a.normalized()
let bNorm = b.normalized()
let cNorm = c.normalized()

print("Euclidean (normalized) A→B: \(aNorm.euclideanDistance(bNorm))")  // 0.0
print("Euclidean (normalized) A→C: \(aNorm.euclideanDistance(cNorm))")  // 0.46
// Now Euclidean agrees with Cosine!
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Using Euclidean on Non-Normalized Embeddings

```swift
// ⚠️ May produce unexpected results
let longDoc = model.embed(veryLongText)    // High norm
let shortDoc = model.embed(shortText)       // Low norm

// Euclidean sees them as far apart even if semantically similar
let distance = longDoc.euclideanDistance(shortDoc)  // Large!
```

**Solution**: Normalize embeddings or use cosine similarity.

### Pitfall 2: Forgetting Cosine Range

```swift
// ❌ WRONG: Treating cosine similarity like a probability
let similarity = a.cosineSimilarity(b)  // Could be -0.3!

if similarity > 0.8 {
    print("Very similar")
} else if similarity > 0 {
    print("Somewhat similar")
} else {
    print("Not similar")  // Negative cosine is possible!
}
```

### Pitfall 3: Mixing Metrics

```swift
// ⚠️ Inconsistent: Using cosine for training, Euclidean for search
let nearestByEuclidean = findNearest(query, metric: .euclidean)
let nearestByCosine = findNearest(query, metric: .cosine)

// These may return different results!
```

**Solution**: Use the same metric throughout your pipeline.

---

## Key Takeaways

1. **Euclidean distance** measures straight-line distance; sensitive to magnitude.

2. **Cosine similarity** measures directional alignment; ignores magnitude.

3. **For text embeddings**, cosine is usually preferred (matches training objective).

4. **Normalized embeddings** make Euclidean and cosine equivalent (up to scaling).

5. **SwiftTopics uses both**: Euclidean for HDBSCAN clustering (after reduction), cosine for topic assignment (in incremental updates).

6. **Consistency matters**: Use the same metric throughout your pipeline.

---

## 💡 Key Insight

The choice of distance metric affects which documents are considered "neighbors." Since clustering is fundamentally about finding neighbors, the metric choice propagates through the entire topic modeling pipeline.

```
Metric choice → Affects nearest neighbors → Affects clusters → Affects topics

Most text embedding models: Trained with cosine → Use cosine (or normalize + Euclidean)
```

---

## Chapter Summary

In this chapter, you learned:

1. **Why embeddings enable topic modeling**: Text → vectors where similarity = proximity
2. **How high-dimensional spaces work**: The curse of dimensionality and why reduction helps
3. **How to measure similarity**: Euclidean vs. cosine and when to use each

With this foundation, we're ready to tackle **dimensionality reduction**—the critical step that makes clustering possible.

---

## Next Chapter

**[→ Chapter 2: Dimensionality Reduction](../02-Dimensionality-Reduction/README.md)**

---

*Guide 1.3 of 1.3 • Chapter 1: Embeddings Foundation*
