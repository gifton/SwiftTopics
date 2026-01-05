# Chapter 2: Dimensionality Reduction

> **Compressing 768 dimensions to 15—making clustering mathematically possible.**

Chapter 1 established that embeddings enable topic modeling by transforming text into vectors. But we also hinted at a problem: **high-dimensional spaces are hostile to clustering**.

This chapter explains why, and how dimensionality reduction solves it.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [2.1 Why Reduce Dimensions](./01-Why-Reduce-Dimensions.md) | The curse of dimensionality | Distance concentration, sparsity |
| [2.2 PCA Explained](./02-PCA-Explained.md) | Principal Component Analysis | Variance, eigenvectors, projection |
| [2.3 UMAP Intuition](./03-UMAP-Intuition.md) | Manifold learning | Local structure, fuzzy topology |
| [2.4 PCA vs UMAP](./04-PCA-vs-UMAP.md) | Choosing a method | Speed vs. quality tradeoffs |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- Why clustering fails in high-dimensional spaces
- How PCA finds directions of maximum variance
- How UMAP preserves local neighborhood structure
- When to use PCA (fast, deterministic) vs. UMAP (better structure)
- How SwiftTopics' reduction stage prepares data for HDBSCAN

---

## The Core Problem

Consider clustering 1,000 documents in 768-dimensional space:

```
768D embedding space:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Every point is roughly the same distance from every other point.       │
│                                                                         │
│  ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   │
│    ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●     │
│  ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   │
│                                                                         │
│  No visible structure. Distance variance is tiny.                       │
│  Clustering algorithms see uniform density everywhere.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

After reduction to 15D:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│      ●●●                              ●●●●                              │
│     ●●●●●                            ●●●●●●                             │
│      ●●●                              ●●●●                              │
│                                                                         │
│                  ●●●●●●                                                 │
│                 ●●●●●●●●                      ●●                        │
│                  ●●●●●●                      ●●●●                       │
│                                               ●●                        │
│                                                                         │
│  Clear clusters emerge! Density varies. Structure visible.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Reduction doesn't create structure—it **reveals** the structure that was hidden by high dimensionality.

---

## Why 768D → 15D?

The number 15 isn't magic; it's a practical balance:

| Output Dimension | Effect |
|------------------|--------|
| **2-3** | Visualization only; loses too much signal |
| **5-10** | May lose important structure |
| **10-20** | Good balance for topic modeling |
| **30-50** | Preserves more signal; slower clustering |
| **100+** | Diminishing returns; curse returns |

SwiftTopics defaults to **15 dimensions** because:
- Empirically works well for topic discovery
- Fast enough for real-time use
- Dense enough for HDBSCAN to find structure

---

## The Two Approaches

### PCA: Find Maximum Variance

```
PCA asks: "Which directions capture the most spread in the data?"

768D data with 3 principal directions:

      PC2
       │
       │    ╱ data spread mostly here
       │   ╱
       │  ╱
       │ ╱
───────┼──────── PC1  (most variance)
       │
       │
      PC3
       (least variance)

Project onto top 15 PCs → preserve most information
```

**Pros**: Fast, deterministic, mathematically elegant
**Cons**: Only captures linear structure; global view

### UMAP: Preserve Local Neighborhoods

```
UMAP asks: "Which points are neighbors? Preserve those relationships."

768D neighborhoods:           15D projection:

  A─B                           A─B
  │╲│    neighbors             │╲│    same neighbors!
  C─D    preserved     →       C─D

UMAP doesn't care about global structure;
it cares that nearby points stay nearby.
```

**Pros**: Preserves local structure; handles non-linear manifolds
**Cons**: Slower, stochastic (results vary with seed)

---

## SwiftTopics Configuration

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

// PCA (default, fast)
let pcaConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .pca
    ),
    // ...
)

// UMAP (better structure, slower)
let umapConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .umap,
        umapConfig: UMAPConfiguration(
            nNeighbors: 15,
            minDist: 0.1,
            nEpochs: 200
        )
    ),
    // ...
)

// Skip reduction (not recommended for high-D)
let noReduction = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        method: .none
    ),
    // ...
)
```

---

## Prerequisites Check

Before starting this chapter, ensure you understand:

- [ ] What embedding vectors are (Chapter 1)
- [ ] Euclidean distance between vectors
- [ ] Basic linear algebra: vectors, matrices, dot products
- [ ] The concept of variance (spread of data)

---

## Start Here

**[→ 2.1 Why Reduce Dimensions](./01-Why-Reduce-Dimensions.md)**

---

*Chapter 2 of 6 • SwiftTopics Learning Guide*
