# 2.4 PCA vs UMAP

> **Choosing the right reduction method for your topic modeling needs.**

---

## The Concept

You've learned two fundamentally different approaches:

| Aspect | PCA | UMAP |
|--------|-----|------|
| **Philosophy** | Maximize variance | Preserve neighborhoods |
| **Structure** | Linear | Non-linear (manifold) |
| **Speed** | Fast | Slow |
| **Determinism** | Deterministic | Stochastic |
| **Parameters** | Few | Several to tune |

The question: **Which should you use for topic modeling?**

---

## Why It Matters

The wrong choice can affect:

```
Bad reduction choice:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  - Clusters that should be separate get merged                          │
│  - Natural topics become split                                          │
│  - More documents marked as outliers                                    │
│  - Lower coherence scores                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Good reduction choice:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  - Clear separation between topic clusters                              │
│  - Natural topics emerge intact                                         │
│  - Fewer outliers                                                       │
│  - Higher coherence scores                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Head-to-Head Comparison

### Speed

```
Dataset: 10,000 documents × 768D embeddings → 15D

Method      Time        Relative
──────────────────────────────────
PCA         ~2 sec      1×
UMAP        ~35 sec     17×

Dataset: 50,000 documents × 768D embeddings → 15D

Method      Time        Relative
──────────────────────────────────
PCA         ~8 sec      1×
UMAP        ~180 sec    22×

UMAP scales worse with N.
```

### Quality (Topic Coherence)

```
Typical results on journal/note datasets:

Method      Topics Found    Avg Coherence    Outlier Rate
────────────────────────────────────────────────────────────
PCA         8-12           0.15-0.25        15-25%
UMAP        10-15          0.18-0.30        10-20%

UMAP often finds 1-3 more topics with slightly better coherence.
The difference varies by dataset.
```

### Reproducibility

```
PCA (same data, same config):
  Run 1: Topics = ["fitness", "coding", "travel", ...]
  Run 2: Topics = ["fitness", "coding", "travel", ...]
  Run 3: Topics = ["fitness", "coding", "travel", ...]
  ✓ Identical

UMAP (same data, same config, no seed):
  Run 1: Topics = ["fitness", "coding", "travel", ...]
  Run 2: Topics = ["fitness", "programming", "vacation", ...]
  Run 3: Topics = ["exercise", "coding", "travel", ...]
  ✗ Similar but not identical

UMAP (same data, same config, fixed seed):
  Run 1-3: Topics = ["fitness", "coding", "travel", ...]
  ✓ Identical (with seed)
```

---

## Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PCA vs UMAP DECISION TREE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Is speed critical?                                                     │
│       │                                                                 │
│       ├── YES → Use PCA                                                 │
│       │                                                                 │
│       └── NO → Continue ↓                                               │
│                    │                                                    │
│  Is reproducibility required (without managing seeds)?                  │
│       │                                                                 │
│       ├── YES → Use PCA                                                 │
│       │                                                                 │
│       └── NO → Continue ↓                                               │
│                    │                                                    │
│  Is dataset > 50,000 documents?                                         │
│       │                                                                 │
│       ├── YES → Use PCA (UMAP too slow)                                 │
│       │                                                                 │
│       └── NO → Continue ↓                                               │
│                    │                                                    │
│  Is topic quality your top priority?                                    │
│       │                                                                 │
│       ├── YES → Try UMAP                                                │
│       │                                                                 │
│       └── NO → Use PCA (simpler, faster)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Use PCA When

✅ **Real-time applications**: Need fast response
✅ **Large datasets**: > 20,000 documents
✅ **Production systems**: Reproducibility matters
✅ **Simplicity**: Fewer parameters to tune
✅ **Iterative development**: Quick feedback loop

### Use UMAP When

✅ **Quality matters most**: Worth the extra time
✅ **Small-medium datasets**: < 10,000 documents
✅ **Research/exploration**: Finding subtle topics
✅ **Complex data**: Suspected non-linear structure
✅ **Visualization**: 2D/3D projections for humans

---

## SwiftTopics Defaults and Recommendations

### Default: PCA

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

public static let `default` = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .pca  // ← Default
    ),
    // ...
)
```

**Why PCA is default:**
- Works well for typical topic modeling
- Fast enough for real-time use
- Deterministic (easier debugging)
- HDBSCAN handles non-linearity in reduced space

### Quality Preset Uses PCA Too

```swift
public static let quality = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 25,  // More dimensions for quality
        method: .pca          // Still PCA (speed + determinism)
    ),
    clustering: HDBSCANConfiguration(
        minClusterSize: 10,
        clusterSelectionMethod: .eom
    ),
    // ...
)
```

### When to Switch to UMAP

```swift
// For maximum topic quality on smaller datasets
let experimentalConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .umap,
        umapConfig: UMAPConfiguration(
            nNeighbors: 15,
            minDist: 0.1,
            nEpochs: 200
        )
    ),
    clustering: HDBSCANConfiguration(minClusterSize: 5),
    seed: 42  // Important for reproducibility!
)
```

---

## The Hybrid Approach

For production systems, consider:

```swift
// Use PCA for real-time
let realtimeModel = TopicModel(
    configuration: .default  // PCA
)

// Use UMAP for nightly batch refresh
let batchModel = TopicModel(
    configuration: TopicModelConfiguration(
        reduction: ReductionConfiguration(
            outputDimension: 15,
            method: .umap,
            umapConfig: .default
        ),
        // ...
        seed: 42
    )
)
```

This gives you:
- Fast real-time topic assignment (PCA model)
- High-quality topic discovery (UMAP model, run overnight)

---

## Empirical Comparison

### Test Setup

```swift
// Same dataset, same clustering, different reduction
let documents = loadJournalEntries()  // 5,000 entries
let embeddings = embedAll(documents)  // 768D

let pcaConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
    clustering: HDBSCANConfiguration(minClusterSize: 5),
    coherence: .default,
    seed: 42
)

let umapConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 15, method: .umap),
    clustering: HDBSCANConfiguration(minClusterSize: 5),
    coherence: .default,
    seed: 42
)
```

### Results

```
PCA Results:
  Time: 1.8 seconds
  Topics found: 9
  Coherence: 0.21
  Outlier rate: 18%
  Top topics:
    - Topic 0: fitness, running, exercise, health
    - Topic 1: swift, code, programming, async
    - Topic 2: anxiety, stress, feeling, mental
    ...

UMAP Results:
  Time: 28.4 seconds
  Topics found: 11
  Coherence: 0.26
  Outlier rate: 14%
  Top topics:
    - Topic 0: fitness, running, marathon, training
    - Topic 1: swift, async, actor, concurrency
    - Topic 2: anxiety, therapy, coping, mindfulness
    ...

UMAP found more refined topics (e.g., separated "swift/async" from general coding)
but took 15× longer.
```

---

## Tuning UMAP for Topic Modeling

If you choose UMAP, these settings work well:

```swift
let optimizedUMAP = UMAPConfiguration(
    nNeighbors: 15,    // Balance local/global (10-30 typical)
    minDist: 0.0,      // Tight clusters for HDBSCAN (0.0-0.1)
    nEpochs: 200,      // Enough for convergence (150-300)
    metric: .euclidean // Match HDBSCAN's metric
)
```

### Parameter Effects

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `nNeighbors` | More local, may split topics | More global, may merge topics |
| `minDist` | Tighter clusters, more outliers | Spread out, fewer outliers |
| `nEpochs` | Faster, possibly unconverged | Slower, better quality |

---

## ⚠️ Common Pitfalls

### Pitfall 1: Using UMAP Without Seed in Production

```swift
// ⚠️ Topics change between runs
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .umap),
    seed: nil  // No seed!
)

// ✅ Always set seed for UMAP
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .umap),
    seed: 42  // Reproducible
)
```

### Pitfall 2: UMAP on Large Datasets

```swift
// ⚠️ 100,000 documents with UMAP
// Expected time: 10+ minutes
// Memory: Several GB

// ✅ Use PCA for large datasets, or sample
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .pca),
    // ...
)
```

### Pitfall 3: Over-optimizing Reduction

```swift
// ⚠️ Spending hours tuning UMAP parameters
// when PCA gets 95% of the quality

// ✓ Start with PCA defaults
// ✓ Only switch to UMAP if topics are poor
// ✓ Often the issue is clustering config, not reduction
```

---

## Key Takeaways

1. **PCA is the safe default**: Fast, deterministic, works well for most cases.

2. **UMAP for quality**: Better topic separation, but slower and stochastic.

3. **Dataset size matters**: PCA for > 20K docs; UMAP for < 10K.

4. **Always seed UMAP**: For reproducibility in production.

5. **The hybrid approach**: PCA for real-time, UMAP for batch quality.

6. **Don't over-optimize**: Start with defaults; tune only if needed.

---

## 💡 Key Insight

The reduction method is **less important than you might think** for topic modeling. HDBSCAN is quite robust, and the difference between PCA and UMAP often comes down to 1-2 extra topics and slightly better coherence.

```
Time spent tuning reduction: diminishing returns

Better uses of that time:
  - Tuning HDBSCAN parameters (minClusterSize)
  - Improving embedding quality
  - Curating document content
```

---

## Chapter Summary

In this chapter, you learned:

1. **Why reduction is necessary**: The curse of dimensionality
2. **PCA**: Linear, fast, finds maximum variance
3. **UMAP**: Non-linear, slower, preserves neighborhoods
4. **When to use each**: Speed/reproducibility → PCA; Quality → UMAP

With dimensionality reduction understood, we're ready for the heart of topic modeling: **HDBSCAN clustering**.

---

## Next Chapter

**[→ Chapter 3: Density-Based Clustering](../03-Density-Based-Clustering/README.md)**

---

*Guide 2.4 of 2.4 • Chapter 2: Dimensionality Reduction*
