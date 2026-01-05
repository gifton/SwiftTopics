# Chapter 3: Density-Based Clustering

> **HDBSCAN—finding clusters by density, not by counting.**

Chapter 2 taught us how to compress embeddings from 768D to 15D. Now those reduced vectors are ready for **clustering**—grouping similar documents together.

But not all clustering algorithms are equal. This chapter explains why HDBSCAN is the right choice for topic modeling.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [3.1 Why Not K-Means](./01-Why-Not-KMeans.md) | Limitations of centroid clustering | Fixed K, spherical assumption, forced assignment |
| [3.2 DBSCAN Foundation](./02-DBSCAN-Foundation.md) | Density-based clustering basics | Core points, border points, noise, epsilon |
| [3.3 HDBSCAN Hierarchy](./03-HDBSCAN-Hierarchy.md) | Hierarchical density clustering | Multi-scale, stability, dendrogram |
| [3.4 Mutual Reachability](./04-Mutual-Reachability.md) | Density-aware distance | Core distance, MST construction |
| [3.5 Cluster Extraction](./05-Cluster-Extraction.md) | EOM vs Leaf selection | Stability scores, condensed tree |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- Why K-Means fails for topic modeling (and when it's appropriate)
- How density-based clustering discovers natural groupings
- The HDBSCAN algorithm step-by-step
- How mutual reachability handles varying-density data
- How cluster extraction chooses the "best" clusters
- How to tune `minClusterSize` and `minSamples`

---

## The Core Problem

Consider trying to find topics in 1,000 journal entries:

```
K-Means approach:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  "I want exactly 8 topics"                                              │
│                                                                         │
│       ●●●●●                    ●●                                       │
│      ●●●●●●●                  ●●●●                                      │
│       ●●●●●                    ●●                                       │
│                                        ●●●●●●●●                         │
│              ●●●●●●●●●●              ●●●●●●●●●●                         │
│             ●●●●●●●●●●●●              ●●●●●●●●                          │
│              ●●●●●●●●●●                                                 │
│  ●                                            ●                ●        │
│                                                                         │
│  K-Means: "I'll force 8 centroids, even if the data has 4 topics."     │
│  Outliers: "Sorry, you belong to SOME cluster, even if you're noise."  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

HDBSCAN approach:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  "Find natural groupings"                                               │
│                                                                         │
│       ●●●●●                    ●●                                       │
│      [Cluster 0]             [Cluster 1]                                │
│       ●●●●●                    ●●                                       │
│                                        ●●●●●●●●                         │
│              ●●●●●●●●●●              [Cluster 2]                        │
│             [Cluster 3]               ●●●●●●●●                          │
│              ●●●●●●●●●●                                                 │
│  ✗                                            ✗                ✗       │
│  [Noise]                                  [Noise]          [Noise]     │
│                                                                         │
│  HDBSCAN: "I found 4 dense regions. The sparse points are outliers."   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why HDBSCAN for Topic Modeling?

### 1. No Predefined K

You don't know how many topics your journal has. 5? 12? 47?

```swift
// K-Means: You must guess
let kmeans = KMeans(k: 10)  // What if the real answer is 7?

// HDBSCAN: The data tells you
let hdbscan = HDBSCANEngine(
    configuration: HDBSCANConfiguration(minClusterSize: 5)
)
let result = try await hdbscan.fit(embeddings)
print("Found \(result.clusterCount) topics")  // Natural discovery
```

### 2. Outlier Detection

Not every document belongs to a topic. Some are unique or transitional.

```swift
// K-Means forces assignment
// "Dear diary, I met a talking dolphin today" → Topic: Fitness?

// HDBSCAN marks as noise
result.label(for: weirdEntry)  // Returns -1 (outlier)
```

### 3. Variable Density

Topic sizes vary. A journaler might have 500 entries about "work" and 20 about "astronomy."

```
K-Means: Splits large topics, merges small ones
         (Biased toward equal-sized clusters)

HDBSCAN: Finds both dense and sparse clusters
         (Respects natural topic sizes)
```

### 4. Arbitrary Shapes

Topics aren't spherical in embedding space. They curve, elongate, branch.

```
Embedding space:

K-Means assumption:      Reality:
    ●●●●                    ●●●●●●●●
   ●●●●●●                  ●●●●●●●●●●
    ●●●●                    ●●●●●●●●●●●
  (Spherical)               ●●●●●●●●●●●●
                             ●●●●●●●●●●
                              ●●●●●●
                             (Elongated manifold)

K-Means splits elongated clusters.
HDBSCAN follows the density.
```

---

## SwiftTopics HDBSCAN Configuration

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

public struct HDBSCANConfiguration: Sendable, Codable {
    /// Minimum size for a cluster to be considered valid.
    /// Smaller = more fine-grained topics.
    /// Larger = broader topics.
    public let minClusterSize: Int  // Default: 5

    /// Minimum samples to define a core point.
    /// Controls density threshold.
    /// Defaults to minClusterSize if nil.
    public let minSamples: Int?

    /// Cluster selection method.
    /// .eom: Excess of Mass (balanced)
    /// .leaf: All leaf clusters (fine-grained)
    public let clusterSelectionMethod: ClusterSelectionMethod  // Default: .eom

    /// Whether to allow a single cluster result.
    public let allowSingleCluster: Bool  // Default: false

    /// Distance metric for clustering.
    public let metric: DistanceMetricType  // Default: .euclidean
}
```

---

## The HDBSCAN Pipeline

SwiftTopics implements HDBSCAN in five stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HDBSCAN Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. CORE DISTANCE                                                       │
│     For each point, find distance to k-th nearest neighbor              │
│     📍 CoreDistance.swift                                               │
│                                                                         │
│  2. MUTUAL REACHABILITY                                                 │
│     Transform distances to be density-aware                             │
│     📍 MutualReachability.swift                                         │
│                                                                         │
│  3. MINIMUM SPANNING TREE                                               │
│     Build MST on mutual reachability graph                              │
│     📍 MinimumSpanningTree.swift                                        │
│                                                                         │
│  4. CLUSTER HIERARCHY                                                   │
│     Convert MST to dendrogram with stability scores                     │
│     📍 ClusterHierarchyBuilder.swift                                    │
│                                                                         │
│  5. CLUSTER EXTRACTION                                                  │
│     Select stable clusters via EOM or leaf method                       │
│     📍 ClusterExtraction.swift                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Each guide in this chapter covers one or more of these stages.

---

## Prerequisites Check

Before starting this chapter, ensure you understand:

- [ ] Why dimensionality reduction is needed (Chapter 2)
- [ ] Euclidean distance between vectors
- [ ] The concept of neighborhoods in metric spaces
- [ ] Tree data structures (for understanding hierarchy)

---

## Start Here

**[→ 3.1 Why Not K-Means](./01-Why-Not-KMeans.md)**

---

*Chapter 3 of 6 • SwiftTopics Learning Guide*
