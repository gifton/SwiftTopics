# 3.1 Why Not K-Means

> **The most popular clustering algorithm—and why it fails for topic modeling.**

---

## The Concept

**K-Means** is the most widely known clustering algorithm. It's simple, fast, and works well for many problems. So why doesn't SwiftTopics use it?

K-Means has four fundamental assumptions that clash with topic modeling:

1. **You know K beforehand** — The number of clusters must be specified
2. **Clusters are spherical** — Points group around centroids in balls
3. **Clusters are similar size** — Equal variance assumed
4. **Every point belongs to a cluster** — No concept of noise/outliers

Topic modeling violates all four.

---

## Why It Matters

Understanding K-Means' limitations clarifies what HDBSCAN provides and when K-Means might still be appropriate.

### The K Problem

```
Real journals have unknown topic counts:

Journal A: 3 topics (work, fitness, family)
Journal B: 12 topics (varied interests)
Journal C: 47 topics (20 years of entries)

K-Means requires:
  KMeans(k: ???)  // What value?

Options:
  1. Guess → Wrong clusters
  2. Try many K values → Slow, still guessing
  3. Use heuristics (elbow method) → Often misleading
```

### The Shape Problem

```
Topics in embedding space aren't spherical:

K-Means assumes:         Topics actually look like:

     ○ ○ ○                    ●●●●●●●●●●
    ○ ● ○ ○                  ●●●●●●●●●●●●
     ○ ○ ○                    ●●●●●●●●●●●●●
                               ●●●●●●●●●●
  (Spherical around             (Elongated manifold
   centroid ●)                   following semantic
                                 similarity)

The "anxiety" topic might curve through embedding space,
not form a neat ball around a center.
```

### The Size Problem

```
Real topics have varying sizes:

[==========] "Work" - 500 entries
[====] "Family" - 200 entries
[==] "Travel" - 100 entries
[=] "Astronomy" - 15 entries

K-Means tends toward equal-sized clusters:

Original:                K-Means result:
[==========]            [======]  ← Split Work
[====]                  [====]
[==]                    [====]  ← Merged Travel+Astronomy
[=]                     [====]  ← Partial Work
```

### The Noise Problem

```
Some entries don't belong anywhere:

"Today I discovered that my neighbor
 is actually three raccoons in a trench coat."

This is:
- Not fitness
- Not work
- Not family
- Not any typical topic

K-Means: Forces into nearest cluster (wrong!)
HDBSCAN: Marks as noise (-1)
```

---

## The Mathematics

### K-Means Objective

K-Means minimizes the **within-cluster sum of squares** (WCSS):

```
WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

Where:
  k = cluster index (1 to K)
  Cₖ = set of points in cluster k
  xᵢ = point i
  μₖ = centroid of cluster k

This is minimized when points are close to their cluster centroid.
```

### The Algorithm

```
K-Means Algorithm:
─────────────────────────────────────────────

1. Choose K random points as initial centroids

2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Recompute centroids as mean of assigned points

3. Return assignments and final centroids

Complexity: O(n × K × d × iterations)
Where: n = points, K = clusters, d = dimensions
```

### Why Spherical?

The distance to centroid metric creates **Voronoi regions**:

```
K=3 centroids (marked ●):

        │       ╱
        │      ╱
     C0 │ C1  ╱
        │    ╱  C2
        │   ╱
────────●──●────────
        │ ╱
        │╱
       ╱│
      ╱ │

Each region contains points closest to that centroid.
Boundaries are straight lines (hyperplanes in high-D).

This creates convex, approximately spherical regions.
Non-spherical clusters get split by these boundaries.
```

---

## The Technique: Comparing Behaviors

### Scenario 1: Natural Spherical Clusters

```
Data with 3 well-separated spherical clusters:

      ●●●                    ●●●
     ●●●●●                  ●●●●●
      ●●●        →          ●●●
                            (Cluster 0)
          ●●●●●                  ●●●●●
         ●●●●●●●               ●●●●●●●
          ●●●●●                  ●●●●●
                               (Cluster 1)
                  ●●●●                  ●●●●
                 ●●●●●●               ●●●●●●
                  ●●●●                  ●●●●
                                      (Cluster 2)

K-Means: Works well! ✓
HDBSCAN: Also works well ✓
```

### Scenario 2: Elongated Cluster

```
Data with one elongated cluster:

    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

K-Means (K=1):
    ●●●●●●●●●●●●●●●●○●●●●●●●●●●●●●●●●●●●●●●●
                   (Single centroid in middle)
    Works! ✓

K-Means (K=2):
    ●●●●●●●●●●○●●●●│●●●●●●○●●●●●●●●●●●●●●●●●
    (Cluster 0)    │     (Cluster 1)
    Splits the natural cluster ✗

K-Means (K=3):
    ●●●●●●○●●●│●●●●○●●●●│●●●●○●●●●●●●●●●●●●●
    (C0)      │   (C1)  │       (C2)
    Further fragmentation ✗

HDBSCAN:
    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
    (All one cluster - follows density) ✓
```

### Scenario 3: Varying Density

```
Dense region + sparse region:

    ●●●●●●●●●●●●              ●        ●
    ●●●●●●●●●●●●●
    ●●●●●●●●●●●●           ●    ●  ●
    (200 points)               (10 points)

K-Means (K=2):
    [=====●=====]           [    ●    ]
    Centroid here           Centroid here
    Works for dense,        Sparse points forced
    but...                  into a "cluster"

What if sparse points are actually 3 different topics?
K-Means merges them.

HDBSCAN:
    [Dense cluster]         ✗ ✗ ✗ ✗ ✗
    Natural grouping        Marked as noise

Or with lower minClusterSize:
    [Dense cluster]         [Small cluster 1]
                           [Small cluster 2]
                           [Small cluster 3]
```

### Scenario 4: Unknown K

```
You have 1000 journal entries. How many topics?

Approach 1: Elbow Method
─────────────────────────
Run K-Means for K = 2, 3, 4, ..., 30
Plot WCSS vs K
Look for "elbow" where improvement slows

    WCSS
      │╲
      │ ╲
      │  ╲
      │   ╲__________     ← Elbow around K=5?
      │              ─────
      └────────────────── K
         5   10  15  20

Problem: Often ambiguous. No clear elbow.

Approach 2: Silhouette Score
────────────────────────────
Measure cluster quality for each K
Pick K with highest silhouette score

Problem: Computationally expensive. Still heuristic.

HDBSCAN Approach:
─────────────────
let result = try await hdbscan.fit(embeddings)
print(result.clusterCount)  // Just get the answer

No iteration. Clusters emerge from data structure.
```

---

## In SwiftTopics

SwiftTopics doesn't use K-Means for topic discovery. Instead:

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift

public actor HDBSCANEngine: ClusteringEngine {
    /// Clusters the given embeddings using HDBSCAN.
    ///
    /// - Parameter embeddings: The embeddings to cluster.
    /// - Returns: Cluster assignments for each embedding.
    public func fit(_ embeddings: [Embedding]) async throws -> ClusterAssignment {
        // 1. Compute core distances
        // 2. Build mutual reachability graph
        // 3. Construct minimum spanning tree
        // 4. Build cluster hierarchy
        // 5. Extract stable clusters
        //
        // No K required. Cluster count emerges.
    }
}
```

### The ClusterAssignment Result

```swift
// 📍 See: Sources/SwiftTopics/Clustering/ClusterAssignment.swift

public struct ClusterAssignment: Sendable {
    /// Cluster labels for each point. -1 indicates noise.
    public let labels: [Int]

    /// Membership probability for each point.
    public let probabilities: [Float]

    /// Outlier score for each point (higher = more outlier-like).
    public let outlierScores: [Float]

    /// Number of clusters found (excluding noise).
    public let clusterCount: Int
}

// Usage:
let result = try await hdbscan.fit(embeddings)

print("Found \(result.clusterCount) topics")

for i in 0..<embeddings.count {
    let label = result.label(for: i)
    if label == -1 {
        print("Document \(i) is an outlier")
    } else {
        print("Document \(i) belongs to topic \(label)")
    }
}
```

---

## When K-Means IS Appropriate

K-Means isn't wrong—it's just wrong for **topic discovery**. Use K-Means when:

### 1. K Is Known

```
Scenario: Categorizing products into predetermined departments

Departments = ["Electronics", "Clothing", "Home", "Food", "Sports"]
K = 5 (known beforehand)

K-Means works well here.
```

### 2. Speed Is Critical

```
K-Means: O(n × K × d × iterations)  — Very fast
HDBSCAN: O(n² log n) or O(n × k) with GPU — Slower

For millions of points where K is acceptable,
K-Means is often the practical choice.
```

### 3. Downstream Requires Fixed K

```
Scenario: Training a classifier with 10 categories

You need exactly 10 cluster centroids as class prototypes.
K-Means guarantees exactly K outputs.
```

### 4. Data Is Well-Behaved

```
When your data actually has:
- Similar-sized clusters
- Roughly spherical distributions
- No significant outliers
- Known number of groups

K-Means is simpler and faster.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Using K-Means for Exploration

```swift
// ❌ WRONG: Using K-Means to discover topics
let kmeans = KMeans(k: 10)  // Why 10? Arbitrary!
let topics = kmeans.fit(embeddings)

// You might have 7 natural topics, but you forced 10.
// Or you might have 30 natural topics, but you limited to 10.
```

### Pitfall 2: Ignoring the Elbow

```swift
// ⚠️ MISLEADING: Blind trust in elbow method
for k in 2...30 {
    let wcss = kmeans(embeddings, k: k).wcss
    print("K=\(k): WCSS=\(wcss)")
}
// Looking for an "elbow" that may not exist
```

### Pitfall 3: Assuming K-Means Outliers Are Valid

```swift
// ❌ WRONG: Treating all K-Means assignments as meaningful
let labels = kmeans.fit(embeddings).labels
// Every point has a label. No outlier detection.
// That weird entry about raccoons? Forced into "Family" topic.
```

### Pitfall 4: K-Means on High-Dimensional Data

```swift
// ⚠️ PROBLEMATIC: K-Means on 768D embeddings
let labels = kmeans.fit(rawEmbeddings)  // 768 dimensions!

// Distance concentration makes all points equidistant.
// Centroids become meaningless.
// Always reduce dimensions first!

// ✅ CORRECT: Reduce first
let reduced = try await pca.transform(rawEmbeddings)  // 768D → 15D
let labels = clustering.fit(reduced)
```

---

## Key Takeaways

1. **K-Means requires K**: You must specify cluster count upfront.

2. **K-Means assumes spherical clusters**: Real topics are irregularly shaped.

3. **K-Means forces assignment**: Every point belongs somewhere—no outlier concept.

4. **K-Means prefers equal sizes**: Biased toward similar-sized clusters.

5. **K-Means is still useful**: When K is known, speed matters, and data is well-behaved.

6. **HDBSCAN addresses all these issues**: Discovers K, handles shapes, marks outliers.

---

## 💡 Key Insight

K-Means is a **partitioning** algorithm—it divides space into K regions. HDBSCAN is a **density** algorithm—it finds regions where points are densely packed.

```
K-Means asks: "How do I best split this space into K pieces?"
HDBSCAN asks: "Where are the dense regions in this space?"

For topic modeling, the second question is the right one.
Your journal entries naturally cluster around themes.
You don't need to decide how many themes beforehand.
```

---

## Next Up

Now that we understand why density-based clustering, let's learn the foundation: **DBSCAN**.

**[→ 3.2 DBSCAN Foundation](./02-DBSCAN-Foundation.md)**

---

*Guide 3.1 of 3.5 • Chapter 3: Density-Based Clustering*
