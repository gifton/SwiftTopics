# 3.3 HDBSCAN Hierarchy

> **Building a dendrogram of clusters across all density levels.**

---

## The Concept

HDBSCAN's key insight: instead of picking **one** density threshold (ε), consider **all** thresholds simultaneously.

```
DBSCAN: "Find clusters at density level ε = 0.5"
        Single snapshot at one density.

HDBSCAN: "Find clusters at ALL density levels, then pick the stable ones"
         Movie of clusters forming and merging.
```

This creates a **hierarchy** (dendrogram) showing how clusters relate across densities:

```
Density (λ = 1/distance):

    High λ                   ●   ●   ●   ●   ●   ●   ●   ●
    (Dense)                   \ / \ /     \ / \ /
                               ●   ●       ●   ●
                                \ /         \ /
                                 ●           ●
                                  \         /
    Low λ                          \       /
    (Sparse)                        ───●───
                                       │
                              (Single cluster at λ=0)

Reading top-to-bottom:
- At high density, many small clusters
- As density decreases, clusters merge
- At very low density, everything is one cluster
```

---

## Why It Matters

The hierarchy solves DBSCAN's variable-density problem:

```
Data with two different densities:

    Dense cluster          Sparse cluster
    ●●●●●●●●●●●           ●     ●
    ●●●●●●●●●●●●             ●
    ●●●●●●●●●●●           ●     ●

DBSCAN at ε=0.2:
    [Dense cluster] ✓      [Noise] ✗

DBSCAN at ε=1.5:
    [Everything merged into one cluster] ✗

HDBSCAN:
    Sees dense cluster BORN at λ=5.0, DIES at λ=0.5
    Sees sparse cluster BORN at λ=0.7, DIES at λ=0.3

    Both clusters are found and kept separate!
```

---

## The Mathematics

### Lambda Space (Density Space)

HDBSCAN works in **lambda space** where λ = 1/distance:

```
λ = 1/distance

distance → λ
   0.1   → 10.0  (Very dense - close together)
   0.5   → 2.0   (Moderately dense)
   1.0   → 1.0
   2.0   → 0.5   (Sparse - far apart)
   10.0  → 0.1   (Very sparse)

Higher λ = higher density = points closer together
```

### Cluster Lifecycle

Each cluster has a **birth** and **death** in lambda space:

```
                    λ (density)
                        ↑
    λ_birth ────────────┼──────────────────── Cluster BORN
                        │                     (First appears at this density)
                        │  ████████████████
                        │  █ Cluster C  ██
                        │  ████████████████
    λ_death ────────────┼──────────────────── Cluster DIES
                        │                     (Merges with another cluster)
                        ↓

Lifespan of C = λ_birth - λ_death

Longer lifespan = more "stable" cluster
(Persists across many density levels)
```

### Stability Score

The stability of a cluster measures its persistence:

```
stability(C) = Σ (λ_death(p) - λ_birth(C))  for all points p in C

Where:
  λ_birth(C) = density level where cluster C first appeared
  λ_death(p) = density level where point p left cluster C (or cluster died)

Intuition:
  - Each point contributes (how long it was in the cluster)
  - Large clusters with long lifespans have high stability
  - Small or fleeting clusters have low stability
```

---

## The Technique: Building the Hierarchy

### Step 1: Sort Edges by Distance

After computing mutual reachability distances (next guide), we have a weighted graph. Sort edges:

```
MST edges (sorted by weight/distance):

Edge      Weight (distance)    λ = 1/weight
─────────────────────────────────────────────
(A,B)          0.1              10.0
(C,D)          0.2               5.0
(B,E)          0.3               3.3
(D,F)          0.4               2.5
(E,G)          0.8               1.25
(A,C)          1.5               0.67
(F,H)          3.0               0.33
```

### Step 2: Process Edges in Order

Starting with each point in its own cluster, process edges:

```
Initial state (λ = ∞): Each point is its own cluster
─────────────────────────────────────────────────────

   A    B    C    D    E    F    G    H
   ●    ●    ●    ●    ●    ●    ●    ●
  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]

Process edge (A,B) at λ = 10.0:
─────────────────────────────────────────────────────

   A────B    C    D    E    F    G    H
    [8]     [2]  [3]  [4]  [5]  [6]  [7]

   Cluster 8 is BORN (merging 0 and 1)
   Birth level: λ = 10.0

Process edge (C,D) at λ = 5.0:
─────────────────────────────────────────────────────

   A────B    C────D    E    F    G    H
    [8]       [9]     [4]  [5]  [6]  [7]

   Cluster 9 is BORN

Process edge (B,E) at λ = 3.3:
─────────────────────────────────────────────────────

   A────B────E    C────D    F    G    H
       [10]         [9]    [5]  [6]  [7]

   Cluster 10 is BORN (merging 8 and 4)
   Cluster 8 DIES at λ = 3.3

... Continue until all points are in one cluster ...

Final state (λ → 0):
─────────────────────────────────────────────────────

                    [Root]
                   ╱      ╲
                  ╱        ╲
               [11]        [H]
              ╱    ╲
            [10]   [9]
           ╱  ╲   ╱  ╲
         [8] [E] [C] [D]
        ╱  ╲
       [A] [B]
```

### Step 3: Build the Dendrogram

The resulting tree (dendrogram) shows all cluster relationships:

```
                          λ (density)
                             │
    10.0 ────────────────────┼──── [8]: {A,B} born
                             │
     5.0 ────────────────────┼──── [9]: {C,D} born
                             │
     3.3 ────────────────────┼──── [8] dies → [10]: {A,B,E}
                             │
     2.5 ────────────────────┼──── [9] dies → [11]: {C,D,F}
                             │
     1.25 ───────────────────┼──── [10] and [11] merge → [12]
                             │
     0.67 ───────────────────┼──── [12] absorbs G
                             │
     0.33 ───────────────────┼──── [13]: All points
                             │
     0.0 ────────────────────┴────

Dendrogram view (rotated):

                [13: Root]
               ╱          ╲
           [12]            [H]
          ╱    ╲
      [10]      [11]
     ╱  |       |  ╲
  [8]  [E]    [9]  [F]
 ╱  ╲        ╱  ╲
[A] [B]    [C] [D]
```

---

## In SwiftTopics

### The Cluster Hierarchy

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchyBuilder.swift

/// A node in the cluster hierarchy.
public struct ClusterHierarchyNode: Sendable, Identifiable {
    /// Unique identifier (points: 0..<n, internal: n..<2n-1)
    public let id: Int

    /// Parent node ID (nil for root)
    public let parent: Int?

    /// Child node IDs (empty for leaves/points)
    public let children: [Int]

    /// Birth level (distance at which cluster formed)
    public let birthLevel: Float

    /// Death level (distance at which cluster merged into parent)
    public let deathLevel: Float

    /// Number of points in this cluster
    public let size: Int

    /// Stability score
    public let stability: Float
}
```

### Building the Hierarchy

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchyBuilder.swift

public struct ClusterHierarchyBuilder: Sendable {
    /// Minimum cluster size for stability computation.
    public let minClusterSize: Int

    /// Builds the cluster hierarchy from an MST.
    public func build(
        from mst: MinimumSpanningTree,
        allowSingleCluster: Bool = false
    ) -> ClusterHierarchy {
        let sortedEdges = mst.sortedEdges  // Ascending by weight

        var clusterState = HierarchyBuildState(pointCount: mst.pointCount)

        // Process edges in order (low distance = high density first)
        for edge in sortedEdges {
            clusterState.mergePoints(
                edge.source,
                edge.target,
                atDistance: edge.weight
            )
        }

        return clusterState.finalize(
            minClusterSize: minClusterSize,
            allowSingleCluster: allowSingleCluster
        )
    }
}
```

### The Stability Calculation

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchyBuilder.swift

/// Computes stability for a single node.
///
/// stability(C) = Σ (λ_death(p) - λ_birth(C)) for points p in C
///
/// Where λ = 1/distance.
private func computeNodeStability(
    node: ClusterHierarchyNode,
    nodes: [ClusterHierarchyNode],
    nodeByID: [Int: Int]
) -> Float {
    let birthDistance = node.birthLevel
    let deathDistance = node.deathLevel

    // Convert to lambda (density) space
    // λ = 1/distance
    let lambdaBirth = birthDistance > Float.ulpOfOne
        ? 1.0 / birthDistance
        : Float.infinity
    let lambdaDeath = deathDistance > Float.ulpOfOne && deathDistance < Float.infinity
        ? 1.0 / deathDistance
        : 0

    // Collect all leaf descendants
    let leaves = collectLeafDescendants(nodeID: node.id, ...)

    var stability: Float = 0
    for (_, leafDeathDistance) in leaves {
        let leafLambdaDeath = leafDeathDistance > Float.ulpOfOne
            ? 1.0 / leafDeathDistance
            : 0

        // Contribution = (λ_death - λ_birth) for this point's membership
        let contribution = max(0, lambdaBirth - max(leafLambdaDeath, lambdaDeath))
        stability += contribution
    }

    return stability
}
```

---

## Visualizing Stability

```
Two potential clusterings from the same hierarchy:

Option A: Select parent cluster [10]
─────────────────────────────────────
           [10]
          ╱    ╲
       [8]      [9]
       (died)   (died)

   Stability([10]) = 0.8
   (Large cluster, long lifespan)

Option B: Select children [8] and [9]
─────────────────────────────────────
        [8]      [9]
       (kept)   (kept)

   Stability([8]) + Stability([9]) = 0.3 + 0.4 = 0.7
   (Two smaller clusters, shorter lifespans)

If Stability([10]) > Stability([8]) + Stability([9]):
   → Select the parent [10]

If Stability([10]) < Stability([8]) + Stability([9]):
   → Select the children [8] and [9]
```

---

## The Condensed Tree

HDBSCAN filters out small clusters to create a **condensed tree**:

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterHierarchyBuilder.swift

/// A condensed view of the cluster tree for cluster extraction.
///
/// The condensed tree removes clusters smaller than minClusterSize,
/// keeping only significant clusters for selection.
public struct CondensedTree: Sendable {
    /// Nodes in the condensed tree.
    public let nodes: [CondensedTreeNode]

    /// Root node ID.
    public let rootID: Int
}

// Before condensing (minClusterSize = 5):
//
//     [Root: 100 pts]
//    ╱      |      ╲
// [40 pts] [30] [30]      [Noise: various small]
//           |      ╲
//         [15]     [15]
//           |
//          [8]
//
// After condensing:
//
//     [Root: 100 pts]
//    ╱      |      ╲
// [40 pts] [30] [30]
//           |      ╲
//         [15]     [15]
//           |
//          [8]
//
// Clusters < 5 points removed from consideration
```

---

## Why Hierarchy Beats Flat Clustering

### Advantage 1: Multi-Resolution

```
Your journal might have:
  - Broad topic: "Health" (200 entries)
  - Sub-topic: "Fitness" (120 entries)
  - Sub-sub-topic: "Running" (50 entries)

Hierarchy captures all levels:

        [Health: 200]
       ╱      |      ╲
  [Fitness] [Mental] [Diet]
      │
  [Running]
      │
  [Marathon Training]

You can extract at any granularity!
```

### Advantage 2: Stable Selection

```
Flat clustering is binary: in or out.
Hierarchical clustering measures confidence.

High stability cluster:
  - Born early (high density)
  - Died late (persisted long)
  - Many points
  → Confident this is a real topic

Low stability cluster:
  - Born late
  - Died quickly
  - Few points
  → Might just be noise
```

### Advantage 3: Outlier Quality

```
Outliers have context in the hierarchy:

Point X is noise because:
  - It never joined a stable cluster
  - It briefly joined [ClusterY] but [ClusterY] was unstable
  - It's far from all dense regions

You can inspect WHY something is an outlier.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Ignoring minClusterSize

```swift
// ⚠️ TOO SMALL: Every noise point becomes a "cluster"
let config = HDBSCANConfiguration(minClusterSize: 2)
// Result: 50 "clusters", most with 2-3 points

// ⚠️ TOO LARGE: Misses real small topics
let config = HDBSCANConfiguration(minClusterSize: 50)
// Result: 3 clusters, everything else is noise

// ✅ BALANCED: Tune based on expected topic size
let config = HDBSCANConfiguration(minClusterSize: 5)
// 5-10 is usually a good starting point
```

### Pitfall 2: Expecting Perfect Nesting

```swift
// ⚠️ WRONG: Assuming hierarchy = topic taxonomy

// The hierarchy shows DENSITY relationships, not SEMANTIC relationships.
// "Running" might merge with "Diet" before "Fitness" if their embeddings
// happen to have overlapping dense regions.

// Don't interpret hierarchy as a topic taxonomy!
```

### Pitfall 3: Forgetting About Lambda Space

```swift
// ⚠️ CONFUSING: Thinking in distance when stability uses λ

// Low distance = high λ (dense, born early)
// High distance = low λ (sparse, born late)

// A cluster "born" at distance 0.1 is born at λ = 10 (dense)
// A cluster "born" at distance 2.0 is born at λ = 0.5 (sparse)
```

---

## Key Takeaways

1. **Hierarchy captures all densities**: No need to pick one ε value.

2. **Lambda space measures density**: λ = 1/distance; higher λ = denser.

3. **Clusters have lifecycles**: Birth (appear), death (merge), lifespan (stability).

4. **Stability scores matter**: Longer-lived clusters are more reliable.

5. **Condensed tree filters noise**: Small clusters removed from consideration.

6. **Multi-resolution output**: Can extract at different granularities.

---

## 💡 Key Insight

The hierarchy is HDBSCAN's secret weapon. Instead of asking "what clusters exist at density X?", it asks "what clusters exist **across all densities** and which are most stable?"

```
DBSCAN: Snapshot photography (one moment in time)
HDBSCAN: Time-lapse video (all moments, find the stable patterns)

Stable patterns = real clusters
Fleeting patterns = noise
```

---

## Next Up

We've seen the big picture. Now let's understand the key enabler: **mutual reachability distance**, which makes the hierarchy handle varying densities.

**[→ 3.4 Mutual Reachability](./04-Mutual-Reachability.md)**

---

*Guide 3.3 of 3.5 • Chapter 3: Density-Based Clustering*
