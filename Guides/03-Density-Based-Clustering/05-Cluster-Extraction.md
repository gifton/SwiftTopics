# 3.5 Cluster Extraction

> **Choosing the optimal flat clustering from the infinite hierarchy.**

---

## The Concept

The hierarchy contains **every possible clustering** at every density level. But we want a **flat** result: each document in exactly one cluster (or marked as noise).

Cluster extraction answers: **Which clusters should we select?**

```
The hierarchy contains many valid clusterings:

                    [Root]
                   ╱      ╲
               [A]          [B]
              ╱   ╲        ╱   ╲
           [C]    [D]   [E]    [F]

Option 1: Select Root         → 1 cluster (everything)
Option 2: Select A, B         → 2 clusters
Option 3: Select C, D, E, F   → 4 clusters
Option 4: Select A, E, F      → 3 clusters (mixed levels)

Which is best?
```

---

## Why It Matters

The extraction method determines your topic granularity:

```
Too coarse (select high in tree):
  Topic 0: "Everything about health, fitness, diet, mental wellness, sleep"
  → Not useful for discovery

Too fine (select low in tree):
  Topic 0: "Running on Tuesdays"
  Topic 1: "Running on Thursdays"
  Topic 2: "Marathon training"
  → Too fragmented

Just right (stable clusters):
  Topic 0: "Fitness and exercise"
  Topic 1: "Mental health"
  Topic 2: "Diet and nutrition"
  → Meaningful, distinct topics
```

---

## The Mathematics

### Excess of Mass (EOM)

EOM is HDBSCAN's default extraction method. It maximizes **total stability**.

```
For each node, decide: select this cluster OR its children?

             [Parent]                    [Parent]
            stability=0.8               (not selected)
                 │                           │
            ─────┴─────                 ─────┴─────
           ╱           ╲               ╱           ╲
      [Child A]    [Child B]      [Child A]    [Child B]
     stability=0.3 stability=0.4   SELECTED    SELECTED

Option A: Select Parent              Option B: Select Children
          Total = 0.8                          Total = 0.3 + 0.4 = 0.7

Since 0.8 > 0.7, EOM chooses Option A (Parent)
```

### Dynamic Programming Decision

```
For each internal node N:

subtree_stability(N) = max(
    stability(N),                           // Select this cluster
    Σ subtree_stability(child) for children // Select children instead
)

Work bottom-up from leaves to root.
At the end, walk top-down selecting winners.
```

### Algorithm

```
EOM Algorithm:
──────────────────────────────────────────────────────────

1. Process nodes BOTTOM-UP (leaves first):

   For each leaf node:
     subtree_stability = own stability
     mark as SELECTED

   For each internal node:
     children_total = sum of children's subtree_stability

     if own_stability > children_total:
       subtree_stability = own_stability
       mark as SELECTED
       mark all descendants as NOT SELECTED
     else:
       subtree_stability = children_total
       mark as NOT SELECTED (children will be selected)

2. Collect all SELECTED nodes → these are the final clusters
```

### Example Walkthrough

```
Hierarchy with stability scores:

                    [Root: 0.1]
                   ╱           ╲
            [A: 0.6]          [B: 0.4]
           ╱       ╲              |
      [C: 0.2]  [D: 0.25]     [E: 0.3]

Step 1: Process leaves (C, D, E)
  subtree[C] = 0.2 (selected)
  subtree[D] = 0.25 (selected)
  subtree[E] = 0.3 (selected)

Step 2: Process A
  children_total = subtree[C] + subtree[D] = 0.2 + 0.25 = 0.45
  own_stability = 0.6
  0.6 > 0.45 → SELECT A, DESELECT C and D
  subtree[A] = 0.6

Step 3: Process B
  children_total = subtree[E] = 0.3
  own_stability = 0.4
  0.4 > 0.3 → SELECT B, DESELECT E
  subtree[B] = 0.4

Step 4: Process Root
  children_total = subtree[A] + subtree[B] = 0.6 + 0.4 = 1.0
  own_stability = 0.1
  0.1 < 1.0 → DON'T SELECT Root, keep A and B
  subtree[Root] = 1.0

Final selection: {A, B}
Total stability: 1.0
```

---

## Leaf Clustering

An alternative to EOM is **leaf clustering**: simply select all leaf clusters.

```
Leaf clustering:

                    [Root]
                   ╱      ╲
               [A]          [B]       ← not selected
              ╱   ╲        ╱   ╲
           [C]    [D]   [E]    [F]    ← ALL selected

Result: {C, D, E, F}

Pros:
  - Maximum granularity
  - Simple to understand
  - Finds fine-grained topics

Cons:
  - May over-fragment coherent topics
  - Ignores stability (selects unstable clusters too)
  - Can produce many small clusters
```

### EOM vs Leaf

```
EOM (Excess of Mass):
─────────────────────────────────
  - Selects stable clusters at any level
  - Balances granularity and stability
  - May select parent over unstable children
  - DEFAULT for topic modeling

Leaf:
─────────────────────────────────
  - Always maximum granularity
  - Ignores stability
  - Good for exploration or when fine detail needed
  - May need post-processing to merge
```

---

## The Technique: Point Assignment

After selecting clusters, assign each point:

```
For each point:
  1. Find its path from leaf to root in hierarchy
  2. Find the first SELECTED cluster on that path
  3. Assign point to that cluster
  4. If no selected cluster found → mark as noise (-1)
```

### Example

```
Selected clusters: {A, B}

Hierarchy:
                    [Root]
                   ╱      ╲
            [A=SEL]       [B=SEL]
           ╱   ╲   ╲          |
         p1   p2   p3        [B']
                            ╱   ╲
                          p4    p5

Point p1: Path is p1 → A → Root
          First selected: A
          Assign to cluster A

Point p4: Path is p4 → B' → B → Root
          First selected: B
          Assign to cluster B

If some point p6 never enters a selected cluster:
          Assign to noise (-1)
```

---

## In SwiftTopics

### The Cluster Extractor

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift

/// Extracts flat cluster assignments from the HDBSCAN hierarchy.
public struct ClusterExtractor: Sendable {

    /// The extraction method.
    public let method: ClusterSelectionMethod

    /// Minimum cluster size.
    public let minClusterSize: Int

    /// Cluster selection epsilon (for merging small clusters).
    public let epsilon: Float

    /// Whether to allow a single cluster result.
    public let allowSingleCluster: Bool
}
```

### EOM Implementation

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift

/// Selects clusters using Excess of Mass method.
private func selectClustersEOM(
    condensedTree: CondensedTree,
    hierarchy: ClusterHierarchy,
    epsilon: Float,
    allowSingleCluster: Bool
) -> Set<Int> {

    // Build lookup maps
    var nodeByID = [Int: CondensedTreeNode]()
    for node in condensedTree.nodes {
        nodeByID[node.id] = node
    }

    // Compute subtree stability for each node (bottom-up DP)
    var subtreeStability = [Int: Float]()
    var isSelected = [Int: Bool]()

    // Process nodes in bottom-up order (leaves first)
    let sortedNodes = condensedTree.nodes.sorted { a, b in
        if a.isLeaf != b.isLeaf { return a.isLeaf }
        return a.size < b.size
    }

    for node in sortedNodes {
        if node.isLeaf {
            // Leaf nodes: subtree stability is just the node's stability
            subtreeStability[node.id] = node.stability
            isSelected[node.id] = true
        } else {
            // Internal nodes: compare own stability vs children's total
            let childrenStability = node.children.reduce(Float(0)) { sum, childID in
                sum + (subtreeStability[childID] ?? 0)
            }

            if node.stability > childrenStability {
                // Select this cluster
                subtreeStability[node.id] = node.stability
                isSelected[node.id] = true

                // Deselect all descendants
                deselect(descendants: node.children, ...)
            } else {
                // Pass through to children
                subtreeStability[node.id] = childrenStability
                isSelected[node.id] = false
            }
        }
    }

    // Collect selected clusters
    return Set(isSelected.filter { $0.value }.map { $0.key })
}
```

### Point Assignment

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift

/// Finds the selected cluster ancestor for a point.
private func findSelectedAncestor(
    pointIndex: Int,
    selectedClusterIDs: Set<Int>,
    hierarchy: ClusterHierarchy
) -> (clusterID: Int?, probability: Float) {

    // Start from the point's leaf node
    var currentID = pointIndex
    var lastValidClusterID: Int? = nil
    var depth = 0

    // Traverse up the hierarchy
    while let node = hierarchy.node(id: currentID) {
        if selectedClusterIDs.contains(currentID) {
            lastValidClusterID = currentID
        }

        guard let parentID = node.parent else { break }
        currentID = parentID
        depth += 1
    }

    // Probability based on depth (higher depth = joined later = lower prob)
    let probability = lastValidClusterID != nil
        ? max(0.1, 1.0 - Float(depth) * 0.1)
        : 0.0

    return (lastValidClusterID, probability)
}
```

### Outlier Scoring

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/ClusterExtraction.swift

/// Computes outlier scores for all points.
private func computeOutlierScores(
    pointLabels: [Int],
    coreDistances: [Float],
    ...
) -> [Float] {
    let maxCoreDistance = coreDistances.max() ?? 1.0

    var outlierScores = [Float](repeating: 0, count: pointLabels.count)

    for i in 0..<pointLabels.count {
        if pointLabels[i] == -1 {
            // Outlier: score based on core distance (normalized)
            outlierScores[i] = coreDistances[i] / maxCoreDistance
        } else {
            // Clustered point: compare to cluster average
            let clusterAvgCore = computeClusterAverageCoreDistance(...)
            let deviation = max(0, coreDistances[i] - clusterAvgCore)
            outlierScores[i] = min(1.0, deviation / maxCoreDistance)
        }
    }

    return outlierScores
}
```

---

## Configuration Options

### Cluster Selection Method

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

public enum ClusterSelectionMethod: String, Sendable, Codable {
    case eom   // Excess of Mass (default)
    case leaf  // Select all leaves
}

// Usage:
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    clusterSelectionMethod: .eom  // or .leaf
)
```

### Selection Epsilon

```swift
// For merging clusters that are very close in the hierarchy
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    clusterSelectionEpsilon: 0.1  // Merge clusters within 0.1 distance
)

// If two selected clusters have birth levels within epsilon,
// merge them into the parent cluster.
```

### Allow Single Cluster

```swift
// By default, HDBSCAN won't return just one cluster
// (usually means everything merged into one topic)

let config = HDBSCANConfiguration(
    minClusterSize: 5,
    allowSingleCluster: false  // Default: try to split into children
)

// If true, a single root cluster is a valid result
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    allowSingleCluster: true  // Allow "everything is one topic"
)
```

---

## Choosing EOM vs Leaf

### Use EOM When

```
✅ General topic modeling
  - Balanced granularity
  - Respects stability

✅ Unknown data characteristics
  - EOM adapts to the hierarchy
  - Doesn't require tuning

✅ Production systems
  - Consistent results
  - Handles edge cases gracefully

✅ Large, diverse datasets
  - Some topics broad, some narrow
  - EOM finds natural boundaries
```

### Use Leaf When

```
✅ Maximum granularity needed
  - Fine-grained analysis
  - Will post-process/merge later

✅ Exploring data structure
  - See all possible sub-topics
  - Investigate hierarchy

✅ Known fine structure
  - Data has many small, distinct clusters
  - Don't want aggregation

✅ Visualization
  - More clusters = more detail
  - Can manually interpret
```

---

## The ClusterAssignment Result

```swift
// 📍 See: Sources/SwiftTopics/Clustering/ClusterAssignment.swift

public struct ClusterAssignment: Sendable {
    /// Cluster labels for each point. -1 indicates noise.
    public let labels: [Int]

    /// Membership probability for each point (0.0 to 1.0).
    public let probabilities: [Float]

    /// Outlier score for each point (0.0 to 1.0, higher = more outlier-like).
    public let outlierScores: [Float]

    /// Number of clusters found (excluding noise).
    public let clusterCount: Int

    /// Gets the label for a specific point.
    public func label(for index: Int) -> Int {
        labels[index]
    }
}

// Example usage:
let result = try await hdbscan.fit(embeddings)

print("Found \(result.clusterCount) clusters")

for i in 0..<embeddings.count {
    let label = result.label(for: i)
    let prob = result.probabilities[i]
    let outlier = result.outlierScores[i]

    if label == -1 {
        print("Point \(i): noise (outlier score: \(outlier))")
    } else {
        print("Point \(i): cluster \(label) (prob: \(prob), outlier: \(outlier))")
    }
}
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Expecting Deterministic Results with Ties

```swift
// ⚠️ When stability scores are equal, selection can vary

// Hierarchy:
//     [A: 0.5]    [B: 0.5]
//
// Both children have same stability.
// Which is selected first? Implementation-dependent.

// ✅ Solution: Use seed for reproducibility
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    seed: 42
)
```

### Pitfall 2: Ignoring Soft Clustering Output

```swift
// ⚠️ Only using hard labels
let labels = result.labels
// Point is in cluster 3 or noise. That's it.

// ✅ Better: Use probabilities for soft clustering
let prob = result.probabilities[i]
if prob < 0.5 {
    print("Point \(i) is borderline—could be noise")
}
```

### Pitfall 3: Not Tuning minClusterSize for Extraction

```swift
// ⚠️ minClusterSize affects extraction, not just stability

// Small minClusterSize = many small clusters extracted
// Large minClusterSize = few large clusters extracted

// The condensed tree filters by minClusterSize!
// Clusters smaller than this are removed from consideration.
```

### Pitfall 4: Forgetting allowSingleCluster

```swift
// ⚠️ Data with one dominant topic
// If all points cluster together, HDBSCAN might return no clusters!

// Default behavior: if only one cluster found, try to split
// This can result in forced sub-clusters or all noise

// ✅ If one cluster is valid for your use case:
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    allowSingleCluster: true
)
```

---

## Key Takeaways

1. **EOM maximizes stability**: Dynamic programming chooses highest total stability.

2. **Leaf selects all leaves**: Maximum granularity, ignores stability.

3. **Points trace up the tree**: Find first selected ancestor to get cluster assignment.

4. **Noise has no selected ancestor**: Points that never enter a selected cluster.

5. **Probabilities indicate confidence**: Higher = more central to cluster.

6. **Outlier scores for all points**: Even clustered points have outlier scores.

---

## 💡 Key Insight

Cluster extraction is where HDBSCAN's **philosophy** becomes concrete. EOM says: "A cluster that persists across many density levels is more real than one that appears briefly."

```
High stability cluster:
  - Appears at high density (tight group)
  - Persists as density decreases
  - Contains many points
  → Likely a real, coherent topic

Low stability cluster:
  - Appears briefly
  - Quickly merges into something else
  - Few points
  → Might be noise masquerading as structure

EOM automatically finds the sweet spot between
"too many tiny clusters" and "one giant cluster."
```

---

## Chapter Summary

In this chapter, you learned:

1. **Why not K-Means**: Fixed K, spherical assumption, no outliers
2. **DBSCAN foundation**: Density-based clustering with ε problem
3. **HDBSCAN hierarchy**: Multi-scale view with stability scores
4. **Mutual reachability**: Density-aware distances
5. **Cluster extraction**: EOM vs Leaf for final clustering

With clusters in hand, we're ready for the next stage: **extracting topic keywords** with c-TF-IDF.

---

## Next Chapter

**[→ Chapter 4: Topic Representation](../04-Topic-Representation/README.md)**

---

*Guide 3.5 of 3.5 • Chapter 3: Density-Based Clustering*
