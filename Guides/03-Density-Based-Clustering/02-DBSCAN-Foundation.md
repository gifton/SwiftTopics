# 3.2 DBSCAN Foundation

> **The original density-based clustering algorithm—and why we needed HDBSCAN.**

---

## The Concept

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) was a breakthrough: the first practical clustering algorithm that didn't require K and could mark outliers.

The core idea is elegant:

> **Clusters are dense regions separated by sparse regions.**

```
What DBSCAN sees:

    Dense region 1        Sparse gap        Dense region 2
    ┌────────────────┐   ┌──────────┐   ┌────────────────┐
    │  ●●●●●●●●●●    │   │          │   │    ●●●●●●●●●●  │
    │ ●●●●●●●●●●●●   │   │  ●    ●  │   │   ●●●●●●●●●●●● │
    │  ●●●●●●●●●●    │   │          │   │    ●●●●●●●●●●  │
    └────────────────┘   └──────────┘   └────────────────┘
         Cluster 0           Noise           Cluster 1
```

---

## Why It Matters

DBSCAN introduced three concepts that HDBSCAN builds upon:

1. **Core Points**: Points with enough neighbors (density indicators)
2. **Border Points**: Points on the edge of clusters
3. **Noise Points**: Points in sparse regions (outliers)

Understanding these is essential for understanding HDBSCAN.

---

## The Mathematics

### Two Parameters

DBSCAN uses two parameters:

```
ε (epsilon): Maximum distance for a point to be considered a neighbor
minPts:      Minimum neighbors to be considered a core point

These define what "dense" means:
- A point is in a dense region if it has ≥ minPts neighbors within distance ε
```

### Point Classification

```
For each point p:

Count neighbors within ε:
  N_ε(p) = { q : distance(p, q) ≤ ε }

If |N_ε(p)| ≥ minPts:
  p is a CORE POINT (in a dense region)

If |N_ε(p)| < minPts but p is within ε of a core point:
  p is a BORDER POINT (on the edge of a cluster)

If |N_ε(p)| < minPts and p is not near any core point:
  p is a NOISE POINT (outlier)
```

### Visual Classification

```
ε = the radius of circles
minPts = 4

     ●───●───●          Legend:
    ╱│╲ ╱ ╲ ╱│╲         ● Core point (≥4 neighbors in ε)
   ● │ ●   ● │ ●        ○ Border point (<4 but near core)
    ╲│╱     ╲│╱         ✗ Noise point (sparse, isolated)
     ●───────●
              ╲
               ○        Border: Only 2 neighbors,
                          but within ε of core ●
       ✗                Noise: Isolated, no core nearby

Each core point can "reach" other core points through chains.
All reachable core points form one cluster.
```

---

## The Technique: DBSCAN Algorithm

### Step by Step

```
DBSCAN Algorithm:
──────────────────────────────────────────────────────────

Input: Points, ε, minPts
Output: Cluster labels (-1 for noise)

1. Mark all points as UNVISITED

2. For each UNVISITED point p:
   a. Mark p as VISITED
   b. Find all neighbors within ε: N = neighbors(p, ε)

   c. If |N| < minPts:
      - Mark p as NOISE (may change later)

   d. If |N| ≥ minPts:
      - Create new cluster C
      - Add p to C
      - For each point q in N:
        * If q is UNVISITED:
          - Mark q as VISITED
          - Find neighbors of q: N' = neighbors(q, ε)
          - If |N'| ≥ minPts: add N' to N (expand cluster)
        * If q is not in any cluster: add q to C

3. Return cluster assignments

Complexity: O(n²) naive, O(n log n) with spatial index
```

### Walkthrough Example

```
Points: A, B, C, D, E, F, G, H
ε = 1.0, minPts = 3

Distance matrix (simplified):
        A     B     C     D     E     F     G     H
    A   0    0.5   0.8   1.5   2.0   2.5   3.0   5.0
    B  0.5    0    0.3   1.0   1.5   2.0   2.5   4.5
    C  0.8   0.3    0    0.7   1.2   1.7   2.2   4.2
    D  1.5   1.0   0.7    0    0.5   1.0   1.5   3.7
    E  2.0   1.5   1.2   0.5    0    0.5   1.0   3.2
    F  2.5   2.0   1.7   1.0   0.5    0    0.5   2.7
    G  3.0   2.5   2.2   1.5   1.0   0.5    0    2.2
    H  5.0   4.5   4.2   3.7   3.2   2.7   2.2    0

Step 1: Process A
  Neighbors(A, ε=1.0) = {B, C}  (distances 0.5, 0.8)
  |neighbors| = 2 < minPts(3)
  Mark A as NOISE (for now)

Step 2: Process B
  Neighbors(B, ε=1.0) = {A, C, D}
  |neighbors| = 3 ≥ minPts(3)
  B is CORE POINT → Start Cluster 0
  Add B to Cluster 0
  Expand: check A, C, D

  A: neighbors = {B, C}, |N|=2 < 3, A is border, add to Cluster 0
  C: neighbors = {A, B, D}, |N|=3 ≥ 3, C is core, add neighbors to search
  D: neighbors = {B, C, E, F}, |N|=4 ≥ 3, D is core, add neighbors

  Continue expanding...

Final result:
  Cluster 0: {A, B, C, D, E, F, G}  (connected dense region)
  Noise: {H}  (isolated point)
```

---

## The Epsilon Problem

DBSCAN has a critical weakness: **ε must be chosen carefully**.

### Too Small ε

```
ε too small:

    Data:                     DBSCAN result (ε=0.3):

       ●●●●●●                    ○ ○ ○ ○ ○ ○
      ●●●●●●●●                  ○ ○ ○ ○ ○ ○ ○ ○
       ●●●●●●                    ○ ○ ○ ○ ○ ○

    Natural cluster            All points become noise!
                              (No point has enough neighbors
                               within the tiny radius)
```

### Too Large ε

```
ε too large:

    Data:                     DBSCAN result (ε=5.0):

    ●●●●●●     ●●●●●●          ●●●●●● ─── ●●●●●●
    (Cluster A) (Cluster B)    (All merged into one!)

    Two separate clusters      ε so large that they connect
```

### The "Right" ε

```
The Goldilocks problem:

ε = 0.3  →  All noise (too small)
ε = 0.5  →  Cluster 1 found, Cluster 2 becomes noise
ε = 0.8  →  Both clusters found! ✓
ε = 1.2  →  Clusters merge (too large)
ε = 2.0  →  Everything is one cluster

The "right" ε depends on data density.
Different regions may need different ε values!
```

---

## The Variable Density Problem

This is DBSCAN's fatal flaw for topic modeling:

```
Real data has VARYING DENSITY:

    Dense cluster:            Sparse cluster:

    ●●●●●●●●●●●●●●●●          ●     ●     ●
    ●●●●●●●●●●●●●●●●●●
    ●●●●●●●●●●●●●●●●           ●       ●
    (Points 0.1 apart)        (Points 1.0 apart)

With ε = 0.2:
  Dense cluster: Found! ✓
  Sparse cluster: All noise ✗

With ε = 1.5:
  Dense cluster: Found ✓
  Sparse cluster: Found ✓
  BUT: Dense cluster expands to connect to noise!

NO SINGLE ε WORKS FOR BOTH.
```

This is why HDBSCAN was invented.

---

## In SwiftTopics

SwiftTopics doesn't use vanilla DBSCAN. But understanding DBSCAN helps understand HDBSCAN's core distance concept:

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/CoreDistance.swift

/// The core distance of a point is the distance to its k-th nearest neighbor,
/// which serves as a measure of local density. Points in dense regions have
/// small core distances; points in sparse regions have large core distances.

// DBSCAN: Fixed ε for all points
// HDBSCAN: Adaptive "ε" per point (core distance)

// DBSCAN check:
let isCore = neighbors.count >= minPts  // within fixed ε

// HDBSCAN approach:
let coreDistance = distanceToKthNeighbor(k: minSamples)
// Each point has its own "ε" based on local density
```

### The Core Distance Evolution

```swift
// 📍 Conceptual evolution from DBSCAN to HDBSCAN:

// DBSCAN:
struct DBSCANPoint {
    let isCore: Bool  // Binary: has ≥ minPts neighbors within ε
}

// HDBSCAN:
struct HDBSCANPoint {
    let coreDistance: Float  // Continuous: distance to minPts-th neighbor
    // Density = 1 / coreDistance
    // Small coreDistance = dense region
    // Large coreDistance = sparse region
}
```

---

## Visualizing Core vs Border vs Noise

```
Example: 15 points with ε = 1.0, minPts = 4

     A(●)─── B(●)─── C(●)
      │╲     │      ╱│
      │ ╲    │     ╱ │
      │  ╲   │    ╱  │
     D(●)── E(●)── F(●)
             │
             │
            G(○)       H(✗)



Point  Neighbors (within ε)    Count   Classification
────────────────────────────────────────────────────────
  A     {B, D, E}               3      Border? (< 4) but wait...
  B     {A, C, E}               3      Border? checking...
  C     {B, E, F}               3      Border?
  D     {A, E}                  2      Border
  E     {A, B, C, D, F, G}      6      CORE ✓
  F     {C, E}                  2      Border
  G     {E}                     1      Border (near core E)
  H     {}                      0      NOISE ✗

Hmm, only E is core. Let's recalculate with minPts = 3:

  A     {B, D, E}               3      CORE ✓
  B     {A, C, E}               3      CORE ✓
  C     {B, E, F}               3      CORE ✓
  D     {A, E}                  2      Border (near core A)
  E     {A, B, C, D, F, G}      6      CORE ✓
  F     {C, E}                  2      Border (near core C)
  G     {E}                     1      Border (near core E)
  H     {}                      0      NOISE ✗

Now we have one cluster containing A,B,C,D,E,F,G and one noise point H.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Not Scaling Data

```swift
// ❌ WRONG: DBSCAN on unscaled features
// Feature 1: salary (10,000 - 1,000,000)
// Feature 2: age (18 - 80)

let dbscan = DBSCAN(epsilon: 100, minPts: 5)
// ε = 100 is huge for age but tiny for salary!

// ✅ CORRECT: Normalize or standardize first
let normalized = standardize(data)
let dbscan = DBSCAN(epsilon: 0.5, minPts: 5)
```

### Pitfall 2: Using Distance Instead of Density Intuition

```swift
// ⚠️ MISLEADING: Thinking of ε as "closeness"

// Wrong mental model:
// "Points within ε are 'close enough' to cluster"

// Better mental model:
// "ε defines what 'dense' means for this dataset"
// Core points are density indicators, not distance indicators.
```

### Pitfall 3: Expecting K Clusters

```swift
// ❌ WRONG: Expecting DBSCAN to find exactly K clusters
let result = dbscan.fit(data)
assert(result.clusterCount == 5)  // May not be 5!

// DBSCAN finds what's there, not what you want.
// If data has 3 dense regions, you get 3 clusters.
// If data is uniform, you may get 1 cluster or all noise.
```

### Pitfall 4: Ignoring Noise Points

```swift
// ⚠️ MISTAKE: Treating noise as "bad" results
let noiseCount = result.labels.filter { $0 == -1 }.count
print("⚠️ \(noiseCount) points failed to cluster!")  // Wrong framing

// Better framing:
print("ℹ️ \(noiseCount) points identified as outliers")
// Noise detection is a feature, not a bug!
```

---

## Key Takeaways

1. **DBSCAN uses density**: Clusters are dense regions; gaps are sparse regions.

2. **Three point types**: Core (dense), border (edge), noise (sparse).

3. **Two parameters**: ε (neighborhood radius) and minPts (density threshold).

4. **No K required**: Cluster count emerges from data.

5. **Handles outliers**: Noise points are explicitly identified.

6. **Fatal flaw**: Single ε can't handle varying-density data.

---

## 💡 Key Insight

DBSCAN's genius is the density perspective. Its weakness is the global ε assumption.

```
DBSCAN's question: "Is this point in a dense region?"
                   (Using fixed definition of "dense")

HDBSCAN's question: "Is this point in a RELATIVELY dense region?"
                    (Compared to its local neighborhood)

HDBSCAN makes ε adaptive through "core distance":
- Dense regions: small effective ε
- Sparse regions: large effective ε

This is the mutual reachability transform we'll learn next.
```

---

## Next Up

Now we understand the foundation. Let's see how HDBSCAN extends DBSCAN to handle varying densities:

**[→ 3.3 HDBSCAN Hierarchy](./03-HDBSCAN-Hierarchy.md)**

---

*Guide 3.2 of 3.5 • Chapter 3: Density-Based Clustering*
