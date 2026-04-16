# Cluster Extraction Performance Optimization Plan

> **Status:** Planning
> **Priority:** Critical
> **Expected Speedup:** 100x (188s → ~2s for 5000 points)

## Executive Summary

Benchmarking revealed that cluster extraction dominates HDBSCAN runtime, taking **188 seconds** for 5000 points—regardless of CPU/GPU acceleration for earlier phases. This document outlines 4 fixes that will reduce cluster extraction to **~1-2 seconds**.

### Benchmark Context

```
HDBSCAN Timing (GPU, 5000 points):
  Core distances:      0.066s   ✅ GPU accelerated
  Mutual reachability: 0.044s   ✅ GPU accelerated
  MST construction:    0.109s   ✅ GPU accelerated
  Hierarchy building:  4.627s   ⚠️  Acceptable
  Cluster extraction:  187.350s ❌ BOTTLENECK
  Total:               192.196s
```

### Root Cause

Multiple **O(n)** linear searches (`nodes.first { $0.id == id }`) are called inside loops, resulting in **O(n²)** or worse algorithmic complexity. With 5000 points creating ~10,000 hierarchy nodes, this means ~100 million operations.

---

## Fix 1: Add Index Dictionary to ClusterHierarchy

**Priority:** P0 - Highest Impact
**Estimated Speedup:** 50-100x for hierarchy traversal
**Files:** `ClusterAssignment.swift`
**Complexity:** Low

### Problem

```swift
// ClusterAssignment.swift:351-352
public func node(id: Int) -> ClusterHierarchyNode? {
    nodes.first { $0.id == id }  // O(n) per call!
}
```

This O(n) lookup is called:
- In `findSelectedAncestor()` for every point (5000 calls)
- Each call traverses up hierarchy (avg 10-15 parent lookups per point)
- Total: 5000 × 12 × O(10,000) = **600 million operations**

### Solution

Add a pre-computed dictionary for O(1) lookups:

```swift
public struct ClusterHierarchy: Sendable, Codable {
    public let nodes: [ClusterHierarchyNode]
    public let rootID: Int

    // NEW: O(1) lookup index
    private let nodeIndex: [Int: Int]  // nodeID → array index

    public init(nodes: [ClusterHierarchyNode], rootID: Int) {
        self.nodes = nodes
        self.rootID = rootID

        // Build index once during construction: O(n)
        var index = [Int: Int]()
        index.reserveCapacity(nodes.count)
        for (i, node) in nodes.enumerated() {
            index[node.id] = i
        }
        self.nodeIndex = index
    }

    public func node(id: Int) -> ClusterHierarchyNode? {
        guard let idx = nodeIndex[id] else { return nil }
        return nodes[idx]  // O(1)!
    }

    // For Codable conformance
    enum CodingKeys: String, CodingKey {
        case nodes, rootID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let nodes = try container.decode([ClusterHierarchyNode].self, forKey: .nodes)
        let rootID = try container.decode(Int.self, forKey: .rootID)
        self.init(nodes: nodes, rootID: rootID)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nodes, forKey: .nodes)
        try container.encode(rootID, forKey: .rootID)
    }
}
```

### Verification

```swift
// Add to HDBSCANBenchmarks.swift
func testHierarchyLookupPerformance() async throws {
    let hierarchy = ... // Build hierarchy with 5000 points

    measure {
        for _ in 0..<10000 {
            let _ = hierarchy.node(id: Int.random(in: 0..<10000))
        }
    }
    // Before: ~2.0s
    // After:  ~0.001s
}
```

### Tasks

- [x] Add `nodeIndex` property to `ClusterHierarchy`
- [x] Update `init(nodes:rootID:)` to build index
- [x] Add custom `Codable` implementation (index is transient)
- [x] Update `node(id:)` to use index
- [ ] Add performance test
- [ ] Verify all tests pass

---

## Fix 2: Eliminate O(n) Lookups in Hierarchy Builder

**Priority:** P0 - High Impact
**Estimated Speedup:** 1000x for finalization phase
**Files:** `ClusterHierarchyBuilder.swift`
**Complexity:** Low

### Problem

```swift
// ClusterHierarchyBuilder.swift:214
if let parentIdx = nodes.firstIndex(where: { $0.id == parentID }) {
    // ...
}
```

Called for every node (~10,000), each doing O(n) search = **O(n²)**.

### Solution

Build index before the loop:

```swift
func finalize(minClusterSize: Int, allowSingleCluster: Bool) -> ClusterHierarchy {
    var nodes = [ClusterHierarchyNode]()

    // ... create leaf and internal nodes ...

    // NEW: Build index for O(1) lookup
    var nodeIndexByID = [Int: Int]()
    nodeIndexByID.reserveCapacity(nodes.count)
    for (idx, node) in nodes.enumerated() {
        nodeIndexByID[node.id] = idx
    }

    // Set parent pointers and death levels
    for i in 0..<nodes.count {
        let nodeID = nodes[i].id

        if let parentID = parentMap[nodeID],
           let parentIdx = nodeIndexByID[parentID] {  // O(1) instead of O(n)!
            let parentBirth = nodes[parentIdx].birthLevel

            nodes[i] = ClusterHierarchyNode(
                id: nodeID,
                parent: parentID,
                children: nodes[i].children,
                birthLevel: nodes[i].birthLevel,
                deathLevel: parentBirth,
                size: nodes[i].size,
                stability: 0
            )
        } else {
            // Root node handling unchanged
        }
    }

    // ... rest unchanged ...
}
```

### Tasks

- [x] Add `nodeIndexByID` dictionary in `finalize()`
- [x] Replace `nodes.firstIndex(where:)` with dictionary lookup
- [ ] Verify timing improvement in benchmarks

---

## Fix 3: Bottom-Up Leaf Descendant Computation

**Priority:** P1 - High Impact
**Estimated Speedup:** 1000x for stability computation
**Files:** `ClusterHierarchyBuilder.swift`
**Complexity:** Medium

### Problem

```swift
// ClusterHierarchyBuilder.swift:375-398
private func collectLeafDescendants(
    nodeID: Int,
    nodes: [ClusterHierarchyNode],
    nodeByID: [Int: Int]
) -> [(leafID: Int, deathDistance: Float)] {
    // Recursively traverses ALL descendants for EACH internal node
    for childID in node.children {
        leaves.append(contentsOf: collectLeafDescendants(...))
    }
}
```

This is called for every internal node during stability computation. Each call traverses all descendants. Total complexity: **Σ(subtree sizes) = O(n²)** for unbalanced trees.

### Solution

Compute leaf descendants once using bottom-up dynamic programming:

```swift
/// Precomputed leaf info per node
private struct LeafInfo {
    var leafIDs: [Int]
    var deathDistances: [Float]

    static let empty = LeafInfo(leafIDs: [], deathDistances: [])
}

/// Build leaf descendants for all nodes in O(n) total
private func buildLeafDescendantsMap(
    nodes: [ClusterHierarchyNode],
    nodeByID: [Int: Int]
) -> [Int: LeafInfo] {
    var leafMap = [Int: LeafInfo]()
    leafMap.reserveCapacity(nodes.count)

    // Process in bottom-up order: leaves first, then by increasing size
    // This ensures children are processed before parents
    let sortedNodes = nodes.sorted { $0.size < $1.size }

    for node in sortedNodes {
        if node.children.isEmpty {
            // Leaf node: contains only itself
            leafMap[node.id] = LeafInfo(
                leafIDs: [node.id],
                deathDistances: [node.deathLevel]
            )
        } else {
            // Internal node: merge children's leaves (already computed)
            var allLeaves = [Int]()
            var allDeaths = [Float]()

            // Pre-allocate based on node size (each point appears once)
            allLeaves.reserveCapacity(node.size)
            allDeaths.reserveCapacity(node.size)

            for childID in node.children {
                if let childInfo = leafMap[childID] {
                    allLeaves.append(contentsOf: childInfo.leafIDs)
                    allDeaths.append(contentsOf: childInfo.deathDistances)
                }
            }

            leafMap[node.id] = LeafInfo(leafIDs: allLeaves, deathDistances: allDeaths)
        }
    }

    return leafMap
}

/// Updated stability computation using precomputed leaf info
private func computeStability(
    nodes: [ClusterHierarchyNode],
    minClusterSize: Int
) -> [ClusterHierarchyNode] {
    var result = nodes

    // Build lookup indices
    var nodeByID = [Int: Int]()
    for (idx, node) in nodes.enumerated() {
        nodeByID[node.id] = idx
    }

    // NEW: Precompute all leaf descendants in O(n)
    let leafMap = buildLeafDescendantsMap(nodes: nodes, nodeByID: nodeByID)

    for i in 0..<result.count {
        let node = result[i]

        guard node.size >= minClusterSize else {
            result[i] = ClusterHierarchyNode(
                id: node.id,
                parent: node.parent,
                children: node.children,
                birthLevel: node.birthLevel,
                deathLevel: node.deathLevel,
                size: node.size,
                stability: 0
            )
            continue
        }

        // NEW: Use precomputed leaf info instead of recursive traversal
        let stability = computeNodeStabilityFast(
            node: node,
            leafInfo: leafMap[node.id] ?? .empty
        )

        result[i] = ClusterHierarchyNode(
            id: node.id,
            parent: node.parent,
            children: node.children,
            birthLevel: node.birthLevel,
            deathLevel: node.deathLevel,
            size: node.size,
            stability: stability
        )
    }

    return result
}

/// Fast stability computation using precomputed leaves
private func computeNodeStabilityFast(
    node: ClusterHierarchyNode,
    leafInfo: LeafInfo
) -> Float {
    let birthDistance = node.birthLevel
    let deathDistance = node.deathLevel

    guard birthDistance > 0 || deathDistance > 0 else { return 0 }

    let lambdaBirth = birthDistance > Float.ulpOfOne
        ? 1.0 / birthDistance
        : Float.infinity
    let lambdaDeath = deathDistance > Float.ulpOfOne && deathDistance < Float.infinity
        ? 1.0 / deathDistance
        : 0

    if leafInfo.leafIDs.isEmpty {
        return max(0, lambdaBirth - lambdaDeath)
    }

    var stability: Float = 0

    for leafDeathDistance in leafInfo.deathDistances {
        let leafLambdaDeath = leafDeathDistance > Float.ulpOfOne && leafDeathDistance < Float.infinity
            ? 1.0 / leafDeathDistance
            : 0

        let effectiveLambdaDeath = max(leafLambdaDeath, lambdaDeath)
        let contribution = max(0, lambdaBirth - effectiveLambdaDeath)
        stability += contribution
    }

    return stability
}
```

### Complexity Analysis

| Approach | Time Complexity | Space Complexity |
|----------|-----------------|------------------|
| Current (recursive) | O(n²) worst case | O(h) stack depth |
| Proposed (bottom-up) | O(n) | O(n) for leaf map |

### Tasks

- [x] Add `LeafInfo` struct
- [x] Implement `buildLeafDescendantsMap()`
- [x] Update `computeStability()` to use precomputed map
- [x] Implement `computeNodeStabilityFast()`
- [x] Remove old `collectLeafDescendants()` method
- [ ] Add unit tests for correctness
- [ ] Benchmark improvement

---

## Fix 4: Memoized Point Assignment

**Priority:** P1 - Medium Impact
**Estimated Speedup:** 10-100x for point assignment
**Files:** `ClusterExtraction.swift`
**Complexity:** Medium

### Problem

```swift
// ClusterExtraction.swift:288-301
for pointIndex in 0..<pointCount {
    let (clusterID, probability) = findSelectedAncestor(
        pointIndex: pointIndex,
        selectedClusterIDs: selectedClusterIDs,
        hierarchy: hierarchy
    )
    // ...
}
```

Each point independently traverses up the hierarchy. Many points share ancestor paths, leading to redundant traversals.

### Solution

Use memoization to cache ancestor lookups:

```swift
/// Assigns points to selected clusters with memoization
private func assignPointsToClustersFast(
    selectedClusterIDs: Set<Int>,
    hierarchy: ClusterHierarchy,
    pointCount: Int,
    coreDistances: [Float]
) -> ClusterAssignment {
    guard !selectedClusterIDs.isEmpty else {
        return ClusterAssignment(
            labels: [Int](repeating: -1, count: pointCount),
            probabilities: [Float](repeating: 0, count: pointCount),
            outlierScores: coreDistances.isEmpty
                ? [Float](repeating: 1, count: pointCount)
                : coreDistances.map { $0 / (coreDistances.max() ?? 1) },
            clusterCount: 0
        )
    }

    // Build cluster label mapping
    let sortedClusterIDs = Array(selectedClusterIDs).sorted()
    var clusterIDToLabel = [Int: Int]()
    for (label, clusterID) in sortedClusterIDs.enumerated() {
        clusterIDToLabel[clusterID] = label
    }

    // NEW: Memoization cache for ancestor lookups
    // Maps nodeID → (selectedAncestorID, depth)
    // -1 means no selected ancestor found
    var ancestorCache = [Int: (ancestorID: Int, depth: Int)]()
    ancestorCache.reserveCapacity(pointCount * 2)  // Estimate for cache size

    func findSelectedAncestorMemoized(nodeID: Int, depth: Int) -> (ancestorID: Int, depth: Int) {
        // Check cache first
        if let cached = ancestorCache[nodeID] {
            return cached
        }

        // Check if this node is a selected cluster
        if selectedClusterIDs.contains(nodeID) {
            let result = (nodeID, depth)
            ancestorCache[nodeID] = result
            return result
        }

        // Traverse to parent
        guard let node = hierarchy.node(id: nodeID),
              let parentID = node.parent else {
            // No selected ancestor found
            let result = (-1, depth)
            ancestorCache[nodeID] = result
            return result
        }

        // Recurse to parent (will use cache if parent already computed)
        let parentResult = findSelectedAncestorMemoized(nodeID: parentID, depth: depth + 1)
        ancestorCache[nodeID] = parentResult
        return parentResult
    }

    // Assign points using memoized lookup
    var pointLabels = [Int](repeating: -1, count: pointCount)
    var pointProbabilities = [Float](repeating: 0, count: pointCount)
    var clusterPoints = [Int: [Int]]()

    for clusterID in selectedClusterIDs {
        clusterPoints[clusterID] = []
    }

    for pointIndex in 0..<pointCount {
        let (ancestorID, depth) = findSelectedAncestorMemoized(nodeID: pointIndex, depth: 0)

        if ancestorID >= 0, let label = clusterIDToLabel[ancestorID] {
            pointLabels[pointIndex] = label
            pointProbabilities[pointIndex] = max(0.1, 1.0 - Float(depth) * 0.1)
            clusterPoints[ancestorID]?.append(pointIndex)
        }
    }

    // Compute outlier scores (unchanged)
    let outlierScores = computeOutlierScores(
        pointLabels: pointLabels,
        coreDistances: coreDistances,
        clusterPoints: clusterPoints,
        clusterIDToLabel: clusterIDToLabel
    )

    return ClusterAssignment(
        labels: pointLabels,
        probabilities: pointProbabilities,
        outlierScores: outlierScores,
        clusterCount: selectedClusterIDs.count
    )
}
```

### Why This Works

Points that share ancestors will hit the cache. Consider points that all belong to the same selected cluster:
- First point: traverses full path, caches each node
- Subsequent points: hit cache at their common ancestor

With typical hierarchies, most paths share significant overlap, reducing total traversals from O(n × h) to approximately O(n + unique paths).

### Tasks

- [ ] Add `ancestorCache` dictionary
- [ ] Implement `findSelectedAncestorMemoized()`
- [ ] Replace `assignPointsToClusters()` call with fast version
- [ ] Benchmark cache hit rate
- [ ] Consider cache size limits for very large hierarchies

---

## Fix 5: Optimize CondensedTree Construction

**Priority:** P2 - Medium Impact
**Estimated Speedup:** 100-900x for condensed tree
**Files:** `ClusterHierarchyBuilder.swift`
**Complexity:** Low

### Problem

```swift
// ClusterHierarchyBuilder.swift:512, 522
currentParent = hierarchy.nodes.first(where: { $0.id == p })?.parent  // O(n)
if let child = hierarchy.nodes.first(where: { $0.id == childID }) {   // O(n)
```

Multiple O(n) lookups in loops during tree condensation.

### Solution

Pre-build a lookup dictionary:

```swift
public init(hierarchy: ClusterHierarchy, minClusterSize: Int) {
    var condensedNodes = [CondensedTreeNode]()

    // NEW: Build lookup index
    var hierarchyNodeByID = [Int: ClusterHierarchyNode]()
    hierarchyNodeByID.reserveCapacity(hierarchy.nodes.count)
    for node in hierarchy.nodes {
        hierarchyNodeByID[node.id] = node
    }

    // Filter to nodes that meet minimum size
    let validNodes = hierarchy.nodes.filter { $0.size >= minClusterSize }
    let validIDs = Set(validNodes.map { $0.id })

    for node in validNodes {
        // Find effective parent (closest ancestor that's valid)
        var effectiveParent: Int? = nil
        var currentParent = node.parent

        while let p = currentParent {
            if validIDs.contains(p) {
                effectiveParent = p
                break
            }
            // NEW: O(1) lookup instead of O(n)
            currentParent = hierarchyNodeByID[p]?.parent
        }

        // Find effective children (valid descendants)
        var effectiveChildren = [Int]()
        var queue = node.children
        while !queue.isEmpty {
            let childID = queue.removeFirst()
            if validIDs.contains(childID) {
                effectiveChildren.append(childID)
            } else if let child = hierarchyNodeByID[childID] {  // O(1)
                queue.append(contentsOf: child.children)
            }
        }

        condensedNodes.append(CondensedTreeNode(
            id: node.id,
            parent: effectiveParent,
            children: effectiveChildren,
            birthLevel: node.birthLevel,
            deathLevel: node.deathLevel,
            size: node.size,
            stability: node.stability
        ))
    }

    self.nodes = condensedNodes
    self.rootID = condensedNodes.first(where: { $0.parent == nil })?.id ?? -1
}
```

### Tasks

- [x] ~~Add `hierarchyNodeByID` dictionary~~ (Not needed - Fix #1 already made `hierarchy.node(id:)` O(1))
- [x] Replace `hierarchy.nodes.first(where:)` calls with `hierarchy.node(id:)` calls
- [ ] Benchmark improvement

---

## Implementation Order

| Order | Fix | Effort | Impact | Dependencies |
|-------|-----|--------|--------|--------------|
| 1 | Fix 1: ClusterHierarchy index | Low | Very High | None |
| 2 | Fix 2: Hierarchy builder index | Low | High | None |
| 3 | Fix 5: CondensedTree index | Low | Medium | None |
| 4 | Fix 3: Bottom-up leaf computation | Medium | Very High | Fix 2 |
| 5 | Fix 4: Memoized point assignment | Medium | High | Fix 1 |

**Recommended approach:** Implement fixes 1, 2, and 5 first (low effort, high impact), then tackle fixes 3 and 4.

---

## Benchmarking Protocol

### Before/After Comparison

```swift
final class ClusterExtractionOptimizationBenchmarks: XCTestCase {

    func testClusterExtractionPerformance() async throws {
        let testSizes = [500, 1000, 2000, 5000]

        for size in testSizes {
            let embeddings = generateRandomEmbeddings(count: size, dimension: 384)
            let engine = HDBSCANEngine(configuration: .default)

            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await engine.fitWithDetails(embeddings)
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime

            print("Size \(size): Extraction = \(result.timingBreakdown?.extractionTime ?? 0)s")
        }
    }

    func testHierarchyLookupScaling() {
        // Test that lookup remains O(1) regardless of hierarchy size
        for size in [1000, 5000, 10000] {
            let hierarchy = buildTestHierarchy(pointCount: size)

            let lookupTime = measure {
                for _ in 0..<10000 {
                    let _ = hierarchy.node(id: Int.random(in: 0..<size * 2))
                }
            }

            print("Size \(size): 10k lookups = \(lookupTime)s")
            // Should be nearly constant (< 0.01s) for all sizes
        }
    }
}
```

### Expected Results

| Size | Before | After Fix 1 | After All Fixes |
|------|--------|-------------|-----------------|
| 500 | ~2s | ~0.5s | ~0.05s |
| 1000 | ~8s | ~1s | ~0.1s |
| 2000 | ~35s | ~3s | ~0.3s |
| 5000 | ~188s | ~15s | ~1.5s |

---

## Validation Checklist

- [ ] All existing unit tests pass
- [ ] Cluster assignments are identical before/after optimization
- [ ] Stability scores match to floating-point precision
- [ ] Outlier scores match to floating-point precision
- [ ] Memory usage does not increase significantly (< 2x)
- [ ] Performance scales linearly with input size

---

## Related Documents

- [BENCHMARK_SUITE_PLAN.md](./BENCHMARK_SUITE_PLAN.md) - Benchmarking framework
- [VECTORACCELERATE_GPU_OPPORTUNITIES.md](./VECTORACCELERATE_GPU_OPPORTUNITIES.md) - GPU acceleration requests
