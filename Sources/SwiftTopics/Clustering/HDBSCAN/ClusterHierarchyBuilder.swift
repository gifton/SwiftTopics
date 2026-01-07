// ClusterHierarchyBuilder.swift
// SwiftTopics
//
// Builds the HDBSCAN cluster hierarchy (dendrogram) from MST

import Foundation

// MARK: - Cluster Hierarchy Builder

/// Builds a cluster hierarchy (dendrogram) from a minimum spanning tree.
///
/// ## Algorithm
///
/// The single-linkage hierarchy is built by processing MST edges in ascending
/// weight (distance) order:
///
/// 1. Initialize each point as its own cluster (singleton)
/// 2. For each MST edge (in ascending weight order):
///    a. Find the clusters containing the two endpoints
///    b. Create a new parent cluster merging them
///    c. Record birth level (current distance) and death levels of children
/// 3. Compute stability for each cluster
///
/// ## Stability
///
/// Cluster stability measures how persistent a cluster is across density levels:
///
/// ```
/// stability(C) = Σ (λ_death(x) - λ_birth(C)) for x in C
/// ```
///
/// Where λ = 1/distance (density). A cluster that persists across many density
/// levels has high stability.
///
/// Simplified form (using distance instead of λ):
/// ```
/// stability(C) = Σ (1/distance_death(x) - 1/distance_birth(C)) for x in C
/// ```
///
/// ## Data Structures
///
/// - Uses Union-Find with parent tracking for cluster membership
/// - Each merge creates a new internal node in the hierarchy
/// - Leaf nodes represent individual points
/// - Internal nodes represent merged clusters
public struct ClusterHierarchyBuilder: Sendable {

    /// Minimum cluster size for stability computation.
    public let minClusterSize: Int

    /// Creates a hierarchy builder.
    ///
    /// - Parameter minClusterSize: Minimum size for a valid cluster.
    public init(minClusterSize: Int) {
        precondition(minClusterSize >= 2, "minClusterSize must be at least 2")
        self.minClusterSize = minClusterSize
    }

    // MARK: - Build Hierarchy

    /// Builds the cluster hierarchy from an MST.
    ///
    /// - Parameters:
    ///   - mst: The minimum spanning tree.
    ///   - allowSingleCluster: Whether to allow a single root cluster.
    /// - Returns: The cluster hierarchy.
    public func build(
        from mst: MinimumSpanningTree,
        allowSingleCluster: Bool = false
    ) -> ClusterHierarchy {
        let n = mst.pointCount

        guard n > 1 else {
            // Single point or empty
            if n == 1 {
                let node = ClusterHierarchyNode(
                    id: 0,
                    parent: nil,
                    children: [],
                    birthLevel: 0,
                    deathLevel: Float.infinity,
                    size: 1,
                    stability: 0
                )
                return ClusterHierarchy(nodes: [node], rootID: 0)
            }
            return ClusterHierarchy(nodes: [], rootID: -1)
        }

        // Sort edges by weight (ascending)
        let sortedEdges = mst.sortedEdges

        // Track cluster structure during merging
        var clusterState = HierarchyBuildState(pointCount: n)

        // Process edges in order
        for edge in sortedEdges {
            clusterState.mergePoints(
                edge.source,
                edge.target,
                atDistance: edge.weight
            )
        }

        // Finalize the hierarchy
        return clusterState.finalize(
            minClusterSize: minClusterSize,
            allowSingleCluster: allowSingleCluster
        )
    }
}

// MARK: - Hierarchy Build State

/// Internal state for hierarchy construction.
private struct HierarchyBuildState {

    /// Union-Find for tracking cluster membership.
    var uf: HierarchyUnionFind

    /// Internal nodes created during merges (indexed by their id - n).
    var internalNodes: [InternalNodeData]

    /// Number of original points.
    let pointCount: Int

    /// Next internal node ID.
    var nextInternalID: Int

    init(pointCount: Int) {
        self.pointCount = pointCount
        self.uf = HierarchyUnionFind(count: pointCount)
        self.internalNodes = []
        self.nextInternalID = pointCount
    }

    /// Merges two points/clusters at a given distance.
    mutating func mergePoints(_ a: Int, _ b: Int, atDistance distance: Float) {
        let rootA = uf.find(a)
        let rootB = uf.find(b)

        guard rootA != rootB else { return }

        // Create new internal node for this merge
        let newNodeID = nextInternalID
        nextInternalID += 1

        // Record children sizes and IDs
        let childA = uf.clusterID[rootA]
        let childB = uf.clusterID[rootB]
        let sizeA = uf.size[rootA]
        let sizeB = uf.size[rootB]

        // Create internal node data
        let node = InternalNodeData(
            id: newNodeID,
            childA: childA,
            childB: childB,
            sizeA: sizeA,
            sizeB: sizeB,
            birthDistance: distance
        )
        internalNodes.append(node)

        // Perform union and update cluster ID
        uf.unionAt(rootA, rootB, newClusterID: newNodeID)
    }

    /// Finalizes the hierarchy and computes stability.
    func finalize(minClusterSize: Int, allowSingleCluster: Bool) -> ClusterHierarchy {
        var nodes = [ClusterHierarchyNode]()

        // Create leaf nodes for individual points
        for i in 0..<pointCount {
            nodes.append(ClusterHierarchyNode(
                id: i,
                parent: nil,  // Will be set later
                children: [],
                birthLevel: 0,
                deathLevel: 0,  // Will be set later
                size: 1,
                stability: 0
            ))
        }

        // Map from node ID to its parent
        var parentMap = [Int: Int]()

        // Create internal nodes and set up parent relationships
        for internalNode in internalNodes {
            // Record parent relationships
            parentMap[internalNode.childA] = internalNode.id
            parentMap[internalNode.childB] = internalNode.id

            let totalSize = internalNode.sizeA + internalNode.sizeB

            nodes.append(ClusterHierarchyNode(
                id: internalNode.id,
                parent: nil,  // Will be set later
                children: [internalNode.childA, internalNode.childB],
                birthLevel: internalNode.birthDistance,
                deathLevel: 0,  // Will be set later
                size: totalSize,
                stability: 0  // Computed below
            ))
        }

        // Build O(1) lookup index: nodeID → array index
        // This replaces O(n) firstIndex(where:) calls with O(1) dictionary lookups
        var nodeIndexByID = [Int: Int]()
        nodeIndexByID.reserveCapacity(nodes.count)
        for (idx, node) in nodes.enumerated() {
            nodeIndexByID[node.id] = idx
        }

        // Set parent pointers and death levels
        for i in 0..<nodes.count {
            let nodeID = nodes[i].id

            if let parentID = parentMap[nodeID] {
                // Find parent node index - O(1) lookup
                if let parentIdx = nodeIndexByID[parentID] {
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
                }
            } else {
                // Root node - death level is infinity (or very large)
                nodes[i] = ClusterHierarchyNode(
                    id: nodeID,
                    parent: nil,
                    children: nodes[i].children,
                    birthLevel: nodes[i].birthLevel,
                    deathLevel: Float.infinity,
                    size: nodes[i].size,
                    stability: 0
                )
            }
        }

        // Compute stability for each node
        nodes = computeStability(
            nodes: nodes,
            minClusterSize: minClusterSize
        )

        // Find root (the node with no parent)
        let rootID = nodes.first(where: { $0.parent == nil && $0.size > 1 })?.id
            ?? nodes.last?.id
            ?? -1

        return ClusterHierarchy(nodes: nodes, rootID: rootID)
    }

    /// Computes stability for all nodes.
    ///
    /// Stability = sum of (1/λ_death - 1/λ_birth) × contribution for each point
    /// where λ = 1/distance.
    ///
    /// This simplifies to: Σ (distance_birth - min(distance_death, distance_birth)) / (distance_birth × distance_death)
    /// But we use a simpler approximation based on persistence × size.
    ///
    /// - Note: Uses bottom-up dynamic programming (Fix #3) for O(n) complexity
    ///   instead of O(n²) recursive traversal.
    private func computeStability(
        nodes: [ClusterHierarchyNode],
        minClusterSize: Int
    ) -> [ClusterHierarchyNode] {
        var result = nodes

        // Index nodes by ID for fast lookup
        var nodeByID = [Int: Int]()
        for (idx, node) in nodes.enumerated() {
            nodeByID[node.id] = idx
        }

        // Fix #3: Precompute all leaf descendants in O(n) using bottom-up DP
        // This replaces O(n²) recursive collectLeafDescendants() calls
        let leafMap = buildLeafDescendantsMap(nodes: nodes, nodeByID: nodeByID)

        // Compute stability using lambda-based formula
        // λ = 1/distance, so we work with inverse distances
        for i in 0..<result.count {
            let node = result[i]

            // Skip if too small to be a valid cluster
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

            // Compute stability using precomputed leaf info (O(n) total)
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

    // MARK: - Bottom-Up Leaf Descendant Computation (Fix #3)

    /// Builds a map of leaf descendants for all nodes in O(n) time.
    ///
    /// This replaces the O(n²) recursive `collectLeafDescendants()` approach
    /// with bottom-up dynamic programming:
    /// 1. Sort nodes by size (smallest first = leaves before parents)
    /// 2. Leaf nodes store only themselves
    /// 3. Internal nodes merge their children's already-computed leaf lists
    ///
    /// - Parameters:
    ///   - nodes: All hierarchy nodes.
    ///   - nodeByID: Pre-built lookup index (nodeID → array index).
    /// - Returns: Map from nodeID to its LeafInfo.
    /// - Complexity: O(n) time and space.
    private func buildLeafDescendantsMap(
        nodes: [ClusterHierarchyNode],
        nodeByID: [Int: Int]
    ) -> [Int: LeafInfo] {
        var leafMap = [Int: LeafInfo]()
        leafMap.reserveCapacity(nodes.count)

        // Process in bottom-up order: leaves first, then by increasing size
        // This ensures children are always processed before their parents
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

                // Pre-allocate based on node size (each point appears exactly once)
                allLeaves.reserveCapacity(node.size)
                allDeaths.reserveCapacity(node.size)

                for childID in node.children {
                    if let childInfo = leafMap[childID] {
                        allLeaves.append(contentsOf: childInfo.leafIDs)
                        allDeaths.append(contentsOf: childInfo.deathDistances)
                    }
                }

                leafMap[node.id] = LeafInfo(
                    leafIDs: allLeaves,
                    deathDistances: allDeaths
                )
            }
        }

        return leafMap
    }

    /// Computes stability for a single node using precomputed leaf info.
    ///
    /// Same mathematical formula as `computeNodeStability()`, but uses
    /// precomputed leaf descendants instead of recursive traversal.
    ///
    /// - Parameters:
    ///   - node: The hierarchy node.
    ///   - leafInfo: Precomputed leaf descendant information.
    /// - Returns: Stability score.
    private func computeNodeStabilityFast(
        node: ClusterHierarchyNode,
        leafInfo: LeafInfo
    ) -> Float {
        let birthDistance = node.birthLevel
        let deathDistance = node.deathLevel

        // Avoid division by zero
        guard birthDistance > 0 || deathDistance > 0 else {
            return 0
        }

        // Convert to lambda (density) space
        // λ = 1/distance, so λ_birth = 1/birth_distance
        let lambdaBirth = birthDistance > Float.ulpOfOne
            ? 1.0 / birthDistance
            : Float.infinity
        let lambdaDeath = deathDistance > Float.ulpOfOne && deathDistance < Float.infinity
            ? 1.0 / deathDistance
            : 0

        // For leaf nodes (empty leafInfo means single point)
        if leafInfo.leafIDs.isEmpty {
            return max(0, lambdaBirth - lambdaDeath)
        }

        // For internal nodes: accumulate contributions from all leaf descendants
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
}

// MARK: - Internal Node Data

/// Data for an internal node during construction.
private struct InternalNodeData {
    let id: Int
    let childA: Int
    let childB: Int
    let sizeA: Int
    let sizeB: Int
    let birthDistance: Float
}

// MARK: - Leaf Info (for Bottom-Up Stability Computation)

/// Precomputed leaf descendant information for a node.
///
/// Used to avoid O(n²) recursive traversal during stability computation.
/// Each node stores its leaf descendants' IDs and death distances.
private struct LeafInfo {
    /// IDs of all leaf descendants.
    var leafIDs: [Int]
    /// Death distances of all leaf descendants (parallel to leafIDs).
    var deathDistances: [Float]

    /// Empty leaf info for convenience.
    static let empty = LeafInfo(leafIDs: [], deathDistances: [])
}

// MARK: - Hierarchy Union Find

/// Union-Find that tracks cluster IDs for hierarchy construction.
private struct HierarchyUnionFind {

    /// Parent pointers.
    var parent: [Int]

    /// Rank for union by rank.
    var rank: [Int]

    /// Size of each set.
    var size: [Int]

    /// Current cluster ID for each root.
    var clusterID: [Int]

    init(count: Int) {
        self.parent = Array(0..<count)
        self.rank = [Int](repeating: 0, count: count)
        self.size = [Int](repeating: 1, count: count)
        // Initially, each point is its own cluster with ID = point index
        self.clusterID = Array(0..<count)
    }

    mutating func find(_ x: Int) -> Int {
        if parent[x] != x {
            parent[x] = find(parent[x])
        }
        return parent[x]
    }

    mutating func unionAt(_ a: Int, _ b: Int, newClusterID: Int) {
        let rootA = find(a)
        let rootB = find(b)

        guard rootA != rootB else { return }

        // Union by rank
        let newRoot: Int
        if rank[rootA] < rank[rootB] {
            parent[rootA] = rootB
            size[rootB] += size[rootA]
            newRoot = rootB
        } else if rank[rootA] > rank[rootB] {
            parent[rootB] = rootA
            size[rootA] += size[rootB]
            newRoot = rootA
        } else {
            parent[rootB] = rootA
            rank[rootA] += 1
            size[rootA] += size[rootB]
            newRoot = rootA
        }

        // Update cluster ID of the new root
        clusterID[newRoot] = newClusterID
    }
}

// MARK: - Condensed Tree

/// A condensed view of the cluster tree for cluster extraction.
///
/// The condensed tree removes clusters smaller than minClusterSize,
/// keeping only significant clusters for selection.
public struct CondensedTree: Sendable {

    /// Nodes in the condensed tree.
    public let nodes: [CondensedTreeNode]

    /// Root node ID.
    public let rootID: Int

    /// Creates a condensed tree from a full hierarchy.
    ///
    /// - Parameters:
    ///   - hierarchy: The full cluster hierarchy.
    ///   - minClusterSize: Minimum size for a node to be retained.
    /// - Complexity: O(n) where n is the number of nodes in the hierarchy.
    public init(hierarchy: ClusterHierarchy, minClusterSize: Int) {
        var condensedNodes = [CondensedTreeNode]()

        // Filter to nodes that meet minimum size
        let validNodes = hierarchy.nodes.filter { $0.size >= minClusterSize }

        // Build parent map for valid nodes
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
                // Find this parent's parent using O(1) lookup
                currentParent = hierarchy.node(id: p)?.parent
            }

            // Find effective children (valid descendants)
            var effectiveChildren = [Int]()
            var queue = node.children
            while !queue.isEmpty {
                let childID = queue.removeFirst()
                if validIDs.contains(childID) {
                    effectiveChildren.append(childID)
                } else if let child = hierarchy.node(id: childID) {
                    // Use O(1) lookup instead of O(n) search
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

    /// Gets a node by ID.
    public func node(id: Int) -> CondensedTreeNode? {
        nodes.first { $0.id == id }
    }
}

// MARK: - Condensed Tree Node

/// A node in the condensed tree.
public struct CondensedTreeNode: Sendable, Identifiable {

    /// Unique identifier.
    public let id: Int

    /// Parent node ID.
    public let parent: Int?

    /// Child node IDs.
    public let children: [Int]

    /// Birth level (distance at which cluster formed).
    public let birthLevel: Float

    /// Death level (distance at which cluster merged).
    public let deathLevel: Float

    /// Number of points.
    public let size: Int

    /// Stability score.
    public let stability: Float

    /// Whether this is a leaf in the condensed tree.
    public var isLeaf: Bool { children.isEmpty }
}
