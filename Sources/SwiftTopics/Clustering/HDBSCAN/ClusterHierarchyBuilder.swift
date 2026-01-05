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

        // Set parent pointers and death levels
        for i in 0..<nodes.count {
            let nodeID = nodes[i].id

            if let parentID = parentMap[nodeID] {
                // Find parent node index
                if let parentIdx = nodes.firstIndex(where: { $0.id == parentID }) {
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

            // Compute stability
            let stability = computeNodeStability(
                node: node,
                nodes: nodes,
                nodeByID: nodeByID
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

    /// Computes stability for a single node.
    ///
    /// The stability of a cluster is the sum over all points in the cluster of
    /// the range of density levels where each point is a member.
    private func computeNodeStability(
        node: ClusterHierarchyNode,
        nodes: [ClusterHierarchyNode],
        nodeByID: [Int: Int]
    ) -> Float {
        let birthDistance = node.birthLevel
        let deathDistance = node.deathLevel

        // Avoid division by zero
        guard birthDistance > 0 || deathDistance > 0 else {
            return 0
        }

        // Convert to lambda (density) space
        // λ = 1/distance, so λ_birth = 1/birth_distance (or 0 if birth at 0)
        let lambdaBirth = birthDistance > Float.ulpOfOne ? 1.0 / birthDistance : Float.infinity
        let lambdaDeath = deathDistance > Float.ulpOfOne && deathDistance < Float.infinity
            ? 1.0 / deathDistance
            : 0

        // For leaf nodes (individual points), stability contribution is based on
        // when they were absorbed into this cluster
        if node.children.isEmpty {
            // Single point - contributes from lambda_birth to lambda_death
            return max(0, lambdaBirth - lambdaDeath)
        }

        // For internal nodes, stability is accumulated from all descendant points
        // Each point contributes (λ_point_death - λ_cluster_birth)
        // where λ_point_death is when the point left this cluster (merged up)
        var stability: Float = 0

        // Collect all leaf descendants and their death levels within this cluster
        let leaves = collectLeafDescendants(
            nodeID: node.id,
            nodes: nodes,
            nodeByID: nodeByID
        )

        for (_, leafDeathDistance) in leaves {
            // This leaf died (left this cluster) at leafDeathDistance
            // It was born into this cluster at birthDistance
            let leafLambdaDeath = leafDeathDistance > Float.ulpOfOne && leafDeathDistance < Float.infinity
                ? 1.0 / leafDeathDistance
                : 0

            // Contribution is (λ_death - λ_birth) for this point's membership
            // But we want the minimum of cluster death and point's individual death
            let effectiveLambdaDeath = max(leafLambdaDeath, lambdaDeath)
            let contribution = max(0, lambdaBirth - effectiveLambdaDeath)
            stability += contribution
        }

        return stability
    }

    /// Collects all leaf descendants of a node with their death distances.
    private func collectLeafDescendants(
        nodeID: Int,
        nodes: [ClusterHierarchyNode],
        nodeByID: [Int: Int]
    ) -> [(leafID: Int, deathDistance: Float)] {
        guard let nodeIdx = nodeByID[nodeID] else { return [] }
        let node = nodes[nodeIdx]

        if node.children.isEmpty {
            // This is a leaf - it dies when absorbed into parent
            return [(nodeID, node.deathLevel)]
        }

        // Recursively collect from children
        var leaves = [(Int, Float)]()
        for childID in node.children {
            leaves.append(contentsOf: collectLeafDescendants(
                nodeID: childID,
                nodes: nodes,
                nodeByID: nodeByID
            ))
        }
        return leaves
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
                // Find this parent's parent in original hierarchy
                currentParent = hierarchy.nodes.first(where: { $0.id == p })?.parent
            }

            // Find effective children (valid descendants)
            var effectiveChildren = [Int]()
            var queue = node.children
            while !queue.isEmpty {
                let childID = queue.removeFirst()
                if validIDs.contains(childID) {
                    effectiveChildren.append(childID)
                } else if let child = hierarchy.nodes.first(where: { $0.id == childID }) {
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
