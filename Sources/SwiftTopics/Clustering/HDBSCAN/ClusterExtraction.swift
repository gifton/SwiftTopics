// ClusterExtraction.swift
// SwiftTopics
//
// Cluster extraction from HDBSCAN hierarchy using EOM or leaf methods

import Foundation

// MARK: - Cluster Extractor

/// Extracts flat cluster assignments from the HDBSCAN hierarchy.
///
/// ## Extraction Methods
///
/// ### Excess of Mass (EOM)
///
/// EOM selects clusters that maximize total stability. It uses dynamic
/// programming to decide at each node whether to:
/// - Select this cluster (take its stability)
/// - Pass through to children (take sum of children's selected stability)
///
/// The choice that maximizes total stability wins.
///
/// ### Leaf Clustering
///
/// Leaf clustering simply takes all leaf clusters in the condensed tree.
/// This produces more fine-grained clusters but may fragment data.
///
/// ## Outlier Detection
///
/// Points that fall out of all selected clusters are marked as outliers
/// (label = -1). The outlier score is based on how much a point deviates
/// from its nearest cluster.
public struct ClusterExtractor: Sendable {

    /// The extraction method.
    public let method: ClusterSelectionMethod

    /// Minimum cluster size.
    public let minClusterSize: Int

    /// Cluster selection epsilon (for merging small clusters).
    public let epsilon: Float

    /// Whether to allow a single cluster result.
    public let allowSingleCluster: Bool

    /// Creates a cluster extractor.
    ///
    /// - Parameters:
    ///   - method: Extraction method (EOM or leaf).
    ///   - minClusterSize: Minimum size for a valid cluster.
    ///   - epsilon: Distance threshold for cluster merging.
    ///   - allowSingleCluster: Whether to allow single cluster results.
    public init(
        method: ClusterSelectionMethod = .eom,
        minClusterSize: Int = 5,
        epsilon: Float = 0.0,
        allowSingleCluster: Bool = false
    ) {
        self.method = method
        self.minClusterSize = minClusterSize
        self.epsilon = epsilon
        self.allowSingleCluster = allowSingleCluster
    }

    // MARK: - Extract Clusters

    /// Extracts cluster assignments from the hierarchy.
    ///
    /// - Parameters:
    ///   - hierarchy: The cluster hierarchy.
    ///   - pointCount: Number of original points.
    ///   - coreDistances: Core distances for outlier scoring.
    /// - Returns: Cluster assignment result.
    public func extract(
        from hierarchy: ClusterHierarchy,
        pointCount: Int,
        coreDistances: [Float]
    ) -> ClusterAssignment {
        // Build condensed tree
        let condensedTree = CondensedTree(
            hierarchy: hierarchy,
            minClusterSize: minClusterSize
        )

        // Select clusters based on method
        let selectedClusterIDs: Set<Int>

        switch method {
        case .eom:
            selectedClusterIDs = selectClustersEOM(
                condensedTree: condensedTree,
                hierarchy: hierarchy,
                epsilon: epsilon,
                allowSingleCluster: allowSingleCluster
            )
        case .leaf:
            selectedClusterIDs = selectClustersLeaf(condensedTree: condensedTree)
        }

        // Assign points to clusters
        return assignPointsToClusters(
            selectedClusterIDs: selectedClusterIDs,
            hierarchy: hierarchy,
            pointCount: pointCount,
            coreDistances: coreDistances
        )
    }

    // MARK: - EOM Selection

    /// Selects clusters using Excess of Mass method.
    ///
    /// This is a dynamic programming algorithm that maximizes total stability.
    private func selectClustersEOM(
        condensedTree: CondensedTree,
        hierarchy: ClusterHierarchy,
        epsilon: Float,
        allowSingleCluster: Bool
    ) -> Set<Int> {
        guard !condensedTree.nodes.isEmpty else {
            return []
        }

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
            // Leaves before internal nodes, then by size
            if a.isLeaf != b.isLeaf {
                return a.isLeaf
            }
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
                    var queue = node.children
                    while !queue.isEmpty {
                        let childID = queue.removeFirst()
                        isSelected[childID] = false
                        if let child = nodeByID[childID] {
                            queue.append(contentsOf: child.children)
                        }
                    }
                } else {
                    // Pass through to children
                    subtreeStability[node.id] = childrenStability
                    isSelected[node.id] = false
                }
            }
        }

        // Apply epsilon merging if specified
        if epsilon > 0 {
            applyEpsilonMerging(
                isSelected: &isSelected,
                nodeByID: nodeByID,
                epsilon: epsilon
            )
        }

        // Collect selected clusters
        var selected = Set<Int>()
        for (nodeID, selected_flag) in isSelected {
            if selected_flag {
                selected.insert(nodeID)
            }
        }

        // Handle single cluster case
        if !allowSingleCluster && selected.count == 1 {
            // Try to split into children if possible
            if let singleID = selected.first,
               let node = nodeByID[singleID],
               !node.children.isEmpty {
                selected.remove(singleID)
                for childID in node.children {
                    selected.insert(childID)
                }
            }
        }

        return selected
    }

    /// Applies epsilon-based cluster merging.
    private func applyEpsilonMerging(
        isSelected: inout [Int: Bool],
        nodeByID: [Int: CondensedTreeNode],
        epsilon: Float
    ) {
        // Merge clusters that are within epsilon distance
        for (nodeID, node) in nodeByID {
            guard isSelected[nodeID] == true else { continue }

            // If this cluster's birth is within epsilon of its parent's birth,
            // deselect it and select the parent instead
            if let parentID = node.parent,
               let parent = nodeByID[parentID] {
                let distance = abs(node.birthLevel - parent.birthLevel)
                if distance < epsilon {
                    isSelected[nodeID] = false
                    isSelected[parentID] = true
                }
            }
        }
    }

    // MARK: - Leaf Selection

    /// Selects all leaf clusters in the condensed tree.
    private func selectClustersLeaf(condensedTree: CondensedTree) -> Set<Int> {
        var selected = Set<Int>()

        for node in condensedTree.nodes {
            if node.isLeaf {
                selected.insert(node.id)
            }
        }

        return selected
    }

    // MARK: - Point Assignment

    /// Assigns points to selected clusters.
    private func assignPointsToClusters(
        selectedClusterIDs: Set<Int>,
        hierarchy: ClusterHierarchy,
        pointCount: Int,
        coreDistances: [Float]
    ) -> ClusterAssignment {
        guard !selectedClusterIDs.isEmpty else {
            // No clusters found - all points are outliers
            return ClusterAssignment(
                labels: [Int](repeating: -1, count: pointCount),
                probabilities: [Float](repeating: 0, count: pointCount),
                outlierScores: coreDistances.isEmpty
                    ? [Float](repeating: 1, count: pointCount)
                    : coreDistances.map { $0 / (coreDistances.max() ?? 1) },
                clusterCount: 0
            )
        }

        // Map selected cluster IDs to 0-indexed labels
        let sortedClusterIDs = Array(selectedClusterIDs).sorted()
        var clusterIDToLabel = [Int: Int]()
        for (label, clusterID) in sortedClusterIDs.enumerated() {
            clusterIDToLabel[clusterID] = label
        }

        // Build point-to-cluster mapping
        // For each point, find which selected cluster it belongs to
        var pointLabels = [Int](repeating: -1, count: pointCount)
        var pointProbabilities = [Float](repeating: 0, count: pointCount)

        // Build ancestor chains for each selected cluster
        var clusterPoints = [Int: [Int]]()  // clusterID -> point indices
        for clusterID in selectedClusterIDs {
            clusterPoints[clusterID] = []
        }

        // For each point, traverse up the hierarchy to find its selected cluster
        for pointIndex in 0..<pointCount {
            let (clusterID, probability) = findSelectedAncestor(
                pointIndex: pointIndex,
                selectedClusterIDs: selectedClusterIDs,
                hierarchy: hierarchy
            )

            if let clusterID = clusterID, let label = clusterIDToLabel[clusterID] {
                pointLabels[pointIndex] = label
                pointProbabilities[pointIndex] = probability
                clusterPoints[clusterID]?.append(pointIndex)
            }
            // If no selected ancestor found, point remains outlier (label = -1)
        }

        // Compute outlier scores
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

    /// Finds the selected cluster ancestor for a point.
    private func findSelectedAncestor(
        pointIndex: Int,
        selectedClusterIDs: Set<Int>,
        hierarchy: ClusterHierarchy
    ) -> (clusterID: Int?, probability: Float) {
        // Start from the point's leaf node (which has id = pointIndex)
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

        // Probability based on how deep the point is in the cluster
        // Higher depth = point joined cluster later = lower probability
        let probability = lastValidClusterID != nil
            ? max(0.1, 1.0 - Float(depth) * 0.1)
            : 0.0

        return (lastValidClusterID, probability)
    }

    /// Computes outlier scores for all points.
    private func computeOutlierScores(
        pointLabels: [Int],
        coreDistances: [Float],
        clusterPoints: [Int: [Int]],
        clusterIDToLabel: [Int: Int]
    ) -> [Float] {
        let n = pointLabels.count

        guard !coreDistances.isEmpty else {
            // If no core distances available, use simple 0/1 scoring
            return pointLabels.map { $0 == -1 ? Float(1.0) : Float(0.0) }
        }

        let maxCoreDistance = coreDistances.max() ?? 1.0
        guard maxCoreDistance > 0 else {
            return [Float](repeating: 0, count: n)
        }

        var outlierScores = [Float](repeating: 0, count: n)

        for i in 0..<n {
            if pointLabels[i] == -1 {
                // Outlier: score based on core distance (normalized)
                outlierScores[i] = coreDistances[i] / maxCoreDistance
            } else {
                // Clustered point: inverse of membership strength
                // Points with high core distances relative to cluster average
                // have higher outlier scores even if clustered
                let clusterLabel = pointLabels[i]

                // Find average core distance in this cluster
                var clusterCoreSum: Float = 0
                var clusterCount = 0

                for (clusterID, points) in clusterPoints {
                    if clusterIDToLabel[clusterID] == clusterLabel {
                        for pointIdx in points {
                            clusterCoreSum += coreDistances[pointIdx]
                            clusterCount += 1
                        }
                    }
                }

                let clusterAvgCore = clusterCount > 0
                    ? clusterCoreSum / Float(clusterCount)
                    : maxCoreDistance

                // Score: how much this point's core distance exceeds cluster average
                let deviation = max(0, coreDistances[i] - clusterAvgCore)
                outlierScores[i] = min(1.0, deviation / maxCoreDistance)
            }
        }

        return outlierScores
    }
}

// MARK: - Membership Probability Calculator

/// Computes soft cluster membership probabilities.
///
/// Unlike hard cluster assignment, this provides a probability distribution
/// over all clusters for each point.
public struct MembershipProbabilityCalculator: Sendable {

    /// Smoothing factor for probability computation.
    public let smoothing: Float

    /// Creates a membership probability calculator.
    ///
    /// - Parameter smoothing: Smoothing factor (default: 0.01).
    public init(smoothing: Float = 0.01) {
        self.smoothing = smoothing
    }

    /// Computes membership probabilities for all points.
    ///
    /// - Parameters:
    ///   - embeddings: Point embeddings.
    ///   - clusterAssignment: The cluster assignment.
    /// - Returns: Matrix of probabilities [pointCount × clusterCount].
    public func compute(
        embeddings: [Embedding],
        clusterAssignment: ClusterAssignment
    ) -> [[Float]] {
        let n = embeddings.count
        let k = clusterAssignment.clusterCount

        guard k > 0 else {
            return [[Float]](repeating: [], count: n)
        }

        // Compute cluster centroids
        var centroids = [[Float]](repeating: [], count: k)
        var clusterSizes = [Int](repeating: 0, count: k)

        for i in 0..<n {
            let label = clusterAssignment.label(for: i)
            guard label >= 0 else { continue }

            if centroids[label].isEmpty {
                centroids[label] = [Float](repeating: 0, count: embeddings[i].dimension)
            }

            for d in 0..<embeddings[i].dimension {
                centroids[label][d] += embeddings[i].vector[d]
            }
            clusterSizes[label] += 1
        }

        // Normalize centroids
        for c in 0..<k {
            if clusterSizes[c] > 0 {
                for d in 0..<centroids[c].count {
                    centroids[c][d] /= Float(clusterSizes[c])
                }
            }
        }

        // Compute probabilities for each point
        var probabilities = [[Float]](repeating: [Float](repeating: 0, count: k), count: n)

        for i in 0..<n {
            // Compute distances to all centroids
            var distances = [Float](repeating: 0, count: k)
            var minDist: Float = .infinity

            for c in 0..<k {
                guard !centroids[c].isEmpty else {
                    distances[c] = .infinity
                    continue
                }

                var dist: Float = 0
                for d in 0..<embeddings[i].dimension {
                    let diff = embeddings[i].vector[d] - centroids[c][d]
                    dist += diff * diff
                }
                distances[c] = dist.squareRoot()
                minDist = min(minDist, distances[c])
            }

            // Convert distances to probabilities using softmax-like transform
            var expSum: Float = 0
            for c in 0..<k {
                let adjusted = max(0, distances[c] - minDist) + smoothing
                let exp = Darwin.exp(-adjusted)
                probabilities[i][c] = exp
                expSum += exp
            }

            // Normalize
            if expSum > 0 {
                for c in 0..<k {
                    probabilities[i][c] /= expSum
                }
            }
        }

        return probabilities
    }
}

// MARK: - Cluster Extraction Result

/// Result of cluster extraction with additional metadata.
public struct ClusterExtractionResult: Sendable {

    /// The cluster assignment.
    public let assignment: ClusterAssignment

    /// IDs of the selected clusters from the hierarchy.
    public let selectedClusterIDs: Set<Int>

    /// The condensed tree used for extraction.
    public let condensedTree: CondensedTree

    /// The extraction method used.
    public let method: ClusterSelectionMethod

    /// Creates a cluster extraction result.
    public init(
        assignment: ClusterAssignment,
        selectedClusterIDs: Set<Int>,
        condensedTree: CondensedTree,
        method: ClusterSelectionMethod
    ) {
        self.assignment = assignment
        self.selectedClusterIDs = selectedClusterIDs
        self.condensedTree = condensedTree
        self.method = method
    }
}
