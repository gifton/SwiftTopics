// ClusterAssignment.swift
// SwiftTopics
//
// Cluster assignment results from HDBSCAN clustering

import Foundation

// MARK: - Cluster Assignment

/// The result of clustering embeddings into topics.
///
/// Contains per-point cluster labels, membership probabilities, and outlier scores.
/// This is the output of the `ClusteringEngine` protocol.
///
/// ## Labels
/// - Cluster labels are 0-indexed integers
/// - A label of `-1` indicates an outlier (no cluster assignment)
///
/// ## Probabilities
/// - Membership probabilities indicate confidence in the cluster assignment
/// - Range: [0, 1] where 1 = highly confident, 0 = very uncertain
/// - Outliers have probability 0
///
/// ## Outlier Scores
/// - Higher scores indicate more "outlier-like" points
/// - Can be used for filtering or visualization
///
/// ## Thread Safety
/// `ClusterAssignment` is `Sendable` and can be safely shared across concurrency domains.
public struct ClusterAssignment: Sendable, Codable {

    /// Cluster label for each point (-1 = outlier).
    public let labels: [Int]

    /// Membership probability for each point's cluster assignment.
    public let probabilities: [Float]

    /// Outlier score for each point (higher = more outlier-like).
    public let outlierScores: [Float]

    /// The number of clusters (excluding outliers).
    public let clusterCount: Int

    /// The number of points.
    public var pointCount: Int {
        labels.count
    }

    /// Creates a cluster assignment result.
    ///
    /// - Parameters:
    ///   - labels: Cluster label per point (-1 = outlier).
    ///   - probabilities: Membership probability per point.
    ///   - outlierScores: Outlier score per point.
    ///   - clusterCount: Number of clusters found.
    public init(
        labels: [Int],
        probabilities: [Float],
        outlierScores: [Float],
        clusterCount: Int
    ) {
        precondition(labels.count == probabilities.count, "Labels and probabilities must have same count")
        precondition(labels.count == outlierScores.count, "Labels and outlier scores must have same count")
        precondition(clusterCount >= 0, "Cluster count must be non-negative")

        self.labels = labels
        self.probabilities = probabilities
        self.outlierScores = outlierScores
        self.clusterCount = clusterCount
    }

    /// Creates a cluster assignment with just labels.
    ///
    /// Probabilities default to 1.0 for clustered points, 0.0 for outliers.
    /// Outlier scores default to 0.0.
    ///
    /// - Parameters:
    ///   - labels: Cluster label per point (-1 = outlier).
    ///   - clusterCount: Number of clusters found.
    public init(labels: [Int], clusterCount: Int) {
        let probabilities = labels.map { $0 >= 0 ? Float(1.0) : Float(0.0) }
        let outlierScores = [Float](repeating: 0, count: labels.count)
        self.init(
            labels: labels,
            probabilities: probabilities,
            outlierScores: outlierScores,
            clusterCount: clusterCount
        )
    }

    // MARK: - Accessors

    /// Returns the cluster label for a point.
    ///
    /// - Parameter index: The point index.
    /// - Returns: The cluster label (-1 for outliers).
    public func label(for index: Int) -> Int {
        labels[index]
    }

    /// Returns whether a point is an outlier.
    ///
    /// - Parameter index: The point index.
    /// - Returns: True if the point is an outlier.
    public func isOutlier(_ index: Int) -> Bool {
        labels[index] == -1
    }

    /// Returns the membership probability for a point.
    ///
    /// - Parameter index: The point index.
    /// - Returns: The membership probability [0, 1].
    public func probability(for index: Int) -> Float {
        probabilities[index]
    }

    /// Returns the outlier score for a point.
    ///
    /// - Parameter index: The point index.
    /// - Returns: The outlier score.
    public func outlierScore(for index: Int) -> Float {
        outlierScores[index]
    }

    // MARK: - Aggregate Queries

    /// Indices of all points assigned to a specific cluster.
    ///
    /// - Parameter clusterID: The cluster ID (0 or greater).
    /// - Returns: Indices of points in that cluster.
    public func pointsInCluster(_ clusterID: Int) -> [Int] {
        labels.indices.filter { labels[$0] == clusterID }
    }

    /// Indices of all outlier points.
    public var outlierIndices: [Int] {
        labels.indices.filter { labels[$0] == -1 }
    }

    /// Number of outliers.
    public var outlierCount: Int {
        labels.filter { $0 == -1 }.count
    }

    /// Outlier rate as a fraction (0-1).
    public var outlierRate: Float {
        guard pointCount > 0 else { return 0 }
        return Float(outlierCount) / Float(pointCount)
    }

    /// Size of each cluster (indexed by cluster ID).
    public var clusterSizes: [Int] {
        var sizes = [Int](repeating: 0, count: clusterCount)
        for label in labels where label >= 0 {
            sizes[label] += 1
        }
        return sizes
    }

    /// The largest cluster size.
    public var maxClusterSize: Int {
        clusterSizes.max() ?? 0
    }

    /// The smallest cluster size.
    public var minClusterSize: Int {
        clusterSizes.min() ?? 0
    }

    /// Average cluster size (non-outlier points / cluster count).
    public var averageClusterSize: Float {
        guard clusterCount > 0 else { return 0 }
        let nonOutlierCount = pointCount - outlierCount
        return Float(nonOutlierCount) / Float(clusterCount)
    }
}

// MARK: - Document Cluster Assignment

/// Maps documents to their cluster assignments.
///
/// Bridges the gap between document IDs and the index-based cluster assignment.
public struct DocumentClusterAssignment: Sendable, Codable {

    /// The underlying cluster assignment.
    public let assignment: ClusterAssignment

    /// Mapping from index to document ID.
    public let documentIDs: [DocumentID]

    /// Creates a document cluster assignment.
    ///
    /// - Parameters:
    ///   - assignment: The cluster assignment result.
    ///   - documentIDs: Document IDs in the same order as the assignment labels.
    public init(assignment: ClusterAssignment, documentIDs: [DocumentID]) {
        precondition(
            assignment.pointCount == documentIDs.count,
            "Assignment and document IDs must have same count"
        )
        self.assignment = assignment
        self.documentIDs = documentIDs
    }

    /// The number of clusters.
    public var clusterCount: Int {
        assignment.clusterCount
    }

    /// The number of documents.
    public var documentCount: Int {
        documentIDs.count
    }

    /// Gets the cluster label for a document.
    ///
    /// - Parameter documentID: The document ID.
    /// - Returns: The cluster label, or nil if document not found.
    public func label(for documentID: DocumentID) -> Int? {
        guard let index = documentIDs.firstIndex(of: documentID) else {
            return nil
        }
        return assignment.label(for: index)
    }

    /// Gets the topic ID for a document.
    ///
    /// - Parameter documentID: The document ID.
    /// - Returns: The topic ID (including outlier), or nil if document not found.
    public func topicID(for documentID: DocumentID) -> TopicID? {
        guard let label = label(for: documentID) else {
            return nil
        }
        return TopicID(value: label)
    }

    /// Gets documents assigned to a specific cluster.
    ///
    /// - Parameter clusterID: The cluster ID.
    /// - Returns: Document IDs in that cluster.
    public func documents(inCluster clusterID: Int) -> [DocumentID] {
        assignment.pointsInCluster(clusterID).map { documentIDs[$0] }
    }

    /// Gets all outlier documents.
    public var outlierDocuments: [DocumentID] {
        assignment.outlierIndices.map { documentIDs[$0] }
    }

    /// Creates a dictionary mapping document IDs to topic IDs.
    public func toDocumentTopicMap() -> [DocumentID: TopicID] {
        var map = [DocumentID: TopicID]()
        for (index, documentID) in documentIDs.enumerated() {
            map[documentID] = TopicID(value: assignment.label(for: index))
        }
        return map
    }
}

// MARK: - Cluster Hierarchy Node

/// A node in the HDBSCAN cluster hierarchy tree.
///
/// The hierarchy represents how clusters merge at different density levels.
/// Used internally by HDBSCAN for cluster extraction.
public struct ClusterHierarchyNode: Sendable, Codable, Identifiable {

    /// Unique identifier for this node.
    public let id: Int

    /// Parent node ID, if any.
    public let parent: Int?

    /// Child node IDs.
    public let children: [Int]

    /// The density level at which this cluster formed (birth).
    public let birthLevel: Float

    /// The density level at which this cluster merged into its parent (death).
    public let deathLevel: Float

    /// Number of points in this cluster.
    public let size: Int

    /// Stability measure (persistence × size).
    ///
    /// Higher stability indicates more robust clusters.
    public var stability: Float

    /// Whether this is a leaf node (no children).
    public var isLeaf: Bool {
        children.isEmpty
    }

    /// The persistence (lifespan) of this cluster.
    ///
    /// Measured as the difference between death and birth levels.
    public var persistence: Float {
        deathLevel - birthLevel
    }

    /// Creates a cluster hierarchy node.
    public init(
        id: Int,
        parent: Int?,
        children: [Int],
        birthLevel: Float,
        deathLevel: Float,
        size: Int,
        stability: Float
    ) {
        self.id = id
        self.parent = parent
        self.children = children
        self.birthLevel = birthLevel
        self.deathLevel = deathLevel
        self.size = size
        self.stability = stability
    }
}

// MARK: - Cluster Hierarchy

/// The complete HDBSCAN cluster hierarchy.
///
/// Contains all nodes in the hierarchy tree, enabling cluster extraction
/// via the Excess of Mass (EOM) or leaf methods.
///
/// ## Performance
///
/// Node lookups are O(1) via an internal index dictionary, enabling efficient
/// hierarchy traversal during cluster extraction.
public struct ClusterHierarchy: Sendable {

    /// All nodes in the hierarchy.
    public let nodes: [ClusterHierarchyNode]

    /// The root node ID.
    public let rootID: Int

    /// O(1) lookup index: nodeID → array index
    ///
    /// Built during initialization, not serialized (reconstructed on decode).
    private let nodeIndex: [Int: Int]

    /// Creates a cluster hierarchy.
    ///
    /// - Parameters:
    ///   - nodes: All hierarchy nodes.
    ///   - rootID: The ID of the root node.
    /// - Complexity: O(n) to build the lookup index.
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

    /// Gets a node by ID.
    ///
    /// - Parameter id: The node ID.
    /// - Returns: The node, or nil if not found.
    /// - Complexity: O(1)
    public func node(id: Int) -> ClusterHierarchyNode? {
        guard let idx = nodeIndex[id] else { return nil }
        return nodes[idx]
    }

    /// Gets the array index for a node ID.
    ///
    /// - Parameter id: The node ID.
    /// - Returns: The array index, or nil if not found.
    /// - Complexity: O(1)
    public func index(for id: Int) -> Int? {
        nodeIndex[id]
    }

    /// Gets the root node.
    public var root: ClusterHierarchyNode? {
        node(id: rootID)
    }

    /// All leaf nodes.
    public var leaves: [ClusterHierarchyNode] {
        nodes.filter(\.isLeaf)
    }

    /// The maximum depth of the hierarchy.
    public var maxDepth: Int {
        func depth(nodeID: Int, current: Int) -> Int {
            guard let node = node(id: nodeID) else { return current }
            if node.children.isEmpty {
                return current
            }
            return node.children.map { depth(nodeID: $0, current: current + 1) }.max() ?? current
        }
        return depth(nodeID: rootID, current: 0)
    }
}

// MARK: - ClusterHierarchy + Codable

extension ClusterHierarchy: Codable {

    enum CodingKeys: String, CodingKey {
        case nodes, rootID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let nodes = try container.decode([ClusterHierarchyNode].self, forKey: .nodes)
        let rootID = try container.decode(Int.self, forKey: .rootID)
        // Delegate to main initializer which builds the index
        self.init(nodes: nodes, rootID: rootID)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nodes, forKey: .nodes)
        try container.encode(rootID, forKey: .rootID)
        // nodeIndex is not encoded - it's rebuilt on decode
    }
}
