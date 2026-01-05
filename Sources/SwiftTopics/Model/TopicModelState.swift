// TopicModelState.swift
// SwiftTopics
//
// Serialization support for fitted topic models

import Foundation

// MARK: - Topic Model State

/// Serializable state of a fitted topic model.
///
/// Use `TopicModelState` to save and load fitted models:
///
/// ```swift
/// // Save
/// let state = try await model.exportState()
/// let data = try JSONEncoder().encode(state)
/// try data.write(to: saveURL)
///
/// // Load
/// let loadedData = try Data(contentsOf: saveURL)
/// let loadedState = try JSONDecoder().decode(TopicModelState.self, from: loadedData)
/// let loadedModel = try await TopicModel.restore(from: loadedState)
/// ```
///
/// ## Contents
///
/// The state includes everything needed to:
/// - Assign new documents to existing topics (transform)
/// - Display topic information (keywords, sizes)
/// - Reproduce the same results (configuration, seed)
///
/// ## Thread Safety
///
/// `TopicModelState` is `Sendable` and `Codable`.
public struct TopicModelState: Sendable, Codable {

    // MARK: - Properties

    /// Version number for compatibility checking.
    ///
    /// Increment when the state format changes in incompatible ways.
    public let version: Int

    /// Configuration used to train the model.
    public let configuration: TopicModelConfiguration

    /// Discovered topics with keywords and metadata.
    public let topics: [Topic]

    /// Cluster assignments for training documents.
    public let assignments: ClusterAssignment

    /// PCA transformation matrix (column-major, d × k).
    ///
    /// Used for projecting new embeddings to the reduced space.
    /// Nil if reduction was skipped or not using PCA.
    public let pcaComponents: [Float]?

    /// PCA mean vector for centering.
    ///
    /// Subtract from embeddings before projecting.
    public let pcaMean: [Float]?

    /// Topic centroids in original embedding space.
    ///
    /// Used for assigning new documents to topics.
    public let centroids: [Embedding]?

    /// Input embedding dimension.
    public let inputDimension: Int

    /// Reduced embedding dimension (after PCA).
    public let reducedDimension: Int

    /// When the model was trained.
    public let trainedAt: Date

    /// Number of documents used for training.
    public let documentCount: Int

    /// Random seed used for reproducibility.
    public let seed: UInt64?

    // MARK: - Current Version

    /// Current state format version.
    public static let currentVersion = 1

    // MARK: - Initialization

    /// Creates a topic model state.
    public init(
        version: Int = TopicModelState.currentVersion,
        configuration: TopicModelConfiguration,
        topics: [Topic],
        assignments: ClusterAssignment,
        pcaComponents: [Float]?,
        pcaMean: [Float]?,
        centroids: [Embedding]?,
        inputDimension: Int,
        reducedDimension: Int,
        trainedAt: Date,
        documentCount: Int,
        seed: UInt64?
    ) {
        self.version = version
        self.configuration = configuration
        self.topics = topics
        self.assignments = assignments
        self.pcaComponents = pcaComponents
        self.pcaMean = pcaMean
        self.centroids = centroids
        self.inputDimension = inputDimension
        self.reducedDimension = reducedDimension
        self.trainedAt = trainedAt
        self.documentCount = documentCount
        self.seed = seed
    }

    // MARK: - Validation

    /// Validates the state for consistency.
    ///
    /// - Throws: `TopicModelError.serializationFailed` if validation fails.
    public func validate() throws {
        // Check version compatibility
        guard version <= TopicModelState.currentVersion else {
            throw TopicModelError.serializationFailed(
                "State version \(version) is newer than supported version \(TopicModelState.currentVersion)"
            )
        }

        // Topics can be empty if all points were outliers
        // So we don't require topics to exist

        // Check centroids match topics if present
        if let centroids = centroids, !topics.isEmpty {
            guard centroids.count == topics.count else {
                throw TopicModelError.serializationFailed(
                    "Centroid count (\(centroids.count)) doesn't match topic count (\(topics.count))"
                )
            }
        }

        // Check dimensions
        guard inputDimension > 0 else {
            throw TopicModelError.serializationFailed("Invalid input dimension: \(inputDimension)")
        }

        guard reducedDimension > 0 else {
            throw TopicModelError.serializationFailed("Invalid reduced dimension: \(reducedDimension)")
        }
    }

    // MARK: - Summary

    /// Human-readable summary of the state.
    public var summary: String {
        var lines: [String] = []

        lines.append("TopicModelState v\(version)")
        lines.append("  Topics: \(topics.count)")
        lines.append("  Documents: \(documentCount)")
        lines.append("  Dimensions: \(inputDimension) → \(reducedDimension)")
        lines.append("  Trained: \(trainedAt)")

        if let seed = seed {
            lines.append("  Seed: \(seed)")
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - TopicModel Serialization Extensions

extension TopicModel {

    /// Exports the current model state for serialization.
    ///
    /// - Returns: The model state.
    /// - Throws: `TopicModelError.notFitted` if model not fitted.
    public func exportState() async throws -> TopicModelState {
        guard isFitted else {
            throw TopicModelError.notFitted
        }

        guard let topics = topics else {
            throw TopicModelError.notFitted
        }

        // Get internal state components
        let stateInfo = getStateInfo()

        return TopicModelState(
            version: TopicModelState.currentVersion,
            configuration: configuration,
            topics: topics,
            assignments: stateInfo.assignment,
            pcaComponents: stateInfo.pcaComponents,
            pcaMean: stateInfo.pcaMean,
            centroids: stateInfo.centroids,
            inputDimension: stateInfo.inputDimension,
            reducedDimension: stateInfo.reducedDimension,
            trainedAt: Date(),
            documentCount: stateInfo.assignment.pointCount,
            seed: configuration.seed
        )
    }

    /// Creates a fitted model from exported state.
    ///
    /// - Parameter state: The saved model state.
    /// - Returns: A fitted topic model.
    /// - Throws: `TopicModelError.serializationFailed` if state is invalid.
    public static func restore(from state: TopicModelState) async throws -> TopicModel {
        // Validate state
        try state.validate()

        // Create model with stored configuration
        let model = TopicModel(configuration: state.configuration)

        // Restore fitted state
        await model.restoreInternalState(
            topics: state.topics,
            assignment: state.assignments,
            pcaComponents: state.pcaComponents,
            pcaMean: state.pcaMean,
            inputDimension: state.inputDimension,
            reducedDimension: state.reducedDimension,
            centroids: state.centroids ?? []
        )

        return model
    }

    // MARK: - Internal State Access

    /// Internal structure for exporting state info.
    private struct StateInfo {
        let assignment: ClusterAssignment
        let pcaComponents: [Float]?
        let pcaMean: [Float]?
        let inputDimension: Int
        let reducedDimension: Int
        let centroids: [Embedding]
    }

    /// Gets internal state for export.
    private func getStateInfo() -> StateInfo {
        if let state = fittedState {
            return StateInfo(
                assignment: state.assignment,
                pcaComponents: state.pcaComponents,
                pcaMean: state.pcaMean,
                inputDimension: state.inputDimension,
                reducedDimension: state.reducedDimension,
                centroids: state.centroids
            )
        }
        return StateInfo(
            assignment: ClusterAssignment(labels: [], probabilities: [], outlierScores: [], clusterCount: 0),
            pcaComponents: nil,
            pcaMean: nil,
            inputDimension: 0,
            reducedDimension: 0,
            centroids: []
        )
    }

    /// Restores internal fitted state from serialized data.
    ///
    /// Note: Documents and embeddings are not serialized due to size constraints.
    /// Restored models will not support `search()` functionality.
    private func restoreInternalState(
        topics: [Topic],
        assignment: ClusterAssignment,
        pcaComponents: [Float]?,
        pcaMean: [Float]?,
        inputDimension: Int,
        reducedDimension: Int,
        centroids: [Embedding]
    ) {
        fittedState = FittedTopicModelState(
            topics: topics,
            assignment: assignment,
            pcaComponents: pcaComponents,
            pcaMean: pcaMean,
            inputDimension: inputDimension,
            reducedDimension: reducedDimension,
            centroids: centroids,
            documents: [],  // Not serialized - search() won't work on restored models
            embeddings: []  // Not serialized - search() won't work on restored models
        )
    }
}

// MARK: - Convenience Methods

extension TopicModelState {

    /// Encodes the state to JSON data.
    ///
    /// - Returns: JSON-encoded data.
    /// - Throws: Encoding errors.
    public func toJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }

    /// Creates a state from JSON data.
    ///
    /// - Parameter data: JSON-encoded data.
    /// - Returns: Decoded state.
    /// - Throws: Decoding errors.
    public static func fromJSON(_ data: Data) throws -> TopicModelState {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(TopicModelState.self, from: data)
    }

    /// Saves the state to a file.
    ///
    /// - Parameter url: File URL to save to.
    /// - Throws: File writing errors.
    public func save(to url: URL) throws {
        let data = try toJSON()
        try data.write(to: url, options: .atomic)
    }

    /// Loads a state from a file.
    ///
    /// - Parameter url: File URL to load from.
    /// - Returns: Loaded state.
    /// - Throws: File reading or decoding errors.
    public static func load(from url: URL) throws -> TopicModelState {
        let data = try Data(contentsOf: url)
        return try fromJSON(data)
    }
}

// MARK: - State Migration

extension TopicModelState {

    /// Migrates state from an older version if needed.
    ///
    /// - Returns: Migrated state compatible with current version.
    /// - Throws: If migration is not possible.
    public func migrateIfNeeded() throws -> TopicModelState {
        // Version 1 is current, no migration needed
        guard version < TopicModelState.currentVersion else {
            return self
        }

        // Future: Add migration logic for each version bump
        // For now, we only support version 1
        throw TopicModelError.serializationFailed(
            "Cannot migrate from version \(version) to \(TopicModelState.currentVersion)"
        )
    }
}
