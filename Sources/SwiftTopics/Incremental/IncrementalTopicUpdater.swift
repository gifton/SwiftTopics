// IncrementalTopicUpdater.swift
// SwiftTopics
//
// Main actor for incremental topic model updates

import Foundation

// MARK: - Incremental Update Error

/// Errors that can occur during incremental updates.
public enum IncrementalUpdateError: Error, Sendable {

    /// No model has been initialized yet.
    case modelNotInitialized

    /// Embedding dimension doesn't match existing model.
    case embeddingDimensionMismatch(expected: Int, got: Int)

    /// Storage operation failed.
    case storageError(underlying: Error)

    /// Training was interrupted.
    case trainingInterrupted(phase: TrainingPhase, progress: Float)

    /// Insufficient documents for the requested operation.
    case insufficientDocuments(required: Int, provided: Int)

    /// Configuration is invalid.
    case invalidConfiguration(issues: [String])

    /// Resume failed because no checkpoint exists.
    case noCheckpointToResume
}

extension IncrementalUpdateError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized:
            return "No topic model has been initialized yet"
        case .embeddingDimensionMismatch(let expected, let got):
            return "Embedding dimension mismatch: expected \(expected), got \(got)"
        case .storageError(let underlying):
            return "Storage error: \(underlying.localizedDescription)"
        case .trainingInterrupted(let phase, let progress):
            return "Training interrupted at \(phase.displayName) (\(Int(progress * 100))%)"
        case .insufficientDocuments(let required, let provided):
            return "Insufficient documents: need \(required), have \(provided)"
        case .invalidConfiguration(let issues):
            return "Invalid configuration: \(issues.joined(separator: "; "))"
        case .noCheckpointToResume:
            return "No checkpoint available to resume"
        }
    }
}

// MARK: - Incremental Topic Updater

/// Main actor for incremental topic model updates.
///
/// `IncrementalTopicUpdater` handles the complete lifecycle of incremental topic modeling:
///
/// 1. **Immediate assignment**: New documents get instant topic assignment via centroid distance
/// 2. **Background micro-retrain**: Buffered documents are periodically incorporated
/// 3. **Full refresh**: Complete retraining when drift is detected
/// 4. **Lifecycle management**: Resume interrupted training across app launches
///
/// ## Usage
///
/// ```swift
/// // Initialize
/// let updater = try await IncrementalTopicUpdater(
///     storage: FileBasedTopicModelStorage(directory: modelDir),
///     modelConfiguration: .default,
///     updateConfiguration: .default
/// )
///
/// // Resume any interrupted training on app launch
/// try await updater.resumeIfNeeded()
///
/// // Process new documents
/// let assignment = try await updater.processDocument(doc, embedding: embedding)
///
/// // Before app termination
/// try await updater.prepareForTermination()
/// ```
///
/// ## Cold Start
///
/// Before the first model is trained, documents receive outlier assignments
/// and are buffered. Once `coldStartThreshold` is reached, initial training
/// creates the first model.
///
/// ## Thread Safety
///
/// `IncrementalTopicUpdater` is an actor, ensuring all state mutations are
/// serialized and thread-safe.
public actor IncrementalTopicUpdater {

    // MARK: - Properties

    /// Storage backend for persistence.
    public let storage: TopicModelStorage

    /// Configuration for topic model training.
    public let modelConfiguration: TopicModelConfiguration

    /// Configuration for incremental updates.
    public let updateConfiguration: IncrementalUpdateConfiguration

    /// Current model state (nil during cold start).
    public private(set) var modelState: IncrementalTopicModelState?

    /// Whether a training operation is in progress.
    public private(set) var isTraining: Bool = false

    // MARK: - Private State

    /// Runner for interruptible training operations.
    private let trainingRunner: InterruptibleTrainingRunner

    /// Merger for combining mini-models with main model.
    private let merger: ModelMerger

    /// Flag to signal training cancellation.
    /// Marked nonisolated(unsafe) because it's read from Sendable closures.
    /// Writes only happen from within the actor.
    private nonisolated(unsafe) var shouldContinueTraining: Bool = true

    /// Background training task (if running).
    private var backgroundTrainingTask: Task<Void, Error>?

    // MARK: - Initialization

    /// Creates an incremental updater.
    ///
    /// Loads existing model state from storage if available.
    ///
    /// - Parameters:
    ///   - storage: Storage backend for persistence.
    ///   - modelConfiguration: Configuration for topic model training.
    ///   - updateConfiguration: Configuration for incremental updates.
    /// - Throws: `IncrementalUpdateError.invalidConfiguration` if configuration is invalid.
    public init(
        storage: TopicModelStorage,
        modelConfiguration: TopicModelConfiguration = .default,
        updateConfiguration: IncrementalUpdateConfiguration = .default
    ) async throws {
        // Validate configuration
        let issues = updateConfiguration.validate()
        if !issues.isEmpty {
            throw IncrementalUpdateError.invalidConfiguration(issues: issues)
        }

        self.storage = storage
        self.modelConfiguration = modelConfiguration
        self.updateConfiguration = updateConfiguration
        self.trainingRunner = InterruptibleTrainingRunner(storage: storage)
        self.merger = ModelMerger()

        // Load existing state if available
        do {
            self.modelState = try await storage.loadModelState()
        } catch {
            throw IncrementalUpdateError.storageError(underlying: error)
        }
    }

    // MARK: - Document Processing

    /// Processes a new document and returns immediate topic assignment.
    ///
    /// This method:
    /// 1. If no model exists and buffer < threshold: buffers and returns outlier
    /// 2. If no model exists and buffer >= threshold: triggers initial training
    /// 3. If model exists: assigns via centroid distance, buffers for micro-retrain
    /// 4. If buffer >= threshold: triggers background micro-retrain
    ///
    /// - Parameters:
    ///   - document: The document to process.
    ///   - embedding: Pre-computed embedding for the document.
    /// - Returns: Topic assignment (immediate result).
    /// - Throws: `IncrementalUpdateError` on failure.
    public func processDocument(
        _ document: Document,
        embedding: Embedding
    ) async throws -> IncrementalTopicAssignment {
        // Validate embedding dimension if model exists
        if let state = modelState {
            guard embedding.dimension == state.inputDimension else {
                throw IncrementalUpdateError.embeddingDimensionMismatch(
                    expected: state.inputDimension,
                    got: embedding.dimension
                )
            }
        }

        // Create buffered entry
        let tokenizedContent = tokenize(document.content)
        let entry = BufferedEntry(
            document: document,
            embedding: embedding,
            tokenizedContent: tokenizedContent
        )

        // Buffer the entry
        do {
            try await storage.appendToPendingBuffer([entry])
            try await storage.appendEmbeddings([(document.id, embedding)])
        } catch {
            throw IncrementalUpdateError.storageError(underlying: error)
        }

        // Handle based on model state
        if let state = modelState {
            // Model exists - assign via centroid
            let assignment = assignViaCentroid(embedding: embedding, state: state)

            // Update drift statistics
            var updatedStats = state.driftStatistics
            updatedStats.observe(
                distance: assignment.distanceToCentroid,
                isOutlier: assignment.isOutlier,
                windowSize: updateConfiguration.driftWindowSize
            )

            // Save updated drift statistics
            let updatedState = state.withDriftStatistics(updatedStats)
            self.modelState = updatedState
            try? await storage.saveModelState(updatedState)

            // Check if micro-retrain threshold reached
            let bufferCount = try await storage.pendingBufferCount()
            if bufferCount >= updateConfiguration.microRetrainThreshold && !isTraining {
                // Trigger background micro-retrain
                triggerBackgroundMicroRetrain()
            }

            return assignment
        } else {
            // No model yet - check if we should do initial training
            let bufferCount = try await storage.pendingBufferCount()

            if bufferCount >= updateConfiguration.coldStartThreshold && !isTraining {
                // Trigger initial training (blocking for first model)
                try await runInitialTraining()

                // Now we have a model - return real assignment
                if let state = modelState {
                    return assignViaCentroid(embedding: embedding, state: state)
                }
            }

            // Return cold start assignment (outlier)
            return .coldStart()
        }
    }

    /// Processes multiple documents in batch.
    ///
    /// More efficient than individual calls for bulk imports.
    ///
    /// - Parameters:
    ///   - documents: Documents to process.
    ///   - embeddings: Pre-computed embeddings (same order as documents).
    /// - Returns: Assignments for each document.
    /// - Throws: `IncrementalUpdateError` on failure.
    public func processDocuments(
        _ documents: [Document],
        embeddings: [Embedding]
    ) async throws -> [IncrementalTopicAssignment] {
        guard documents.count == embeddings.count else {
            throw IncrementalUpdateError.insufficientDocuments(
                required: documents.count,
                provided: embeddings.count
            )
        }

        var assignments = [IncrementalTopicAssignment]()
        assignments.reserveCapacity(documents.count)

        for (document, embedding) in zip(documents, embeddings) {
            let assignment = try await processDocument(document, embedding: embedding)
            assignments.append(assignment)
        }

        return assignments
    }

    // MARK: - Training Control

    /// Forces a micro-retrain with current buffer.
    ///
    /// Call this if you want to incorporate buffered documents
    /// before the automatic threshold is reached.
    ///
    /// - Throws: `IncrementalUpdateError` on failure.
    public func triggerMicroRetrain() async throws {
        guard modelState != nil else {
            throw IncrementalUpdateError.modelNotInitialized
        }

        guard !isTraining else {
            // Already training - skip
            return
        }

        let bufferedEntries = try await storage.drainPendingBuffer()
        guard !bufferedEntries.isEmpty else {
            // Nothing to train on
            return
        }

        try await runMicroRetrain(bufferedEntries: bufferedEntries)
    }

    /// Triggers a full model refresh.
    ///
    /// This is a long-running operation (~15-60s for 1000 docs).
    /// Should be called from background processing context.
    ///
    /// - Parameter progress: Optional progress callback.
    /// - Throws: `IncrementalUpdateError` on failure.
    public func triggerFullRefresh(
        progress: (@Sendable (TrainingProgress) async -> Void)? = nil
    ) async throws {
        guard !isTraining else {
            // Already training - skip
            return
        }

        try await runFullRefresh(progress: progress)
    }

    /// Cancels any in-progress training.
    ///
    /// Training will checkpoint and stop at next opportunity.
    public func cancelTraining() {
        shouldContinueTraining = false
        backgroundTrainingTask?.cancel()
    }

    // MARK: - Lifecycle

    /// Resumes interrupted training from checkpoint.
    ///
    /// Call this on app launch to continue any interrupted training.
    ///
    /// - Returns: True if training was resumed, false if no checkpoint.
    /// - Throws: `IncrementalUpdateError` on failure.
    @discardableResult
    public func resumeIfNeeded() async throws -> Bool {
        guard let checkpoint = try await storage.loadCheckpoint() else {
            return false
        }

        guard checkpoint.canResume else {
            // Checkpoint exhausted - clear it and start fresh
            try await storage.clearCheckpoint()
            return false
        }

        // Load documents for resume
        let documentIDs = checkpoint.documentIDs
        let embeddings = try await storage.loadEmbeddings(for: documentIDs)

        // We need documents for training - create placeholder documents
        // In a real implementation, you'd load the full documents from storage
        let documents = documentIDs.enumerated().map { index, id in
            Document(id: id, content: "")  // Placeholder - content from buffer
        }

        isTraining = true
        shouldContinueTraining = true

        defer { isTraining = false }

        let result = try await trainingRunner.resumeTraining(
            from: checkpoint,
            documents: documents,
            embeddings: embeddings,
            configuration: modelConfiguration,
            shouldContinue: { [weak self] in
                self?.shouldContinueTraining ?? false
            }
        )

        if result.isComplete, let state = result.state {
            self.modelState = state
            try await storage.saveModelState(state)
        }

        return true
    }

    /// Prepares for app termination.
    ///
    /// Saves checkpoint if training is in progress and cancels background tasks.
    public func prepareForTermination() async throws {
        cancelTraining()

        // Wait briefly for background task to checkpoint
        if let task = backgroundTrainingTask {
            _ = try? await withTimeout(seconds: 2.0) {
                try? await task.value
            }
        }
    }

    // MARK: - Queries

    /// Checks if full refresh is recommended based on drift metrics.
    ///
    /// Considers:
    /// - Growth ratio since last full retrain
    /// - Time since last full retrain
    /// - Drift statistics (distance and outlier rate)
    ///
    /// - Returns: True if full refresh is recommended.
    public func shouldTriggerFullRefresh() -> Bool {
        guard let state = modelState else { return false }

        // Check growth ratio
        if state.growthRatio >= updateConfiguration.fullRefreshGrowthRatio {
            return true
        }

        // Check time since last refresh
        if let interval = updateConfiguration.fullRefreshMaxInterval,
           let elapsed = state.timeSinceLastRetrain,
           elapsed >= interval {
            return true
        }

        // Check drift statistics
        if state.driftStatistics.needsRefresh(
            driftThreshold: updateConfiguration.driftRatioThreshold,
            outlierThreshold: updateConfiguration.outlierRateThreshold
        ) {
            return true
        }

        return false
    }

    /// Returns all topics in current model.
    ///
    /// - Returns: Topics array, or nil if no model exists.
    public func getTopics() -> [Topic]? {
        modelState?.topics
    }

    /// Returns a specific topic by ID.
    ///
    /// - Parameter id: The topic ID to find.
    /// - Returns: The topic, or nil if not found.
    public func getTopic(id: TopicID) -> Topic? {
        modelState?.topics.first { $0.id == id }
    }

    /// Returns drift statistics for monitoring.
    ///
    /// - Returns: Drift statistics, or nil if no model exists.
    public func getDriftStatistics() -> DriftStatistics? {
        modelState?.driftStatistics
    }

    /// Returns the current buffer count.
    ///
    /// - Returns: Number of entries awaiting incorporation.
    public func getPendingBufferCount() async throws -> Int {
        try await storage.pendingBufferCount()
    }

    /// Returns whether a model is currently loaded.
    public var hasModel: Bool {
        modelState != nil
    }

    /// Returns the total document count in the model.
    public var totalDocumentCount: Int {
        modelState?.totalDocumentCount ?? 0
    }

    // MARK: - Private Implementation

    /// Assigns document to nearest topic centroid.
    private func assignViaCentroid(
        embedding: Embedding,
        state: IncrementalTopicModelState
    ) -> IncrementalTopicAssignment {
        guard !state.centroids.isEmpty else {
            return .outlier(isTransformAssignment: true)
        }

        // Compute similarity to each centroid
        var bestIndex = -1
        var bestSimilarity: Float = -Float.infinity
        var similarities = [Float]()
        similarities.reserveCapacity(state.centroids.count)

        for (index, centroid) in state.centroids.enumerated() {
            let similarity = embedding.cosineSimilarity(centroid)
            similarities.append(similarity)

            if similarity > bestSimilarity {
                bestSimilarity = similarity
                bestIndex = index
            }
        }

        // Check if best match is good enough
        if bestSimilarity < updateConfiguration.transformOutlierThreshold {
            return .outlier(bestSimilarity: bestSimilarity, isTransformAssignment: true)
        }

        // Compute confidence based on margin over second-best
        let sortedSimilarities = similarities.sorted(by: >)
        let margin: Float
        if sortedSimilarities.count > 1 {
            margin = sortedSimilarities[0] - sortedSimilarities[1]
        } else {
            margin = sortedSimilarities[0]
        }

        // Confidence combines similarity and margin
        // Scale to 0-1 range
        let confidence = min(1.0, (bestSimilarity + margin) / 2)

        // Find the topic and get keywords
        let topic = state.topics.first { $0.id.value == bestIndex }
        let keywords = topic?.keywords.prefix(5).map(\.term) ?? []

        // Get the actual topic ID (may differ from array index)
        let topicID = topic?.id ?? TopicID(value: bestIndex)

        return IncrementalTopicAssignment(
            topicID: topicID,
            confidence: confidence,
            similarity: bestSimilarity,
            isTransformAssignment: true,
            topicKeywords: Array(keywords)
        )
    }

    /// Triggers background micro-retrain.
    private func triggerBackgroundMicroRetrain() {
        backgroundTrainingTask = Task {
            do {
                try await self.triggerMicroRetrain()
            } catch {
                // Log error but don't propagate - this is background work
                // In production, you'd want proper error reporting
            }
        }
    }

    /// Runs initial training on buffered documents.
    private func runInitialTraining() async throws {
        let bufferedEntries = try await storage.drainPendingBuffer()

        guard bufferedEntries.count >= updateConfiguration.coldStartThreshold else {
            // Not enough documents - put them back
            try await storage.appendToPendingBuffer(bufferedEntries)
            return
        }

        isTraining = true
        shouldContinueTraining = true

        defer { isTraining = false }

        // Create documents and embeddings from buffer
        let documents = bufferedEntries.map { entry in
            Document(id: entry.documentID, content: entry.tokenizedContent.joined(separator: " "))
        }
        let embeddings = bufferedEntries.map(\.embedding)

        let result = try await trainingRunner.runTraining(
            documents: documents,
            embeddings: embeddings,
            configuration: modelConfiguration,
            type: .fullRefresh,
            shouldContinue: { [weak self] in
                self?.shouldContinueTraining ?? false
            }
        )

        if result.isComplete, let state = result.state {
            self.modelState = state
            try await storage.saveModelState(state)
        } else if let checkpoint = result.checkpoint {
            throw IncrementalUpdateError.trainingInterrupted(
                phase: checkpoint.currentPhase,
                progress: checkpoint.overallProgress
            )
        }
    }

    /// Runs micro-retrain on buffered documents.
    private func runMicroRetrain(bufferedEntries: [BufferedEntry]) async throws {
        guard let mainState = modelState else {
            throw IncrementalUpdateError.modelNotInitialized
        }

        isTraining = true
        shouldContinueTraining = true

        defer { isTraining = false }

        // Create documents and embeddings from buffer
        let documents = bufferedEntries.map { entry in
            Document(id: entry.documentID, content: entry.tokenizedContent.joined(separator: " "))
        }
        let embeddings = bufferedEntries.map(\.embedding)

        // Run training on just the new documents
        let trainingResult = try await trainingRunner.runTraining(
            documents: documents,
            embeddings: embeddings,
            configuration: modelConfiguration,
            type: .microRetrain,
            shouldContinue: { [weak self] in
                self?.shouldContinueTraining ?? false
            }
        )

        guard trainingResult.isComplete, let miniState = trainingResult.state else {
            // Training interrupted - entries already removed from buffer
            // Put them back for next attempt
            try await storage.appendToPendingBuffer(bufferedEntries)

            if let checkpoint = trainingResult.checkpoint {
                throw IncrementalUpdateError.trainingInterrupted(
                    phase: checkpoint.currentPhase,
                    progress: checkpoint.overallProgress
                )
            }
            return
        }

        // Create mini-model result
        let miniModel = MiniModelResult(
            topics: miniState.topics,
            centroids: miniState.centroids,
            assignments: miniState.assignments,
            vocabulary: miniState.vocabulary
        )

        // Match and merge
        let matchConfig = TopicMatcher.Configuration(
            similarityThreshold: updateConfiguration.topicMatchingSimilarityThreshold
        )

        let mergeResult = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainState,
            matchConfig: matchConfig
        )

        // Update state with merged results
        let updatedState = mainState.afterMicroRetrain(
            topics: mergeResult.topics,
            assignments: mergeResult.newDocumentAssignments,
            centroids: mergeResult.centroids,
            vocabulary: mergeResult.vocabulary,
            newDocumentCount: bufferedEntries.count,
            driftStatistics: mainState.driftStatistics
        )

        self.modelState = updatedState
        try await storage.saveModelState(updatedState)
    }

    /// Runs full refresh on all documents.
    private func runFullRefresh(
        progress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws {
        isTraining = true
        shouldContinueTraining = true

        defer { isTraining = false }

        // Load all embeddings
        let allEmbeddings = try await storage.loadAllEmbeddings()

        guard allEmbeddings.count >= updateConfiguration.coldStartThreshold else {
            throw IncrementalUpdateError.insufficientDocuments(
                required: updateConfiguration.coldStartThreshold,
                provided: allEmbeddings.count
            )
        }

        // We need document content for c-TF-IDF
        // In a real implementation, you'd have a way to load document content
        // For now, create placeholder documents
        let documents = allEmbeddings.map { id, _ in
            Document(id: id, content: "")  // Placeholder
        }
        let embeddings = allEmbeddings.map(\.1)

        let result = try await trainingRunner.runTraining(
            documents: documents,
            embeddings: embeddings,
            configuration: modelConfiguration,
            type: .fullRefresh,
            shouldContinue: { [weak self] in
                self?.shouldContinueTraining ?? false
            },
            onProgress: progress
        )

        if result.isComplete, let state = result.state {
            // For full refresh, update the state with proper metadata
            let updatedState = state.afterFullRefresh(
                topics: state.topics,
                assignments: state.assignments,
                centroids: state.centroids,
                vocabulary: state.vocabulary,
                documentCount: allEmbeddings.count
            )

            self.modelState = updatedState
            try await storage.saveModelState(updatedState)
        } else if let checkpoint = result.checkpoint {
            throw IncrementalUpdateError.trainingInterrupted(
                phase: checkpoint.currentPhase,
                progress: checkpoint.overallProgress
            )
        }
    }

    /// Simple whitespace tokenization.
    private func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty && $0.count > 2 }
    }
}

// MARK: - Timeout Helper

/// Executes an async operation with a timeout.
private func withTimeout<T: Sendable>(
    seconds: TimeInterval,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }

        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw CancellationError()
        }

        guard let result = try await group.next() else {
            throw CancellationError()
        }

        group.cancelAll()
        return result
    }
}
