// InterruptibleTrainingRunner.swift
// SwiftTopics
//
// Orchestrates interruptible training with checkpoint support

import Foundation

// MARK: - Interruptible Training Runner

/// Orchestrates interruptible topic model training with checkpoint support.
///
/// This runner manages the complete training pipeline, supporting:
/// - **Interruption**: Graceful stop at any point
/// - **Checkpointing**: Periodic saves of training state
/// - **Resumption**: Continue from any saved checkpoint
///
/// ## Training Pipeline
///
/// The runner executes these phases in order:
/// 1. UMAP k-NN graph construction
/// 2. Fuzzy simplicial set computation
/// 3. UMAP optimization (interruptible)
/// 4. HDBSCAN core distance computation
/// 5. MST construction (interruptible)
/// 6. Cluster extraction
/// 7. Representation (c-TF-IDF)
/// 8. Topic matching (for retrains)
///
/// ## Checkpoint Strategy
///
/// - After each complete phase: Always checkpoint
/// - During UMAP optimization: Every `umapCheckpointEpochs` epochs
/// - During MST construction: Every `mstCheckpointEdges` edges
/// - Time-based: Every `checkpointTimeInterval` seconds
///
/// ## Thread Safety
///
/// `InterruptibleTrainingRunner` is an actor for thread-safe state management.
public actor InterruptibleTrainingRunner {

    // MARK: - Configuration

    /// Directory for storing checkpoint files.
    private let checkpointDirectory: URL

    /// Interval (epochs) between UMAP checkpoints.
    private let umapCheckpointEpochs: Int

    /// Interval (edges) between MST checkpoints.
    private let mstCheckpointEdges: Int

    /// Time interval between checkpoints (seconds).
    private let checkpointTimeInterval: TimeInterval

    /// Storage for persisting checkpoints.
    private let storage: TopicModelStorage

    /// Last checkpoint time for time-based checkpointing.
    private var lastCheckpointTime: Date

    // MARK: - Initialization

    /// Creates an interruptible training runner.
    ///
    /// - Parameters:
    ///   - storage: Storage for persisting checkpoints and intermediate state.
    ///   - checkpointDirectory: Directory for checkpoint files (defaults to storage directory).
    ///   - umapCheckpointEpochs: Epochs between UMAP checkpoints. Default is 50.
    ///   - mstCheckpointEdges: Edges between MST checkpoints. Default is 100.
    ///   - checkpointTimeInterval: Seconds between checkpoints. Default is 3.0.
    public init(
        storage: TopicModelStorage,
        checkpointDirectory: URL? = nil,
        umapCheckpointEpochs: Int = 50,
        mstCheckpointEdges: Int = 100,
        checkpointTimeInterval: TimeInterval = 3.0
    ) {
        self.storage = storage
        self.checkpointDirectory = checkpointDirectory ?? FileManager.default.temporaryDirectory
        self.umapCheckpointEpochs = umapCheckpointEpochs
        self.mstCheckpointEdges = mstCheckpointEdges
        self.checkpointTimeInterval = checkpointTimeInterval
        self.lastCheckpointTime = Date()
    }

    // MARK: - Training Result

    /// Result of interruptible training.
    public struct TrainingResult: Sendable {
        /// The trained model state (nil if training didn't complete).
        public let state: IncrementalTopicModelState?

        /// Whether training completed all phases.
        public let isComplete: Bool

        /// The checkpoint if training was interrupted.
        public let checkpoint: TrainingCheckpoint?

        /// The phase that was reached.
        public let reachedPhase: TrainingPhase

        /// Creates a training result.
        public init(
            state: IncrementalTopicModelState?,
            isComplete: Bool,
            checkpoint: TrainingCheckpoint?,
            reachedPhase: TrainingPhase
        ) {
            self.state = state
            self.isComplete = isComplete
            self.checkpoint = checkpoint
            self.reachedPhase = reachedPhase
        }
    }

    // MARK: - Training Context

    /// Internal context passed through training phases.
    private struct TrainingContext {
        let documents: [Document]
        let embeddings: [Embedding]
        let configuration: TopicModelConfiguration
        let trainingType: TrainingType
        let documentIDs: [DocumentID]

        // Intermediate results
        var knnGraph: NearestNeighborGraph?
        var fuzzySet: FuzzySimplicialSet?
        var reducedEmbeddings: [[Float]]?
        var coreDistances: [Float]?
        var mutualReachabilityGraph: MutualReachabilityGraph?
        var mstEdges: [MSTEdge]?
        var mstInMST: [Bool]?
        var clusterAssignment: ClusterAssignment?
        var topics: [Topic]?
    }

    // MARK: - Run Training

    /// Runs training from scratch.
    ///
    /// - Parameters:
    ///   - documents: Documents to train on.
    ///   - embeddings: Pre-computed embeddings.
    ///   - configuration: Training configuration.
    ///   - type: Training type (microRetrain or fullRefresh).
    ///   - shouldContinue: Closure checked between operations. Return false to interrupt.
    ///   - onProgress: Progress callback.
    /// - Returns: Training result with state or checkpoint.
    public func runTraining(
        documents: [Document],
        embeddings: [Embedding],
        configuration: TopicModelConfiguration,
        type: TrainingType,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)? = nil
    ) async throws -> TrainingResult {
        let documentIDs = documents.map(\.id)

        // Create initial checkpoint
        let checkpoint = TrainingCheckpoint.initial(trainingType: type, documentIDs: documentIDs)
        try await storage.saveCheckpoint(checkpoint)

        // Create training context
        var context = TrainingContext(
            documents: documents,
            embeddings: embeddings,
            configuration: configuration,
            trainingType: type,
            documentIDs: documentIDs
        )

        return try await runFromPhase(
            .umapKNN,
            context: &context,
            checkpoint: checkpoint,
            shouldContinue: shouldContinue,
            onProgress: onProgress
        )
    }

    /// Resumes training from a checkpoint.
    ///
    /// - Parameters:
    ///   - checkpoint: The checkpoint to resume from.
    ///   - documents: Documents being trained on.
    ///   - embeddings: Pre-computed embeddings.
    ///   - configuration: Training configuration.
    ///   - shouldContinue: Closure checked between operations. Return false to interrupt.
    ///   - onProgress: Progress callback.
    /// - Returns: Training result with state or new checkpoint.
    public func resumeTraining(
        from checkpoint: TrainingCheckpoint,
        documents: [Document],
        embeddings: [Embedding],
        configuration: TopicModelConfiguration,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)? = nil
    ) async throws -> TrainingResult {
        // Check if checkpoint can be resumed
        guard checkpoint.canResume else {
            throw TrainingError.checkpointExhausted
        }

        // Decrement attempts
        let updatedCheckpoint = checkpoint.withDecrementedAttempts()
        try await storage.saveCheckpoint(updatedCheckpoint)

        // Create context and restore state
        var context = TrainingContext(
            documents: documents,
            embeddings: embeddings,
            configuration: configuration,
            trainingType: checkpoint.trainingType,
            documentIDs: checkpoint.documentIDs
        )

        // Restore intermediate state from checkpoint
        try await restoreContext(&context, from: checkpoint)

        // Resume from current phase
        return try await runFromPhase(
            checkpoint.currentPhase,
            context: &context,
            checkpoint: updatedCheckpoint,
            shouldContinue: shouldContinue,
            onProgress: onProgress
        )
    }

    // MARK: - Phase Execution

    /// Runs training starting from a specific phase.
    private func runFromPhase(
        _ startPhase: TrainingPhase,
        context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws -> TrainingResult {
        var currentCheckpoint = checkpoint
        var currentPhase = startPhase

        while currentPhase != .complete {
            // Check for interruption
            guard shouldContinue() else {
                return TrainingResult(
                    state: nil,
                    isComplete: false,
                    checkpoint: currentCheckpoint,
                    reachedPhase: currentPhase
                )
            }

            // Report progress
            if let onProgress = onProgress {
                let progress = TrainingProgress(
                    phase: currentPhase,
                    phaseProgress: currentCheckpoint.currentPhaseProgress,
                    overallProgress: currentCheckpoint.overallProgress,
                    estimatedTimeRemaining: currentCheckpoint.estimatedTimeRemaining,
                    canInterrupt: currentPhase.supportsPartialCheckpoint
                )
                await onProgress(progress)
            }

            // Execute current phase
            let phaseResult: PhaseResult
            do {
                phaseResult = try await executePhase(
                    currentPhase,
                    context: &context,
                    checkpoint: currentCheckpoint,
                    shouldContinue: shouldContinue,
                    onProgress: onProgress
                )
            } catch {
                throw TrainingError.phaseError(phase: currentPhase, underlying: error)
            }

            // Handle phase result
            switch phaseResult {
            case .completed(let newCheckpoint):
                currentCheckpoint = newCheckpoint
                currentPhase = currentPhase.next ?? .complete

            case .interrupted(let newCheckpoint):
                try await storage.saveCheckpoint(newCheckpoint)
                return TrainingResult(
                    state: nil,
                    isComplete: false,
                    checkpoint: newCheckpoint,
                    reachedPhase: currentPhase
                )
            }

            // Save checkpoint after each completed phase
            try await storage.saveCheckpoint(currentCheckpoint)
            lastCheckpointTime = Date()
        }

        // Training complete - build final state
        let finalState = try await buildFinalState(from: context)

        // Clear checkpoint
        try await storage.clearCheckpoint()

        return TrainingResult(
            state: finalState,
            isComplete: true,
            checkpoint: nil,
            reachedPhase: .complete
        )
    }

    /// Result of executing a single phase.
    private enum PhaseResult {
        case completed(TrainingCheckpoint)
        case interrupted(TrainingCheckpoint)
    }

    /// Executes a single training phase.
    private func executePhase(
        _ phase: TrainingPhase,
        context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws -> PhaseResult {
        switch phase {
        case .umapKNN:
            return try await executeUMAPKNN(&context, checkpoint: checkpoint, shouldContinue: shouldContinue)

        case .umapFuzzySet:
            return try await executeUMAPFuzzySet(&context, checkpoint: checkpoint)

        case .umapOptimization:
            return try await executeUMAPOptimization(
                &context,
                checkpoint: checkpoint,
                shouldContinue: shouldContinue,
                onProgress: onProgress
            )

        case .hdbscanCoreDistance:
            return try await executeHDBSCANCoreDistance(&context, checkpoint: checkpoint, shouldContinue: shouldContinue)

        case .hdbscanMST:
            return try await executeHDBSCANMST(
                &context,
                checkpoint: checkpoint,
                shouldContinue: shouldContinue,
                onProgress: onProgress
            )

        case .clusterExtraction:
            return try await executeClusterExtraction(&context, checkpoint: checkpoint)

        case .representation:
            return try await executeRepresentation(&context, checkpoint: checkpoint)

        case .topicMatching:
            return try await executeTopicMatching(&context, checkpoint: checkpoint)

        case .complete:
            return .completed(checkpoint)
        }
    }

    // MARK: - Phase Implementations

    private func executeUMAPKNN(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool
    ) async throws -> PhaseResult {
        guard shouldContinue() else {
            return .interrupted(checkpoint)
        }

        // Build k-NN graph
        let umapConfig = context.configuration.reduction.umapConfig ?? .default
        let metric = convertMetric(umapConfig.metric)
        let graph = try await NearestNeighborGraph.build(
            embeddings: context.embeddings,
            k: umapConfig.nNeighbors,
            metric: metric
        )
        context.knnGraph = graph

        // Save graph to file for checkpoint
        let graphPath = checkpointDirectory.appendingPathComponent("knn_graph.bin")
        try await saveKNNGraph(graph, to: graphPath)

        let newCheckpoint = TrainingCheckpoint(
            runID: checkpoint.runID,
            trainingType: checkpoint.trainingType,
            startedAt: checkpoint.startedAt,
            documentIDs: checkpoint.documentIDs,
            lastCompletedPhase: .umapKNN,
            currentPhase: .umapFuzzySet,
            currentPhaseProgress: 0,
            knnGraphPath: graphPath.path
        )

        return .completed(newCheckpoint)
    }

    private func executeUMAPFuzzySet(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint
    ) async throws -> PhaseResult {
        guard let knnGraph = context.knnGraph else {
            throw TrainingError.missingIntermediateState("knnGraph")
        }

        // Build fuzzy simplicial set
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)
        context.fuzzySet = fuzzySet

        let newCheckpoint = checkpoint.withCompletedPhase(.umapFuzzySet)
        return .completed(newCheckpoint)
    }

    private func executeUMAPOptimization(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws -> PhaseResult {
        guard let fuzzySet = context.fuzzySet else {
            throw TrainingError.missingIntermediateState("fuzzySet")
        }

        let umapConfig = context.configuration.reduction.umapConfig ?? .default
        let nEpochs = umapConfig.nEpochs ?? computeAutoEpochs(n: context.embeddings.count)
        let nComponents = context.configuration.reduction.outputDimension

        // Initialize or restore embedding
        var initialEmbedding: [[Float]]
        var startingEpoch = 0
        var samplingScheduleState: [Float]?

        if let savedPath = checkpoint.umapEmbeddingPath,
           let savedEpoch = checkpoint.umapCurrentEpoch,
           savedEpoch > 0 {
            // Resume from checkpoint
            let url = URL(fileURLWithPath: savedPath)
            if let savedEmbedding = try? CheckpointSerializer.loadEmbedding(from: url) {
                initialEmbedding = savedEmbedding
                startingEpoch = savedEpoch

                // Load sampling schedule state if available
                let stateURL = url.deletingLastPathComponent().appendingPathComponent("umap_state.json")
                if let stateData = try? Data(contentsOf: stateURL),
                   let state = try? CheckpointSerializer.deserializeUMAPState(stateData) {
                    samplingScheduleState = state.samplingScheduleState
                }
            } else {
                // Can't load checkpoint, start fresh
                initialEmbedding = initializeSpectralEmbedding(
                    pointCount: context.embeddings.count,
                    dimensions: nComponents,
                    seed: context.configuration.seed
                )
            }
        } else {
            // Fresh start with spectral initialization
            initialEmbedding = initializeSpectralEmbedding(
                pointCount: context.embeddings.count,
                dimensions: nComponents,
                seed: context.configuration.seed
            )
        }

        // Create optimizer
        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: umapConfig.minDist,
            seed: context.configuration.seed
        )

        // Run interruptible optimization
        let embeddingPath = checkpointDirectory.appendingPathComponent("umap_embedding.bin")
        let statePath = checkpointDirectory.appendingPathComponent("umap_state.json")

        // Capture immutable values for the closure
        let initialCheckpoint = checkpoint
        let checkpointInterval = checkpointTimeInterval
        let totalEpochs = nEpochs
        lastCheckpointTime = Date()

        // Default negative sample rate (same as UMAP.swift)
        let negativeSampleRate = 5

        let result = await optimizer.optimizeInterruptible(
            fuzzySet: fuzzySet,
            nEpochs: nEpochs,
            learningRate: umapConfig.learningRate,
            negativeSampleRate: negativeSampleRate,
            startingEpoch: startingEpoch,
            samplingScheduleState: samplingScheduleState,
            checkpointInterval: umapCheckpointEpochs,
            shouldContinue: shouldContinue,
            onCheckpoint: { [weak self] info in
                guard let self = self else { return }

                // Check if we should save based on time
                let now = Date()
                let elapsed = now.timeIntervalSince(await self.lastCheckpointTime)
                guard elapsed >= checkpointInterval || info.epoch == totalEpochs - 1 else {
                    return
                }

                // Save embedding
                try? CheckpointSerializer.saveEmbedding(info.embedding, to: embeddingPath)

                // Save state
                let state = CheckpointSerializer.UMAPCheckpointState(
                    currentEpoch: info.epoch,
                    totalEpochs: info.totalEpochs,
                    samplingScheduleState: info.samplingScheduleState,
                    randomSeed: nil
                )
                if let stateData = try? CheckpointSerializer.serializeUMAPState(state) {
                    try? stateData.write(to: statePath)
                }

                // Create and save checkpoint
                let newCheckpoint = initialCheckpoint.withUMAPState(
                    embeddingPath: embeddingPath.path,
                    currentEpoch: info.epoch,
                    totalEpochs: info.totalEpochs
                )
                try? await self.storage.saveCheckpoint(newCheckpoint)
                await self.updateLastCheckpointTime(now)

                // Report progress
                if let onProgress = onProgress {
                    let progress = TrainingProgress(
                        phase: .umapOptimization,
                        phaseProgress: info.progress,
                        overallProgress: newCheckpoint.overallProgress,
                        estimatedTimeRemaining: newCheckpoint.estimatedTimeRemaining,
                        canInterrupt: true
                    )
                    await onProgress(progress)
                }
            }
        )

        context.reducedEmbeddings = result.embedding

        if result.isComplete {
            let newCheckpoint = checkpoint.withCompletedPhase(.umapOptimization)
            return .completed(newCheckpoint)
        } else {
            // Save final state
            try CheckpointSerializer.saveEmbedding(result.embedding, to: embeddingPath)
            let finalCheckpoint = checkpoint.withUMAPState(
                embeddingPath: embeddingPath.path,
                currentEpoch: result.completedEpoch,
                totalEpochs: result.totalEpochs
            )
            return .interrupted(finalCheckpoint)
        }
    }

    private func executeHDBSCANCoreDistance(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool
    ) async throws -> PhaseResult {
        guard shouldContinue() else {
            return .interrupted(checkpoint)
        }

        guard let reducedEmbeddings = context.reducedEmbeddings else {
            throw TrainingError.missingIntermediateState("reducedEmbeddings")
        }

        // Convert to Embedding type
        let embeddings = reducedEmbeddings.map { Embedding(vector: $0) }

        // Compute core distances
        let minSamples = context.configuration.clustering.effectiveMinSamples
        let coreComputer = CoreDistanceComputer(minSamples: minSamples)
        let coreDistances = try await coreComputer.compute(embeddings: embeddings, gpuContext: nil)

        context.coreDistances = coreDistances

        // Build mutual reachability graph
        let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)
        context.mutualReachabilityGraph = mrGraph

        // Save core distances
        let corePath = checkpointDirectory.appendingPathComponent("core_distances.bin")
        try CheckpointSerializer.saveFloatArray(coreDistances, to: corePath)

        let newCheckpoint = TrainingCheckpoint(
            runID: checkpoint.runID,
            trainingType: checkpoint.trainingType,
            startedAt: checkpoint.startedAt,
            documentIDs: checkpoint.documentIDs,
            lastCompletedPhase: .hdbscanCoreDistance,
            currentPhase: .hdbscanMST,
            currentPhaseProgress: 0,
            knnGraphPath: checkpoint.knnGraphPath,
            coreDistancesPath: corePath.path
        )

        return .completed(newCheckpoint)
    }

    private func executeHDBSCANMST(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint,
        shouldContinue: @escaping @Sendable () -> Bool,
        onProgress: (@Sendable (TrainingProgress) async -> Void)?
    ) async throws -> PhaseResult {
        guard let mrGraph = context.mutualReachabilityGraph else {
            throw TrainingError.missingIntermediateState("mutualReachabilityGraph")
        }

        // Check for existing partial MST
        var startingEdges: [MSTEdge]?
        var startingInMST: [Bool]?

        if let mstPath = checkpoint.mstStatePath,
           checkpoint.mstEdgesCompleted ?? 0 > 0 {
            let edgesURL = URL(fileURLWithPath: mstPath)
            let inMSTURL = edgesURL.deletingLastPathComponent().appendingPathComponent("mst_inmst.bin")

            startingEdges = try? CheckpointSerializer.loadMSTEdges(from: edgesURL)
            startingInMST = try? CheckpointSerializer.loadBoolArray(from: inMSTURL)
        }

        // Use stored partial state if available from previous interrupt in same run
        if let edges = context.mstEdges, let inMST = context.mstInMST {
            startingEdges = edges
            startingInMST = inMST
        }

        let mstBuilder = InterruptibleMSTBuilder(checkpointEdgeInterval: mstCheckpointEdges)
        let edgesPath = checkpointDirectory.appendingPathComponent("mst_edges.bin")
        let inMSTPath = checkpointDirectory.appendingPathComponent("mst_inmst.bin")

        // Capture immutable values for the closure
        let initialCheckpoint = checkpoint
        let checkpointInterval = checkpointTimeInterval
        lastCheckpointTime = Date()

        let result = await mstBuilder.build(
            from: mrGraph,
            startingEdges: startingEdges,
            startingInMST: startingInMST,
            shouldContinue: shouldContinue,
            onCheckpoint: { [weak self] info in
                guard let self = self else { return }

                // Check if we should save based on time
                let now = Date()
                let elapsed = now.timeIntervalSince(await self.lastCheckpointTime)
                guard elapsed >= checkpointInterval else {
                    return
                }

                // Save edges and inMST
                try? CheckpointSerializer.saveMSTEdges(info.edges, to: edgesPath)
                try? CheckpointSerializer.saveBoolArray(info.inMST, to: inMSTPath)

                // Create and save checkpoint
                let newCheckpoint = initialCheckpoint.withMSTState(
                    statePath: edgesPath.path,
                    edgesCompleted: info.edges.count,
                    totalEdges: info.pointCount - 1
                )
                try? await self.storage.saveCheckpoint(newCheckpoint)
                await self.updateLastCheckpointTime(now)

                // Report progress
                if let onProgress = onProgress {
                    let progress = TrainingProgress(
                        phase: .hdbscanMST,
                        phaseProgress: info.progress,
                        overallProgress: newCheckpoint.overallProgress,
                        estimatedTimeRemaining: newCheckpoint.estimatedTimeRemaining,
                        canInterrupt: true
                    )
                    await onProgress(progress)
                }
            }
        )

        context.mstEdges = result.edges
        context.mstInMST = result.inMST

        if result.isComplete {
            let newCheckpoint = checkpoint.withCompletedPhase(.hdbscanMST)
            return .completed(newCheckpoint)
        } else {
            // Save final state
            try CheckpointSerializer.saveMSTEdges(result.edges, to: edgesPath)
            try CheckpointSerializer.saveBoolArray(result.inMST, to: inMSTPath)

            let finalCheckpoint = checkpoint.withMSTState(
                statePath: edgesPath.path,
                edgesCompleted: result.edges.count,
                totalEdges: result.pointCount - 1
            )
            return .interrupted(finalCheckpoint)
        }
    }

    private func executeClusterExtraction(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint
    ) async throws -> PhaseResult {
        guard let mstEdges = context.mstEdges,
              let reducedEmbeddings = context.reducedEmbeddings,
              let coreDistances = context.coreDistances else {
            throw TrainingError.missingIntermediateState("mstEdges, reducedEmbeddings, or coreDistances")
        }

        // Create MST
        let mst = MinimumSpanningTree(edges: mstEdges, pointCount: reducedEmbeddings.count)

        // Build cluster hierarchy
        let config = context.configuration.clustering
        let hierarchyBuilder = ClusterHierarchyBuilder(
            minClusterSize: config.minClusterSize
        )
        let hierarchy = hierarchyBuilder.build(
            from: mst,
            allowSingleCluster: config.allowSingleCluster
        )

        // Extract clusters
        let extractor = ClusterExtractor(
            method: config.clusterSelectionMethod,
            minClusterSize: config.minClusterSize,
            epsilon: config.clusterSelectionEpsilon,
            allowSingleCluster: config.allowSingleCluster
        )
        let assignment = extractor.extract(
            from: hierarchy,
            pointCount: reducedEmbeddings.count,
            coreDistances: coreDistances
        )
        context.clusterAssignment = assignment

        let newCheckpoint = checkpoint.withCompletedPhase(.clusterExtraction)
        return .completed(newCheckpoint)
    }

    private func executeRepresentation(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint
    ) async throws -> PhaseResult {
        guard let assignment = context.clusterAssignment else {
            throw TrainingError.missingIntermediateState("clusterAssignment")
        }

        // Extract topic representations
        let representer = CTFIDFRepresenter(configuration: context.configuration.representation)
        let topics = try await representer.represent(
            documents: context.documents,
            embeddings: context.embeddings,
            assignment: assignment
        )
        context.topics = topics

        let newCheckpoint = checkpoint.withCompletedPhase(.representation)
        return .completed(newCheckpoint)
    }

    private func executeTopicMatching(
        _ context: inout TrainingContext,
        checkpoint: TrainingCheckpoint
    ) async throws -> PhaseResult {
        // Topic matching is primarily for incremental updates
        // For now, just pass through
        let newCheckpoint = checkpoint.withCompletedPhase(.topicMatching)
        return .completed(newCheckpoint)
    }

    // MARK: - State Management

    private func restoreContext(
        _ context: inout TrainingContext,
        from checkpoint: TrainingCheckpoint
    ) async throws {
        // Restore k-NN graph if available
        if let path = checkpoint.knnGraphPath {
            let url = URL(fileURLWithPath: path)
            context.knnGraph = try await loadKNNGraph(from: url)
        }

        // Restore core distances if available
        if let path = checkpoint.coreDistancesPath {
            let url = URL(fileURLWithPath: path)
            context.coreDistances = try CheckpointSerializer.loadFloatArray(from: url)

            // Rebuild mutual reachability graph
            if let coreDistances = context.coreDistances,
               let reducedEmbeddings = context.reducedEmbeddings {
                let embeddings = reducedEmbeddings.map { Embedding(vector: $0) }
                context.mutualReachabilityGraph = MutualReachabilityGraph(
                    embeddings: embeddings,
                    coreDistances: coreDistances
                )
            }
        }

        // Restore MST state if available
        if let path = checkpoint.mstStatePath {
            let url = URL(fileURLWithPath: path)
            context.mstEdges = try CheckpointSerializer.loadMSTEdges(from: url)

            let inMSTURL = url.deletingLastPathComponent().appendingPathComponent("mst_inmst.bin")
            context.mstInMST = try CheckpointSerializer.loadBoolArray(from: inMSTURL)
        }
    }

    private func buildFinalState(from context: TrainingContext) async throws -> IncrementalTopicModelState {
        guard let topics = context.topics,
              let assignment = context.clusterAssignment else {
            throw TrainingError.missingIntermediateState("topics or clusterAssignment")
        }

        // Compute centroids
        let centroids = computeCentroids(
            topics: topics,
            embeddings: context.embeddings,
            assignment: assignment
        )

        // Build vocabulary
        let vocabulary = buildVocabulary(
            documents: context.documents,
            assignment: assignment
        )

        return IncrementalTopicModelState.initial(
            configuration: context.configuration,
            topics: topics,
            assignments: assignment,
            centroids: centroids,
            vocabulary: vocabulary,
            inputDimension: context.embeddings.first?.dimension ?? 0,
            reducedDimension: context.configuration.reduction.outputDimension,
            documentCount: context.documents.count
        )
    }

    // MARK: - Helper Methods

    private func initializeSpectralEmbedding(
        pointCount: Int,
        dimensions: Int,
        seed: UInt64?
    ) -> [[Float]] {
        // Simple random initialization (spectral would require eigenvector computation)
        var rng = RandomState(seed: seed)
        var embedding = [[Float]]()
        embedding.reserveCapacity(pointCount)

        for _ in 0..<pointCount {
            var row = [Float]()
            row.reserveCapacity(dimensions)
            for _ in 0..<dimensions {
                row.append(rng.nextFloat() * 20.0 - 10.0)  // [-10, 10]
            }
            embedding.append(row)
        }

        return embedding
    }

    private func computeCentroids(
        topics: [Topic],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) -> [Embedding] {
        var centroids = [Embedding]()
        centroids.reserveCapacity(topics.count)

        for topic in topics {
            if let centroid = topic.centroid {
                centroids.append(centroid)
            } else {
                // Compute from assignments
                let dim = embeddings.first?.dimension ?? 0
                var sum = [Float](repeating: 0, count: dim)
                var count = 0

                for (i, embedding) in embeddings.enumerated() {
                    if assignment.label(for: i) == topic.id.value {
                        for d in 0..<dim {
                            sum[d] += embedding.vector[d]
                        }
                        count += 1
                    }
                }

                if count > 0 {
                    let scale = 1.0 / Float(count)
                    for d in 0..<dim {
                        sum[d] *= scale
                    }
                }

                centroids.append(Embedding(vector: sum))
            }
        }

        return centroids
    }

    private func buildVocabulary(
        documents: [Document],
        assignment: ClusterAssignment
    ) -> IncrementalVocabulary {
        var vocabulary = IncrementalVocabulary()

        for (i, document) in documents.enumerated() {
            let topic = assignment.label(for: i)
            guard topic >= 0 else { continue }

            let terms = tokenize(document.content)
            vocabulary.addDocument(terms: terms, topic: topic)
        }

        return vocabulary
    }

    private func tokenize(_ text: String) -> [String] {
        // Simple whitespace tokenization
        text.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty && $0.count > 2 }
    }

    private func saveKNNGraph(_ graph: NearestNeighborGraph, to url: URL) async throws {
        // Serialize k-NN graph (simplified format)
        var data = Data()

        // Header
        let pointCount = Int32(graph.pointCount)
        let k = Int32(graph.k)
        withUnsafeBytes(of: pointCount) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: k) { data.append(contentsOf: $0) }

        // Neighbors and distances
        for i in 0..<graph.pointCount {
            for j in 0..<graph.k {
                let neighbor = Int32(graph.neighbors[i][j])
                let distance = graph.distances[i][j]
                withUnsafeBytes(of: neighbor) { data.append(contentsOf: $0) }
                withUnsafeBytes(of: distance) { data.append(contentsOf: $0) }
            }
        }

        try data.write(to: url, options: .atomic)
    }

    private func loadKNNGraph(from url: URL) async throws -> NearestNeighborGraph {
        let data = try Data(contentsOf: url)

        return data.withUnsafeBytes { buffer in
            let ptr = buffer.bindMemory(to: UInt8.self).baseAddress!

            let pointCount = Int(ptr.withMemoryRebound(to: Int32.self, capacity: 1) { $0.pointee })
            let k = Int((ptr + 4).withMemoryRebound(to: Int32.self, capacity: 1) { $0.pointee })

            var neighbors = [[Int]]()
            var distances = [[Float]]()
            neighbors.reserveCapacity(pointCount)
            distances.reserveCapacity(pointCount)

            var offset = 8
            for _ in 0..<pointCount {
                var pointNeighbors = [Int]()
                var pointDistances = [Float]()
                pointNeighbors.reserveCapacity(k)
                pointDistances.reserveCapacity(k)

                for _ in 0..<k {
                    let neighbor = Int((ptr + offset).withMemoryRebound(to: Int32.self, capacity: 1) { $0.pointee })
                    let distance = (ptr + offset + 4).withMemoryRebound(to: Float.self, capacity: 1) { $0.pointee }
                    pointNeighbors.append(neighbor)
                    pointDistances.append(distance)
                    offset += 8
                }

                neighbors.append(pointNeighbors)
                distances.append(pointDistances)
            }

            return NearestNeighborGraph(neighbors: neighbors, distances: distances, k: k)
        }
    }

    private func updateLastCheckpointTime(_ time: Date) {
        lastCheckpointTime = time
    }
}

// MARK: - Training Error

/// Errors that can occur during interruptible training.
public enum TrainingError: Error, Sendable {
    /// Checkpoint has exhausted all retry attempts.
    case checkpointExhausted

    /// Required intermediate state is missing.
    case missingIntermediateState(String)

    /// A phase failed with an underlying error.
    case phaseError(phase: TrainingPhase, underlying: Error)

    /// Invalid configuration for training.
    case invalidConfiguration(String)
}

extension TrainingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .checkpointExhausted:
            return "Checkpoint has exhausted all retry attempts"
        case .missingIntermediateState(let state):
            return "Missing intermediate state: \(state)"
        case .phaseError(let phase, let underlying):
            return "Phase \(phase.displayName) failed: \(underlying.localizedDescription)"
        case .invalidConfiguration(let message):
            return "Invalid training configuration: \(message)"
        }
    }
}

// MARK: - Helpers

/// Converts DistanceMetricType to DistanceMetric.
private func convertMetric(_ type: DistanceMetricType) -> DistanceMetric {
    switch type {
    case .euclidean:
        return .euclidean
    case .cosine:
        return .cosine
    case .manhattan:
        return .manhattan
    case .dotProduct:
        return .cosine  // Approximate with cosine
    }
}

/// Computes automatic epoch count based on dataset size.
///
/// Matches the formula in UMAP.swift: 500 for small datasets, scaling down for larger ones.
private func computeAutoEpochs(n: Int) -> Int {
    if n <= 10_000 {
        return 500
    } else if n <= 100_000 {
        return 200
    } else {
        return 100
    }
}
