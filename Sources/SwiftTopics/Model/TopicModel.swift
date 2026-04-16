// TopicModel.swift
// SwiftTopics
//
// Main topic modeling orchestrator

import Foundation

// MARK: - Topic Model

/// The main topic modeling orchestrator.
///
/// `TopicModel` coordinates the full topic modeling pipeline:
/// 1. **Embeddings**: Accept pre-computed embeddings or use an EmbeddingProvider
/// 2. **Reduction**: Reduce dimensionality using PCA (or UMAP)
/// 3. **Clustering**: Cluster embeddings using HDBSCAN
/// 4. **Representation**: Extract keywords using c-TF-IDF
/// 5. **Evaluation**: Compute coherence scores using NPMI
///
/// ## Basic Usage
///
/// ```swift
/// let model = TopicModel(configuration: .default)
/// let result = try await model.fit(documents: documents, embeddings: embeddings)
///
/// print("Found \(result.topics.count) topics")
/// for topic in result.topics {
///     print("Topic \(topic.id): \(topic.keywordSummary())")
/// }
/// ```
///
/// ## With Embedding Provider
///
/// ```swift
/// let model = TopicModel()
/// let result = try await model.fit(
///     documents: documents,
///     embeddingProvider: myProvider
/// )
/// ```
///
/// ## Transform New Documents
///
/// ```swift
/// let assignments = try await model.transform(
///     documents: newDocuments,
///     embeddings: newEmbeddings
/// )
/// ```
///
/// ## Thread Safety
///
/// `TopicModel` is an actor, ensuring thread-safe access to fitted state.
/// All methods are async and can be called from any concurrency context.
public actor TopicModel {

    // MARK: - Properties

    /// The dimensionality reducer used by this model.
    public var reducer: any DimensionReducer

    /// The clustering engine used by this model.
    public let clusterer: any ClusteringEngine

    /// The topic representer used by this model.
    public let representer: any TopicRepresenter

    /// The configuration used by this model.
    public nonisolated let configuration: TopicModelConfiguration

    /// Whether the model has been fitted.
    public var isFitted: Bool {
        fittedState != nil
    }

    /// The topics discovered during fitting.
    public var topics: [Topic]? {
        fittedState?.topics
    }

    /// The fitted state (nil before fitting).
    internal var fittedState: FittedTopicModelState?

    /// The embedding provider used during fitting (for findTopics).
    private var storedEmbeddingProvider: (any EmbeddingProvider)?

    /// Progress handler for reporting progress.
    private var progressHandler: TopicModelProgressHandler?

    // MARK: - Initialization

    /// Creates a topic model with explicit dependencies.
    ///
    /// - Parameters:
    ///   - reducer: The component to reduce embedding dimensionality.
    ///   - clusterer: The engine to cluster embeddings.
    ///   - representer: The component to extract keywords.
    ///   - configuration: Fallback configuration for metadata and coherence evaluation.
    public init(
        reducer: any DimensionReducer,
        clusterer: any ClusteringEngine,
        representer: any TopicRepresenter,
        configuration: TopicModelConfiguration = .default
    ) {
        self.reducer = reducer
        self.clusterer = clusterer
        self.representer = representer
        self.configuration = configuration
    }

    /// Creates a topic model with the given configuration.
    ///
    /// - Parameter configuration: The configuration to use.
    public init(configuration: TopicModelConfiguration = .default) {
        self.reducer = configuration.buildReducer()
        self.clusterer = configuration.buildClusterer()
        self.representer = configuration.buildRepresenter()
        self.configuration = configuration
    }

    /// Sets a progress handler for receiving updates during fitting.
    ///
    /// - Parameter handler: The handler to call with progress updates.
    public func setProgressHandler(_ handler: @escaping TopicModelProgressHandler) {
        self.progressHandler = handler
    }

    // MARK: - Fit

    /// Fits the model on documents with pre-computed embeddings.
    ///
    /// This is the main entry point for topic modeling when you have
    /// embeddings already computed (e.g., from a database or external model).
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster into topics.
    ///   - embeddings: Pre-computed document embeddings.
    /// - Returns: Complete topic modeling result.
    /// - Throws: `TopicModelError` if fitting fails.
    public func fit(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> TopicModelResult {
        // Validate configuration
        try configuration.validate()

        // Validate inputs
        try validateInputs(documents: documents, embeddings: embeddings)

        let startTime = Date()
        let tracker = ProgressTracker(handler: progressHandler)

        // Create result builder
        let resultBuilder = TopicModelResultBuilder()

        // Step 1: Reduce dimensionality
        await tracker.enterStage(.reduction)
        let reducedEmbeddings: [Embedding]
        let pcaComponents: [Float]?

        do {
            let (reduced, components) = try await reduceEmbeddings(embeddings)
            reducedEmbeddings = reduced
            pcaComponents = components
        } catch {
            throw TopicModelError.pipelineError(stage: "reduction", underlying: error)
        }

        await tracker.reportInStageProgress(1.0)

        // Step 2: Cluster
        await tracker.enterStage(.clustering)
        let clusterAssignment: ClusterAssignment

        do {
            clusterAssignment = try await clusterEmbeddings(reducedEmbeddings)
        } catch {
            throw TopicModelError.pipelineError(stage: "clustering", underlying: error)
        }

        await tracker.reportInStageProgress(1.0)

        // Step 3: Extract topic representations
        await tracker.enterStage(.representation)
        var topics: [Topic]

        do {
            topics = try await extractTopics(
                documents: documents,
                embeddings: embeddings,
                assignment: clusterAssignment
            )
        } catch {
            throw TopicModelError.pipelineError(stage: "representation", underlying: error)
        }

        await tracker.reportInStageProgress(1.0)

        // Step 4: Evaluate coherence (optional)
        var coherenceScore: Float?

        if configuration.coherence != nil && !topics.isEmpty {
            await tracker.enterStage(.evaluation)

            do {
                let (updatedTopics, meanCoherence) = try await evaluateCoherence(
                    topics: topics,
                    documents: documents
                )
                topics = updatedTopics
                coherenceScore = meanCoherence
            } catch {
                // Coherence evaluation failure is non-fatal
                // Just continue without coherence scores
                coherenceScore = nil
            }

            await tracker.reportInStageProgress(1.0)
        }

        // Step 5: Build document-topic assignments
        let documentTopics = buildDocumentTopics(
            documents: documents,
            embeddings: embeddings,
            assignment: clusterAssignment,
            topics: topics
        )

        // Step 6: Compute centroids for topics
        let centroids = computeTopicCentroids(
            topics: topics,
            embeddings: embeddings,
            assignment: clusterAssignment
        )

        // Store fitted state
        fittedState = FittedTopicModelState(
            topics: topics,
            assignment: clusterAssignment,
            pcaComponents: pcaComponents,
            pcaMean: nil,  // Would need to extract from PCAReducer
            inputDimension: embeddings.first?.dimension ?? 0,
            reducedDimension: reducedEmbeddings.first?.dimension ?? 0,
            centroids: centroids,
            documents: documents,
            embeddings: embeddings
        )

        // Build metadata
        let trainingDuration = Date().timeIntervalSince(startTime)
        let metadata = TopicModelMetadata(
            libraryVersion: SwiftTopics.version,
            trainedAt: Date(),
            trainingDuration: trainingDuration,
            configuration: configuration.toSnapshot(),
            documentCount: documents.count,
            embeddingDimension: embeddings.first?.dimension ?? 0,
            reducedDimension: reducedEmbeddings.first?.dimension ?? 0,
            randomSeed: configuration.seed
        )

        resultBuilder.setMetadata(metadata)
        resultBuilder.setTopics(topics)
        resultBuilder.setAssignments(documentTopics)
        if let score = coherenceScore {
            resultBuilder.setCoherenceScore(score)
        }

        await tracker.complete()

        return resultBuilder.build()
    }

    /// Fits the model on documents using an embedding provider.
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster into topics.
    ///   - embeddingProvider: Provider to compute embeddings.
    /// - Returns: Complete topic modeling result.
    /// - Throws: `TopicModelError` if fitting fails.
    public func fit(
        documents: [Document],
        embeddingProvider: any EmbeddingProvider
    ) async throws -> TopicModelResult {
        // Store the embedding provider for later use by findTopics(for:)
        storedEmbeddingProvider = embeddingProvider

        let tracker = ProgressTracker(handler: progressHandler)

        // Step 0: Compute embeddings
        await tracker.enterStage(.embedding(current: 0, total: documents.count))

        let embeddings: [Embedding]
        do {
            let texts = documents.map(\.content)
            embeddings = try await embeddingProvider.embedBatch(texts)
        } catch {
            throw TopicModelError.pipelineError(stage: "embedding", underlying: error)
        }

        await tracker.reportInStageProgress(1.0)

        // Continue with normal fitting
        return try await fit(documents: documents, embeddings: embeddings)
    }

    // MARK: - Transform

    /// Assigns new documents to existing topics.
    ///
    /// The model must be fitted first. New documents are assigned to
    /// the topic whose centroid is closest.
    ///
    /// - Parameters:
    ///   - documents: New documents to assign.
    ///   - embeddings: Pre-computed embeddings.
    /// - Returns: Topic assignments for each document.
    /// - Throws: `TopicModelError.notFitted` if model not fitted.
    public func transform(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> [TopicAssignment] {
        guard let state = fittedState else {
            throw TopicModelError.notFitted
        }

        // Validate dimensions
        if let firstEmbedding = embeddings.first {
            guard firstEmbedding.dimension == state.inputDimension else {
                throw TopicModelError.embeddingDimensionMismatch(
                    expected: state.inputDimension,
                    got: firstEmbedding.dimension
                )
            }
        }

        var assignments: [TopicAssignment] = []
        assignments.reserveCapacity(embeddings.count)

        for embedding in embeddings {
            let assignment = assignToTopic(
                embedding: embedding,
                centroids: state.centroids,
                topics: state.topics
            )
            assignments.append(assignment)
        }

        return assignments
    }

    /// Fits the model and returns assignments (combines fit + transform).
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster into topics.
    ///   - embeddings: Pre-computed embeddings.
    /// - Returns: Complete topic modeling result.
    public func fitTransform(
        documents: [Document],
        embeddings: [Embedding]
    ) async throws -> TopicModelResult {
        // fitTransform is the same as fit for this implementation
        // since fit already returns the assignments
        try await fit(documents: documents, embeddings: embeddings)
    }

    // MARK: - Topic Discovery & Search

    /// Finds topic assignments for arbitrary text.
    ///
    /// Embeds the text and computes similarity to each topic centroid,
    /// returning all topics ranked by probability.
    ///
    /// - Parameter text: The text to classify.
    /// - Returns: Topic assignments sorted by probability (highest first).
    /// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
    /// - Throws: `TopicModelError.noEmbeddingProvider` if model was fitted with pre-computed embeddings.
    ///
    /// ## Example
    /// ```swift
    /// let model = TopicModel()
    /// _ = try await model.fit(documents: docs, embeddingProvider: provider)
    ///
    /// let assignments = try await model.findTopics(for: "machine learning is fascinating")
    /// if let top = assignments.first {
    ///     print("Most likely topic: \(top.topicID), probability: \(top.probability)")
    /// }
    /// ```
    public func findTopics(for text: String) async throws -> [TopicAssignment] {
        guard let state = fittedState else {
            throw TopicModelError.notFitted
        }

        guard let provider = storedEmbeddingProvider else {
            throw TopicModelError.noEmbeddingProvider
        }

        guard !state.centroids.isEmpty else {
            return []
        }

        // Embed the text
        let embedding = try await provider.embed(text)

        // Compute distances to all centroids
        var distances: [(topicIndex: Int, distance: Float)] = []
        distances.reserveCapacity(state.centroids.count)

        for (i, centroid) in state.centroids.enumerated() {
            let dist = embedding.euclideanDistance(centroid)
            distances.append((i, dist))
        }

        // Convert distances to probabilities using a Student-t style transform:
        // p_i ∝ 1 / (1 + d_i^2), then normalized. This is more stable in high-D.
        let weights = distances.map { 1.0 / (1.0 + $0.distance * $0.distance) }
        let weightSum = weights.reduce(0, +)
        let probabilities = weightSum > 0 ? weights.map { $0 / weightSum } : Array(repeating: 0.0 as Float, count: weights.count)

        // Build topic assignments with alternatives
        var primaryAssignment: TopicAssignment?
        var alternatives: [AlternativeAssignment] = []

        // Sort by probability (descending) - which means sort by distance (ascending)
        let sorted = distances.enumerated().sorted { probabilities[$0.offset] > probabilities[$1.offset] }

        for (rank, (offset, (topicIndex, distance))) in sorted.enumerated() {
            let topicID = state.topics[topicIndex].id
            let probability = probabilities[offset]

            if rank == 0 {
                primaryAssignment = TopicAssignment(
                    topicID: topicID,
                    probability: probability,
                    distanceToCentroid: distance,
                    alternatives: nil  // Will be set below
                )
            } else {
                alternatives.append(AlternativeAssignment(topicID: topicID, probability: probability))
            }
        }

        // Return all assignments as a sorted array
        var result: [TopicAssignment] = []
        if let primary = primaryAssignment {
            // Create the primary with alternatives attached
            let primaryWithAlternatives = TopicAssignment(
                topicID: primary.topicID,
                probability: primary.probability,
                distanceToCentroid: primary.distanceToCentroid,
                alternatives: alternatives.isEmpty ? nil : alternatives
            )
            result.append(primaryWithAlternatives)

            // Also add each alternative as a standalone assignment
            for alt in alternatives {
                let dist = distances.first { state.topics[$0.topicIndex].id == alt.topicID }?.distance
                result.append(TopicAssignment(
                    topicID: alt.topicID,
                    probability: alt.probability,
                    distanceToCentroid: dist,
                    alternatives: nil
                ))
            }
        }

        return result
    }

    /// Searches for documents similar to the query.
    ///
    /// Performs semantic search by embedding the query and computing
    /// cosine similarity to all stored document embeddings.
    ///
    /// - Parameters:
    ///   - query: Search query text.
    ///   - topK: Maximum number of results (default: 10).
    /// - Returns: Documents with similarity scores, sorted by relevance (highest first).
    /// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
    /// - Throws: `TopicModelError.noEmbeddingProvider` if model was fitted with pre-computed embeddings.
    ///
    /// ## Example
    /// ```swift
    /// let results = try await model.search(query: "neural networks", topK: 5)
    /// for (doc, score) in results {
    ///     print("\(doc.content.prefix(50))... (score: \(score))")
    /// }
    /// ```
    public func search(query: String, topK: Int = 10) async throws -> [(document: Document, score: Float)] {
        guard let state = fittedState else {
            throw TopicModelError.notFitted
        }

        guard let provider = storedEmbeddingProvider else {
            throw TopicModelError.noEmbeddingProvider
        }

        guard !state.embeddings.isEmpty else {
            return []
        }

        // Embed the query
        let queryEmbedding = try await provider.embed(query)

        // Compute cosine similarity to all document embeddings
        var scores: [(index: Int, score: Float)] = []
        scores.reserveCapacity(state.embeddings.count)

        for (i, docEmbedding) in state.embeddings.enumerated() {
            let similarity = queryEmbedding.cosineSimilarity(docEmbedding)
            scores.append((i, similarity))
        }

        // Sort by similarity (descending) and take top K
        scores.sort { $0.score > $1.score }
        let topResults = scores.prefix(topK)

        // Build result array
        return topResults.map { (state.documents[$0.index], $0.score) }
    }

    // MARK: - Topic Manipulation

    /// Merges multiple topics into a single topic.
    ///
    /// All documents assigned to the specified topics are reassigned to the new merged topic.
    /// Keywords are recomputed using c-TF-IDF over the combined documents.
    ///
    /// - Parameter topicIds: IDs of topics to merge (must have at least 2).
    /// - Returns: The newly created merged topic.
    /// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
    /// - Throws: `TopicModelError.invalidInput` if any ID doesn't exist or fewer than 2 IDs provided.
    ///
    /// ## Example
    /// ```swift
    /// // Merge topics 0 and 2 into a single topic
    /// let merged = try await model.merge(topics: [0, 2])
    /// print("Merged topic has \(merged.size) documents")
    /// ```
    public func merge(topics topicIds: [Int]) async throws -> Topic {
        guard let state = fittedState else {
            throw TopicModelError.notFitted
        }

        // Validate at least 2 topics
        guard topicIds.count >= 2 else {
            throw TopicModelError.invalidInput("Must provide at least 2 topic IDs to merge")
        }

        // Convert to TopicIDs and validate
        let topicIDSet = Set(topicIds.map { TopicID(value: $0) })

        // Validate no outlier topic (-1)
        if topicIDSet.contains(.outlier) {
            throw TopicModelError.invalidInput("Cannot merge the outlier topic (-1)")
        }

        // Validate all topic IDs exist
        let existingIDs = Set(state.topics.map(\.id))
        for topicID in topicIDSet {
            guard existingIDs.contains(topicID) else {
                throw TopicModelError.invalidInput("Topic ID \(topicID.value) does not exist")
            }
        }

        // Collect document indices for all topics being merged
        var mergedDocIndices: [Int] = []
        for (docIdx, label) in state.assignment.labels.enumerated() {
            if topicIDSet.contains(TopicID(value: label)) {
                mergedDocIndices.append(docIdx)
            }
        }

        // Use the lowest topic ID as the new merged topic ID
        let newTopicID = topicIds.min()!

        // Update assignment labels
        var newLabels = state.assignment.labels
        for docIdx in mergedDocIndices {
            newLabels[docIdx] = newTopicID
        }

        // Compute new centroid as mean of merged document embeddings
        let newCentroid = computeMergedCentroid(docIndices: mergedDocIndices, embeddings: state.embeddings)

        // Recompute keywords using c-TF-IDF on merged documents
        let mergedDocuments = mergedDocIndices.map { state.documents[$0] }
        let mergedKeywords = try await recomputeKeywords(
            documents: mergedDocuments,
            allDocuments: state.documents,
            allLabels: newLabels,
            targetCluster: newTopicID
        )

        // Build the new merged topic
        let mergedTopic = Topic(
            id: TopicID(value: newTopicID),
            keywords: mergedKeywords,
            size: mergedDocIndices.count,
            coherenceScore: nil,  // Would need to recompute
            representativeDocuments: selectRepresentativeDocs(
                docIndices: mergedDocIndices,
                documents: state.documents,
                embeddings: state.embeddings,
                centroid: newCentroid
            ),
            centroid: newCentroid
        )

        // Build new topics array: remove merged topics, add new merged topic
        var newTopics = state.topics.filter { !topicIDSet.contains($0.id) }
        newTopics.append(mergedTopic)
        newTopics.sort { $0.id < $1.id }

        // Rebuild centroids array to match new topics
        var newCentroids: [Embedding] = []
        for topic in newTopics {
            if let centroid = topic.centroid {
                newCentroids.append(centroid)
            } else {
                // Compute centroid for topics without one
                let dim = state.embeddings.first?.dimension ?? 0
                newCentroids.append(Embedding(vector: [Float](repeating: 0, count: dim)))
            }
        }

        // Create new cluster assignment with updated labels and reduced cluster count
        let newClusterCount = newTopics.count
        let newAssignment = ClusterAssignment(
            labels: newLabels,
            probabilities: state.assignment.probabilities,
            outlierScores: state.assignment.outlierScores,
            clusterCount: newClusterCount
        )

        // Update fitted state
        fittedState = FittedTopicModelState(
            topics: newTopics,
            assignment: newAssignment,
            pcaComponents: state.pcaComponents,
            pcaMean: state.pcaMean,
            inputDimension: state.inputDimension,
            reducedDimension: state.reducedDimension,
            centroids: newCentroids,
            documents: state.documents,
            embeddings: state.embeddings
        )

        return mergedTopic
    }

    /// Reduces the number of topics by merging similar ones.
    ///
    /// Uses hierarchical agglomerative clustering to iteratively merge
    /// the most similar topic pairs until the target count is reached.
    /// Similarity is measured by centroid distance in embedding space.
    ///
    /// - Parameter count: Target number of topics (must be >= 1 and < current count).
    /// - Returns: Array of final topics after reduction.
    /// - Throws: `TopicModelError.notFitted` if model hasn't been trained.
    /// - Throws: `TopicModelError.invalidInput` if count is invalid.
    ///
    /// ## Example
    /// ```swift
    /// // Reduce from 10 topics to 5
    /// let reducedTopics = try await model.reduce(to: 5)
    /// print("Now have \(reducedTopics.count) topics")
    /// ```
    public func reduce(to count: Int) async throws -> [Topic] {
        guard let state = fittedState else {
            throw TopicModelError.notFitted
        }

        // Get current non-outlier topic count
        let currentTopicCount = state.topics.filter { !$0.isOutlierTopic }.count

        // Validate count
        guard count >= 1 else {
            throw TopicModelError.invalidInput("Target count must be at least 1")
        }

        guard count < currentTopicCount else {
            throw TopicModelError.invalidInput(
                "Target count (\(count)) must be less than current topic count (\(currentTopicCount))"
            )
        }

        // Iteratively merge most similar pairs until target count reached
        while let currentState = fittedState {
            let currentCount = currentState.topics.filter { !$0.isOutlierTopic }.count
            if currentCount <= count {
                break
            }

            // Find the most similar pair (smallest centroid distance)
            let mostSimilarPair = findMostSimilarTopicPair(
                topics: currentState.topics,
                centroids: currentState.centroids
            )

            guard let pair = mostSimilarPair else {
                break  // No more pairs to merge
            }

            // Merge the pair
            _ = try await merge(topics: [pair.0, pair.1])
        }

        return fittedState?.topics ?? []
    }

    // MARK: - Private Merge Helpers

    /// Computes the centroid for merged documents.
    private func computeMergedCentroid(docIndices: [Int], embeddings: [Embedding]) -> Embedding? {
        guard !docIndices.isEmpty, !embeddings.isEmpty else { return nil }

        let dim = embeddings[docIndices[0]].dimension
        var sum = [Float](repeating: 0, count: dim)

        for idx in docIndices {
            let emb = embeddings[idx]
            for d in 0..<dim {
                sum[d] += emb.vector[d]
            }
        }

        let scale = 1.0 / Float(docIndices.count)
        for d in 0..<dim {
            sum[d] *= scale
        }

        return Embedding(vector: sum)
    }

    /// Recomputes keywords for a merged cluster using c-TF-IDF.
    private func recomputeKeywords(
        documents: [Document],
        allDocuments: [Document],
        allLabels: [Int],
        targetCluster: Int
    ) async throws -> [TopicKeyword] {
        // Build a temporary assignment for the representer
        // Only the target cluster gets the merged documents
        let clusterCount = (allLabels.max() ?? -1) + 1

        guard clusterCount > 0 else {
            return []
        }

        // Create representer and compute keywords for all clusters
        let representer = CTFIDFRepresenter(configuration: configuration.representation)

        // We need to create a proper ClusterAssignment
        let tempAssignment = ClusterAssignment(labels: allLabels, clusterCount: clusterCount)

        // Extract topics (this gives us keywords for all clusters)
        let topics = try await representer.represent(documents: allDocuments, assignment: tempAssignment)

        // Return keywords for our target cluster
        return topics.first { $0.id.value == targetCluster }?.keywords ?? []
    }

    /// Selects representative documents closest to the centroid.
    private func selectRepresentativeDocs(
        docIndices: [Int],
        documents: [Document],
        embeddings: [Embedding],
        centroid: Embedding?,
        maxCount: Int = 3
    ) -> [DocumentID] {
        guard !docIndices.isEmpty else { return [] }

        let count = min(maxCount, docIndices.count)

        guard let centroid = centroid else {
            return docIndices.prefix(count).map { documents[$0].id }
        }

        // Sort by distance to centroid
        let sorted = docIndices.sorted { idxA, idxB in
            let distA = embeddings[idxA].euclideanDistance(centroid)
            let distB = embeddings[idxB].euclideanDistance(centroid)
            return distA < distB
        }

        return sorted.prefix(count).map { documents[$0].id }
    }

    /// Finds the most similar pair of topics by centroid distance.
    private func findMostSimilarTopicPair(
        topics: [Topic],
        centroids: [Embedding]
    ) -> (Int, Int)? {
        let nonOutlierTopics = topics.filter { !$0.isOutlierTopic }

        guard nonOutlierTopics.count >= 2 else { return nil }

        var minDistance: Float = .infinity
        var bestPair: (Int, Int)?

        // Compare all pairs
        for i in 0..<nonOutlierTopics.count {
            for j in (i + 1)..<nonOutlierTopics.count {
                let topicA = nonOutlierTopics[i]
                let topicB = nonOutlierTopics[j]

                // Find centroids for these topics
                guard let idxA = topics.firstIndex(where: { $0.id == topicA.id }),
                      let idxB = topics.firstIndex(where: { $0.id == topicB.id }),
                      idxA < centroids.count, idxB < centroids.count else {
                    continue
                }

                let distance = centroids[idxA].euclideanDistance(centroids[idxB])
                if distance < minDistance {
                    minDistance = distance
                    bestPair = (topicA.id.value, topicB.id.value)
                }
            }
        }

        return bestPair
    }

    // MARK: - Private Pipeline Methods

    private func validateInputs(
        documents: [Document],
        embeddings: [Embedding]
    ) throws {
        guard !documents.isEmpty else {
            throw TopicModelError.invalidInput("Documents array is empty")
        }

        guard !embeddings.isEmpty else {
            throw TopicModelError.invalidInput("Embeddings array is empty")
        }

        guard documents.count == embeddings.count else {
            throw TopicModelError.invalidInput(
                "Document count (\(documents.count)) doesn't match embedding count (\(embeddings.count))"
            )
        }

        // Validate consistent embedding dimensions
        let dimension = embeddings[0].dimension
        for embedding in embeddings {
            guard embedding.dimension == dimension else {
                throw TopicModelError.embeddingDimensionMismatch(
                    expected: dimension,
                    got: embedding.dimension
                )
            }
        }
    }

    private func reduceEmbeddings(
        _ embeddings: [Embedding]
    ) async throws -> (reduced: [Embedding], components: [Float]?) {
        // Don't reduce if already low-dimensional and we have a target dimension > 0
        let inputDim = embeddings.first?.dimension ?? 0
        if self.reducer.outputDimension > 0 && inputDim <= self.reducer.outputDimension {
            return (embeddings, nil)
        }

        try await self.reducer.fit(embeddings)
        let reduced = try await self.reducer.transform(embeddings)

        // PCA has extractable components
        let components = (self.reducer as? PCAReducer)?.principalComponents
        return (reduced, components)
    }

    private func clusterEmbeddings(
        _ embeddings: [Embedding]
    ) async throws -> ClusterAssignment {
        return try await self.clusterer.fit(embeddings)
    }

    private func extractTopics(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) async throws -> [Topic] {
        return try await self.representer.represent(
            documents: documents,
            embeddings: embeddings,
            assignment: assignment
        )
    }

    private func evaluateCoherence(
        topics: [Topic],
        documents: [Document]
    ) async throws -> (topics: [Topic], meanCoherence: Float) {
        guard let coherenceConfig = configuration.coherence else {
            return (topics, 0)
        }

        let evaluator = NPMICoherenceEvaluator(configuration: coherenceConfig)
        let result = await evaluator.evaluate(topics: topics, documents: documents)

        // Update topics with coherence scores
        var updatedTopics: [Topic] = []
        updatedTopics.reserveCapacity(topics.count)

        for (i, topic) in topics.enumerated() {
            let coherenceScore = i < result.topicScores.count ? result.topicScores[i] : nil
            let updatedTopic = Topic(
                id: topic.id,
                keywords: topic.keywords,
                size: topic.size,
                coherenceScore: coherenceScore,
                representativeDocuments: topic.representativeDocuments,
                centroid: topic.centroid
            )
            updatedTopics.append(updatedTopic)
        }

        return (updatedTopics, result.meanCoherence)
    }

    private func buildDocumentTopics(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment,
        topics: [Topic]
    ) -> [DocumentID: TopicAssignment] {
        var result: [DocumentID: TopicAssignment] = [:]
        result.reserveCapacity(documents.count)

        // Pre-compute centroids for distance calculation
        var centroids: [Int: Embedding] = [:]
        for topic in topics {
            if let centroid = topic.centroid {
                centroids[topic.id.value] = centroid
            }
        }

        for (i, document) in documents.enumerated() {
            let label = assignment.label(for: i)
            let probability = assignment.probability(for: i)

            // Compute distance to centroid if available
            let distanceToCentroid: Float?
            if label >= 0, let centroid = centroids[label] {
                distanceToCentroid = embeddings[i].euclideanDistance(centroid)
            } else {
                distanceToCentroid = nil
            }

            let topicAssignment = TopicAssignment(
                topicID: TopicID(value: label),
                probability: probability,
                distanceToCentroid: distanceToCentroid,
                alternatives: nil
            )

            result[document.id] = topicAssignment
        }

        return result
    }

    private func computeTopicCentroids(
        topics: [Topic],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) -> [Embedding] {
        var centroids: [Embedding] = []
        centroids.reserveCapacity(topics.count)

        for topic in topics {
            if let centroid = topic.centroid {
                centroids.append(centroid)
            } else {
                // Compute centroid from assignments
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

    private func assignToTopic(
        embedding: Embedding,
        centroids: [Embedding],
        topics: [Topic]
    ) -> TopicAssignment {
        guard !centroids.isEmpty else {
            return TopicAssignment.outlier
        }

        var minDist: Float = .infinity
        var nearestIndex = -1

        for (i, centroid) in centroids.enumerated() {
            let dist = embedding.euclideanDistance(centroid)
            if dist < minDist {
                minDist = dist
                nearestIndex = i
            }
        }

        guard nearestIndex >= 0 && nearestIndex < topics.count else {
            return TopicAssignment.outlier
        }

        // Simple probability based on UMAP's Student-t distribution
        // This provides more stable gradients in high-dimensional space
        let probability: Float = 1.0 / (1.0 + minDist * minDist)

        return TopicAssignment(
            topicID: topics[nearestIndex].id,
            probability: probability,
            distanceToCentroid: minDist,
            alternatives: nil
        )
    }
}

// MARK: - Fitted State

/// Internal state of a fitted topic model.
///
/// This struct stores all the information needed to:
/// - Transform new embeddings using the fitted PCA model
/// - Assign new documents to topics using centroids
/// - Export the model state for serialization
/// - Search for similar documents
internal struct FittedTopicModelState: Sendable {

    /// The discovered topics.
    let topics: [Topic]

    /// Cluster assignments from training.
    let assignment: ClusterAssignment

    /// PCA transformation matrix (for transforming new data).
    let pcaComponents: [Float]?

    /// PCA mean vector (for centering new data).
    let pcaMean: [Float]?

    /// Input embedding dimension.
    let inputDimension: Int

    /// Reduced embedding dimension.
    let reducedDimension: Int

    /// Topic centroids for assignment.
    let centroids: [Embedding]

    /// Training documents (for search functionality).
    let documents: [Document]

    /// Training embeddings (for search functionality).
    let embeddings: [Embedding]
}

// MARK: - Convenience Extensions

extension Array where Element == Document {

    /// Discovers topics in these documents using pre-computed embeddings.
    ///
    /// - Parameters:
    ///   - embeddings: Pre-computed document embeddings.
    ///   - configuration: Topic modeling configuration.
    /// - Returns: Topic modeling result.
    public func discoverTopics(
        embeddings: [Embedding],
        configuration: TopicModelConfiguration = .default
    ) async throws -> TopicModelResult {
        let model = TopicModel(configuration: configuration)
        return try await model.fit(documents: self, embeddings: embeddings)
    }

    /// Discovers topics in these documents using an embedding provider.
    ///
    /// - Parameters:
    ///   - embeddingProvider: Provider to compute embeddings.
    ///   - configuration: Topic modeling configuration.
    /// - Returns: Topic modeling result.
    public func discoverTopics(
        embeddingProvider: any EmbeddingProvider,
        configuration: TopicModelConfiguration = .default
    ) async throws -> TopicModelResult {
        let model = TopicModel(configuration: configuration)
        return try await model.fit(documents: self, embeddingProvider: embeddingProvider)
    }
}
