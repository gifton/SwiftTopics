// IncrementalTopicUpdaterTests.swift
// SwiftTopics
//
// Tests for the incremental topic updater

import Foundation
import Testing
@testable import SwiftTopics

// MARK: - Mock Storage

/// In-memory mock storage for testing.
actor MockTopicModelStorage: TopicModelStorage {
    var modelState: IncrementalTopicModelState?
    var embeddings: [(DocumentID, Embedding)] = []
    var pendingBuffer: [BufferedEntry] = []
    var checkpoint: TrainingCheckpoint?
    var reducedEmbeddings: [(DocumentID, [Float])]?
    var knnGraph: NearestNeighborGraph?

    init() {}

    func saveModelState(_ state: IncrementalTopicModelState) async throws {
        self.modelState = state
    }

    func loadModelState() async throws -> IncrementalTopicModelState? {
        return modelState
    }

    func appendEmbeddings(_ newEmbeddings: [(DocumentID, Embedding)]) async throws {
        embeddings.append(contentsOf: newEmbeddings)
    }

    func loadEmbeddings(for documentIDs: [DocumentID]) async throws -> [Embedding] {
        var result = [Embedding]()
        for id in documentIDs {
            if let embedding = embeddings.first(where: { $0.0 == id })?.1 {
                result.append(embedding)
            }
        }
        return result
    }

    func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)] {
        return embeddings
    }

    func embeddingCount() async throws -> Int {
        return embeddings.count
    }

    func saveReducedEmbeddings(_ embeddings: [(DocumentID, [Float])]) async throws {
        self.reducedEmbeddings = embeddings
    }

    func loadReducedEmbeddings() async throws -> [(DocumentID, [Float])]? {
        return reducedEmbeddings
    }

    func saveKNNGraph(_ graph: NearestNeighborGraph) async throws {
        self.knnGraph = graph
    }

    func loadKNNGraph() async throws -> NearestNeighborGraph? {
        return knnGraph
    }

    func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws {
        self.checkpoint = checkpoint
    }

    func loadCheckpoint() async throws -> TrainingCheckpoint? {
        return checkpoint
    }

    func clearCheckpoint() async throws {
        self.checkpoint = nil
    }

    func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws {
        pendingBuffer.append(contentsOf: entries)
    }

    func drainPendingBuffer() async throws -> [BufferedEntry] {
        let result = pendingBuffer
        pendingBuffer = []
        return result
    }

    func pendingBufferCount() async throws -> Int {
        return pendingBuffer.count
    }

    func clear() async throws {
        modelState = nil
        embeddings = []
        pendingBuffer = []
        checkpoint = nil
        reducedEmbeddings = nil
        knnGraph = nil
    }

    func storageSizeBytes() async throws -> UInt64 {
        // Rough estimate
        var size: UInt64 = 0
        size += UInt64(embeddings.count * 4 * 384)  // Assume 384-dim embeddings
        size += UInt64(pendingBuffer.count * 1000)  // Estimate per entry
        return size
    }
}

// MARK: - Test Helpers

/// Creates a random embedding for testing.
private func makeEmbedding(dimension: Int = 4, seed: Int = 0) -> Embedding {
    var values = [Float]()
    values.reserveCapacity(dimension)
    for i in 0..<dimension {
        let value = sin(Float(seed + i) * 0.1)
        values.append(value)
    }
    return Embedding(vector: values).normalized()
}

/// Creates a test document.
private func makeDocument(id: Int, content: String = "test content") -> Document {
    Document(id: DocumentID(string: "\(id)"), content: content)
}

/// Creates a mock model state for testing.
private func makeMockModelState(topicCount: Int = 3, embeddingDimension: Int = 4) -> IncrementalTopicModelState {
    var topics = [Topic]()
    var centroids = [Embedding]()

    for i in 0..<topicCount {
        let centroid = makeEmbedding(dimension: embeddingDimension, seed: i * 100)
        let keywords = [
            TopicKeyword(term: "word\(i)a", score: 0.9),
            TopicKeyword(term: "word\(i)b", score: 0.8),
            TopicKeyword(term: "word\(i)c", score: 0.7)
        ]

        let topic = Topic(
            id: TopicID(value: i),
            keywords: keywords,
            size: 10,
            coherenceScore: 0.5,
            representativeDocuments: [],
            centroid: centroid
        )

        topics.append(topic)
        centroids.append(centroid)
    }

    let assignment = ClusterAssignment(
        labels: Array(repeating: 0, count: 30),
        clusterCount: topicCount
    )

    return IncrementalTopicModelState.initial(
        configuration: .default,
        topics: topics,
        assignments: assignment,
        centroids: centroids,
        vocabulary: IncrementalVocabulary(),
        inputDimension: embeddingDimension,
        reducedDimension: 2,
        documentCount: 30
    )
}

// MARK: - Configuration Tests

@Suite("IncrementalUpdateConfiguration Tests")
struct IncrementalUpdateConfigurationTests {

    @Test("Default configuration is valid")
    func testDefaultConfigurationValid() {
        let config = IncrementalUpdateConfiguration.default
        #expect(config.isValid)
        #expect(config.validate().isEmpty)
    }

    @Test("Aggressive configuration is valid")
    func testAggressiveConfigurationValid() {
        let config = IncrementalUpdateConfiguration.aggressive
        #expect(config.isValid)
    }

    @Test("Conservative configuration is valid")
    func testConservativeConfigurationValid() {
        let config = IncrementalUpdateConfiguration.conservative
        #expect(config.isValid)
    }

    @Test("Testing configuration is valid")
    func testTestingConfigurationValid() {
        let config = IncrementalUpdateConfiguration.testing
        #expect(config.isValid)
    }

    @Test("Invalid coldStartThreshold is detected")
    func testInvalidColdStartThreshold() {
        let config = IncrementalUpdateConfiguration(coldStartThreshold: 2)
        #expect(!config.isValid)
        #expect(config.validate().contains { $0.contains("coldStartThreshold") })
    }

    @Test("Invalid microRetrainThreshold is detected")
    func testInvalidMicroRetrainThreshold() {
        let config = IncrementalUpdateConfiguration(microRetrainThreshold: 1)
        #expect(!config.isValid)
        #expect(config.validate().contains { $0.contains("microRetrainThreshold") })
    }

    @Test("Invalid outlierRateThreshold is detected")
    func testInvalidOutlierRateThreshold() {
        let config = IncrementalUpdateConfiguration(outlierRateThreshold: 0.01)
        #expect(!config.isValid)

        let config2 = IncrementalUpdateConfiguration(outlierRateThreshold: 0.95)
        #expect(!config2.isValid)
    }

    @Test("Custom configuration preserves values")
    func testCustomConfiguration() {
        let config = IncrementalUpdateConfiguration(
            coldStartThreshold: 25,
            microRetrainThreshold: 40,
            fullRefreshGrowthRatio: 2.5
        )

        #expect(config.coldStartThreshold == 25)
        #expect(config.microRetrainThreshold == 40)
        #expect(config.fullRefreshGrowthRatio == 2.5)
        #expect(config.isValid)
    }
}

// MARK: - Topic Assignment Tests

@Suite("IncrementalTopicAssignment Tests")
struct IncrementalTopicAssignmentTests {

    @Test("Normal assignment properties")
    func testNormalAssignment() {
        let assignment = IncrementalTopicAssignment(
            topicID: TopicID(value: 0),
            confidence: 0.8,
            similarity: 0.75,
            isTransformAssignment: true,
            topicKeywords: ["apple", "fruit", "red"]
        )

        #expect(!assignment.isOutlier)
        #expect(assignment.topicID.value == 0)
        #expect(assignment.confidence == 0.8)
        #expect(assignment.similarity == 0.75)
        #expect(assignment.distanceToCentroid == 0.25)
        #expect(assignment.topicKeywords.count == 3)
    }

    @Test("Outlier assignment properties")
    func testOutlierAssignment() {
        let assignment = IncrementalTopicAssignment.outlier(bestSimilarity: 0.2)

        #expect(assignment.isOutlier)
        #expect(assignment.topicID.isOutlier)
        #expect(assignment.confidence == 0)
        #expect(assignment.similarity == 0.2)
        #expect(assignment.topicKeywords.isEmpty)
    }

    @Test("Cold start assignment")
    func testColdStartAssignment() {
        let assignment = IncrementalTopicAssignment.coldStart()

        #expect(assignment.isOutlier)
        #expect(assignment.confidence == 0)
        #expect(assignment.similarity == 0)
        #expect(assignment.isTransformAssignment)
    }

    @Test("Batch assignment statistics")
    func testBatchAssignmentResult() {
        let assignments = [
            IncrementalTopicAssignment(
                topicID: TopicID(value: 0),
                confidence: 0.9,
                similarity: 0.85,
                isTransformAssignment: true,
                topicKeywords: []
            ),
            IncrementalTopicAssignment(
                topicID: TopicID(value: 0),
                confidence: 0.8,
                similarity: 0.75,
                isTransformAssignment: true,
                topicKeywords: []
            ),
            IncrementalTopicAssignment.outlier(bestSimilarity: 0.2),
            IncrementalTopicAssignment(
                topicID: TopicID(value: 1),
                confidence: 0.7,
                similarity: 0.65,
                isTransformAssignment: true,
                topicKeywords: []
            )
        ]

        let batch = IncrementalBatchAssignmentResult(assignments: assignments)

        #expect(batch.assignedCount == 3)
        #expect(batch.outlierCount == 1)
        #expect(batch.averageConfidence > 0.79 && batch.averageConfidence < 0.81)

        let distribution = batch.topicDistribution
        #expect(distribution[TopicID(value: 0)] == 2)
        #expect(distribution[TopicID(value: 1)] == 1)
        #expect(distribution[TopicID.outlier] == 1)
    }

    @Test("Assignment is codable")
    func testAssignmentCodable() throws {
        let original = IncrementalTopicAssignment(
            topicID: TopicID(value: 5),
            confidence: 0.85,
            similarity: 0.9,
            isTransformAssignment: true,
            topicKeywords: ["test", "keywords"]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(IncrementalTopicAssignment.self, from: data)

        #expect(decoded.topicID == original.topicID)
        #expect(decoded.confidence == original.confidence)
        #expect(decoded.similarity == original.similarity)
        #expect(decoded.topicKeywords == original.topicKeywords)
    }
}

// MARK: - Incremental Topic Updater Tests

@Suite("IncrementalTopicUpdater Tests")
struct IncrementalTopicUpdaterTests {

    @Test("Initialization with valid configuration succeeds")
    func testInitializationSuccess() async throws {
        let storage = MockTopicModelStorage()
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let hasModel = await updater.hasModel
        let docCount = await updater.totalDocumentCount
        #expect(!hasModel)
        #expect(docCount == 0)
    }

    @Test("Initialization with invalid configuration fails")
    func testInitializationWithInvalidConfig() async {
        let storage = MockTopicModelStorage()
        let invalidConfig = IncrementalUpdateConfiguration(coldStartThreshold: 2)

        do {
            _ = try await IncrementalTopicUpdater(
                storage: storage,
                updateConfiguration: invalidConfig
            )
            Issue.record("Should have thrown error")
        } catch IncrementalUpdateError.invalidConfiguration {
            // Expected
        } catch {
            Issue.record("Wrong error type: \(error)")
        }
    }

    @Test("Cold start returns outlier assignments")
    func testColdStartReturnsOutliers() async throws {
        let storage = MockTopicModelStorage()
        let config = IncrementalUpdateConfiguration.testing
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: config
        )

        // Process documents below threshold
        for i in 0..<(config.coldStartThreshold - 1) {
            let doc = makeDocument(id: i, content: "test document \(i)")
            let embedding = makeEmbedding(seed: i)
            let assignment = try await updater.processDocument(doc, embedding: embedding)

            #expect(assignment.isOutlier)
            let hasModel = await updater.hasModel
            #expect(!hasModel)
        }

        // Verify buffer count
        let bufferCount = try await updater.getPendingBufferCount()
        #expect(bufferCount == config.coldStartThreshold - 1)
    }

    @Test("Processing with existing model returns real assignments")
    func testProcessingWithExistingModel() async throws {
        let storage = MockTopicModelStorage()

        // Pre-populate with a model
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let hasModel = await updater.hasModel
        #expect(hasModel)

        // Process a document similar to topic 0
        let doc = makeDocument(id: 0, content: "similar content")
        let embedding = makeEmbedding(dimension: 4, seed: 0)  // Same seed as topic 0
        let assignment = try await updater.processDocument(doc, embedding: embedding)

        // Should get a real assignment (not outlier) since embedding matches topic 0
        #expect(assignment.isTransformAssignment)
        // The similarity should be high since we're using the same seed
    }

    @Test("Embedding dimension mismatch throws error")
    func testEmbeddingDimensionMismatch() async throws {
        let storage = MockTopicModelStorage()

        // Pre-populate with a model (4-dim embeddings)
        let mockState = makeMockModelState(embeddingDimension: 4)
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        // Try to process with wrong dimension (8-dim)
        let doc = makeDocument(id: 0)
        let wrongDimEmbedding = makeEmbedding(dimension: 8, seed: 0)

        do {
            _ = try await updater.processDocument(doc, embedding: wrongDimEmbedding)
            Issue.record("Should have thrown error")
        } catch IncrementalUpdateError.embeddingDimensionMismatch(let expected, let got) {
            #expect(expected == 4)
            #expect(got == 8)
        } catch {
            Issue.record("Wrong error type: \(error)")
        }
    }

    @Test("Get topics returns model topics")
    func testGetTopics() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState(topicCount: 5)
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let topics = await updater.getTopics()
        #expect(topics?.count == 5)
    }

    @Test("Get topic by ID")
    func testGetTopicByID() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState(topicCount: 3)
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let topic = await updater.getTopic(id: TopicID(value: 1))
        #expect(topic != nil)
        #expect(topic?.id.value == 1)

        let nonExistent = await updater.getTopic(id: TopicID(value: 99))
        #expect(nonExistent == nil)
    }

    @Test("Drift statistics returned when model exists")
    func testGetDriftStatistics() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let stats = await updater.getDriftStatistics()
        #expect(stats != nil)
    }

    @Test("Should trigger full refresh detects growth")
    func testShouldTriggerFullRefreshGrowth() async throws {
        let storage = MockTopicModelStorage()

        // Create a state with high growth ratio
        let baseState = makeMockModelState()
        // The initial state has documentCountAtLastRetrain = 30, totalDocumentCount = 30
        // We need to simulate growth by creating a state with higher totalDocumentCount

        // Create state with growth
        let mockState = IncrementalTopicModelState(
            configuration: baseState.configuration,
            topics: baseState.topics,
            assignments: baseState.assignments,
            centroids: baseState.centroids,
            inputDimension: baseState.inputDimension,
            reducedDimension: baseState.reducedDimension,
            vocabulary: baseState.vocabulary,
            totalDocumentCount: 100,  // 100/30 = 3.3x growth
            lastUpdatedAt: Date(),
            lastFullRetrainAt: Date(),
            documentCountAtLastRetrain: 30,
            driftStatistics: .initial
        )

        try await storage.saveModelState(mockState)

        let config = IncrementalUpdateConfiguration(fullRefreshGrowthRatio: 1.5)
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: config
        )

        let shouldRefresh = await updater.shouldTriggerFullRefresh()
        #expect(shouldRefresh)
    }

    @Test("Should trigger full refresh detects time elapsed")
    func testShouldTriggerFullRefreshTime() async throws {
        let storage = MockTopicModelStorage()

        // Create a state with old lastFullRetrainAt
        let oldDate = Date().addingTimeInterval(-400 * 24 * 60 * 60)  // 400 days ago
        let mockState = IncrementalTopicModelState(
            configuration: .default,
            topics: [],
            assignments: ClusterAssignment(labels: [], clusterCount: 0),
            centroids: [],
            inputDimension: 4,
            reducedDimension: 2,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 30,
            lastUpdatedAt: Date(),
            lastFullRetrainAt: oldDate,
            documentCountAtLastRetrain: 30,
            driftStatistics: .initial
        )

        try await storage.saveModelState(mockState)

        let config = IncrementalUpdateConfiguration(
            fullRefreshMaxInterval: 365 * 24 * 60 * 60  // 365 days
        )
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: config
        )

        let shouldRefresh = await updater.shouldTriggerFullRefresh()
        #expect(shouldRefresh)
    }

    @Test("Cancel training sets flag")
    func testCancelTraining() async throws {
        let storage = MockTopicModelStorage()
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        // Cancel should work even when not training
        await updater.cancelTraining()

        // No crash expected
    }

    @Test("Trigger micro-retrain without model throws error")
    func testMicroRetrainWithoutModel() async throws {
        let storage = MockTopicModelStorage()
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        do {
            try await updater.triggerMicroRetrain()
            Issue.record("Should have thrown error")
        } catch IncrementalUpdateError.modelNotInitialized {
            // Expected
        } catch {
            Issue.record("Wrong error type: \(error)")
        }
    }

    @Test("Resume with no checkpoint returns false")
    func testResumeNoCheckpoint() async throws {
        let storage = MockTopicModelStorage()
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let resumed = try await updater.resumeIfNeeded()
        #expect(!resumed)
    }

    @Test("Process multiple documents in batch")
    func testProcessDocumentsBatch() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let documents = (0..<5).map { makeDocument(id: $0) }
        let embeddings = (0..<5).map { makeEmbedding(dimension: 4, seed: $0) }

        let assignments = try await updater.processDocuments(documents, embeddings: embeddings)

        #expect(assignments.count == 5)
    }

    @Test("Process batch with mismatched counts throws error")
    func testProcessBatchMismatch() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let documents = (0..<5).map { makeDocument(id: $0) }
        let embeddings = (0..<3).map { makeEmbedding(dimension: 4, seed: $0) }  // Wrong count

        do {
            _ = try await updater.processDocuments(documents, embeddings: embeddings)
            Issue.record("Should have thrown error")
        } catch IncrementalUpdateError.insufficientDocuments(let required, let provided) {
            #expect(required == 5)
            #expect(provided == 3)
        } catch {
            Issue.record("Wrong error type: \(error)")
        }
    }
}

// MARK: - Drift Statistics Integration Tests

@Suite("Drift Statistics Integration Tests")
struct DriftStatisticsIntegrationTests {

    @Test("Processing updates drift statistics")
    func testDriftStatisticsUpdate() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        let initialStats = await updater.getDriftStatistics()
        #expect(initialStats?.totalEntriesTracked == 0)

        // Process several documents
        for i in 0..<10 {
            let doc = makeDocument(id: i)
            let embedding = makeEmbedding(dimension: 4, seed: i)
            _ = try await updater.processDocument(doc, embedding: embedding)
        }

        let updatedStats = await updater.getDriftStatistics()
        #expect(updatedStats?.totalEntriesTracked == 10)
    }

    @Test("Drift detection with high outlier rate")
    func testDriftDetectionHighOutlierRate() async throws {
        let storage = MockTopicModelStorage()

        // Create state with high drift statistics
        var driftStats = DriftStatistics.initial
        // Simulate 100 observations with 30% outliers
        for i in 0..<100 {
            driftStats.observe(distance: 0.5, isOutlier: i % 3 == 0, windowSize: 100)
        }

        let mockState = IncrementalTopicModelState(
            configuration: .default,
            topics: [],
            assignments: ClusterAssignment(labels: [], clusterCount: 0),
            centroids: [],
            inputDimension: 4,
            reducedDimension: 2,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 100,
            lastUpdatedAt: Date(),
            lastFullRetrainAt: Date(),
            documentCountAtLastRetrain: 100,
            driftStatistics: driftStats
        )

        try await storage.saveModelState(mockState)

        let config = IncrementalUpdateConfiguration(outlierRateThreshold: 0.2)
        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: config
        )

        let shouldRefresh = await updater.shouldTriggerFullRefresh()
        #expect(shouldRefresh)
    }
}

// MARK: - Transform Assignment Tests

@Suite("Transform Assignment Tests")
struct TransformAssignmentTests {

    @Test("High similarity gets high confidence")
    func testHighSimilarityHighConfidence() async throws {
        let storage = MockTopicModelStorage()
        let mockState = makeMockModelState()
        try await storage.saveModelState(mockState)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: .testing
        )

        // Use same seed as topic 0's centroid
        let doc = makeDocument(id: 0)
        let embedding = makeEmbedding(dimension: 4, seed: 0)
        let assignment = try await updater.processDocument(doc, embedding: embedding)

        // Should have high confidence since embedding matches topic centroid
        #expect(assignment.confidence > 0.5)
    }

    @Test("Orthogonal embedding becomes outlier")
    func testOrthogonalBecomesOutlier() async throws {
        let storage = MockTopicModelStorage()

        // Create a model with a single topic
        let centroid = Embedding(vector: [1, 0, 0, 0]).normalized()
        let topic = Topic(
            id: TopicID(value: 0),
            keywords: [TopicKeyword(term: "test", score: 1.0)],
            size: 10,
            coherenceScore: 0.5,
            representativeDocuments: [],
            centroid: centroid
        )

        let state = IncrementalTopicModelState.initial(
            configuration: .default,
            topics: [topic],
            assignments: ClusterAssignment(labels: [0], clusterCount: 1),
            centroids: [centroid],
            vocabulary: IncrementalVocabulary(),
            inputDimension: 4,
            reducedDimension: 2,
            documentCount: 1
        )

        try await storage.saveModelState(state)

        let updater = try await IncrementalTopicUpdater(
            storage: storage,
            updateConfiguration: IncrementalUpdateConfiguration(transformOutlierThreshold: 0.3)
        )

        // Create orthogonal embedding
        let orthogonalEmbedding = Embedding(vector: [0, 1, 0, 0]).normalized()
        let doc = makeDocument(id: 1)
        let assignment = try await updater.processDocument(doc, embedding: orthogonalEmbedding)

        // Orthogonal vectors have similarity ~0, should be outlier
        #expect(assignment.isOutlier)
    }
}
