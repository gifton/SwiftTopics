import Testing
import Foundation
@testable import SwiftTopics

// MARK: - File-Based Storage Tests

@Suite("FileBasedTopicModelStorage")
struct FileBasedStorageTests {

    // MARK: - Test Helpers

    /// Creates a temporary directory for testing.
    func createTempDirectory() throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SwiftTopicsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir,
            withIntermediateDirectories: true
        )
        return tempDir
    }

    /// Cleans up a test directory.
    func cleanup(directory: URL) {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: - Initialization Tests

    @Test("Creates storage directory if it doesn't exist")
    func testCreatesDirectory() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SwiftTopicsTests-\(UUID().uuidString)")

        // Directory shouldn't exist yet
        #expect(!FileManager.default.fileExists(atPath: tempDir.path))

        // Create storage - should create directory
        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Directory should now exist
        #expect(FileManager.default.fileExists(atPath: tempDir.path))

        // Verify storage works
        let count = try await storage.embeddingCount()
        #expect(count == 0)

        cleanup(directory: tempDir)
    }

    // MARK: - Embedding Storage Tests

    @Test("Appends and loads embeddings")
    func testAppendAndLoadEmbeddings() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Create test embeddings
        let doc1 = DocumentID()
        let doc2 = DocumentID()
        let emb1 = Embedding(vector: [1.0, 2.0, 3.0, 4.0])
        let emb2 = Embedding(vector: [5.0, 6.0, 7.0, 8.0])

        // Append embeddings
        try await storage.appendEmbeddings([(doc1, emb1), (doc2, emb2)])

        // Verify count
        let count = try await storage.embeddingCount()
        #expect(count == 2)

        // Load specific embeddings
        let loaded = try await storage.loadEmbeddings(for: [doc2, doc1])
        #expect(loaded.count == 2)

        // Verify order matches request order
        #expect(loaded[0].vector == emb2.vector)
        #expect(loaded[1].vector == emb1.vector)
    }

    @Test("Loads all embeddings")
    func testLoadAllEmbeddings() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Create and append embeddings
        let docs = (0..<5).map { _ in DocumentID() }
        let embeddings = docs.map { doc in
            (doc, Embedding.random(dimension: 8))
        }

        try await storage.appendEmbeddings(embeddings)

        // Load all
        let all = try await storage.loadAllEmbeddings()
        #expect(all.count == 5)

        // Verify all docs are present
        let loadedIDs = Set(all.map { $0.0 })
        let expectedIDs = Set(docs)
        #expect(loadedIDs == expectedIDs)
    }

    @Test("Appends to existing embeddings file")
    func testAppendIncremental() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // First batch
        let doc1 = DocumentID()
        let emb1 = Embedding(vector: [1.0, 2.0, 3.0])
        try await storage.appendEmbeddings([(doc1, emb1)])

        #expect(try await storage.embeddingCount() == 1)

        // Second batch
        let doc2 = DocumentID()
        let emb2 = Embedding(vector: [4.0, 5.0, 6.0])
        try await storage.appendEmbeddings([(doc2, emb2)])

        #expect(try await storage.embeddingCount() == 2)

        // Verify both are loadable
        let all = try await storage.loadAllEmbeddings()
        #expect(all.count == 2)
    }

    @Test("Throws error for non-existent document")
    func testLoadNonExistentDocument() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Add one document
        let doc1 = DocumentID()
        try await storage.appendEmbeddings([(doc1, Embedding(vector: [1.0]))])

        // Try to load non-existent
        let nonExistent = DocumentID()

        do {
            _ = try await storage.loadEmbeddings(for: [nonExistent])
            Issue.record("Expected error to be thrown")
        } catch is TopicModelStorageError {
            // Expected
        }
    }

    // MARK: - Pending Buffer Tests

    @Test("Appends to and drains pending buffer")
    func testPendingBuffer() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Initially empty
        #expect(try await storage.pendingBufferCount() == 0)

        // Add entries
        let entry1 = BufferedEntry(
            documentID: DocumentID(),
            embedding: Embedding(vector: [1.0, 2.0]),
            tokenizedContent: ["hello", "world"]
        )
        let entry2 = BufferedEntry(
            documentID: DocumentID(),
            embedding: Embedding(vector: [3.0, 4.0]),
            tokenizedContent: ["test"]
        )

        try await storage.appendToPendingBuffer([entry1, entry2])

        #expect(try await storage.pendingBufferCount() == 2)

        // Drain buffer
        let drained = try await storage.drainPendingBuffer()
        #expect(drained.count == 2)

        // Buffer should be empty after drain
        #expect(try await storage.pendingBufferCount() == 0)
    }

    // MARK: - Checkpoint Tests

    @Test("Saves and loads checkpoint")
    func testCheckpoint() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Initially no checkpoint
        let initial = try await storage.loadCheckpoint()
        #expect(initial == nil)

        // Save checkpoint
        let checkpoint = TrainingCheckpoint.initial(
            trainingType: .microRetrain,
            documentIDs: [DocumentID(), DocumentID()]
        )

        try await storage.saveCheckpoint(checkpoint)

        // Load checkpoint
        let loaded = try await storage.loadCheckpoint()
        #expect(loaded != nil)
        #expect(loaded?.runID == checkpoint.runID)
        #expect(loaded?.trainingType == .microRetrain)
        #expect(loaded?.documentIDs.count == 2)

        // Clear checkpoint
        try await storage.clearCheckpoint()
        let cleared = try await storage.loadCheckpoint()
        #expect(cleared == nil)
    }

    // MARK: - Model State Tests

    @Test("Saves and loads model state")
    func testModelState() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Initially no state
        let initial = try await storage.loadModelState()
        #expect(initial == nil)

        // Create and save state
        let state = IncrementalTopicModelState.initial(
            configuration: .default,
            topics: [],
            assignments: ClusterAssignment(labels: [], clusterCount: 0),
            centroids: [],
            vocabulary: IncrementalVocabulary(),
            inputDimension: 512,
            reducedDimension: 15,
            documentCount: 0
        )

        try await storage.saveModelState(state)

        // Load state
        let loaded = try await storage.loadModelState()
        #expect(loaded != nil)
        #expect(loaded?.inputDimension == 512)
        #expect(loaded?.reducedDimension == 15)
    }

    // MARK: - Maintenance Tests

    @Test("Clears all storage")
    func testClear() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Add various data
        try await storage.appendEmbeddings([
            (DocumentID(), Embedding(vector: [1.0]))
        ])
        try await storage.appendToPendingBuffer([
            BufferedEntry(
                documentID: DocumentID(),
                embedding: Embedding(vector: [1.0]),
                tokenizedContent: []
            )
        ])

        // Verify data exists
        #expect(try await storage.embeddingCount() == 1)
        #expect(try await storage.pendingBufferCount() == 1)

        // Clear all
        try await storage.clear()

        // Verify cleared
        #expect(try await storage.embeddingCount() == 0)
        #expect(try await storage.pendingBufferCount() == 0)
    }

    @Test("Reports storage size")
    func testStorageSize() async throws {
        let tempDir = try createTempDirectory()
        defer { cleanup(directory: tempDir) }

        let storage = try FileBasedTopicModelStorage(directory: tempDir)

        // Initially empty
        let initialSize = try await storage.storageSizeBytes()
        #expect(initialSize == 0)

        // Add embeddings
        let largeEmbedding = Embedding(vector: [Float](repeating: 1.0, count: 512))
        try await storage.appendEmbeddings([
            (DocumentID(), largeEmbedding),
            (DocumentID(), largeEmbedding),
            (DocumentID(), largeEmbedding)
        ])

        // Should have non-trivial size now
        let size = try await storage.storageSizeBytes()
        #expect(size > 0)

        // Size should be approximately 3 × 512 × 4 + overhead
        let expectedMinSize: UInt64 = 3 * 512 * 4
        #expect(size >= expectedMinSize)
    }
}

// MARK: - Buffered Entry Tests

@Suite("BufferedEntry")
struct BufferedEntryTests {

    @Test("Creates buffered entry with correct properties")
    func testCreation() {
        let docID = DocumentID()
        let embedding = Embedding(vector: [1.0, 2.0, 3.0])
        let tokens = ["hello", "world"]
        let now = Date()

        let entry = BufferedEntry(
            documentID: docID,
            embedding: embedding,
            tokenizedContent: tokens,
            addedAt: now
        )

        #expect(entry.documentID == docID)
        #expect(entry.embedding.vector == embedding.vector)
        #expect(entry.tokenizedContent == tokens)
        #expect(entry.addedAt == now)
        #expect(entry.id == docID)
    }

    @Test("Buffered entries are comparable by date")
    func testComparable() {
        let older = BufferedEntry(
            documentID: DocumentID(),
            embedding: Embedding(vector: [1.0]),
            tokenizedContent: [],
            addedAt: Date(timeIntervalSinceNow: -100)
        )

        let newer = BufferedEntry(
            documentID: DocumentID(),
            embedding: Embedding(vector: [1.0]),
            tokenizedContent: [],
            addedAt: Date()
        )

        #expect(older < newer)
    }
}

// MARK: - Training Checkpoint Tests

@Suite("TrainingCheckpoint")
struct TrainingCheckpointTests {

    @Test("Creates initial checkpoint")
    func testInitial() {
        let docs = [DocumentID(), DocumentID()]
        let checkpoint = TrainingCheckpoint.initial(
            trainingType: .fullRefresh,
            documentIDs: docs
        )

        #expect(checkpoint.trainingType == .fullRefresh)
        #expect(checkpoint.documentIDs.count == 2)
        #expect(checkpoint.currentPhase == .umapKNN)
        #expect(checkpoint.lastCompletedPhase == nil)
        #expect(checkpoint.currentPhaseProgress == 0)
        #expect(checkpoint.attemptsRemaining == 3)
        #expect(checkpoint.canResume == true)
    }

    @Test("Updates progress correctly")
    func testProgressUpdate() {
        let initial = TrainingCheckpoint.initial(
            trainingType: .microRetrain,
            documentIDs: [DocumentID()]
        )

        let updated = initial.withProgress(phase: .umapOptimization, progress: 0.5)

        #expect(updated.currentPhase == .umapOptimization)
        #expect(updated.currentPhaseProgress == 0.5)
        #expect(updated.runID == initial.runID)
    }

    @Test("Completes phase correctly")
    func testPhaseCompletion() {
        let initial = TrainingCheckpoint.initial(
            trainingType: .microRetrain,
            documentIDs: [DocumentID()]
        )

        let afterPhase1 = initial.withCompletedPhase(.umapKNN)

        #expect(afterPhase1.lastCompletedPhase == .umapKNN)
        #expect(afterPhase1.currentPhase == .umapFuzzySet)
        #expect(afterPhase1.currentPhaseProgress == 0)
    }

    @Test("Decrements attempts")
    func testDecrementAttempts() {
        var checkpoint = TrainingCheckpoint.initial(
            trainingType: .microRetrain,
            documentIDs: [DocumentID()]
        )

        #expect(checkpoint.attemptsRemaining == 3)
        #expect(checkpoint.canResume == true)

        checkpoint = checkpoint.withDecrementedAttempts()
        #expect(checkpoint.attemptsRemaining == 2)
        #expect(checkpoint.canResume == true)

        checkpoint = checkpoint.withDecrementedAttempts()
        checkpoint = checkpoint.withDecrementedAttempts()
        #expect(checkpoint.attemptsRemaining == 0)
        #expect(checkpoint.canResume == false)
    }

    @Test("Training phase has correct properties")
    func testTrainingPhase() {
        #expect(TrainingPhase.umapKNN.displayName == "Building neighbor graph")
        #expect(TrainingPhase.umapOptimization.supportsPartialCheckpoint == true)
        #expect(TrainingPhase.clusterExtraction.supportsPartialCheckpoint == false)

        // Test phase navigation
        #expect(TrainingPhase.umapKNN.next == .umapFuzzySet)
        #expect(TrainingPhase.umapKNN.previous == nil)
        #expect(TrainingPhase.complete.next == nil)
    }
}

// MARK: - Drift Statistics Tests

@Suite("DriftStatistics")
struct DriftStatisticsTests {

    @Test("Initial statistics are zeroed")
    func testInitial() {
        let stats = DriftStatistics.initial

        #expect(stats.recentAverageDistance == 0)
        #expect(stats.overallAverageDistance == 0)
        #expect(stats.recentOutlierRate == 0)
        #expect(stats.recentWindowSize == 0)
        #expect(stats.totalEntriesTracked == 0)
    }

    @Test("Observes distances correctly")
    func testObserve() {
        var stats = DriftStatistics.initial

        // Observe some distances
        stats.observe(distance: 1.0, isOutlier: false)
        stats.observe(distance: 2.0, isOutlier: false)
        stats.observe(distance: 3.0, isOutlier: true)

        #expect(stats.totalEntriesTracked == 3)
        #expect(stats.overallAverageDistance > 0)
    }

    @Test("Calculates drift ratio")
    func testDriftRatio() {
        var stats = DriftStatistics.initial

        // Establish baseline
        for _ in 0..<50 {
            stats.observe(distance: 1.0, isOutlier: false, windowSize: 20)
        }

        let baselineRatio = stats.driftRatio

        // Now add higher distance entries
        for _ in 0..<50 {
            stats.observe(distance: 3.0, isOutlier: false, windowSize: 20)
        }

        // Drift ratio should increase
        #expect(stats.driftRatio > baselineRatio)
    }

    @Test("Detects need for refresh")
    func testNeedsRefresh() {
        var stats = DriftStatistics.initial

        // With few entries, should not need refresh
        for _ in 0..<10 {
            stats.observe(distance: 100.0, isOutlier: true)
        }
        #expect(stats.needsRefresh() == false)

        // With enough entries and high outlier rate, should need refresh
        for _ in 0..<30 {
            stats.observe(distance: 100.0, isOutlier: true)
        }
        #expect(stats.needsRefresh(outlierThreshold: 0.5) == true)
    }
}

// MARK: - Incremental Vocabulary Tests

@Suite("IncrementalVocabulary")
struct IncrementalVocabularyTests {

    @Test("Creates empty vocabulary")
    func testEmpty() {
        let vocab = IncrementalVocabulary()

        #expect(vocab.termCount == 0)
        #expect(vocab.topicCount == 0)
        #expect(vocab.totalDocuments == 0)
    }

    @Test("Adds terms and tracks frequencies")
    func testAddDocument() {
        var vocab = IncrementalVocabulary()

        vocab.addDocument(terms: ["hello", "world", "hello"], topic: 0)

        #expect(vocab.termCount == 2)
        #expect(vocab.topicCount == 1)
        #expect(vocab.totalDocuments == 1)

        // "hello" appears twice in topic 0
        let helloIdx = vocab.termToIndex["hello"]!
        #expect(vocab.topicTermFrequencies[0][helloIdx] == 2)
    }

    @Test("Computes IDF correctly")
    func testIDF() {
        var vocab = IncrementalVocabulary()

        // Add documents - "hello" in all, "rare" in one
        vocab.addDocument(terms: ["hello", "rare"], topic: 0)
        vocab.addDocument(terms: ["hello", "world"], topic: 0)
        vocab.addDocument(terms: ["hello"], topic: 0)

        let helloIdx = vocab.termToIndex["hello"]!
        let rareIdx = vocab.termToIndex["rare"]!

        // IDF of "hello" should be lower (appears in all docs)
        // IDF of "rare" should be higher (appears in 1 doc)
        let helloIDF = vocab.idf(for: helloIdx)
        let rareIDF = vocab.idf(for: rareIdx)

        #expect(rareIDF > helloIDF)
    }

    @Test("Supports multiple topics")
    func testMultipleTopics() {
        var vocab = IncrementalVocabulary()

        vocab.addDocument(terms: ["tech", "computer"], topic: 0)
        vocab.addDocument(terms: ["sports", "game"], topic: 1)
        vocab.addDocument(terms: ["tech", "mobile"], topic: 0)

        #expect(vocab.topicCount == 2)

        // "tech" should only be in topic 0
        let techIdx = vocab.termToIndex["tech"]!
        #expect(vocab.topicTermFrequencies[0][techIdx] == 2)
        #expect(vocab.topicTermFrequencies[1][techIdx] == 0)
    }
}
