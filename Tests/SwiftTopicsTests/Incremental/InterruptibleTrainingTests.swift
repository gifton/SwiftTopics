// InterruptibleTrainingTests.swift
// SwiftTopicsTests
//
// Tests for interruptible training components

import Testing
import Foundation
@testable import SwiftTopics

// MARK: - Test State Actor

/// Thread-safe state for tests using callbacks.
private actor TestState<T: Sendable> {
    private var value: T

    init(_ initial: T) {
        self.value = initial
    }

    func get() -> T { value }

    func set(_ newValue: T) { value = newValue }

    func update(_ transform: (inout T) -> Void) {
        transform(&value)
    }
}

// MARK: - Interruptible UMAP Tests

@Suite("Interruptible UMAP Optimizer")
struct InterruptibleUMAPTests {

    @Test("UMAP completes full run without interruption")
    func testUMAPFullRun() async throws {
        // Create test data
        let points = generateTestPoints(count: 30, dimensions: 8, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        // Build k-NN graph and fuzzy set
        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: 5,
            metric: .euclidean
        )
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

        // Initialize random embedding
        var rng = RandomState(seed: 42)
        let initialEmbedding = (0..<30).map { _ in
            (0..<2).map { _ in rng.nextFloat(in: -10...10) }
        }

        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: 0.1,
            seed: 42
        )

        let checkpointState = TestState<[Int]>([])

        let result = await optimizer.optimizeInterruptible(
            fuzzySet: fuzzySet,
            nEpochs: 50,
            learningRate: 1.0,
            negativeSampleRate: 5,
            checkpointInterval: 10,
            shouldContinue: { true },
            onCheckpoint: { info in
                await checkpointState.update { $0.append(info.epoch) }
            }
        )

        #expect(result.isComplete)
        #expect(result.completedEpoch == 49)
        #expect(result.embedding.count == 30)

        let epochs = await checkpointState.get()
        #expect(epochs.count >= 4)  // At least 4 checkpoints at interval 10
    }

    @Test("UMAP result has correct dimensions")
    func testUMAPResultDimensions() async throws {
        let points = generateTestPoints(count: 20, dimensions: 10, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: 5,
            metric: .euclidean
        )
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

        var rng = RandomState(seed: 42)
        let initialEmbedding = (0..<20).map { _ in
            (0..<3).map { _ in rng.nextFloat(in: -10...10) }
        }

        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: 0.1,
            seed: 42
        )

        let result = await optimizer.optimizeInterruptible(
            fuzzySet: fuzzySet,
            nEpochs: 30,
            learningRate: 1.0,
            negativeSampleRate: 5,
            checkpointInterval: 15,
            shouldContinue: { true }
        )

        #expect(result.embedding.count == 20)
        #expect(result.embedding[0].count == 3)
    }

    @Test("UMAP progress is monotonic")
    func testUMAPProgressMonotonic() async throws {
        let points = generateTestPoints(count: 25, dimensions: 6, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let knnGraph = try await NearestNeighborGraph.build(
            embeddings: embeddings,
            k: 5,
            metric: .euclidean
        )
        let fuzzySet = FuzzySimplicialSet.build(from: knnGraph)

        var rng = RandomState(seed: 42)
        let initialEmbedding = (0..<25).map { _ in
            (0..<2).map { _ in rng.nextFloat(in: -10...10) }
        }

        let optimizer = UMAPOptimizer(
            initialEmbedding: initialEmbedding,
            minDist: 0.1,
            seed: 42
        )

        let progressState = TestState<[Float]>([])

        let result = await optimizer.optimizeInterruptible(
            fuzzySet: fuzzySet,
            nEpochs: 40,
            learningRate: 1.0,
            negativeSampleRate: 5,
            checkpointInterval: 5,
            shouldContinue: { true },
            onCheckpoint: { info in
                await progressState.update { $0.append(info.progress) }
            }
        )

        #expect(result.isComplete)

        let progresses = await progressState.get()
        for i in 1..<progresses.count {
            #expect(progresses[i] >= progresses[i-1], "Progress should be monotonic")
        }
    }
}

// MARK: - Interruptible MST Tests

@Suite("Interruptible MST Builder")
struct InterruptibleMSTTests {

    @Test("MST completes full run")
    func testMSTFullRun() async throws {
        let points = generateTestPoints(count: 50, dimensions: 4, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let coreComputer = CoreDistanceComputer(minSamples: 3)
        let coreDistances = try await coreComputer.compute(embeddings: embeddings, gpuContext: nil)
        let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)

        let builder = InterruptibleMSTBuilder(checkpointEdgeInterval: 10)
        let result = await builder.build(
            from: mrGraph,
            shouldContinue: { true }
        )

        #expect(result.isComplete)
        #expect(result.edges.count == 49)  // n-1 edges for n points
        #expect(result.pointCount == 50)
    }

    @Test("MST result matches non-interruptible Prim")
    func testMSTMatchesPrim() async throws {
        let points = generateTestPoints(count: 40, dimensions: 4, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let coreComputer = CoreDistanceComputer(minSamples: 3)
        let coreDistances = try await coreComputer.compute(embeddings: embeddings, gpuContext: nil)
        let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)

        // Build using interruptible builder
        let interruptibleBuilder = InterruptibleMSTBuilder(checkpointEdgeInterval: 10)
        let interruptibleResult = await interruptibleBuilder.build(
            from: mrGraph,
            shouldContinue: { true }
        )

        // Build using standard Prim
        let primBuilder = PrimMSTBuilder()
        let primMST = primBuilder.build(from: mrGraph)

        #expect(interruptibleResult.isComplete)
        #expect(interruptibleResult.edges.count == primMST.edges.count)

        // Total weight should be the same
        let interruptibleWeight = interruptibleResult.edges.reduce(0) { $0 + $1.weight }
        let primWeight = primMST.edges.reduce(0) { $0 + $1.weight }
        #expect(abs(interruptibleWeight - primWeight) < 0.001)
    }

    @Test("MST progress is monotonic")
    func testMSTProgressMonotonic() async throws {
        let points = generateTestPoints(count: 30, dimensions: 3, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let coreComputer = CoreDistanceComputer(minSamples: 3)
        let coreDistances = try await coreComputer.compute(embeddings: embeddings, gpuContext: nil)
        let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)

        let progressState = TestState<[Float]>([])

        let builder = InterruptibleMSTBuilder(checkpointEdgeInterval: 5)
        let result = await builder.build(
            from: mrGraph,
            shouldContinue: { true },
            onCheckpoint: { info in
                await progressState.update { $0.append(info.progress) }
            }
        )

        #expect(result.isComplete)

        let progresses = await progressState.get()
        for i in 1..<progresses.count {
            #expect(progresses[i] >= progresses[i-1], "Progress should be monotonic")
        }
    }

    @Test("MST can be resumed from partial state")
    func testMSTResumeFromPartial() async throws {
        let points = generateTestPoints(count: 40, dimensions: 3, clusters: 2)
        let embeddings = points.map { Embedding(vector: $0) }

        let coreComputer = CoreDistanceComputer(minSamples: 3)
        let coreDistances = try await coreComputer.compute(embeddings: embeddings, gpuContext: nil)
        let mrGraph = MutualReachabilityGraph(embeddings: embeddings, coreDistances: coreDistances)

        // First run: build a complete MST
        let builder1 = InterruptibleMSTBuilder(checkpointEdgeInterval: 100)
        let result1 = await builder1.build(
            from: mrGraph,
            shouldContinue: { true }
        )

        // Test resumption with known partial state
        // Take the first 15 edges from a complete run
        #expect(result1.isComplete)
        #expect(result1.edges.count >= 15)

        let startingEdges = Array(result1.edges.prefix(15))
        var startingInMST = [Bool](repeating: false, count: 40)
        startingInMST[0] = true  // Point 0 is always first
        for edge in startingEdges {
            startingInMST[edge.target] = true
        }

        // Resume with partial state
        let builder2 = InterruptibleMSTBuilder(checkpointEdgeInterval: 10)
        let result2 = await builder2.build(
            from: mrGraph,
            startingEdges: startingEdges,
            startingInMST: startingInMST,
            shouldContinue: { true }
        )

        #expect(result2.isComplete)
        #expect(result2.edges.count == 39)  // n-1 edges
    }
}

// MARK: - Checkpoint Serializer Tests

@Suite("Checkpoint Serializer")
struct CheckpointSerializerTests {

    @Test("Serializes and deserializes embeddings")
    func testEmbeddingSerialization() throws {
        let original: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        let data = CheckpointSerializer.serializeEmbedding(original)
        let deserialized = CheckpointSerializer.deserializeEmbedding(data)

        #expect(deserialized != nil)
        #expect(deserialized?.count == 3)
        #expect(deserialized?[0].count == 3)

        for i in 0..<3 {
            for j in 0..<3 {
                #expect(abs(deserialized![i][j] - original[i][j]) < 0.0001)
            }
        }
    }

    @Test("Serializes and deserializes MST edges")
    func testMSTEdgeSerialization() throws {
        let original: [MSTEdge] = [
            MSTEdge(source: 0, target: 1, weight: 1.5),
            MSTEdge(source: 1, target: 2, weight: 2.5),
            MSTEdge(source: 2, target: 3, weight: 0.5)
        ]

        let data = CheckpointSerializer.serializeMSTEdges(original)
        let deserialized = CheckpointSerializer.deserializeMSTEdges(data)

        #expect(deserialized != nil)
        #expect(deserialized?.count == 3)

        for i in 0..<3 {
            #expect(deserialized?[i].source == original[i].source)
            #expect(deserialized?[i].target == original[i].target)
            #expect(abs(deserialized![i].weight - original[i].weight) < 0.0001)
        }
    }

    @Test("Serializes and deserializes boolean arrays")
    func testBoolArraySerialization() throws {
        let original = [true, false, true, true, false, true, false, false, true, true]

        let data = CheckpointSerializer.serializeBoolArray(original)
        let deserialized = CheckpointSerializer.deserializeBoolArray(data)

        #expect(deserialized != nil)
        #expect(deserialized?.count == original.count)

        for i in 0..<original.count {
            #expect(deserialized?[i] == original[i])
        }
    }

    @Test("Serializes and deserializes float arrays")
    func testFloatArraySerialization() throws {
        let original: [Float] = [1.0, 2.5, 3.14159, -42.0, 0.0]

        let data = CheckpointSerializer.serializeFloatArray(original)
        let deserialized = CheckpointSerializer.deserializeFloatArray(data)

        #expect(deserialized != nil)
        #expect(deserialized?.count == original.count)

        for i in 0..<original.count {
            #expect(abs(deserialized![i] - original[i]) < 0.0001)
        }
    }

    @Test("Handles empty embedding")
    func testEmptyEmbedding() throws {
        let original: [[Float]] = []
        let data = CheckpointSerializer.serializeEmbedding(original)
        #expect(data.isEmpty)
    }

    @Test("File-based serialization works")
    func testFileSerialization() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_embedding_\(UUID().uuidString).bin")

        defer {
            try? FileManager.default.removeItem(at: testFile)
        }

        let original: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ]

        try CheckpointSerializer.saveEmbedding(original, to: testFile)
        let loaded = try CheckpointSerializer.loadEmbedding(from: testFile)

        #expect(loaded != nil)
        #expect(loaded?.count == 2)
        #expect(loaded?[0].count == 4)
    }
}

// MARK: - Helper Functions

/// Generates random test points in clusters.
private func generateTestPoints(count: Int, dimensions: Int, clusters: Int) -> [[Float]] {
    var rng = RandomState(seed: 42)
    var points: [[Float]] = []
    points.reserveCapacity(count)

    let pointsPerCluster = count / clusters

    for c in 0..<clusters {
        // Generate cluster center
        let center = (0..<dimensions).map { _ in rng.nextFloat() * 10 - 5 }

        // Generate points around center
        let clusterSize = c < clusters - 1 ? pointsPerCluster : count - points.count
        for _ in 0..<clusterSize {
            let point = center.map { $0 + rng.nextGaussian(standardDeviation: 0.5) }
            points.append(point)
        }
    }

    // Shuffle points
    rng.shuffle(&points)
    return points
}
