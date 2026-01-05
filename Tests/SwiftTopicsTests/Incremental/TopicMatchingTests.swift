// TopicMatchingTests.swift
// SwiftTopicsTests
//
// Tests for Phase 3: Topic Matching & Merging components

import Testing
import Foundation
@testable import SwiftTopics

// MARK: - Hungarian Algorithm Tests

@Suite("Hungarian Algorithm")
struct HungarianMatcherTests {

    @Test("Finds optimal assignment for square matrix")
    func testSquareMatrixOptimal() throws {
        let costs: [[Float]] = [
            [0.1, 0.5, 0.3],
            [0.4, 0.2, 0.6],
            [0.3, 0.4, 0.1]
        ]

        let matcher = HungarianMatcher()
        let (assignments, totalCost) = matcher.solveWithCost(costs: costs)

        #expect(assignments.count == 3)

        // Optimal: (0,0), (1,1), (2,2) with cost 0.1 + 0.2 + 0.1 = 0.4
        #expect(abs(totalCost - 0.4) < 0.001)

        // Verify assignment is valid
        let rows = Set(assignments.map(\.row))
        let cols = Set(assignments.map(\.col))
        #expect(rows.count == 3)
        #expect(cols.count == 3)
    }

    @Test("Handles rectangular matrix (more rows)")
    func testRectangularMoreRows() throws {
        let costs: [[Float]] = [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ]

        let matcher = HungarianMatcher()
        let assignments = matcher.solve(costs: costs)

        // Can only match 2 rows (min of 3 rows, 2 cols)
        #expect(assignments.count == 2)

        // Verify no duplicate columns
        let cols = Set(assignments.map(\.col))
        #expect(cols.count == 2)
    }

    @Test("Handles rectangular matrix (more columns)")
    func testRectangularMoreCols() throws {
        let costs: [[Float]] = [
            [0.1, 0.8, 0.5],
            [0.7, 0.2, 0.4]
        ]

        let matcher = HungarianMatcher()
        let assignments = matcher.solve(costs: costs)

        // Can only match 2 columns (min of 2 rows, 3 cols)
        #expect(assignments.count == 2)

        // Optimal: (0,0), (1,1) with cost 0.3
        let totalCost = assignments.reduce(Float(0)) { $0 + costs[$1.row][$1.col] }
        #expect(abs(totalCost - 0.3) < 0.001)
    }

    @Test("Returns empty for empty matrix")
    func testEmptyMatrix() throws {
        let matcher = HungarianMatcher()
        let assignments = matcher.solve(costs: [])

        #expect(assignments.isEmpty)
    }

    @Test("Handles single element matrix")
    func testSingleElement() throws {
        let costs: [[Float]] = [[0.5]]

        let matcher = HungarianMatcher()
        let assignments = matcher.solve(costs: costs)

        #expect(assignments.count == 1)
        #expect(assignments[0].row == 0)
        #expect(assignments[0].col == 0)
    }

    @Test("Cost matrix from similarities inverts correctly")
    func testSimilarityToCost() throws {
        let similarities: [[Float]] = [
            [0.9, 0.1],
            [0.2, 0.8]
        ]

        let costs = HungarianMatcher.costMatrixFromSimilarities(similarities)

        #expect(abs(costs[0][0] - 0.1) < 0.001)  // 1 - 0.9
        #expect(abs(costs[0][1] - 0.9) < 0.001)  // 1 - 0.1
        #expect(abs(costs[1][0] - 0.8) < 0.001)  // 1 - 0.2
        #expect(abs(costs[1][1] - 0.2) < 0.001)  // 1 - 0.8
    }
}

// MARK: - Topic Matcher Tests

@Suite("Topic Matcher")
struct TopicMatcherTests {

    /// Creates test embeddings that are similar within clusters.
    private func createClusterEmbeddings(
        centers: [[Float]],
        variance: Float = 0.1
    ) -> [Embedding] {
        centers.map { center in
            // Add small variance to make them not identical
            let varied = center.map { $0 + Float.random(in: -variance...variance) }
            return Embedding(vector: varied)
        }
    }

    @Test("Matches identical centroids perfectly")
    func testIdenticalCentroids() throws {
        let oldCentroids = [
            Embedding(vector: [1, 0, 0, 0]),
            Embedding(vector: [0, 1, 0, 0]),
            Embedding(vector: [0, 0, 1, 0])
        ]

        let oldTopics = oldCentroids.enumerated().map { idx, centroid in
            Topic(
                id: TopicID(value: idx),
                keywords: [],
                size: 10,
                centroid: centroid
            )
        }

        // New centroids are identical to old
        let newCentroids = oldCentroids

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics
        )

        // All should match
        #expect(result.newTopicIndices.isEmpty)
        #expect(result.retiredTopicIDs.isEmpty)

        // Verify correct mappings
        for i in 0..<3 {
            #expect(result.newToOld[i] == TopicID(value: i))
        }
    }

    @Test("Creates new topics for dissimilar centroids")
    func testNewTopics() throws {
        let oldCentroids = [
            Embedding(vector: [1, 0, 0, 0])
        ]

        let oldTopics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [],
                size: 10,
                centroid: oldCentroids[0]
            )
        ]

        // New centroid is orthogonal (cosine similarity = 0)
        let newCentroids = [
            Embedding(vector: [0, 1, 0, 0])
        ]

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics
        )

        // Should be treated as new topic (similarity 0 < threshold 0.7)
        #expect(result.newTopicIndices.contains(0))
        #expect(result.retiredTopicIDs.contains(TopicID(value: 0)))
    }

    @Test("Detects topic merges")
    func testMergeDetection() throws {
        // Two old topics that are somewhat similar
        let oldCentroids = [
            Embedding(vector: [1, 0.2, 0, 0]).normalized(),
            Embedding(vector: [1, 0.3, 0, 0]).normalized()
        ]

        let oldTopics = oldCentroids.enumerated().map { idx, centroid in
            Topic(
                id: TopicID(value: idx),
                keywords: [],
                size: 10,
                centroid: centroid
            )
        }

        // One new centroid that matches both old topics
        let newCentroids = [
            Embedding(vector: [1, 0.25, 0, 0]).normalized()
        ]

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics
        )

        // Should detect a merge
        #expect(result.merges.count == 1)
        #expect(result.merges[0].oldIDs.count == 2)
        #expect(result.merges[0].newIndex == 0)
    }

    @Test("Detects topic splits")
    func testSplitDetection() throws {
        // One old topic
        let oldCentroids = [
            Embedding(vector: [1, 0.2, 0, 0]).normalized()
        ]

        let oldTopics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [],
                size: 100,
                centroid: oldCentroids[0]
            )
        ]

        // Two new centroids that both match the old topic
        let newCentroids = [
            Embedding(vector: [1, 0.15, 0, 0]).normalized(),
            Embedding(vector: [1, 0.25, 0, 0]).normalized()
        ]

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics
        )

        // Should detect a split
        #expect(result.splits.count == 1)
        #expect(result.splits[0].oldID == TopicID(value: 0))
        #expect(result.splits[0].newIndices.count == 2)
    }

    @Test("Respects similarity threshold")
    func testSimilarityThreshold() throws {
        let oldCentroids = [
            Embedding(vector: [1, 0, 0, 0])
        ]

        let oldTopics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [],
                size: 10,
                centroid: oldCentroids[0]
            )
        ]

        // New centroid with similarity 0.6
        // cosine([1,0,0,0], [0.6,0.8,0,0]) = 0.6 (below threshold 0.7)
        let newCentroids = [
            Embedding(vector: [0.6, 0.8, 0, 0]).normalized()
        ]

        // With default threshold (0.7), should NOT match
        let matcherDefault = TopicMatcher()
        let resultDefault = matcherDefault.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics,
            configuration: .default
        )
        #expect(resultDefault.newTopicIndices.contains(0))

        // With lenient threshold (0.5), SHOULD match
        let resultLenient = matcherDefault.match(
            newCentroids: newCentroids,
            oldTopics: oldTopics,
            configuration: .lenient
        )
        #expect(resultLenient.newToOld[0] == TopicID(value: 0))
    }

    @Test("Handles empty old topics")
    func testEmptyOldTopics() throws {
        let newCentroids = [
            Embedding(vector: [1, 0, 0, 0]),
            Embedding(vector: [0, 1, 0, 0])
        ]

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: newCentroids,
            oldTopics: []
        )

        // All are new topics
        #expect(result.newTopicIndices.count == 2)
        #expect(result.retiredTopicIDs.isEmpty)
    }

    @Test("Handles empty new centroids")
    func testEmptyNewCentroids() throws {
        let oldTopics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [],
                size: 10,
                centroid: Embedding(vector: [1, 0, 0, 0])
            )
        ]

        let matcher = TopicMatcher()
        let result = matcher.match(
            newCentroids: [],
            oldTopics: oldTopics
        )

        // All old topics are retired
        #expect(result.retiredTopicIDs.count == 1)
        #expect(result.newToOld.isEmpty)
    }
}

// MARK: - Topic ID Generator Tests

@Suite("Topic ID Generator")
struct TopicIDGeneratorTests {

    @Test("Generates sequential IDs from start")
    func testSequentialGeneration() throws {
        var generator = TopicIDGenerator(startingFrom: 0)

        let id1 = generator.next()
        let id2 = generator.next()
        let id3 = generator.next()

        #expect(id1.value == 0)
        #expect(id2.value == 1)
        #expect(id3.value == 2)
    }

    @Test("Observes existing IDs")
    func testObserveIDs() throws {
        var generator = TopicIDGenerator()

        generator.observe(TopicID(value: 5))
        generator.observe(TopicID(value: 3))

        // Next ID should be 6 (one higher than max observed)
        let nextID = generator.next()
        #expect(nextID.value == 6)
    }

    @Test("Ignores outlier IDs")
    func testIgnoresOutliers() throws {
        var generator = TopicIDGenerator()

        generator.observe(TopicID.outlier)  // -1
        generator.observe(TopicID(value: 2))

        let nextID = generator.next()
        #expect(nextID.value == 3)  // Should be 3, not affected by -1
    }

    @Test("Initializes from topics")
    func testInitFromTopics() throws {
        let topics = [
            Topic(id: TopicID(value: 5), keywords: [], size: 10),
            Topic(id: TopicID(value: 2), keywords: [], size: 10),
            Topic(id: TopicID(value: 8), keywords: [], size: 10)
        ]

        var generator = TopicIDGenerator(observing: topics)
        let nextID = generator.next()

        #expect(nextID.value == 9)  // One higher than max (8)
    }

    @Test("Creates ID mappings from match result")
    func testCreateIDMappings() throws {
        let matchResult = TopicMatcher.MatchResult(
            newToOld: [
                0: TopicID(value: 3),  // Matched to existing topic 3
                1: nil,                 // New topic
                2: TopicID(value: 7)   // Matched to existing topic 7
            ],
            merges: [],
            splits: [],
            newTopicIndices: [1],
            retiredTopicIDs: [],
            similarityMatrix: []
        )

        var generator = TopicIDGenerator(startingFrom: 0)
        let mappings = generator.createIDMappings(from: matchResult)

        #expect(mappings[0]?.value == 3)  // Keeps existing ID
        #expect(mappings[2]?.value == 7)  // Keeps existing ID
        #expect(mappings[1]?.value == 8)  // New ID (7 was observed, so next is 8)
    }

    @Test("Peek does not consume ID")
    func testPeekNonConsuming() throws {
        var generator = TopicIDGenerator(startingFrom: 5)

        let peek1 = generator.peek()
        let peek2 = generator.peek()
        let actual = generator.next()

        #expect(peek1.value == 5)
        #expect(peek2.value == 5)
        #expect(actual.value == 5)

        // After consuming, peek should show 6
        let peekAfter = generator.peek()
        #expect(peekAfter.value == 6)
    }

    @Test("Generator is Codable")
    func testCodable() throws {
        var original = TopicIDGenerator(startingFrom: 10)
        _ = original.next()  // Consume one ID, now at 11

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        var decoded = try decoder.decode(TopicIDGenerator.self, from: data)

        #expect(decoded.next().value == 11)
    }
}

// MARK: - Model Merger Tests

@Suite("Model Merger")
struct ModelMergerTests {

    /// Creates a simple test state for merging.
    private func createTestMainModel() -> IncrementalTopicModelState {
        let topics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [TopicKeyword(term: "travel", score: 0.9)],
                size: 50,
                centroid: Embedding(vector: [1, 0, 0, 0])
            ),
            Topic(
                id: TopicID(value: 1),
                keywords: [TopicKeyword(term: "food", score: 0.85)],
                size: 30,
                centroid: Embedding(vector: [0, 1, 0, 0])
            )
        ]

        let centroids = topics.compactMap(\.centroid)

        return IncrementalTopicModelState(
            configuration: TopicModelConfiguration(),
            topics: topics,
            assignments: ClusterAssignment(labels: [0, 0, 1, 1, 1], clusterCount: 2),
            centroids: centroids,
            inputDimension: 128,
            reducedDimension: 4,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 80
        )
    }

    /// Creates a mini-model result for testing.
    private func createTestMiniModel(
        centroids: [Embedding],
        labels: [Int]
    ) -> MiniModelResult {
        let topics = centroids.enumerated().map { idx, centroid in
            Topic(
                id: TopicID(value: idx),  // Temporary IDs
                keywords: [TopicKeyword(term: "new\(idx)", score: 0.8)],
                size: labels.filter { $0 == idx }.count,
                centroid: centroid
            )
        }

        return MiniModelResult(
            topics: topics,
            centroids: centroids,
            assignments: ClusterAssignment(
                labels: labels,
                clusterCount: centroids.count
            ),
            vocabulary: IncrementalVocabulary()
        )
    }

    @Test("Preserves topic IDs for matched topics")
    func testPreservesMatchedIDs() throws {
        let mainModel = createTestMainModel()

        // Mini-model with centroids matching existing topics
        let miniModel = createTestMiniModel(
            centroids: [
                Embedding(vector: [1, 0, 0, 0]),  // Matches topic 0
                Embedding(vector: [0, 1, 0, 0])   // Matches topic 1
            ],
            labels: [0, 0, 1, 1]
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // Both topics should keep their original IDs
        let topicIDs = Set(result.topics.map(\.id))
        #expect(topicIDs.contains(TopicID(value: 0)))
        #expect(topicIDs.contains(TopicID(value: 1)))

        // Check summary
        #expect(result.summary.topicsUpdated == 2)
        #expect(result.summary.topicsCreated == 0)
    }

    @Test("Creates new IDs for unmatched topics")
    func testCreatesNewIDs() throws {
        let mainModel = createTestMainModel()

        // Mini-model with a completely new topic
        let miniModel = createTestMiniModel(
            centroids: [
                Embedding(vector: [0, 0, 1, 0])  // New topic (orthogonal to existing)
            ],
            labels: [0, 0, 0]
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // Should have original 2 topics + 1 new
        #expect(result.topics.count == 3)

        // New topic should have ID 2 (next after 0, 1)
        let newTopic = result.topics.first { $0.id.value >= 2 }
        #expect(newTopic != nil)
        #expect(newTopic?.id.value == 2)

        #expect(result.summary.topicsCreated == 1)
    }

    @Test("Merges centroids with weighted average")
    func testWeightedAverageCentroids() throws {
        let mainModel = createTestMainModel()  // Topic 0 has size 50

        // Mini-model with centroid that matches topic 0
        let newCentroid = Embedding(vector: [0.9, 0.1, 0, 0]).normalized()
        let miniModel = createTestMiniModel(
            centroids: [newCentroid],
            labels: [0, 0, 0, 0, 0]  // 5 new documents
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel,
            matchConfig: .lenient  // Lower threshold to allow the match
        )

        // Find the merged topic
        if let mergedTopic = result.topics.first(where: { $0.id.value == 0 }) {
            // Size should be combined
            #expect(mergedTopic.size > 50)

            // Centroid should be weighted average (heavily toward original due to size 50 vs 5)
            if let centroid = mergedTopic.centroid {
                // Original was [1, 0, 0, 0], new is ~[0.9, 0.1, 0, 0]
                // With 50:5 weighting, result should be close to original
                #expect(centroid.vector[0] > 0.95)
            }
        }
    }

    @Test("Combines keywords using union strategy")
    func testKeywordUnion() throws {
        let topics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [
                    TopicKeyword(term: "travel", score: 0.9),
                    TopicKeyword(term: "vacation", score: 0.7)
                ],
                size: 50,
                centroid: Embedding(vector: [1, 0, 0, 0])
            )
        ]

        let mainModel = IncrementalTopicModelState(
            configuration: TopicModelConfiguration(),
            topics: topics,
            assignments: ClusterAssignment(labels: [0], clusterCount: 1),
            centroids: [Embedding(vector: [1, 0, 0, 0])],
            inputDimension: 128,
            reducedDimension: 4,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 50
        )

        let miniTopics = [
            Topic(
                id: TopicID(value: 0),
                keywords: [
                    TopicKeyword(term: "trip", score: 0.85),
                    TopicKeyword(term: "travel", score: 0.8)  // Duplicate
                ],
                size: 10,
                centroid: Embedding(vector: [1, 0, 0, 0])
            )
        ]

        let miniModel = MiniModelResult(
            topics: miniTopics,
            centroids: [Embedding(vector: [1, 0, 0, 0])],
            assignments: ClusterAssignment(labels: [0], clusterCount: 1),
            vocabulary: IncrementalVocabulary()
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        if let mergedTopic = result.topics.first {
            let terms = Set(mergedTopic.keywords.map(\.term))
            // Should have all unique terms
            #expect(terms.contains("travel"))
            #expect(terms.contains("vacation"))
            #expect(terms.contains("trip"))

            // "travel" should appear only once with the higher score
            let travelKeywords = mergedTopic.keywords.filter { $0.term == "travel" }
            #expect(travelKeywords.count == 1)
            #expect(travelKeywords[0].score == 0.9)  // Higher of 0.9 and 0.8
        }
    }

    @Test("Remaps new document assignments")
    func testAssignmentRemapping() throws {
        let mainModel = createTestMainModel()

        // Mini-model that matches existing topics
        let miniModel = createTestMiniModel(
            centroids: [
                Embedding(vector: [1, 0, 0, 0]),  // Matches topic 0
                Embedding(vector: [0, 1, 0, 0])   // Matches topic 1
            ],
            labels: [0, 1, 0, 1, -1]  // Mix of assignments, one outlier
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // New assignments should use stable IDs
        let labels = result.newDocumentAssignments.labels
        #expect(labels[0] == 0)   // Mapped to stable ID 0
        #expect(labels[1] == 1)   // Mapped to stable ID 1
        #expect(labels[4] == -1)  // Outlier remains outlier
    }

    @Test("Handles merge summary correctly")
    func testMergeSummary() throws {
        let mainModel = createTestMainModel()

        // Mini-model with one matched topic and one new topic
        let miniModel = createTestMiniModel(
            centroids: [
                Embedding(vector: [1, 0, 0, 0]),  // Matches topic 0
                Embedding(vector: [0, 0, 1, 0])   // New topic
            ],
            labels: [0, 0, 1, 1]
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        #expect(result.summary.topicsUpdated >= 1)
        #expect(result.summary.topicsCreated == 1)
        #expect(result.summary.totalTopics == 3)
    }

    @Test("ID generator state is persisted in result")
    func testIDGeneratorPersistence() throws {
        let mainModel = createTestMainModel()

        // Mini-model with one new topic
        let miniModel = createTestMiniModel(
            centroids: [
                Embedding(vector: [0, 0, 1, 0])
            ],
            labels: [0, 0]
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // The ID generator should be ready to generate the next ID
        var generator = result.idGenerator
        let nextID = generator.next()

        // Should be 3 (after 0, 1, 2)
        #expect(nextID.value == 3)
    }
}

// MARK: - Integration Tests

@Suite("Topic Matching Integration")
struct TopicMatchingIntegrationTests {

    @Test("Topics maintain IDs after micro-retrain with similar data")
    func testTopicIDStability() async throws {
        // Simulate main model with 3 topics
        let travelCentroid = Embedding(vector: [1, 0, 0, 0])
        let foodCentroid = Embedding(vector: [0, 1, 0, 0])
        let workCentroid = Embedding(vector: [0, 0, 1, 0])

        let mainTopics = [
            Topic(id: TopicID(value: 0), keywords: [TopicKeyword(term: "travel", score: 0.9)], size: 50, centroid: travelCentroid),
            Topic(id: TopicID(value: 1), keywords: [TopicKeyword(term: "food", score: 0.85)], size: 30, centroid: foodCentroid),
            Topic(id: TopicID(value: 2), keywords: [TopicKeyword(term: "work", score: 0.8)], size: 40, centroid: workCentroid)
        ]

        let mainModel = IncrementalTopicModelState(
            configuration: TopicModelConfiguration(),
            topics: mainTopics,
            assignments: ClusterAssignment(labels: Array(repeating: 0, count: 120), clusterCount: 3),
            centroids: [travelCentroid, foodCentroid, workCentroid],
            inputDimension: 128,
            reducedDimension: 4,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 120
        )

        // Mini-model from micro-retrain with similar topics (slightly different centroids)
        let newTravelCentroid = Embedding(vector: [0.98, 0.02, 0, 0]).normalized()
        let newFoodCentroid = Embedding(vector: [0.02, 0.98, 0, 0]).normalized()
        let newWorkCentroid = Embedding(vector: [0, 0.02, 0.98, 0]).normalized()

        let miniModel = MiniModelResult(
            topics: [
                Topic(id: TopicID(value: 0), keywords: [], size: 10, centroid: newTravelCentroid),
                Topic(id: TopicID(value: 1), keywords: [], size: 8, centroid: newFoodCentroid),
                Topic(id: TopicID(value: 2), keywords: [], size: 12, centroid: newWorkCentroid)
            ],
            centroids: [newTravelCentroid, newFoodCentroid, newWorkCentroid],
            assignments: ClusterAssignment(labels: [0, 0, 1, 1, 2, 2, 2, 2], clusterCount: 3),
            vocabulary: IncrementalVocabulary()
        )

        // Merge
        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // All 3 original topic IDs should still exist
        let topicIDs = Set(result.topics.map(\.id.value))
        #expect(topicIDs.contains(0))
        #expect(topicIDs.contains(1))
        #expect(topicIDs.contains(2))

        // No new topics should be created
        #expect(result.summary.topicsCreated == 0)
        #expect(result.summary.topicsUpdated == 3)
    }

    @Test("New cluster gets new topic ID")
    func testNewTopicCreation() async throws {
        // Main model with 2 topics
        let mainTopics = [
            Topic(id: TopicID(value: 0), keywords: [], size: 50, centroid: Embedding(vector: [1, 0, 0, 0])),
            Topic(id: TopicID(value: 1), keywords: [], size: 30, centroid: Embedding(vector: [0, 1, 0, 0]))
        ]

        let mainModel = IncrementalTopicModelState(
            configuration: TopicModelConfiguration(),
            topics: mainTopics,
            assignments: ClusterAssignment(labels: [], clusterCount: 2),
            centroids: mainTopics.compactMap(\.centroid),
            inputDimension: 128,
            reducedDimension: 4,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 80
        )

        // Mini-model with a completely new topic (orthogonal to existing)
        let newTopicCentroid = Embedding(vector: [0, 0, 1, 0])

        let miniModel = MiniModelResult(
            topics: [
                Topic(id: TopicID(value: 0), keywords: [TopicKeyword(term: "fitness", score: 0.9)], size: 15, centroid: newTopicCentroid)
            ],
            centroids: [newTopicCentroid],
            assignments: ClusterAssignment(labels: Array(repeating: 0, count: 15), clusterCount: 1),
            vocabulary: IncrementalVocabulary()
        )

        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel
        )

        // Should have 3 topics now
        #expect(result.topics.count == 3)

        // New topic should have ID 2
        let newTopic = result.topics.first { $0.id.value == 2 }
        #expect(newTopic != nil)
        #expect(newTopic?.keywords.first?.term == "fitness")

        #expect(result.summary.topicsCreated == 1)
        #expect(result.summary.topicsRetired == 0)
    }

    @Test("Below-threshold matches become new topics")
    func testSimilarityThresholdEnforcement() async throws {
        // Main model with one topic
        let mainTopics = [
            Topic(id: TopicID(value: 0), keywords: [], size: 50, centroid: Embedding(vector: [1, 0, 0, 0]))
        ]

        let mainModel = IncrementalTopicModelState(
            configuration: TopicModelConfiguration(),
            topics: mainTopics,
            assignments: ClusterAssignment(labels: [], clusterCount: 1),
            centroids: mainTopics.compactMap(\.centroid),
            inputDimension: 128,
            reducedDimension: 4,
            vocabulary: IncrementalVocabulary(),
            totalDocumentCount: 50
        )

        // Mini-model with a topic that has low similarity (0.5) to existing
        // Using [0.5, 0.866, 0, 0] gives cosine similarity ~0.5 with [1, 0, 0, 0]
        let lowSimilarityCentroid = Embedding(vector: [0.5, 0.866, 0, 0]).normalized()

        let miniModel = MiniModelResult(
            topics: [
                Topic(id: TopicID(value: 0), keywords: [], size: 10, centroid: lowSimilarityCentroid)
            ],
            centroids: [lowSimilarityCentroid],
            assignments: ClusterAssignment(labels: [0], clusterCount: 1),
            vocabulary: IncrementalVocabulary()
        )

        // With default threshold (0.7), this should be treated as new
        let merger = ModelMerger()
        let result = merger.matchAndMerge(
            miniModel: miniModel,
            mainModel: mainModel,
            matchConfig: .default  // threshold = 0.7
        )

        // Should have 2 topics (original + new)
        #expect(result.topics.count == 2)
        #expect(result.summary.topicsCreated == 1)
    }
}
