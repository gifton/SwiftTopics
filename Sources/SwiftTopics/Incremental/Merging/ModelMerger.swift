// ModelMerger.swift
// SwiftTopics
//
// Merges mini-model results into the main topic model

import Foundation

// MARK: - Mini Model Result

/// Result from training a mini-model on new documents.
///
/// This represents the output of a micro-retrain: a topic model trained
/// on just the new documents, before merging with the main model.
public struct MiniModelResult: Sendable {

    /// Topics discovered in the new documents.
    ///
    /// These have temporary indices (0, 1, 2...) that need to be mapped
    /// to stable IDs via `TopicMatcher`.
    public let topics: [Topic]

    /// Centroids for each topic.
    public let centroids: [Embedding]

    /// Document assignments for the new documents.
    ///
    /// Labels are indices into the `topics` array.
    public let assignments: ClusterAssignment

    /// Vocabulary built from the new documents.
    public let vocabulary: IncrementalVocabulary

    /// Creates a mini-model result.
    public init(
        topics: [Topic],
        centroids: [Embedding],
        assignments: ClusterAssignment,
        vocabulary: IncrementalVocabulary
    ) {
        self.topics = topics
        self.centroids = centroids
        self.assignments = assignments
        self.vocabulary = vocabulary
    }
}

// MARK: - Model Merger

/// Merges a mini-model into the main topic model.
///
/// After a micro-retrain produces a mini-model on new documents, `ModelMerger`
/// combines it with the existing main model to produce an updated model with:
///
/// - Stable topic IDs (matched topics keep their IDs)
/// - Merged centroids (weighted by document count)
/// - Combined keywords (re-ranked by c-TF-IDF)
/// - Updated vocabulary (terms from both models)
///
/// ## Merge Process
///
/// 1. **Topic Matching**: Use `TopicMatcher` to find correspondences
/// 2. **ID Assignment**: Map mini-model topic indices to stable IDs
/// 3. **Centroid Merge**: Combine centroids using weighted average
/// 4. **Keyword Merge**: Combine and re-rank keywords
/// 5. **Assignment Update**: Map new document assignments to stable IDs
///
/// ## Thread Safety
///
/// `ModelMerger` is `Sendable` and stateless.
public struct ModelMerger: Sendable {

    // MARK: - Configuration

    /// Configuration for model merging.
    public struct Configuration: Sendable {

        /// How to update centroids when merging topics.
        public let centroidMergeStrategy: CentroidMergeStrategy

        /// How to combine keywords when merging topics.
        public let keywordMergeStrategy: KeywordMergeStrategy

        /// Maximum number of keywords to keep per topic.
        public let maxKeywordsPerTopic: Int

        /// Default configuration.
        public static let `default` = Configuration(
            centroidMergeStrategy: .weightedAverage,
            keywordMergeStrategy: .unionReranked,
            maxKeywordsPerTopic: 10
        )

        /// Creates a custom configuration.
        public init(
            centroidMergeStrategy: CentroidMergeStrategy = .weightedAverage,
            keywordMergeStrategy: KeywordMergeStrategy = .unionReranked,
            maxKeywordsPerTopic: Int = 10
        ) {
            self.centroidMergeStrategy = centroidMergeStrategy
            self.keywordMergeStrategy = keywordMergeStrategy
            self.maxKeywordsPerTopic = maxKeywordsPerTopic
        }
    }

    /// Strategy for merging topic centroids.
    public enum CentroidMergeStrategy: Sendable {
        /// Weighted average by document count.
        ///
        /// mergedCentroid = (oldCentroid * oldSize + newCentroid * newSize) / totalSize
        case weightedAverage

        /// Use the larger cluster's centroid.
        case largerCluster

        /// Use the new centroid (fully replace).
        case useNew
    }

    /// Strategy for merging topic keywords.
    public enum KeywordMergeStrategy: Sendable {
        /// Union of keywords, re-ranked by combined c-TF-IDF.
        case unionReranked

        /// Keep existing keywords first, append new unique ones.
        case existingFirst

        /// Use only the new keywords (fully replace).
        case useNew
    }

    // MARK: - Merge Result

    /// Result of merging a mini-model into the main model.
    public struct MergeResult: Sendable {

        /// Updated topics with stable IDs.
        public let topics: [Topic]

        /// Updated centroids aligned with topics.
        public let centroids: [Embedding]

        /// Updated vocabulary combining both models.
        public let vocabulary: IncrementalVocabulary

        /// Updated assignments for the new documents only.
        ///
        /// Labels are now stable topic IDs (TopicID.value).
        public let newDocumentAssignments: ClusterAssignment

        /// The ID generator state after merge (for persistence).
        public let idGenerator: TopicIDGenerator

        /// Summary of the merge operation.
        public let summary: MergeSummary
    }

    /// Summary of merge changes.
    public struct MergeSummary: Sendable, Codable {
        /// Topics that remained unchanged.
        public let topicsUnchanged: Int

        /// Topics that were updated (new documents added).
        public let topicsUpdated: Int

        /// Genuinely new topics created.
        public let topicsCreated: Int

        /// Topics that merged together.
        public let topicsMerged: Int

        /// Topics that were retired (no longer exist).
        public let topicsRetired: Int

        /// Total topics in the merged model.
        public var totalTopics: Int {
            topicsUnchanged + topicsUpdated + topicsCreated
        }

        public var description: String {
            """
            Merge Summary:
              Unchanged: \(topicsUnchanged)
              Updated: \(topicsUpdated)
              Created: \(topicsCreated)
              Merged: \(topicsMerged)
              Retired: \(topicsRetired)
              Total: \(totalTopics)
            """
        }
    }

    // MARK: - Initialization

    /// Creates a new model merger.
    public init() {}

    // MARK: - Public API

    /// Merges a mini-model into the main model.
    ///
    /// - Parameters:
    ///   - miniModel: Topics and assignments from training on new documents.
    ///   - mainModel: Current main model state.
    ///   - matchResult: Result from `TopicMatcher.match()`.
    ///   - configuration: Merge configuration.
    /// - Returns: Merged result with stable topic IDs.
    public func merge(
        miniModel: MiniModelResult,
        mainModel: IncrementalTopicModelState,
        matchResult: TopicMatcher.MatchResult,
        configuration: Configuration = .default
    ) -> MergeResult {
        // Initialize ID generator from existing topics
        var idGenerator = TopicIDGenerator(observing: mainModel.topics)

        // Create ID mappings for mini-model topics
        let idMappings = idGenerator.createIDMappings(from: matchResult)

        // Merge topics
        let (mergedTopics, mergedCentroids) = mergeTopics(
            mainTopics: mainModel.topics,
            mainCentroids: mainModel.centroids,
            miniTopics: miniModel.topics,
            miniCentroids: miniModel.centroids,
            miniAssignments: miniModel.assignments,
            idMappings: idMappings,
            matchResult: matchResult,
            configuration: configuration
        )

        // Merge vocabulary
        let mergedVocabulary = mergeVocabulary(
            mainVocab: mainModel.vocabulary,
            miniVocab: miniModel.vocabulary,
            idMappings: idMappings,
            matchResult: matchResult
        )

        // Update new document assignments to use stable IDs
        let newAssignments = remapAssignments(
            assignments: miniModel.assignments,
            idMappings: idMappings
        )

        // Compute summary
        let summary = computeSummary(
            mainTopicCount: mainModel.topics.count,
            matchResult: matchResult,
            mergedTopicCount: mergedTopics.count
        )

        return MergeResult(
            topics: mergedTopics,
            centroids: mergedCentroids,
            vocabulary: mergedVocabulary,
            newDocumentAssignments: newAssignments,
            idGenerator: idGenerator,
            summary: summary
        )
    }

    // MARK: - Private Implementation

    /// Merges topics from main and mini models.
    ///
    /// For micro-retrain scenarios, we keep all main topics that weren't matched
    /// (they still have their old documents). Only matched topics are merged.
    /// The `retiredTopicIDs` is informational only - in micro-retrain, those topics
    /// just didn't have any matching new documents, but they still exist.
    private func mergeTopics(
        mainTopics: [Topic],
        mainCentroids: [Embedding],
        miniTopics: [Topic],
        miniCentroids: [Embedding],
        miniAssignments: ClusterAssignment,
        idMappings: [Int: TopicID],
        matchResult: TopicMatcher.MatchResult,
        configuration: Configuration
    ) -> (topics: [Topic], centroids: [Embedding]) {
        var topics = [Topic]()
        var centroids = [Embedding]()

        // Track which main topics were matched (will be merged with mini topics)
        let matchedMainIDs = Set(matchResult.newToOld.values.compactMap { $0 })

        // First, add unmatched main topics (unchanged - keep even if not matched)
        // In micro-retrain, these topics still have their old documents
        for (index, topic) in mainTopics.enumerated() {
            if !matchedMainIDs.contains(topic.id) {
                topics.append(topic)
                if index < mainCentroids.count {
                    centroids.append(mainCentroids[index])
                }
            }
        }

        // Now process mini-model topics
        for (miniIndex, miniTopic) in miniTopics.enumerated() {
            guard let stableID = idMappings[miniIndex] else { continue }

            if let matchedOldID = matchResult.newToOld[miniIndex], matchedOldID != nil {
                // This mini-topic matches an existing topic - merge them
                if let mainIndex = mainTopics.firstIndex(where: { $0.id == matchedOldID }) {
                    let mainTopic = mainTopics[mainIndex]
                    let mainCentroid = mainIndex < mainCentroids.count ? mainCentroids[mainIndex] : nil
                    let miniCentroid = miniIndex < miniCentroids.count ? miniCentroids[miniIndex] : nil

                    let mergedTopic = mergeSingleTopic(
                        mainTopic: mainTopic,
                        mainCentroid: mainCentroid,
                        miniTopic: miniTopic,
                        miniCentroid: miniCentroid,
                        miniSize: miniAssignments.clusterSizes.indices.contains(miniIndex)
                            ? miniAssignments.clusterSizes[miniIndex] : miniTopic.size,
                        stableID: stableID,
                        configuration: configuration
                    )

                    topics.append(mergedTopic.topic)
                    if let centroid = mergedTopic.centroid {
                        centroids.append(centroid)
                    }
                }
            } else {
                // This is a new topic - add with stable ID
                let newTopic = Topic(
                    id: stableID,
                    keywords: Array(miniTopic.keywords.prefix(configuration.maxKeywordsPerTopic)),
                    size: miniTopic.size,
                    coherenceScore: miniTopic.coherenceScore,
                    representativeDocuments: miniTopic.representativeDocuments,
                    centroid: miniIndex < miniCentroids.count ? miniCentroids[miniIndex] : nil
                )
                topics.append(newTopic)
                if miniIndex < miniCentroids.count {
                    centroids.append(miniCentroids[miniIndex])
                }
            }
        }

        // Sort by topic ID for consistency
        let sortedIndices = topics.indices.sorted { topics[$0].id < topics[$1].id }
        topics = sortedIndices.map { topics[$0] }
        centroids = sortedIndices.compactMap { idx in
            idx < centroids.count ? centroids[idx] : nil
        }

        return (topics, centroids)
    }

    /// Merges a single topic pair.
    private func mergeSingleTopic(
        mainTopic: Topic,
        mainCentroid: Embedding?,
        miniTopic: Topic,
        miniCentroid: Embedding?,
        miniSize: Int,
        stableID: TopicID,
        configuration: Configuration
    ) -> (topic: Topic, centroid: Embedding?) {
        let mainSize = mainTopic.size
        let totalSize = mainSize + miniSize

        // Merge centroid
        let mergedCentroid: Embedding?
        switch configuration.centroidMergeStrategy {
        case .weightedAverage:
            mergedCentroid = weightedAverageCentroid(
                centroid1: mainCentroid,
                size1: mainSize,
                centroid2: miniCentroid,
                size2: miniSize
            )
        case .largerCluster:
            mergedCentroid = mainSize >= miniSize ? mainCentroid : miniCentroid
        case .useNew:
            mergedCentroid = miniCentroid ?? mainCentroid
        }

        // Merge keywords
        let mergedKeywords: [TopicKeyword]
        switch configuration.keywordMergeStrategy {
        case .unionReranked:
            mergedKeywords = unionRerankedKeywords(
                keywords1: mainTopic.keywords,
                keywords2: miniTopic.keywords,
                maxCount: configuration.maxKeywordsPerTopic
            )
        case .existingFirst:
            mergedKeywords = existingFirstKeywords(
                existing: mainTopic.keywords,
                new: miniTopic.keywords,
                maxCount: configuration.maxKeywordsPerTopic
            )
        case .useNew:
            mergedKeywords = Array(miniTopic.keywords.prefix(configuration.maxKeywordsPerTopic))
        }

        // Average coherence scores if both exist
        let mergedCoherence: Float?
        if let mainCoh = mainTopic.coherenceScore, let miniCoh = miniTopic.coherenceScore {
            mergedCoherence = (mainCoh * Float(mainSize) + miniCoh * Float(miniSize)) / Float(totalSize)
        } else {
            mergedCoherence = mainTopic.coherenceScore ?? miniTopic.coherenceScore
        }

        let mergedTopic = Topic(
            id: stableID,
            keywords: mergedKeywords,
            size: totalSize,
            coherenceScore: mergedCoherence,
            representativeDocuments: mainTopic.representativeDocuments + miniTopic.representativeDocuments,
            centroid: mergedCentroid
        )

        return (mergedTopic, mergedCentroid)
    }

    /// Computes weighted average of two centroids.
    private func weightedAverageCentroid(
        centroid1: Embedding?,
        size1: Int,
        centroid2: Embedding?,
        size2: Int
    ) -> Embedding? {
        guard let c1 = centroid1 else { return centroid2 }
        guard let c2 = centroid2 else { return centroid1 }
        guard c1.dimension == c2.dimension else { return c1 }

        let totalSize = Float(size1 + size2)
        guard totalSize > 0 else { return c1 }

        let w1 = Float(size1) / totalSize
        let w2 = Float(size2) / totalSize

        var merged = [Float]()
        merged.reserveCapacity(c1.dimension)

        for i in 0..<c1.dimension {
            merged.append(c1.vector[i] * w1 + c2.vector[i] * w2)
        }

        return Embedding(vector: merged)
    }

    /// Combines keywords with union and re-ranking by score.
    private func unionRerankedKeywords(
        keywords1: [TopicKeyword],
        keywords2: [TopicKeyword],
        maxCount: Int
    ) -> [TopicKeyword] {
        // Create map of term -> best score
        var termToKeyword = [String: TopicKeyword]()

        for kw in keywords1 {
            if let existing = termToKeyword[kw.term] {
                if kw.score > existing.score {
                    termToKeyword[kw.term] = kw
                }
            } else {
                termToKeyword[kw.term] = kw
            }
        }

        for kw in keywords2 {
            if let existing = termToKeyword[kw.term] {
                // For union, we could add scores or take max
                // Here we take max for simplicity
                if kw.score > existing.score {
                    termToKeyword[kw.term] = kw
                }
            } else {
                termToKeyword[kw.term] = kw
            }
        }

        // Sort by score descending
        let sorted = termToKeyword.values.sorted { $0.score > $1.score }
        return Array(sorted.prefix(maxCount))
    }

    /// Keeps existing keywords first, appends new unique ones.
    private func existingFirstKeywords(
        existing: [TopicKeyword],
        new: [TopicKeyword],
        maxCount: Int
    ) -> [TopicKeyword] {
        var result = Array(existing.prefix(maxCount))
        let existingTerms = Set(existing.map(\.term))

        for kw in new {
            if result.count >= maxCount { break }
            if !existingTerms.contains(kw.term) {
                result.append(kw)
            }
        }

        return result
    }

    /// Merges vocabularies from main and mini models.
    private func mergeVocabulary(
        mainVocab: IncrementalVocabulary,
        miniVocab: IncrementalVocabulary,
        idMappings: [Int: TopicID],
        matchResult: TopicMatcher.MatchResult
    ) -> IncrementalVocabulary {
        // For now, we use the main vocabulary as base and add new terms
        // A more sophisticated approach would recompute frequencies
        var merged = mainVocab

        // Add terms from mini vocabulary
        for term in miniVocab.indexToTerm {
            _ = merged.getOrAddTerm(term)
        }

        // Update document count
        merged = IncrementalVocabulary(
            termToIndex: merged.termToIndex,
            indexToTerm: merged.indexToTerm,
            topicTermFrequencies: merged.topicTermFrequencies,
            documentFrequencies: merged.documentFrequencies,
            totalDocuments: mainVocab.totalDocuments + miniVocab.totalDocuments
        )

        return merged
    }

    /// Remaps mini-model assignments to use stable topic IDs.
    private func remapAssignments(
        assignments: ClusterAssignment,
        idMappings: [Int: TopicID]
    ) -> ClusterAssignment {
        let remappedLabels = assignments.labels.map { label -> Int in
            if label < 0 {
                return -1  // Outlier
            }
            if let stableID = idMappings[label] {
                return stableID.value
            }
            return label  // Fallback (shouldn't happen)
        }

        // Compute new cluster count from unique non-outlier labels
        let uniqueLabels = Set(remappedLabels.filter { $0 >= 0 })
        let clusterCount = uniqueLabels.count

        return ClusterAssignment(
            labels: remappedLabels,
            probabilities: assignments.probabilities,
            outlierScores: assignments.outlierScores,
            clusterCount: clusterCount
        )
    }

    /// Computes summary statistics for the merge.
    ///
    /// For micro-retrain, topics in `retiredTopicIDs` are kept as unchanged
    /// (they still have their old documents, just no new documents matched them).
    private func computeSummary(
        mainTopicCount: Int,
        matchResult: TopicMatcher.MatchResult,
        mergedTopicCount: Int
    ) -> MergeSummary {
        let matchedCount = matchResult.newToOld.values.compactMap { $0 }.count
        let newCount = matchResult.newTopicIndices.count
        let mergeCount = matchResult.merges.count

        // For micro-retrain: topics that weren't matched are unchanged (not retired)
        // retiredTopicIDs in the match result just means "no matching new topic"
        // but those topics still exist with their old documents
        let unchangedCount = mainTopicCount - matchedCount

        // In micro-retrain, nothing is truly retired - topics without new matches are just unchanged
        // For full refresh scenarios, this would need different logic
        let retiredCount = 0

        return MergeSummary(
            topicsUnchanged: max(0, unchangedCount),
            topicsUpdated: matchedCount,
            topicsCreated: newCount,
            topicsMerged: mergeCount,
            topicsRetired: retiredCount
        )
    }
}

// MARK: - Convenience Extensions

extension ModelMerger {

    /// Performs full topic matching and merging in one step.
    ///
    /// This is a convenience method that combines `TopicMatcher.match()` and
    /// `merge()` into a single call.
    ///
    /// - Parameters:
    ///   - miniModel: The mini-model from micro-retrain.
    ///   - mainModel: The current main model state.
    ///   - matchConfig: Configuration for topic matching.
    ///   - mergeConfig: Configuration for merging.
    /// - Returns: The merged result.
    public func matchAndMerge(
        miniModel: MiniModelResult,
        mainModel: IncrementalTopicModelState,
        matchConfig: TopicMatcher.Configuration = .default,
        mergeConfig: Configuration = .default
    ) -> MergeResult {
        let matcher = TopicMatcher()

        let matchResult = matcher.match(
            newCentroids: miniModel.centroids,
            oldTopics: mainModel.topics,
            configuration: matchConfig
        )

        return merge(
            miniModel: miniModel,
            mainModel: mainModel,
            matchResult: matchResult,
            configuration: mergeConfig
        )
    }
}
