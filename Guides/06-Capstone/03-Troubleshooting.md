# 6.3 Troubleshooting

> **Common problems and how to fix them.**

---

## The Challenge

Topic modeling involves many steps, and problems can arise at any stage:

```
Documents → Embeddings → Reduction → Clustering → Keywords → Evaluation
    ↓           ↓           ↓           ↓           ↓           ↓
  Bad input  Dim mismatch  Info loss  No clusters  Noise words  Low scores
```

This guide helps you diagnose and fix common issues.

---

## Quick Diagnostic Checklist

Before diving into specific problems, run through this checklist:

```swift
// 1. Check document count
print("Documents: \(documents.count)")
// < 50 is very small, may get poor results

// 2. Check embedding consistency
let dims = Set(embeddings.map(\.dimension))
print("Embedding dimensions: \(dims)")
// Should be exactly one value

// 3. Check result basics
print("Topics: \(result.topicCount)")
print("Outliers: \(result.statistics.outlierCount)")
print("Outlier rate: \(result.statistics.outlierRate * 100)%")
// > 50% outliers is concerning

// 4. Check coherence
if let coherence = result.coherenceScore {
    print("Mean coherence: \(coherence)")
    // < 0.0 indicates poor topics
}

// 5. Check topic sizes
for topic in result.topics where !topic.isOutlierTopic {
    print("Topic \(topic.id.value): \(topic.size) docs, \(topic.keywordSummary())")
}
```

---

## Problem: No Topics Found

### Symptoms

```swift
result.topicCount == 0
// All documents are outliers
```

### Causes and Solutions

#### Cause 1: Corpus Too Small

```
Problem: HDBSCAN needs enough documents to find density patterns.

If documents.count < 50:
  - Every point looks like an outlier
  - No dense regions form
```

```swift
// Solution: Lower minClusterSize
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(
        minClusterSize: 3,        // Minimum allowed
        minSamples: 2,
        allowSingleCluster: true
    )
)

// Or: Use smallCorpus preset
let config = TopicModelConfiguration.smallCorpus
```

#### Cause 2: minClusterSize Too High

```
Problem: If minClusterSize > largest natural cluster,
         no clusters can form.

Example: 200 documents, minClusterSize = 50
         But largest natural cluster has 40 docs
         → No topics found
```

```swift
// Solution: Lower minClusterSize
// Rule of thumb: minClusterSize ≤ documents.count / 5
let minSize = max(3, documents.count / 10)
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(minClusterSize: minSize)
)
```

#### Cause 3: Embeddings Don't Capture Semantics

```
Problem: If embeddings are random or low-quality,
         similar documents won't be near each other.
```

```swift
// Diagnostic: Check embedding similarity for known-similar docs
func cosineSimilarity(_ a: Embedding, _ b: Embedding) -> Float {
    var dot: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    for i in 0..<a.dimension {
        dot += a.vector[i] * b.vector[i]
        normA += a.vector[i] * a.vector[i]
        normB += b.vector[i] * b.vector[i]
    }
    return dot / (sqrt(normA) * sqrt(normB))
}

// Check similarity between documents that SHOULD be similar
let sim = cosineSimilarity(embeddings[0], embeddings[1])
print("Similarity between doc 0 and 1: \(sim)")
// Should be high (> 0.7) if they're about the same topic

// Solution: Use better embedding model
// - Apple's NLContextualEmbedding
// - OpenAI text-embedding-3
// - Fine-tuned domain-specific model
```

#### Cause 4: Reduction Lost Too Much Information

```
Problem: Reducing from 768D to 5D may lose critical structure.
```

```swift
// Solution: Increase output dimensions
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 30,  // Higher dimension
        method: .pca
    )
)

// Or: Skip reduction if embeddings are already low-dimensional
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .none)
)
```

---

## Problem: Too Many Outliers

### Symptoms

```swift
result.statistics.outlierRate > 0.5  // > 50% are outliers
```

### Causes and Solutions

#### Cause 1: minSamples Too High

```
Problem: High minSamples requires very dense neighborhoods
         for points to be "core points".
```

```swift
// Solution: Lower minSamples
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(
        minClusterSize: 10,
        minSamples: 3  // Lower = more core points
    )
)
```

#### Cause 2: Data Is Actually Sparse

```
Problem: Your corpus genuinely has many unique documents
         that don't fit established topics.

Example: 100 journal entries, each about different things
         → Not a bug, that's the data
```

```swift
// Solution A: Accept some outliers (20-30% is normal)

// Solution B: Lower thresholds if you want more assigned
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(
        minClusterSize: 3,
        minSamples: 2,
        clusterSelectionEpsilon: 0.5  // Merges nearby clusters
    )
)
```

#### Cause 3: Embeddings Are Too Spread Out

```
Problem: After reduction, points are uniformly distributed
         instead of forming clusters.
```

```swift
// Solution: Try UMAP instead of PCA
// UMAP actively optimizes for cluster structure
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .umap,
        umapConfig: UMAPConfiguration(
            nNeighbors: 10,
            minDist: 0.0  // Tighter clusters
        )
    )
)
```

---

## Problem: Poor Topic Quality (Low Coherence)

### Symptoms

```swift
result.coherenceScore ?? 0 < 0.0
// Negative or very low coherence scores
```

### Causes and Solutions

#### Cause 1: Topics Are Mixing Unrelated Concepts

```
Problem: Clustering grouped documents that shouldn't be together.
```

```swift
// Solution: Increase minClusterSize for more robust clusters
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(
        minClusterSize: 10,  // Requires more documents per topic
        clusterSelectionMethod: .eom
    )
)
```

#### Cause 2: Keywords Don't Reflect Topic Content

```
Problem: c-TF-IDF selected rare or generic words.
```

```swift
// Solution: Filter keywords more aggressively
let config = TopicModelConfiguration(
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 10,
        minDocumentFrequency: 3,       // Remove rare terms
        maxDocumentFrequencyRatio: 0.8  // Remove too-common terms
    )
)
```

#### Cause 3: Coherence Window Is Wrong Size

```
Problem: Window size doesn't match how keywords co-occur
         in your documents.
```

```swift
// For short documents (tweets, headlines): smaller window
let config = TopicModelConfiguration(
    coherence: CoherenceConfiguration(windowSize: 5)
)

// For long documents (articles, essays): larger window
let config = TopicModelConfiguration(
    coherence: CoherenceConfiguration(windowSize: 30)
)

// For very varied lengths: document-level
let config = TopicModelConfiguration(
    coherence: CoherenceConfiguration(useDocumentCooccurrence: true)
)
```

#### Cause 4: Stop Words Aren't Filtered

```
Problem: Generic words ("the", "and", "is") appear in keywords.
```

```swift
// Solution: Enable stop word removal (usually on by default)
let config = TopicModelConfiguration(
    representation: CTFIDFConfiguration(
        tokenizer: TokenizerConfiguration(
            removeStopWords: true,
            customStopWords: ["really", "very", "just"]  // Add domain-specific
        )
    )
)
```

---

## Problem: Topics Look Redundant

### Symptoms

```swift
// Multiple topics have overlapping keywords:
// Topic 0: running, exercise, fitness, workout
// Topic 1: fitness, gym, training, exercise
// Topic 2: workout, training, strength, exercise
```

### Causes and Solutions

#### Cause 1: Too Many Fine-Grained Topics

```
Problem: HDBSCAN found many small clusters that are semantically similar.
```

```swift
// Solution A: Increase minClusterSize
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(minClusterSize: 15)
)

// Solution B: Manually merge similar topics
let mergedTopic = try await model.merge(topics: [0, 1, 2])

// Solution C: Reduce topic count programmatically
let reducedTopics = try await model.reduce(to: 5)
```

#### Cause 2: Use EOM Instead of Leaf

```
Problem: Leaf selection produces maximum clusters,
         some may be sub-topics of others.
```

```swift
// Solution: Use EOM (default)
let config = TopicModelConfiguration(
    clustering: HDBSCANConfiguration(
        clusterSelectionMethod: .eom  // Balances stability and granularity
    )
)
```

#### Cause 3: Check Diversity Score

```swift
// Use diversity evaluation to quantify redundancy
let diversityResult = result.topics.evaluateDiversity(topKeywords: 10)

print("Diversity: \(diversityResult.diversity)")
// < 0.7 indicates significant overlap

print("Mean redundancy: \(diversityResult.meanRedundancy)")
// > 0.3 indicates topics sharing many keywords

// Find the most overlapping pairs
if let matrix = diversityResult.overlapMatrix {
    // Find pairs with > 50% overlap
    for i in 0..<matrix.count {
        for j in (i+1)..<matrix[i].count {
            if matrix[i][j] > 0.5 {
                print("Topics \(i) and \(j) overlap \(Int(matrix[i][j] * 100))%")
            }
        }
    }
}
```

---

## Problem: Slow Performance

### Symptoms

```swift
// Fitting takes > 30 seconds for moderate corpus
// Or: UI freezes during topic modeling
```

### Causes and Solutions

#### Cause 1: UMAP Is Slow

```
Problem: UMAP is O(N²) for graph construction and
         has expensive optimization loop.
```

```swift
// Solution: Use PCA during development
let devConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .pca)
)

// Only use UMAP for final production run
let prodConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .umap)
)
```

#### Cause 2: Coherence Evaluation Adds Time

```
Problem: NPMI requires counting all co-occurrences in corpus.
```

```swift
// Solution: Disable coherence for iteration
let fastConfig = TopicModelConfiguration(
    coherence: nil
)

// Re-enable for final evaluation
let fullConfig = TopicModelConfiguration(
    coherence: .default
)
```

#### Cause 3: Large Embedding Dimension

```
Problem: Very high-dimensional embeddings (> 1024) slow down
         all distance calculations.
```

```swift
// Solution: More aggressive reduction
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 10,  // Lower output dimension
        method: .pca
    )
)
```

#### Cause 4: Blocking Main Thread

```
Problem: Async operations not properly dispatched.
```

```swift
// ❌ WRONG: Blocking UI
let result = await model.fit(documents: docs, embeddings: embs)
// If called from main actor, UI freezes

// ✅ CORRECT: Background task
Task.detached {
    let result = try await model.fit(documents: docs, embeddings: embs)
    await MainActor.run {
        self.updateUI(with: result)
    }
}
```

---

## Problem: Memory Issues

### Symptoms

```swift
// App crashes with memory warnings
// Or: System becomes unresponsive
```

### Causes and Solutions

#### Cause 1: Large Corpus Exhausts Memory

```
Problem: 50,000 documents × 768 dimensions × 4 bytes = 150 MB
         just for embeddings in memory.

         Distance matrices are O(N²):
         50,000² × 4 bytes = 10 GB
```

```swift
// Solution A: Batch processing
let batchSize = 5000
var allTopics: [Topic] = []

for batch in documents.chunked(into: batchSize) {
    let batchEmbeddings = // ... get embeddings for batch
    let batchResult = try await model.fit(documents: batch, embeddings: batchEmbeddings)
    allTopics.append(contentsOf: batchResult.topics)
}

// Solution B: Incremental training (see Chapter 6.4)
let updater = try await IncrementalTopicUpdater(storage: storage)
for (doc, embedding) in zip(documents, embeddings) {
    _ = try await updater.processDocument(doc, embedding: embedding)
}
```

#### Cause 2: Retaining Unnecessary Data

```swift
// ❌ Keeping all intermediate results
var allEmbeddings: [[Float]] = []
var reducedEmbeddings: [[Float]] = []
var distances: [[Float]] = []
// Memory: 3× embedding size

// ✅ Let intermediate data be released
let embeddings = computeEmbeddings()
let result = try await model.fit(documents: docs, embeddings: embeddings)
// Intermediate data freed after fit() completes
```

---

## Problem: Inconsistent Results

### Symptoms

```swift
// Same documents give different topics each run
// Topic IDs change between runs
```

### Causes and Solutions

#### Cause: Missing Random Seed

```
Problem: UMAP and some PCA variants use random initialization.
         Without a seed, results vary.
```

```swift
// Solution: Set a seed
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(seed: 42),
    clustering: HDBSCANConfiguration(seed: 42),
    seed: 42
)

// Now results will be reproducible
```

---

## Problem: Transform Gives Wrong Topics

### Symptoms

```swift
// New documents about "fitness" assigned to topic "programming"
```

### Causes and Solutions

#### Cause 1: Different Embedding Provider

```
Problem: Fitted with provider A, transforming with provider B.
         Embedding spaces don't match.
```

```swift
// ❌ WRONG
let result = try await model.fit(docs, embeddingProvider: nlProvider)
let newEmbedding = try await openAIProvider.embed(newDoc)  // Different!
let assignment = try await model.transform([newDoc], [newEmbedding])

// ✅ CORRECT
let result = try await model.fit(docs, embeddingProvider: nlProvider)
let newEmbedding = try await nlProvider.embed(newDoc)  // Same provider
let assignment = try await model.transform([newDoc], [newEmbedding])
```

#### Cause 2: Topic Drift

```
Problem: Model was fitted long ago, topics no longer represent
         current document themes.
```

```swift
// Solution: Retrain periodically
// Or: Use incremental training (Chapter 6.4)

let updater = try await IncrementalTopicUpdater(storage: storage)

// Check if refresh is needed
if updater.shouldTriggerFullRefresh() {
    try await updater.triggerFullRefresh()
}
```

---

## Error Reference

### TopicModelError.invalidInput

```swift
// Cause: Empty documents or embeddings, count mismatch
// Fix: Validate inputs before fitting

guard !documents.isEmpty else { throw ... }
guard documents.count == embeddings.count else { throw ... }
```

### TopicModelError.embeddingDimensionMismatch

```swift
// Cause: Embeddings have different dimensions
// Fix: Ensure consistent embedding provider

let dims = Set(embeddings.map(\.dimension))
assert(dims.count == 1, "All embeddings must have same dimension")
```

### TopicModelError.invalidConfiguration

```swift
// Cause: Configuration validation failed
// Fix: Check the error message

do {
    try config.validate()
} catch TopicModelError.invalidConfiguration(let message) {
    print("Config error: \(message)")
    // Common issues:
    // - minClusterSize < 2
    // - coherence.topKeywords > representation.keywordsPerTopic
    // - outputDimension <= 0
}
```

### TopicModelError.notFitted

```swift
// Cause: Calling transform/search before fit
// Fix: Fit first

if await !model.isFitted {
    _ = try await model.fit(documents: docs, embeddings: embs)
}
let assignment = try await model.transform(...)
```

### TopicModelError.noEmbeddingProvider

```swift
// Cause: Calling findTopics/search when fitted with pre-computed embeddings
// Fix: Provide embedding provider to fit

// ❌ Can't search without provider
let result = try await model.fit(docs, embeddings: precomputedEmbeddings)
let results = try await model.search(query: "test")  // Throws!

// ✅ Provide provider
let result = try await model.fit(docs, embeddingProvider: provider)
let results = try await model.search(query: "test")  // Works
```

---

## Key Takeaways

1. **Start with diagnostics**: Check document count, outlier rate, coherence.

2. **Scale minClusterSize to corpus**: Smaller corpus needs smaller minClusterSize.

3. **Use PCA for development**: Switch to UMAP only for final quality.

4. **Set a seed for reproducibility**: Essential for debugging.

5. **Match embedding providers**: Same provider for fit and transform.

6. **Monitor diversity**: Low diversity indicates redundant topics.

7. **Profile before optimizing**: Know where time is spent.

---

## 💡 Key Insight

```
Debugging topic models is about:

1. DATA QUALITY
   - Are documents long enough?
   - Are embeddings meaningful?
   - Is there actual structure to discover?

2. PARAMETER MATCHING
   - Does minClusterSize match your data?
   - Are window sizes appropriate for doc length?
   - Are you filtering the right terms?

3. PIPELINE INSPECTION
   - What happens at each stage?
   - Where does quality drop?
   - Is the problem in clustering or representation?

When in doubt:
  - Visualize the reduced embeddings (2D plot)
  - Look at specific cluster assignments
  - Compare keyword lists to actual documents
  - Check if "similar" documents are near each other
```

---

## Next Up

You've mastered the basics and can troubleshoot problems. Now let's look at advanced topics like incremental updates and custom extensions.

**[→ 6.4 Next Steps](./04-Next-Steps.md)**

---

*Guide 6.3 of 6.4 • Chapter 6: Capstone*
