# 6.1 The Complete Pipeline

> **End-to-end walkthrough: from documents to topics in production code.**

---

## The Concept

You've learned each component. Now let's see them work together.

```
The Complete Flow:

  Documents           Embeddings            TopicModel            Result
  ─────────────       ─────────────         ──────────────       ─────────

  ┌─────────┐         ┌─────────┐          ┌─────────────┐       ┌─────┐
  │ "Had a  │         │[0.23,   │          │             │       │Topic│
  │ great   │──────►  │ -0.15,  │ ──────►  │    fit()    │ ────► │  0  │
  │ run..." │         │  0.89]  │          │             │       │     │
  └─────────┘         └─────────┘          │  reduction  │       ├─────┤
  ┌─────────┐         ┌─────────┐          │  clustering │       │Topic│
  │ "Swift  │         │[0.71,   │          │  represent  │       │  1  │
  │ async   │──────►  │ -0.02,  │ ──────►  │  evaluate   │ ────► │     │
  │ is..."  │         │  0.33]  │          │             │       ├─────┤
  └─────────┘         └─────────┘          └─────────────┘       │ ... │
      ...                 ...                                    └─────┘
```

This guide shows you the complete code path.

---

## Step 1: Prepare Documents

```swift
// 📍 See: Sources/SwiftTopics/Core/Document.swift

import SwiftTopics

// Create documents from your source data
let documents = journalEntries.map { entry in
    Document(
        id: DocumentID(string: entry.id),  // Optional: provide your own ID
        content: entry.text,
        metadata: DocumentMetadata([
            "date": .date(entry.date),
            "mood": .string(entry.mood)
        ])
    )
}

// Simpler version without metadata
let simpleDocuments = texts.map { text in
    Document(content: text)
}
```

### Document ID Strategies

```swift
// Strategy 1: Auto-generated UUIDs (default)
Document(content: "Hello")  // Gets random UUID

// Strategy 2: String-based (deterministic)
Document(stringID: "entry-001", content: "Hello")
// Same string → same ID (reproducible)

// Strategy 3: Existing UUIDs
Document(id: DocumentID(uuid: myUUID), content: "Hello")
```

### Metadata for Filtering

```swift
// Add metadata for post-processing
let doc = Document(
    content: "Finished my marathon training today!",
    metadata: DocumentMetadata([
        "category": .string("fitness"),
        "year": .int(2025),
        "tags": .strings(["running", "training"])
    ])
)

// Later, filter documents by metadata
let fitnessTopics = result.topics.filter { topic in
    let topicDocs = result.documents(for: topic.id)
    return topicDocs.contains { docID in
        // Retrieve original document and check metadata
        documents.first { $0.id == docID }?.metadata?["category"]?.stringValue == "fitness"
    }
}
```

---

## Step 2: Compute Embeddings

SwiftTopics is embedding-agnostic. You provide embeddings from any source.

```swift
// Your embedding provider (not part of SwiftTopics core)
let embeddings: [Embedding]

// Option A: Use SwiftTopicsApple (Apple's NLContextualEmbedding)
import SwiftTopicsApple
let provider = NLEmbeddingProvider()
embeddings = try await provider.embedBatch(documents.map(\.content))

// Option B: Use a Core ML model
let bertModel = try BERTEncoder()
embeddings = try await bertModel.embedBatch(documents.map(\.content))

// Option C: Use an external API
embeddings = try await openAIClient.embed(
    texts: documents.map(\.content),
    model: "text-embedding-3-small"
)

// Option D: Pre-computed embeddings from a database
embeddings = try database.loadEmbeddings(for: documents.map(\.id))
```

### Embedding Dimension

```
Common embedding dimensions:

  Apple NLContextualEmbedding: 512
  BERT base:                   768
  OpenAI ada-002:             1536
  OpenAI text-embedding-3-small: 1536

SwiftTopics handles any dimension—it will reduce to 15D by default.
```

### Validating Embeddings

```swift
// Embeddings must match document count
assert(embeddings.count == documents.count)

// Embeddings must have consistent dimensions
let dimension = embeddings[0].dimension
for embedding in embeddings {
    assert(embedding.dimension == dimension)
}

// SwiftTopics validates this for you and throws TopicModelError.invalidInput
```

---

## Step 3: Configure the Model

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

// Option A: Use a preset
let config = TopicModelConfiguration.default

// Option B: Custom configuration
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 15,
        method: .pca
    ),
    clustering: HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3,
        clusterSelectionMethod: .eom
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 10,
        diversify: true,
        diversityWeight: 0.3
    ),
    coherence: CoherenceConfiguration(
        windowSize: 10,
        topKeywords: 10
    ),
    seed: 42  // For reproducibility
)

// Option C: Use the builder pattern
let config = TopicModelConfigurationBuilder()
    .reductionDimension(20)
    .minClusterSize(8)
    .keywordsPerTopic(12)
    .enableCoherence(true)
    .seed(42)
    .build()
```

### Configuration Validation

```swift
// Configuration is validated before fitting
do {
    try config.validate()
} catch TopicModelError.invalidConfiguration(let message) {
    print("Invalid config: \(message)")
}

// Common validation errors:
// - "Reduction output dimension must be positive"
// - "Minimum cluster size must be at least 2"
// - "Coherence topKeywords exceeds representation keywordsPerTopic"
```

---

## Step 4: Fit the Model

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModel.swift

let model = TopicModel(configuration: config)

// Fit the model (async operation)
let result = try await model.fit(
    documents: documents,
    embeddings: embeddings
)

// That's it! The model:
// 1. Reduces embeddings from 768D → 15D (PCA)
// 2. Clusters with HDBSCAN
// 3. Extracts keywords with c-TF-IDF
// 4. Computes coherence scores (if enabled)
```

### Progress Reporting

For large corpora, track progress:

```swift
// Set up progress handler
await model.setProgressHandler { progress in
    print(progress.description)
    // "Reducing dimensions (15%)"
    // "Clustering embeddings (45%)"
    // "Extracting keywords (80%)"
    // "Evaluating coherence (95%)"
}

// Then fit
let result = try await model.fit(documents: documents, embeddings: embeddings)
```

### Pipeline Stages

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         fit() Pipeline Stages                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Stage        Weight    What Happens                                      │
│  ─────        ──────    ────────────                                      │
│                                                                           │
│  embedding    30%       If using EmbeddingProvider, compute embeddings    │
│  reduction    15%       PCA or UMAP reduces dimensions                    │
│  clustering   30%       HDBSCAN discovers clusters                        │
│  represent    15%       c-TF-IDF extracts keywords per cluster            │
│  evaluation   10%       NPMI coherence scoring (if enabled)               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Step 5: Use the Results

### Exploring Topics

```swift
// 📍 See: Sources/SwiftTopics/Core/TopicModelResult.swift

print("Found \(result.topicCount) topics")
print("Outlier rate: \(result.statistics.outlierRate * 100)%")

if let coherence = result.coherenceScore {
    print("Mean coherence: \(coherence)")
}

// Iterate through topics
for topic in result.topics {
    guard !topic.isOutlierTopic else { continue }

    print("\nTopic \(topic.id.value):")
    print("  Keywords: \(topic.keywordSummary())")
    print("  Size: \(topic.size) documents")

    if let coherence = topic.coherenceScore {
        print("  Coherence: \(String(format: "%.3f", coherence))")
    }
}
```

### Finding Document Topics

```swift
// Get topic for a specific document
let docID = documents[0].id
if let assignment = result.topicAssignment(for: docID) {
    print("Document is in topic \(assignment.topicID.value)")
    print("Confidence: \(assignment.probability)")

    if let topic = result.topic(for: docID) {
        print("Topic keywords: \(topic.keywordSummary())")
    }
}

// Get all documents in a topic
let topic0Docs = result.documents(for: TopicID(value: 0))
print("Topic 0 has \(topic0Docs.count) documents")

// Get representative documents (closest to centroid)
let representatives = result.representativeDocuments(for: TopicID(value: 0), count: 5)
```

### Sorting Topics

```swift
// By size (largest first)
let bySize = result.topicsBySizeDescending
print("Largest topic: \(bySize.first?.keywordSummary() ?? "none")")

// By coherence (most coherent first)
let byCoherence = result.topicsByCoherenceDescending
print("Most coherent: \(byCoherence.first?.keywordSummary() ?? "none")")
```

---

## Step 6: Transform New Documents

After fitting, assign new documents to existing topics:

```swift
// New document comes in
let newDoc = Document(content: "Just finished a 10K run in the park!")
let newEmbedding = try await embeddingProvider.embed(newDoc.content)

// Transform assigns to nearest topic
let assignments = try await model.transform(
    documents: [newDoc],
    embeddings: [newEmbedding]
)

let assignment = assignments[0]
print("Assigned to topic \(assignment.topicID.value)")
print("Distance to centroid: \(assignment.distanceToCentroid ?? 0)")

// Get the topic details
if let topic = await model.topics?.first(where: { $0.id == assignment.topicID }) {
    print("Topic: \(topic.keywordSummary())")
}
```

### Transform vs. Fit

```
fit():       Discovers topics from data
transform(): Assigns to existing topics

Use transform() when:
- You have a fitted model
- New documents should use existing topics
- You don't want to retrain

Use fit() again when:
- Topics are stale (corpus has changed significantly)
- You want to rediscover topics
- New documents represent new themes
```

---

## Step 7: Search and Discovery

### Semantic Search

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModel.swift

// Search requires an embedding provider
let model = TopicModel(configuration: .default)
_ = try await model.fit(documents: documents, embeddingProvider: provider)

// Search for similar documents
let results = try await model.search(query: "running and exercise", topK: 5)

for (doc, score) in results {
    print("Score \(score): \(doc.content.prefix(50))...")
}
```

### Finding Topics for Text

```swift
// Find which topic best matches arbitrary text
let assignments = try await model.findTopics(for: "machine learning is fascinating")

if let top = assignments.first {
    print("Best topic: \(top.topicID.value)")
    print("Probability: \(top.probability)")

    // Get alternatives
    if let alternatives = top.alternatives {
        for alt in alternatives.prefix(3) {
            print("  Alternative: Topic \(alt.topicID.value) (\(alt.probability))")
        }
    }
}
```

---

## Step 8: Topic Manipulation

### Merging Topics

```swift
// If two topics seem to overlap, merge them
let merged = try await model.merge(topics: [0, 2])

print("Merged topic: \(merged.keywordSummary())")
print("Combined size: \(merged.size)")

// The model's state is updated
// Topics 0 and 2 are replaced with the merged topic
```

### Reducing Topic Count

```swift
// Hierarchical reduction to target count
let reducedTopics = try await model.reduce(to: 5)

print("Reduced to \(reducedTopics.count) topics")
for topic in reducedTopics {
    print("  \(topic.keywordSummary())")
}

// This iteratively merges the most similar pairs
// until the target count is reached
```

---

## Complete Example

Here's a full working example:

```swift
import SwiftTopics

// 1. Prepare data
let journalEntries: [(id: String, text: String, date: Date)] = loadEntries()

let documents = journalEntries.map { entry in
    Document(
        stringID: entry.id,
        content: entry.text,
        metadata: DocumentMetadata(["date": .date(entry.date)])
    )
}

// 2. Compute embeddings
let embeddingProvider = NLEmbeddingProvider()  // Or your preferred provider
let embeddings = try await embeddingProvider.embedBatch(documents.map(\.content))

// 3. Configure and fit
let config = TopicModelConfiguration(
    reduction: .init(outputDimension: 15, method: .pca),
    clustering: .init(minClusterSize: 5),
    representation: .init(keywordsPerTopic: 10),
    coherence: .default,
    seed: 42
)

let model = TopicModel(configuration: config)
let result = try await model.fit(documents: documents, embeddings: embeddings)

// 4. Analyze results
print("📊 Topic Model Results")
print("══════════════════════")
print("Topics discovered: \(result.topicCount)")
print("Documents: \(result.documentCount)")
print("Outliers: \(result.statistics.outlierCount) (\(Int(result.statistics.outlierRate * 100))%)")

if let coherence = result.coherenceScore {
    print("Mean coherence: \(String(format: "%.3f", coherence))")
}

print("\n📌 Topics:")
for topic in result.topicsBySizeDescending {
    let quality: String
    if let c = topic.coherenceScore {
        quality = c > 0.3 ? "✓" : c > 0 ? "○" : "✗"
    } else {
        quality = "?"
    }

    print("  \(quality) Topic \(topic.id.value) [\(topic.size) docs]: \(topic.keywordSummary())")
}

// 5. Query specific documents
print("\n🔍 Sample Assignments:")
for doc in documents.prefix(5) {
    if let assignment = result.topicAssignment(for: doc.id),
       !assignment.isOutlier {
        print("  \"\(doc.content.prefix(30))...\" → Topic \(assignment.topicID.value)")
    }
}

// 6. Test transformation on new content
let testTexts = [
    "Great workout at the gym today",
    "Debugging Swift async code",
    "Feeling stressed about deadlines"
]

for text in testTexts {
    let embedding = try await embeddingProvider.embed(text)
    let assignments = try await model.transform(
        documents: [Document(content: text)],
        embeddings: [embedding]
    )
    if let a = assignments.first, !a.isOutlier {
        let topic = result.topic(id: a.topicID)
        print("  \"\(text)\" → \(topic?.keywordSummary() ?? "Topic \(a.topicID.value)")")
    }
}
```

---

## Error Handling

```swift
do {
    let result = try await model.fit(documents: documents, embeddings: embeddings)
} catch TopicModelError.invalidInput(let message) {
    // Document/embedding count mismatch, empty arrays, etc.
    print("Invalid input: \(message)")

} catch TopicModelError.embeddingDimensionMismatch(let expected, let got) {
    // Inconsistent embedding dimensions
    print("Dimension mismatch: expected \(expected), got \(got)")

} catch TopicModelError.invalidConfiguration(let message) {
    // Configuration validation failed
    print("Config error: \(message)")

} catch TopicModelError.pipelineError(let stage, let underlying) {
    // Error in a specific pipeline stage
    print("Error in \(stage): \(underlying)")

} catch TopicModelError.notFitted {
    // Trying to transform before fitting
    print("Model must be fitted first")

} catch TopicModelError.noEmbeddingProvider {
    // Trying to search/findTopics without embedding provider
    print("Search requires embedding provider")
}
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Mismatched Counts

```swift
// ❌ WRONG: Different counts
let docs = [doc1, doc2, doc3]
let embs = [emb1, emb2]  // Missing one!

// ✅ CORRECT: Same counts
assert(docs.count == embs.count)
```

### Pitfall 2: Inconsistent Embedding Provider

```swift
// ❌ WRONG: Different providers for fit and transform
let result = try await model.fit(documents: docs, embeddings: bertEmbeddings)
let newEmb = try await openAIClient.embed(newDoc.content)  // Different model!
let assignment = try await model.transform(documents: [newDoc], embeddings: [newEmb])
// Assignments will be meaningless

// ✅ CORRECT: Same provider
let result = try await model.fit(documents: docs, embeddings: bertEmbeddings)
let newEmb = try await bertModel.embed(newDoc.content)  // Same model
let assignment = try await model.transform(documents: [newDoc], embeddings: [newEmb])
```

### Pitfall 3: Forgetting to Check isFitted

```swift
// ❌ WRONG: Transform on unfitted model
let model = TopicModel()
let assignments = try await model.transform(...)  // Throws .notFitted

// ✅ CORRECT: Check first
if await model.isFitted {
    let assignments = try await model.transform(...)
} else {
    // Fit first or handle the case
}
```

### Pitfall 4: Very Small Corpora

```swift
// ⚠️ Fewer than 50 documents often yields poor topics
// HDBSCAN may not find meaningful clusters

// ✅ Use smallCorpus configuration
let config = TopicModelConfiguration.smallCorpus
// - minClusterSize = 3
// - allowSingleCluster = true
```

---

## Key Takeaways

1. **Prepare documents with proper IDs** to track assignments back to source data.

2. **Embeddings are your responsibility**—use consistent providers for fit and transform.

3. **Configuration presets** cover most use cases; customize only when needed.

4. **Progress reporting** helps users understand long-running operations.

5. **Result contains everything**—topics, assignments, statistics, metadata.

6. **Transform for classification**—assign new documents to existing topics.

7. **Search for discovery**—find similar documents by semantic similarity.

---

## 💡 Key Insight

```
The TopicModel is an orchestrator, not an algorithm.

It coordinates:
  - PCAReducer or UMAPReducer (dimensionality reduction)
  - HDBSCANEngine (clustering)
  - CTFIDFRepresenter (keyword extraction)
  - NPMICoherenceEvaluator (quality scoring)

Each component is independent and testable.
The TopicModel just wires them together.

This means:
  - You can use components individually
  - You can swap implementations
  - You can debug each stage separately
```

---

## Next Up

Now that you can run the complete pipeline, let's explore all the configuration options in detail.

**[→ 6.2 Configuration Guide](./02-Configuration-Guide.md)**

---

*Guide 6.1 of 6.4 • Chapter 6: Capstone*
