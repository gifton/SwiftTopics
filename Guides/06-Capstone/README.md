# Chapter 6: Capstone

> **Bringing it all together—complete pipeline walkthrough and production guidance.**

---

## The Challenge

You now understand each component of the topic modeling pipeline:

```
Documents → Embeddings → UMAP/PCA → HDBSCAN → c-TF-IDF → Topics
```

But how do you use them together in real code?

```swift
// What you want to do:
let topics = discoverTopics(from: documents)

// What you need to know:
// - How to configure each stage
// - How to handle errors
// - How to tune for your use case
// - How to add new documents over time
```

This chapter shows you how to go from theory to production.

---

## What You'll Learn

### The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TopicModel API                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   TopicModelConfiguration                        │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │   │
│  │  │ Reduction │ │ Clustering│ │Representa-│ │  Coherence    │   │   │
│  │  │ Config    │ │  Config   │ │tion Config│ │  Config       │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      TopicModel                                  │   │
│  │                                                                  │   │
│  │  fit(documents:embeddings:) → TopicModelResult                  │   │
│  │  transform(documents:embeddings:) → [TopicAssignment]           │   │
│  │  search(query:documents:embeddings:topK:) → [(Doc, Float)]      │   │
│  │  merge(topics:documents:embeddings:) → Topic                    │   │
│  │  reduce(to:documents:embeddings:) → [Topic]                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   TopicModelResult                               │   │
│  │                                                                  │   │
│  │  topics: [Topic]                 ← Discovered topics            │   │
│  │  documentTopics: [DocID: TopicAssignment]  ← Assignments        │   │
│  │  coherenceScore: Float?          ← Quality metric               │   │
│  │  statistics: TopicStatistics     ← Summary stats                │   │
│  │  metadata: TopicModelMetadata    ← Training info                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Guides

| Guide | You'll Learn | Use Case |
|-------|-------------|----------|
| [6.1 The Complete Pipeline](./01-The-Complete-Pipeline.md) | End-to-end usage | First implementation |
| [6.2 Configuration Guide](./02-Configuration-Guide.md) | All configuration options | Tuning for your data |
| [6.3 Troubleshooting](./03-Troubleshooting.md) | Common issues & solutions | Debugging problems |
| [6.4 Next Steps](./04-Next-Steps.md) | Incremental updates & advanced topics | Production systems |

---

## Quick Start

Here's the complete pattern you'll learn:

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModel.swift

import SwiftTopics

// 1. Prepare documents
let documents = journalEntries.map { entry in
    Document(content: entry.text)
}

// 2. Get embeddings (from your embedding provider)
let embeddings = try await embeddingProvider.embedBatch(
    documents.map(\.content)
)

// 3. Configure the model
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
    clustering: HDBSCANConfiguration(minClusterSize: 5),
    representation: CTFIDFConfiguration(keywordsPerTopic: 10),
    coherence: .default
)

// 4. Fit the model
let model = TopicModel(configuration: config)
let result = try await model.fit(documents: documents, embeddings: embeddings)

// 5. Use the results
print("Found \(result.topicCount) topics")

for topic in result.topics where !topic.isOutlierTopic {
    print("Topic \(topic.id.value): \(topic.keywordSummary())")
    print("  Size: \(topic.size) documents")
    if let coherence = topic.coherenceScore {
        print("  Coherence: \(coherence)")
    }
}

// 6. Find topic for a new document
let newAssignment = try await model.transform(
    documents: [Document(content: "New journal entry...")],
    embeddings: [newEmbedding]
)
```

---

## The SwiftTopics Implementation

### Core Components

```
Sources/SwiftTopics/
├── Model/
│   ├── TopicModel.swift           ← Main orchestrator
│   ├── TopicModelConfiguration.swift  ← All config options
│   └── TopicModelProgress.swift   ← Progress reporting
│
├── Core/
│   ├── TopicModelResult.swift     ← Complete output
│   ├── Document.swift             ← Input type
│   ├── Topic.swift                ← Topic representation
│   ├── Embedding.swift            ← Vector type
│   └── ClusterAssignment.swift    ← Clustering output
│
├── Reduction/
│   ├── PCA.swift                  ← PCA implementation
│   └── UMAP/                      ← UMAP implementation
│
├── Clustering/HDBSCAN/            ← HDBSCAN implementation
│
├── Representation/
│   ├── CTFIDFRepresenter.swift    ← Keyword extraction
│   ├── cTFIDF.swift               ← Score computation
│   └── Tokenizer.swift            ← Text tokenization
│
├── Evaluation/                    ← Quality metrics
│
└── Incremental/                   ← Incremental updates
    ├── IncrementalTopicUpdater.swift
    └── Storage/
```

---

## The Mental Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Topic Modeling Mental Model                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BATCH PROCESSING (one-time analysis)                                   │
│  ────────────────────────────────────                                   │
│                                                                         │
│     Documents ──► fit() ──► Result                                      │
│                                                                         │
│     Use case: "What themes exist in my 5000 journal entries?"           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TRANSFORM (classify new documents)                                     │
│  ─────────────────────────────────────                                  │
│                                                                         │
│     New Doc ──► transform() ──► Topic Assignment                        │
│                                                                         │
│     Use case: "Which topic does this new entry belong to?"              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  INCREMENTAL (growing corpus)                                           │
│  ─────────────────────────────────                                      │
│                                                                         │
│     Stream ──► IncrementalTopicUpdater ──► Live Topics                  │
│                                                                         │
│     Use case: "Process entries as they're written, update topics"       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before starting this chapter, you should understand:

- ✅ How embeddings represent semantic meaning (Chapter 1)
- ✅ Why dimensionality reduction is needed (Chapter 2)
- ✅ How HDBSCAN discovers clusters (Chapter 3)
- ✅ How c-TF-IDF extracts keywords (Chapter 4)
- ✅ How NPMI measures topic quality (Chapter 5)

If any of these are unclear, review the relevant chapter first.

---

## Configuration Presets Reference

SwiftTopics provides presets for common scenarios:

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

// Balanced (default)
TopicModelConfiguration.default
// - Reduction: PCA to 15D
// - Clustering: minClusterSize = 5
// - Representation: 10 keywords
// - Coherence: enabled

// Speed-optimized
TopicModelConfiguration.fast
// - Reduction: PCA to 10D
// - Clustering: minClusterSize = 3
// - Representation: 5 keywords
// - Coherence: disabled (skip for speed)

// Quality-optimized
TopicModelConfiguration.quality
// - Reduction: PCA to 25D
// - Clustering: minClusterSize = 10, EOM selection
// - Representation: 15 keywords with diversification
// - Coherence: enabled with larger window

// Small corpus (< 100 docs)
TopicModelConfiguration.smallCorpus
// - Smaller cluster sizes
// - Allow single cluster

// Large corpus (> 10,000 docs)
TopicModelConfiguration.largeCorpus
// - Larger cluster sizes
// - Higher min document frequency
```

---

## Result Structure Reference

```swift
// 📍 See: Sources/SwiftTopics/Core/TopicModelResult.swift

struct TopicModelResult {

    // Discovered topics
    let topics: [Topic]

    // Document assignments
    let documentTopics: [DocumentID: TopicAssignment]

    // Quality score
    let coherenceScore: Float?

    // Summary statistics
    let statistics: TopicStatistics

    // Training metadata
    let metadata: TopicModelMetadata

    // Accessors
    var topicCount: Int                    // Non-outlier topic count
    var documentCount: Int                 // Total documents
    var outlierDocuments: [DocumentID]     // Unclassified docs

    func topicAssignment(for: DocumentID) -> TopicAssignment?
    func topic(for: DocumentID) -> Topic?
    func topic(id: TopicID) -> Topic?
    func documents(for: TopicID) -> [DocumentID]
}
```

---

## Key Insights

### Insight 1: The Actor Model

```swift
// TopicModel is an actor for thread-safe fitting
public actor TopicModel {
    // All methods are async
    public func fit(...) async throws -> TopicModelResult
    public func transform(...) async throws -> [TopicAssignment]
}

// Safe to call from any concurrency context
Task {
    let result = try await model.fit(documents: docs, embeddings: embs)
}
```

### Insight 2: Embeddings Are Your Responsibility

```swift
// SwiftTopics is embedding-agnostic
// You provide embeddings from any source:

// Option 1: External API
let embeddings = try await openAIClient.embed(texts)

// Option 2: Core ML model
let embeddings = try await bertModel.embed(texts)

// Option 3: SwiftTopicsApple (Apple's NLContextualEmbedding)
let provider = NLEmbeddingProvider()
let embeddings = try await provider.embedBatch(texts)
```

### Insight 3: Quality Over Quantity

```swift
// More topics isn't better—coherent topics are better

// Bad: 50 incoherent topics
// - Harder to understand
// - Keywords overlap
// - Outlier rate high

// Good: 8 coherent topics
// - Clear themes
// - Distinct vocabularies
// - Low outlier rate

// Use quality metrics to guide tuning
if result.coherenceScore ?? 0 < 0.1 {
    print("Consider increasing minClusterSize")
}
```

---

## Let's Begin

Ready to put it all together?

**[→ 6.1 The Complete Pipeline](./01-The-Complete-Pipeline.md)**

---

*Chapter 6 of 6 • SwiftTopics Learning Guide*
