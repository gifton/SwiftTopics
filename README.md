# SwiftTopics

High-fidelity, on-device topic extraction for Apple platforms with GPU acceleration.

SwiftTopics is a pure-Swift topic modeling library inspired by [BERTopic](https://maartengr.github.io/BERTopic/), optimized for Apple's ecosystem with Metal 4 GPU acceleration via [VectorAccelerate](https://github.com/gifton/VectorAccelerate).

## Requirements

- iOS 26.0+ / macOS 26.0+ / visionOS 26.0+
- Swift 6.2+
- Xcode 26+

## Installation

Add SwiftTopics to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/gifton/SwiftTopics.git", from: "0.1.0-beta.1")
]
```

Then import the target you need:

```swift
// Core functionality (BYOE - Bring Your Own Embeddings)
import SwiftTopics

// Apple platform integrations (EmbedKit adapter)
import SwiftTopicsApple
```

## Quick Start

### Basic Usage

```swift
import SwiftTopics

// 1. Create documents
let documents = [
    Document(content: "Machine learning models require training data"),
    Document(content: "Neural networks use backpropagation for learning"),
    Document(content: "Swift is a programming language by Apple"),
    // ... more documents
]

// 2. Provide embeddings (from your embedding service)
let embeddings: [Embedding] = await yourEmbeddingService.embed(documents)

// 3. Discover topics
let model = TopicModel(configuration: .default)
let result = try await model.fit(documents: documents, embeddings: embeddings)

// 4. Inspect results
print("Found \(result.topicCount) topics")
for topic in result.topics where !topic.isOutlierTopic {
    print("Topic \(topic.id): \(topic.keywordSummary())")
}
```

### Using SwiftTopicsApple with EmbedKit

```swift
import SwiftTopics
import SwiftTopicsApple
import EmbedKit

// Use Apple's NLContextualEmbedding via EmbedKit
let nlModel = try AppleNLContextualModel(language: "en")
let provider = EmbedKitAdapter(model: nlModel)

let model = TopicModel(configuration: .default)
let result = try await model.fit(
    documents: documents,
    embeddingProvider: provider
)
```

## Pipeline Architecture

SwiftTopics implements a modular 4-stage pipeline:

```
Documents → Embeddings → Reduction → Clustering → Representation
              ↓            ↓            ↓             ↓
         [Float] vectors   PCA/UMAP    HDBSCAN      c-TF-IDF
```

### Stage 1: Embeddings

Embeddings are provided externally (BYOE pattern). SwiftTopics works with any embedding source:

- **Apple NL**: Via `SwiftTopicsApple.EmbedKitAdapter`
- **CoreML models**: Via EmbedKit
- **Remote APIs**: Pre-compute and provide as `[Embedding]`

### Stage 2: Dimensionality Reduction

Reduces high-dimensional embeddings (384-1536D) to low-dimensional space for clustering:

| Method | Use Case |
|--------|----------|
| **PCA** (default) | Fast, deterministic, good for most cases |
| **UMAP** | Better local structure preservation, slower |

### Stage 3: Clustering (HDBSCAN)

Density-based clustering with automatic topic count discovery:

- No predefined K required
- Outlier detection (sparse regions marked, not forced)
- Variable-density cluster support
- GPU-accelerated core distance computation

### Stage 4: Topic Representation (c-TF-IDF)

Extracts representative keywords using class-based TF-IDF:

```swift
for topic in result.topics {
    let keywords = topic.keywords.prefix(5)
    // Each keyword has: term, score, frequency
}
```

## Configuration

### Presets

```swift
// Balanced quality and speed
TopicModelConfiguration.default

// Prioritizes speed (fewer components, skip coherence)
TopicModelConfiguration.fast

// Prioritizes quality (more components, stricter clustering)
TopicModelConfiguration.quality

// Optimized for small corpora (< 100 docs)
TopicModelConfiguration.smallCorpus

// Optimized for large corpora (> 10,000 docs)
TopicModelConfiguration.largeCorpus
```

### Custom Configuration

```swift
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(
        outputDimension: 20,
        method: .pca
    ),
    clustering: HDBSCANConfiguration(
        minClusterSize: 8,
        minSamples: 4,
        clusterSelectionMethod: .eom
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 12,
        diversify: true
    ),
    coherence: .default,  // nil to skip
    seed: 42  // For reproducibility
)
```

### Builder Pattern

```swift
let config = TopicModelConfigurationBuilder()
    .reductionDimension(15)
    .minClusterSize(5)
    .keywordsPerTopic(10)
    .enableCoherence(true)
    .seed(42)
    .build()
```

## GPU Acceleration

SwiftTopics uses VectorAccelerate's Metal 4 kernels for GPU-accelerated computation, providing **25-125x speedup** for HDBSCAN and **10-50x speedup** for UMAP optimization.

### Enabling GPU Acceleration

```swift
import SwiftTopics

// Create GPU context
let gpuContext = await TopicsGPUContext.create(allowCPUFallback: true)

// Pass to TopicModel or individual engines
let model = TopicModel(configuration: .default, gpuContext: gpuContext)

// Or use with UMAP directly
let umap = UMAPReducer(
    configuration: .default,
    nComponents: 15,
    gpuContext: gpuContext
)
```

### GPU Configuration

```swift
let gpuConfig = TopicsGPUConfiguration(
    preferHighPerformance: true,
    maxBufferPoolMemory: 1024 * 1024 * 1024,  // 1GB
    enableProfiling: false,
    gpuMinPointsThreshold: 100  // Use GPU for datasets >= 100 points
)

let gpuContext = try await TopicsGPUContext(configuration: gpuConfig)
```

### Performance Characteristics

| Operation | Dataset Size | CPU Time | GPU Time | Speedup |
|-----------|--------------|----------|----------|---------|
| HDBSCAN MST | 500 points | ~2s | ~50ms | 40x |
| HDBSCAN MST | 1,000 points | ~8s | ~150ms | 53x |
| HDBSCAN MST | 5,000 points | ~200s | ~2s | 100x |
| UMAP Epoch | 1,000 points | ~80ms | ~13ms | 6x |
| UMAP Full (100 epochs) | 1,000 points | ~8s | ~1.3s | 6x |

### Timing Breakdown

Enable detailed timing logs for performance analysis:

```swift
let config = HDBSCANConfiguration(
    minClusterSize: 5,
    logTiming: true  // Logs per-phase timing breakdown
)

// Results include timing breakdown
let result = try await engine.fitWithDetails(embeddings)
if let timing = result.timingBreakdown {
    print(timing.summary)
    // HDBSCAN Timing (GPU, 1000 points):
    //   Core distances:      0.009s
    //   Mutual reachability: 0.006s
    //   MST construction:    0.016s
    //   Hierarchy building:  0.002s
    //   Cluster extraction:  0.001s
    //   Total:               0.034s
}
```

### GPU Threshold Tuning

The GPU is only used when the dataset exceeds `gpuMinPointsThreshold`:

| Threshold | Use Case |
|-----------|----------|
| 50 | Testing, small datasets where GPU may still help |
| 100 (default) | Balanced - GPU overhead justified at this scale |
| 200+ | Conservative - only use GPU for larger datasets |

### Graceful CPU Fallback

GPU operations automatically fall back to CPU if:
- GPU is unavailable (older hardware)
- Dataset is below the threshold
- GPU computation fails for any reason

A warning is logged on fallback:

```
[SwiftTopics.HDBSCAN] GPU HDBSCAN computation failed, falling back to CPU: ...
```

## Core Operations

### Transform (Assign New Documents)

```swift
// After fitting, assign new documents to existing topics
let newDocs = [Document(content: "New text about machine learning")]
let newEmbeddings = await yourService.embed(newDocs)

let assignments = try await model.transform(
    documents: newDocs,
    embeddings: newEmbeddings
)

for assignment in assignments {
    print("Topic: \(assignment.topicID), Confidence: \(assignment.probability)")
}
```

### Semantic Search

```swift
// Search documents by semantic similarity
let results = try await model.search(query: "neural networks", topK: 5)

for (document, score) in results {
    print("Score \(score): \(document.content.prefix(50))...")
}
```

### Topic Operations

```swift
// Merge similar topics
let mergedTopic = try await model.merge(topics: [0, 2])

// Reduce total topic count
let reducedTopics = try await model.reduce(to: 5)

// Get topic assignment for arbitrary text
let assignments = try await model.findTopics(for: "quantum computing")
```

## Incremental Updates

For applications with streaming data (journals, notes), `IncrementalTopicUpdater` provides lifecycle management:

```swift
import SwiftTopics

let storage = FileBasedTopicModelStorage(directory: modelDirectory)
let updater = try await IncrementalTopicUpdater(
    storage: storage,
    modelConfiguration: .default,
    updateConfiguration: .default
)

// On app launch: resume any interrupted training
try await updater.resumeIfNeeded()

// Process new documents as they arrive
let assignment = try await updater.processDocument(doc, embedding: embedding)
// Returns immediate assignment via centroid distance

// Before app termination
try await updater.prepareForTermination()
```

### Update Flow

1. **Cold Start**: Documents buffer until `coldStartThreshold` (default: 30), then initial model is trained
2. **Immediate Assignment**: New documents get instant topic assignment via centroid similarity
3. **Micro-Retrain**: After `microRetrainThreshold` (default: 30) documents buffer, background retrain incorporates them
4. **Full Refresh**: Triggered by growth ratio, time interval, or drift detection

### Configuration Presets

```swift
IncrementalUpdateConfiguration.default      // Balanced (30 entries/retrain)
IncrementalUpdateConfiguration.aggressive   // Frequent updates (20 entries)
IncrementalUpdateConfiguration.conservative // Less frequent (50 entries)
```

## Result Types

### TopicModelResult

```swift
let result: TopicModelResult = try await model.fit(...)

// Access topics
result.topics                    // [Topic] - all discovered topics
result.topicCount               // Int - count excluding outliers
result.topicsBySizeDescending   // [Topic] - sorted by document count

// Access assignments
result.topicAssignment(for: documentID)  // TopicAssignment?
result.documents(for: topicID)           // [DocumentID]
result.outlierDocuments                  // [DocumentID]

// Quality metrics
result.coherenceScore           // Float? - aggregate NPMI
result.statistics               // TopicStatistics
```

### Topic

```swift
let topic: Topic

topic.id                      // TopicID
topic.keywords                // [TopicKeyword] - ranked by c-TF-IDF score
topic.size                    // Int - document count
topic.coherenceScore          // Float? - NPMI score [-1, +1]
topic.isOutlierTopic          // Bool
topic.keywordSummary(count: 5) // String - comma-separated top keywords
```

### TopicAssignment

```swift
let assignment: TopicAssignment

assignment.topicID           // TopicID
assignment.probability       // Float [0, 1]
assignment.distanceToCentroid // Float?
assignment.isOutlier         // Bool
assignment.alternatives      // [AlternativeAssignment]?
```

## Best Practices

### Document Preparation

- **Minimum corpus size**: 30+ documents for meaningful topics
- **Content length**: Works best with 50-500 word documents
- **Language consistency**: Keep documents in a single language per model

### Configuration Tuning

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `minClusterSize` | Fewer, larger topics; more outliers |
| `minSamples` | Stricter core point definition |
| `reductionDimension` | More signal preserved; higher compute |
| `keywordsPerTopic` | More keyword coverage; potential noise |

### Performance Guidelines

- **< 1,000 docs**: `.default` or `.fast` configuration
- **1,000-10,000 docs**: `.default` with GPU acceleration
- **> 10,000 docs**: `.largeCorpus` configuration
- Use `EmbedKitAdapter.highThroughput(model:)` for bulk embedding

### Coherence Interpretation

| NPMI Score | Interpretation |
|------------|----------------|
| > 0.3 | Highly coherent topics |
| 0.1 - 0.3 | Good topics |
| 0.0 - 0.1 | Acceptable topics |
| < 0.0 | Topics may need refinement |

## Dependencies

- [VectorAccelerate](https://github.com/gifton/VectorAccelerate) (0.3.1+) - GPU-accelerated vector operations
- [EmbedKit](https://github.com/gifton/EmbedKit) (0.2.7+) - Embedding model framework (SwiftTopicsApple only)

## License

MIT License - see LICENSE file for details.
