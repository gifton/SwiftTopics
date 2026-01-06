# 6.2 Configuration Guide

> **Every knob explained—tune SwiftTopics for your specific use case.**

---

## The Concept

SwiftTopics has many configuration options. Understanding what each one does helps you:

- Get better topics from your data
- Balance quality vs. speed
- Debug unexpected results
- Optimize for your corpus size

```
Configuration Hierarchy:

TopicModelConfiguration
├── ReductionConfiguration     ← How to compress embeddings
├── HDBSCANConfiguration       ← How to cluster
├── CTFIDFConfiguration        ← How to extract keywords
├── CoherenceConfiguration     ← How to measure quality
└── seed                       ← For reproducibility
```

---

## Configuration Presets

Start with a preset, then customize if needed.

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

// Default: Balanced quality and speed
TopicModelConfiguration.default

// Fast: Speed over quality
TopicModelConfiguration.fast

// Quality: Quality over speed
TopicModelConfiguration.quality

// Small corpus (< 100 documents)
TopicModelConfiguration.smallCorpus

// Large corpus (> 10,000 documents)
TopicModelConfiguration.largeCorpus
```

### Preset Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Configuration Presets                              │
├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│ Setting      │ default  │ fast     │ quality  │ small    │ large        │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Reduction    │ PCA 15D  │ PCA 10D  │ PCA 25D  │ PCA 10D  │ PCA 30D      │
│ minCluster   │ 5        │ 3        │ 10       │ 3        │ 20           │
│ minSamples   │ nil (=5) │ 2        │ 5        │ 2        │ 10           │
│ Selection    │ EOM      │ EOM      │ EOM      │ EOM      │ EOM          │
│ Keywords     │ 10       │ 5        │ 15       │ 8        │ 15           │
│ Diversify    │ false    │ false    │ true     │ false    │ false        │
│ Coherence    │ enabled  │ disabled │ enabled  │ enabled  │ enabled      │
│ Window size  │ 10       │ -        │ 20       │ 10       │ 15           │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

---

## Reduction Configuration

Controls how high-dimensional embeddings are compressed.

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift
//         Sources/SwiftTopics/Reduction/PCA.swift
//         Sources/SwiftTopics/Reduction/UMAP/UMAP.swift

public struct ReductionConfiguration: Sendable, Codable {
    /// Target number of dimensions after reduction.
    public let outputDimension: Int

    /// Reduction method: .pca, .umap, or .none
    public let method: ReductionMethod

    /// Random seed for reproducibility.
    public let seed: UInt64?

    /// PCA-specific configuration.
    public let pcaConfig: PCAConfiguration?

    /// UMAP-specific configuration.
    public let umapConfig: UMAPConfiguration?
}
```

### Output Dimension

```
The "Goldilocks" problem:

Too few dimensions (< 5):
  - Loses too much information
  - Different concepts collapse together
  - Poor clustering quality

Too many dimensions (> 50):
  - Curse of dimensionality
  - All points become equidistant
  - HDBSCAN fails to find clusters

Just right (10-30):
  - Preserves key relationships
  - Enables meaningful distances
  - HDBSCAN finds natural clusters

Default: 15 dimensions
```

```swift
// Tuning output dimension
let config = ReductionConfiguration(
    outputDimension: 15,  // Default: balanced
    // outputDimension: 5,   // Very small corpus, few topics expected
    // outputDimension: 30,  // Large corpus, many topics expected
    method: .pca
)
```

### Reduction Method

```swift
// PCA: Fast, linear reduction
// Good for: Most use cases, fast iteration
ReductionConfiguration(outputDimension: 15, method: .pca)

// UMAP: Slower, preserves local structure better
// Good for: When clusters are oddly shaped, semantic nuance matters
ReductionConfiguration(outputDimension: 15, method: .umap)

// None: Skip reduction (use if embeddings already low-dimensional)
ReductionConfiguration(outputDimension: 15, method: .none)
```

### PCA Configuration

```swift
public struct PCAConfiguration: Sendable, Codable {
    /// Whether to whiten the output (decorrelate and normalize).
    public let whiten: Bool

    /// Target variance ratio to retain (e.g., 0.95 for 95%).
    /// If set, outputDimension is a maximum.
    public let varianceRatio: Float?
}

// Example: Retain 95% of variance
let pcaConfig = PCAConfiguration(whiten: false, varianceRatio: 0.95)
let reduction = ReductionConfiguration(
    outputDimension: 50,  // Maximum dimensions
    method: .pca,
    pcaConfig: pcaConfig
)
// Actual output dimension will be whatever retains 95% variance
```

### UMAP Configuration

```swift
public struct UMAPConfiguration: Sendable, Codable {
    /// Number of neighbors to consider (affects local/global balance).
    public let nNeighbors: Int          // Default: 15

    /// Minimum distance between points in output.
    public let minDist: Float           // Default: 0.1

    /// Distance metric.
    public let metric: DistanceMetric   // Default: .euclidean

    /// Number of optimization epochs.
    public let nEpochs: Int             // Default: 200
}

// Tighter clusters (more local structure)
let umapConfig = UMAPConfiguration(nNeighbors: 5, minDist: 0.0)

// Broader clusters (more global structure)
let umapConfig = UMAPConfiguration(nNeighbors: 50, minDist: 0.5)
```

### When to Use UMAP vs PCA

```
Use PCA when:
  - Speed matters
  - Embeddings are already well-clustered
  - Quick iteration during development
  - Corpus is small (< 1000 docs)

Use UMAP when:
  - Quality matters more than speed
  - Clusters have complex shapes
  - You need fine-grained topic distinctions
  - You're doing final production training
```

---

## Clustering Configuration (HDBSCAN)

Controls how documents are grouped into topics.

```swift
// 📍 See: Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift

public struct HDBSCANConfiguration: Sendable, Codable {
    /// Minimum cluster size.
    public let minClusterSize: Int

    /// Minimum samples for core points (nil = minClusterSize).
    public let minSamples: Int?

    /// Cluster selection method.
    public let clusterSelectionMethod: ClusterSelectionMethod

    /// Cluster selection epsilon.
    public let clusterSelectionEpsilon: Float

    /// Whether to allow a single cluster.
    public let allowSingleCluster: Bool

    /// Distance metric.
    public let metric: DistanceMetric

    /// Random seed.
    public let seed: UInt64?
}
```

### Minimum Cluster Size

The most important parameter. Sets the smallest allowed topic size.

```
minClusterSize = 5 (default):
  - Allows small, niche topics
  - May produce some noisy topics
  - Good for diverse corpora

minClusterSize = 10:
  - More robust topics
  - Misses niche topics
  - Good for focused corpora

minClusterSize = 20+:
  - Only broad themes
  - Few topics
  - Good for very large corpora
```

```swift
// Guidelines based on corpus size:
let minClusterSize: Int
switch documents.count {
case 0..<100:
    minClusterSize = 3
case 100..<500:
    minClusterSize = 5
case 500..<2000:
    minClusterSize = 8
case 2000..<10000:
    minClusterSize = 15
default:
    minClusterSize = 20
}
```

### Minimum Samples

Controls density requirements for "core" points.

```
minSamples = nil (default):
  - Uses minClusterSize as minSamples
  - Balanced behavior

minSamples < minClusterSize:
  - More points become core points
  - Clusters extend further
  - More documents assigned (fewer outliers)

minSamples > minClusterSize:
  - Stricter core point requirements
  - Denser cluster cores
  - More outliers
```

```swift
// Reduce outlier rate
let config = HDBSCANConfiguration(
    minClusterSize: 10,
    minSamples: 3  // Lower = more docs assigned to clusters
)

// Higher quality clusters (more outliers)
let config = HDBSCANConfiguration(
    minClusterSize: 10,
    minSamples: 10  // Same as minClusterSize
)
```

### Cluster Selection Method

```swift
public enum ClusterSelectionMethod: String, Sendable, Codable {
    /// Excess of Mass - tends to produce more, smaller clusters.
    case eom

    /// Leaf - produces clusters at the finest granularity.
    case leaf
}
```

```
EOM (Excess of Mass) - default:
  - Balances cluster stability and size
  - Produces hierarchical structure
  - Good for most use cases

Leaf:
  - Maximum number of clusters
  - Finest granularity
  - May produce very small clusters
  - Good when you want many specific topics
```

### Cluster Selection Epsilon

Flattens the hierarchy to a minimum distance threshold.

```swift
// Default: 0.0 (no flattening)
HDBSCANConfiguration(clusterSelectionEpsilon: 0.0)

// Merge clusters closer than 0.5
HDBSCANConfiguration(clusterSelectionEpsilon: 0.5)
// Results in fewer, larger clusters
```

### Allow Single Cluster

```swift
// Default: false - requires at least 2 clusters
HDBSCANConfiguration(allowSingleCluster: false)

// Allow all documents to be one cluster
HDBSCANConfiguration(allowSingleCluster: true)
// Useful for: Checking if corpus is homogeneous
// Useful for: Very small corpora
```

---

## Representation Configuration (c-TF-IDF)

Controls how keywords are extracted for each topic.

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

public struct CTFIDFConfiguration: Sendable, Codable {
    /// Number of keywords per topic.
    public let keywordsPerTopic: Int

    /// Minimum document frequency for a term to be considered.
    public let minDocumentFrequency: Int

    /// Maximum document frequency ratio (excludes very common terms).
    public let maxDocumentFrequencyRatio: Float

    /// Whether to use n-grams in addition to unigrams.
    public let useNGrams: Bool

    /// Maximum n-gram length.
    public let maxNGramLength: Int

    /// Whether to diversify keywords using MMR.
    public let diversify: Bool

    /// Diversity weight (0 = relevance only, 1 = diversity only).
    public let diversityWeight: Float

    /// Tokenizer configuration.
    public let tokenizer: TokenizerConfiguration
}
```

### Keywords Per Topic

```swift
// Fewer keywords: Focused, but may miss nuances
CTFIDFConfiguration(keywordsPerTopic: 5)

// Default: Good balance
CTFIDFConfiguration(keywordsPerTopic: 10)

// More keywords: Complete, but some may be less relevant
CTFIDFConfiguration(keywordsPerTopic: 15)
```

### Document Frequency Filtering

```swift
// Remove very rare terms (typos, names)
CTFIDFConfiguration(minDocumentFrequency: 2)
// Term must appear in at least 2 documents

// Remove very common terms (generic words that slipped through)
CTFIDFConfiguration(maxDocumentFrequencyRatio: 0.9)
// Term must appear in less than 90% of documents
```

### N-Grams

```swift
// Include bigrams ("machine learning", "natural language")
CTFIDFConfiguration(
    useNGrams: true,
    maxNGramLength: 2
)

// Include trigrams too
CTFIDFConfiguration(
    useNGrams: true,
    maxNGramLength: 3
)
// Warning: Increases vocabulary size significantly
```

### Keyword Diversification (MMR)

```swift
// Without diversification:
// Keywords might be: [running, run, runs, runner, jogging]
// All variations of the same concept

// With diversification (MMR - Maximal Marginal Relevance):
// Keywords might be: [running, fitness, morning, training, goals]
// More diverse, covers more aspects of the topic

CTFIDFConfiguration(
    keywordsPerTopic: 10,
    diversify: true,
    diversityWeight: 0.3  // 0-1, higher = more diversity
)
```

### Tokenizer Configuration

```swift
public struct TokenizerConfiguration: Sendable, Codable {
    /// Language for tokenization.
    public let language: String           // Default: "en"

    /// Whether to lowercase text.
    public let lowercase: Bool            // Default: true

    /// Whether to remove stop words.
    public let removeStopWords: Bool      // Default: true

    /// Minimum token length.
    public let minTokenLength: Int        // Default: 2

    /// Maximum token length.
    public let maxTokenLength: Int        // Default: 50

    /// Custom stop words to add.
    public let customStopWords: Set<String>?

    /// Tokens to never remove.
    public let protectedTokens: Set<String>?
}

// Custom tokenizer
let tokenizer = TokenizerConfiguration(
    language: "en",
    lowercase: true,
    removeStopWords: true,
    minTokenLength: 3,
    customStopWords: ["lol", "gonna", "wanna"],  // App-specific
    protectedTokens: ["AI", "ML", "iOS"]         // Keep these
)
```

---

## Coherence Configuration

Controls quality evaluation (NPMI scoring).

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

public struct CoherenceConfiguration: Sendable, Codable {
    /// Sliding window size for co-occurrence counting.
    public let windowSize: Int              // Default: 10

    /// Whether to use document-level co-occurrence.
    public let useDocumentCooccurrence: Bool // Default: false

    /// Number of top keywords to evaluate.
    public let topKeywords: Int             // Default: 10

    /// Smoothing epsilon for probability calculations.
    public let epsilon: Float               // Default: 1e-12
}
```

### Window Size

```
Small window (5-10):
  - Captures local context
  - Words must appear close together
  - Stricter coherence requirements

Large window (20-50):
  - Captures thematic relationships
  - Words can be further apart
  - More lenient coherence

Document-level:
  - Entire document is one window
  - Any co-occurrence counts
  - Most lenient
```

```swift
// Strict coherence
CoherenceConfiguration(windowSize: 5)

// Default: balanced
CoherenceConfiguration(windowSize: 10)

// Thematic coherence
CoherenceConfiguration(windowSize: 50)

// Document-level
CoherenceConfiguration(useDocumentCooccurrence: true)
```

### Top Keywords for Evaluation

```swift
// Evaluate top 5 keywords only (faster, focuses on most important)
CoherenceConfiguration(topKeywords: 5)

// Evaluate all 10 keywords (default)
CoherenceConfiguration(topKeywords: 10)

// Note: topKeywords should not exceed keywordsPerTopic
// This is validated by TopicModelConfiguration.validate()
```

### Disabling Coherence

```swift
// Skip coherence evaluation for speed
let config = TopicModelConfiguration(
    reduction: .default,
    clustering: .default,
    representation: .default,
    coherence: nil  // ← Disabled
)

// Use TopicModelConfiguration.fast which has coherence disabled
```

---

## Random Seed

For reproducible results:

```swift
let config = TopicModelConfiguration(
    reduction: .default,
    clustering: .default,
    representation: .default,
    coherence: .default,
    seed: 42  // ← Same seed = same results
)

// Without seed (nil), results may vary slightly between runs
// due to random initialization in PCA/UMAP
```

---

## Using the Builder Pattern

For incremental configuration:

```swift
// 📍 See: Sources/SwiftTopics/Model/TopicModelConfiguration.swift

let config = TopicModelConfigurationBuilder()
    .reductionDimension(20)
    .reductionMethod(.pca)
    .minClusterSize(8)
    .minSamples(4)
    .clusterSelectionMethod(.eom)
    .keywordsPerTopic(12)
    .diversify(true)
    .enableCoherence(true)
    .coherenceTopKeywords(10)
    .seed(42)
    .build()
```

---

## Configuration by Use Case

### Journal/Diary Analysis

```swift
// Personal journals: diverse topics, varying entry lengths
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 15, method: .pca),
    clustering: HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3,
        clusterSelectionMethod: .eom
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 10,
        diversify: true,
        diversityWeight: 0.2
    ),
    coherence: .default
)
```

### Customer Reviews

```swift
// Reviews: often short, specific terminology
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 20, method: .pca),
    clustering: HDBSCANConfiguration(
        minClusterSize: 10,
        minSamples: 5,
        clusterSelectionMethod: .eom
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 8,
        useNGrams: true,  // Catch "battery life", "screen quality"
        maxNGramLength: 2,
        minDocumentFrequency: 3
    ),
    coherence: .default
)
```

### Technical Documentation

```swift
// Docs: specific terminology, longer texts
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 25, method: .umap),
    clustering: HDBSCANConfiguration(
        minClusterSize: 8,
        clusterSelectionMethod: .leaf  // Fine-grained topics
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 15,
        useNGrams: true,
        maxNGramLength: 3,  // "natural language processing"
        tokenizer: TokenizerConfiguration(
            protectedTokens: ["API", "SDK", "iOS", "macOS"]
        )
    ),
    coherence: CoherenceConfiguration(windowSize: 20)
)
```

### Social Media Posts

```swift
// Posts: very short, informal language, high noise
let config = TopicModelConfiguration(
    reduction: ReductionConfiguration(outputDimension: 10, method: .pca),
    clustering: HDBSCANConfiguration(
        minClusterSize: 15,     // Larger to handle noise
        minSamples: 5
    ),
    representation: CTFIDFConfiguration(
        keywordsPerTopic: 8,
        minDocumentFrequency: 5,  // Filter rare terms
        tokenizer: TokenizerConfiguration(
            minTokenLength: 3,
            customStopWords: ["rt", "lol", "omg", "gonna"]
        )
    ),
    coherence: nil  // Skip - short posts have poor coherence anyway
)
```

---

## ⚠️ Common Configuration Mistakes

### Mistake 1: minClusterSize Too Large for Small Corpus

```swift
// ❌ 100 documents with minClusterSize = 20
// Result: Maybe 2-3 topics, most docs are outliers

// ✅ Scale to corpus size
let minClusterSize = max(3, documents.count / 20)
```

### Mistake 2: Forgetting Seed for Reproducibility

```swift
// ❌ No seed - results vary between runs
let config = TopicModelConfiguration.default

// ✅ Set seed for reproducible experiments
let config = TopicModelConfiguration(
    ...
    seed: 42
)
```

### Mistake 3: Too Many Keywords with Small Windows

```swift
// ❌ 15 keywords but window size 5
// Few keyword pairs will co-occur in such small windows
// → Low coherence scores even for good topics

// ✅ Match window size to keyword count
let config = TopicModelConfiguration(
    representation: CTFIDFConfiguration(keywordsPerTopic: 15),
    coherence: CoherenceConfiguration(windowSize: 20, topKeywords: 10)
)
```

### Mistake 4: UMAP for Interactive Use

```swift
// ❌ UMAP during interactive sessions
// User waits 30+ seconds for results

// ✅ Use PCA for development, UMAP for final training
let devConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .pca)
)

let prodConfig = TopicModelConfiguration(
    reduction: ReductionConfiguration(method: .umap)
)
```

---

## Key Takeaways

1. **Start with presets**—`.default`, `.fast`, `.quality`, `.smallCorpus`, `.largeCorpus`.

2. **minClusterSize is the #1 tuning parameter**—scale it to your corpus size.

3. **PCA vs UMAP**: PCA for speed, UMAP for quality.

4. **Diversification** reduces keyword redundancy—enable for interpretable topics.

5. **Coherence evaluation** can be disabled for speed during development.

6. **Set a seed** when you need reproducible results.

7. **Validate configuration** before fitting to catch errors early.

---

## 💡 Key Insight

```
Configuration is about trade-offs:

  Speed       ←──────────────→ Quality
  PCA 10D                       UMAP 30D
  minClusterSize=3             minClusterSize=20
  5 keywords                    15 keywords
  No coherence                  Full coherence

  Generic     ←──────────────→ Specific
  Fewer topics                  More topics
  Broad themes                  Niche concepts
  More docs per topic           Fewer docs per topic

There's no "best" configuration—only the best
configuration for YOUR use case.

Iterate:
1. Start with a preset
2. Look at the results
3. Adjust one parameter
4. Compare
5. Repeat
```

---

## Next Up

What do you do when things go wrong? Let's look at common problems and solutions.

**[→ 6.3 Troubleshooting](./03-Troubleshooting.md)**

---

*Guide 6.2 of 6.4 • Chapter 6: Capstone*
