# 1.1 Why Embeddings for Topics

> **Transforming text into a geometric space where clustering becomes possible.**

---

## The Concept

**Embeddings** are dense numerical vectors that represent text. A sentence like "Had a great run this morning" becomes an array of ~768 floating-point numbers:

```
"Had a great run this morning"
        │
        ▼
[0.023, -0.156, 0.089, 0.234, -0.067, ..., 0.041]
        │
        └── 768 numbers encoding the sentence's meaning
```

The key insight: **semantically similar texts produce geometrically close vectors**.

```
Text A: "Had a great run this morning"      → Vector A: [0.023, -0.156, ...]
Text B: "Went jogging at sunrise"           → Vector B: [0.019, -0.148, ...]
Text C: "The compiler threw an error"       → Vector C: [0.567, 0.234, ...]

Distance(A, B) ≈ 0.12   (similar meaning → close vectors)
Distance(A, C) ≈ 1.87   (different meaning → far vectors)
```

---

## Why It Matters

Topic modeling requires grouping "similar" documents. But what does "similar" mean for text?

### The Problem with String Comparison

```swift
let text1 = "Had a great run this morning"
let text2 = "Went jogging at sunrise"

// String equality? No.
text1 == text2  // false

// Levenshtein distance? Poor signal.
levenshteinDistance(text1, text2)  // 25 (very different strings!)

// Word overlap? Misses synonyms.
Set(text1.words).intersection(Set(text2.words))  // {} (no overlap!)
```

These texts are **semantically identical** (both describe morning exercise) but **textually different**. String-based similarity fails completely.

### The Embedding Solution

Embedding models learn to map text to vectors where semantic relationships become geometric ones:

| Semantic Relationship | Geometric Manifestation |
|----------------------|-------------------------|
| Similar meaning | Close vectors (small distance) |
| Different meaning | Far vectors (large distance) |
| Related concepts | Vectors in same region |
| Antonyms | Specific directional relationship |

This transformation enables standard clustering algorithms (designed for points in space) to discover semantic groups (topics).

---

## The Mathematics

### What Embeddings Encode

An embedding model `E` is a function:

```
E: Text → ℝᴰ

Where:
  - Text is any string (word, sentence, paragraph)
  - ℝᴰ is D-dimensional real vector space (e.g., D = 768)
```

The model learns this mapping from massive text corpora, encoding:

- **Semantics**: What the text means
- **Context**: How words relate to each other
- **World knowledge**: Learned relationships (Paris→France, CEO→company)

### The Similarity Property

A well-trained embedding model satisfies:

```
If text₁ ≈ text₂ semantically, then:
    distance(E(text₁), E(text₂)) is small

If text₁ ≠ text₂ semantically, then:
    distance(E(text₁), E(text₂)) is large
```

This is what makes topic modeling possible. The geometric structure of embedding space reflects the semantic structure of language.

---

## The Technique: From Text to Topics

Here's how embeddings enable the topic modeling pipeline:

### Step 1: Embed All Documents

```swift
// Each document becomes a point in high-dimensional space
let documents = [
    "Had a great run this morning",
    "New Swift concurrency features are amazing",
    "Went jogging at sunrise",
    "The async/await syntax is clean",
    "5K personal best today",
    "Metal shaders for GPU compute"
]

let embeddings = documents.map { embeddingModel.embed($0) }
// 6 documents → 6 points in 768-dimensional space
```

### Step 2: Visualize the Structure

If we could see 768-dimensional space (we can't), we'd observe:

```
768D Embedding Space:

    ┌─────────────────────────────────────────┐
    │                                         │
    │    ★ "great run"                        │
    │       "jogging at sunrise" ★            │
    │          ★ "5K personal best"           │
    │                                         │  ← Fitness cluster
    │─────────────────────────────────────────│
    │                                         │
    │              ★ "Swift concurrency"      │
    │                 "async/await" ★         │
    │                    ★ "Metal shaders"    │
    │                                         │  ← Programming cluster
    └─────────────────────────────────────────┘
```

### Step 3: Cluster the Points

Since similar documents are nearby, clustering algorithms can find groups:

```swift
// HDBSCAN finds dense regions (topics)
let clusters = hdbscan.fit(embeddings)

// Result:
// Cluster 0: ["great run", "jogging at sunrise", "5K personal best"]
// Cluster 1: ["Swift concurrency", "async/await", "Metal shaders"]
```

The clusters correspond to **topics**—groups of semantically related documents.

---

## In SwiftTopics

SwiftTopics defines the `Embedding` type to wrap vectors with dimension safety:

```swift
// 📍 See: Sources/SwiftTopics/Core/Embedding.swift:31-50

public struct Embedding: Sendable, Codable, Hashable {
    /// The vector components.
    public let vector: [Float]

    /// The dimensionality of this embedding.
    @inlinable
    public var dimension: Int {
        vector.count
    }

    /// Creates an embedding from a float array.
    public init(vector: [Float]) {
        precondition(!vector.isEmpty, "Embedding vector cannot be empty")
        self.vector = vector
    }
}
```

The `EmbeddingProvider` protocol abstracts embedding sources:

```swift
// 📍 See: Sources/SwiftTopics/Protocols/EmbeddingProvider.swift

public protocol EmbeddingProvider: Sendable {
    /// The dimension of embeddings produced by this provider.
    var dimension: Int { get }

    /// Embeds a single text string.
    func embed(_ text: String) async throws -> Embedding

    /// Embeds multiple texts in a batch.
    func embedBatch(_ texts: [String]) async throws -> [Embedding]
}
```

### Using SwiftTopicsApple with EmbedKit

```swift
import SwiftTopics
import SwiftTopicsApple
import EmbedKit

// Create an embedding provider using Apple's NL framework
let nlModel = try AppleNLContextualModel(language: "en")
let provider = EmbedKitAdapter(model: nlModel)

// Use with TopicModel
let model = TopicModel(configuration: .default)
let result = try await model.fit(
    documents: documents,
    embeddingProvider: provider
)
```

### Using Pre-computed Embeddings

```swift
// If you've already computed embeddings (e.g., stored in database)
let documents: [Document] = loadDocuments()
let embeddings: [Embedding] = loadEmbeddings()  // From your storage

let result = try await model.fit(
    documents: documents,
    embeddings: embeddings
)
```

---

## Common Embedding Models

Different embedding models have different characteristics:

| Model | Dimension | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| **all-MiniLM-L6-v2** | 384 | Good | Fast | General purpose |
| **Apple NLContextualEmbedding** | 512 | Good | Fast | On-device, English |
| **all-mpnet-base-v2** | 768 | Better | Medium | Higher quality |
| **OpenAI text-embedding-3-small** | 1536 | Excellent | API call | Production |
| **OpenAI text-embedding-3-large** | 3072 | Best | API call | Maximum quality |

### Dimension vs. Quality Tradeoff

```
Higher dimension:
  ✓ More capacity to encode nuance
  ✓ Better separation of similar concepts
  ✗ More memory usage
  ✗ Curse of dimensionality (addressed in Chapter 2)
  ✗ Slower processing

Lower dimension:
  ✓ Faster processing
  ✓ Less memory
  ✗ May conflate distinct concepts
  ✗ Less nuanced representation
```

SwiftTopics handles any dimension through its reduction stage (Chapter 2).

---

## ⚠️ Common Pitfalls

### Pitfall 1: Mixing Embedding Models

```swift
// ❌ WRONG: Different models produce incompatible embeddings
let embedding1 = modelA.embed("Hello")  // 384D from MiniLM
let embedding2 = modelB.embed("World")  // 768D from MPNet

// These cannot be compared or clustered together!
```

**Rule**: All documents in a corpus must use the same embedding model.

### Pitfall 2: Ignoring Document Length

Most embedding models have a maximum input length (e.g., 512 tokens). Long documents get truncated:

```swift
// ⚠️ 5000-word document truncated to first 512 tokens
let embedding = model.embed(veryLongDocument)
// Only embeds the beginning!
```

**Solutions**:
- Chunk long documents and average embeddings
- Use the most representative portion
- Use models with longer context windows

### Pitfall 3: Language Mismatch

Embedding models are language-specific or multilingual:

```swift
// ⚠️ English model on French text
let englishModel = AppleNLContextualModel(language: "en")
let embedding = englishModel.embed("Bonjour le monde")
// Poor quality embedding!
```

**Rule**: Match the embedding model's language to your documents.

---

## Key Takeaways

1. **Embeddings transform text into geometry**: Semantic similarity becomes geometric proximity, enabling clustering algorithms.

2. **The transformation preserves meaning**: Similar texts → close vectors; different texts → far vectors.

3. **SwiftTopics is embedding-agnostic**: Bring embeddings from any source (Apple NL, CoreML, remote APIs).

4. **Consistency matters**: All documents must use the same embedding model.

5. **Dimension varies by model**: 384D to 3072D common; SwiftTopics handles any dimension via reduction.

---

## 💡 Key Insight

Embeddings are the **foundation** of modern NLP. Once you have them, many tasks become geometric problems:

| NLP Task | Geometric Operation |
|----------|---------------------|
| Similarity search | Nearest neighbor |
| Topic modeling | Clustering |
| Classification | Region assignment |
| Anomaly detection | Distance from centroid |

Understanding embeddings unlocks all of these.

---

## Next Up

Now that we understand *what* embeddings are, let's explore their structure:

**[→ 1.2 Embedding Spaces](./02-Embedding-Spaces.md)**

---

*Guide 1.1 of 1.3 • Chapter 1: Embeddings Foundation*
