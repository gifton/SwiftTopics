# 4.1 From Clusters to Topics

> **Clusters are groups of points. Topics are ideas with names.**

---

## The Concept

After HDBSCAN, you have **cluster assignments**—each document belongs to a cluster (or is noise). But clusters are just numerical labels:

```
HDBSCAN output:

Document 0  →  Cluster 0
Document 1  →  Cluster 1
Document 2  →  Cluster 2
Document 3  →  Cluster 2
Document 4  →  Cluster 0
Document 5  →  Cluster -1 (noise)
Document 6  →  Cluster 2
Document 7  →  Cluster 0

What does "Cluster 0" mean? 🤷
```

A **topic** is more than a cluster label. It includes:

1. **Keywords** — Terms that characterize the cluster
2. **Size** — How many documents belong
3. **Coherence** — How well the keywords fit together
4. **Representative documents** — Exemplar entries

```
From this:                        To this:

Cluster 0: [0, 4, 7]             Topic 0:
                                   Keywords: fitness, running, exercise
                                   Size: 127 documents
                                   Coherence: 0.42
                                   Representative: "Great 5K run today..."
```

---

## Why It Matters

### The User Perspective

Users don't care about embeddings or clusters. They want to understand:

```
"What themes does my journal contain?"

Bad answer:                      Good answer:
─────────────────────            ─────────────────────────────────────
You have 5 clusters              Your journal has 5 main themes:
with IDs 0-4.                    • Fitness & Exercise (127 entries)
Cluster 0 contains               • Work Projects (89 entries)
127 documents.                   • Anxiety & Mental Health (54 entries)
                                • Family & Relationships (67 entries)
                                • Travel & Adventure (43 entries)
```

### The Technical Perspective

Keywords enable:

1. **Topic browsing** — Users explore by keyword
2. **Search** — Find documents related to a topic
3. **Visualization** — Label chart axes and legends
4. **Quality assessment** — Coherent keywords = good clustering

---

## The Problem: What Makes a Good Keyword?

### Attempt 1: Most Frequent Words

```
Cluster 0 contains documents about running.

Frequency approach:
  "the" - 847 occurrences
  "and" - 623 occurrences
  "to"  - 589 occurrences
  "I"   - 502 occurrences
  "a"   - 478 occurrences
  ...
  "running" - 42 occurrences ← What we actually want

Problem: Common words dominate.
```

### Attempt 2: Filter Stop Words

```
After removing stop words:

  "day"     - 156 occurrences
  "today"   - 142 occurrences
  "morning" - 98 occurrences
  "went"    - 87 occurrences
  "running" - 42 occurrences ← Still not at the top

Problem: Generic words still dominate.
```

### Attempt 3: Unique Words

```
Only words that appear ONLY in this cluster:

  "5k"        - 3 occurrences
  "marathon"  - 2 occurrences
  "treadmill" - 1 occurrence

Problem: Rare words might be too specific.
         Missing "running" which appears in other clusters too.
```

### The Solution: Distinctive Words

```
What we need: Words that appear MORE OFTEN in this cluster
              RELATIVE to other clusters.

"running" appears:
  - 42 times in Cluster 0 (127 docs)  → 0.33 per doc
  - 3 times in Cluster 1 (89 docs)    → 0.03 per doc
  - 1 time in Cluster 2 (54 docs)     → 0.02 per doc

"running" is DISTINCTIVE to Cluster 0!

This is the TF-IDF intuition applied to clusters.
```

---

## The Representation Pipeline

```
┌───────────────────────────────────────────────────────────────────────┐
│                     Topic Representation Steps                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 1: TOKENIZE                                                     │
│  ─────────────────                                                    │
│  "Had a great run this morning" → ["great", "run", "morning"]         │
│                                                                       │
│  - Lowercase                                                          │
│  - Remove punctuation                                                 │
│  - Filter stop words ("had", "a", "this")                             │
│  - Optionally create bigrams                                          │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 2: AGGREGATE BY CLUSTER                                         │
│  ────────────────────────────                                         │
│  Cluster 0 docs: ["great run morning", "5k today", "new shoes"]       │
│  Cluster 0 tokens: ["great", "run", "morning", "5k", "today", ...]    │
│                                                                       │
│  Treat the entire cluster as ONE mega-document.                       │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 3: BUILD VOCABULARY                                             │
│  ────────────────────────                                             │
│  All unique terms → indices                                           │
│  {"running": 0, "exercise": 1, "meeting": 2, ...}                     │
│                                                                       │
│  Filter:                                                              │
│  - Too rare (< minDocumentFrequency)                                  │
│  - Too common (> maxDocumentFrequencyRatio)                           │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 4: COMPUTE c-TF-IDF                                             │
│  ────────────────────────                                             │
│  For each term in each cluster:                                       │
│    score = tf(term, cluster) × log(1 + A / tf(term, corpus))          │
│                                                                       │
│  Sort by score descending.                                            │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 5: EXTRACT TOP-K                                                │
│  ─────────────────────                                                │
│  Cluster 0: [running, exercise, fitness, morning, workout]            │
│  Cluster 1: [meeting, project, deadline, client, review]              │
│  Cluster 2: [anxiety, stress, feeling, sleep, worried]                │
│                                                                       │
│  Optional: Apply MMR diversification to reduce redundancy.            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## In SwiftTopics

### The Topic Struct

```swift
// 📍 See: Sources/SwiftTopics/Core/Topic.swift

/// A discovered topic in the corpus.
public struct Topic: Sendable, Codable, Hashable, Identifiable {

    /// Unique identifier for this topic.
    public let id: TopicID

    /// Keywords that characterize this topic, ranked by relevance.
    public let keywords: [TopicKeyword]

    /// Number of documents assigned to this topic.
    public let size: Int

    /// NPMI coherence score for this topic.
    public let coherenceScore: Float?

    /// IDs of documents most representative of this topic.
    public let representativeDocuments: [DocumentID]

    /// The centroid embedding of this topic.
    public let centroid: Embedding?

    /// Returns the top N keywords as a comma-separated string.
    public func keywordSummary(count: Int = 5) -> String {
        keywords.prefix(count).map(\.term).joined(separator: ", ")
    }
}
```

### The TopicKeyword Struct

```swift
// 📍 See: Sources/SwiftTopics/Core/Topic.swift

/// A keyword that characterizes a topic.
public struct TopicKeyword: Sendable, Codable, Hashable {

    /// The keyword term.
    public let term: String

    /// The c-TF-IDF score indicating relevance to the topic.
    public let score: Float

    /// The raw term frequency within the topic's documents.
    public let frequency: Int?
}
```

### The TopicRepresenter Protocol

```swift
// 📍 See: Sources/SwiftTopics/Protocols/TopicRepresenter.swift

/// A component that extracts interpretable keywords for each topic.
public protocol TopicRepresenter: Sendable {

    /// The configuration for this representer.
    associatedtype Configuration: RepresentationConfiguration
    var configuration: Configuration { get }

    /// Extracts topic representations from clustered documents.
    func represent(
        documents: [Document],
        assignment: ClusterAssignment
    ) async throws -> [Topic]

    /// Extracts topics with embeddings for centroid computation.
    func represent(
        documents: [Document],
        embeddings: [Embedding],
        assignment: ClusterAssignment
    ) async throws -> [Topic]
}
```

### Basic Usage

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

// After clustering...
let assignment: ClusterAssignment = try await hdbscan.fit(embeddings)

// Extract topics with keywords
let representer = CTFIDFRepresenter(configuration: .default)
let topics = try await representer.represent(
    documents: documents,
    assignment: assignment
)

// Use the topics
for topic in topics {
    print("Topic \(topic.id.value): \(topic.keywordSummary())")
    print("  Size: \(topic.size) documents")
    if let coherence = topic.coherenceScore {
        print("  Coherence: \(coherence)")
    }
}
```

---

## Topic Quality Indicators

### Good Topics

```
Topic: [running, exercise, fitness, workout, training]

✓ Coherent: Keywords relate to each other
✓ Specific: Not too generic
✓ Distinctive: Wouldn't describe other topics
✓ Comprehensive: Covers the topic's breadth
```

### Problematic Topics

```
Topic: [day, today, time, thing, people]

✗ Generic: Could describe any topic
✗ Not distinctive: Common across all clusters
✗ Suggests poor clustering or poor filtering
```

```
Topic: [ultramarathon]

✗ Too specific: Only one keyword
✗ Suggests cluster is too small
✗ Might need to merge with similar topics
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Ignoring Outliers

```swift
// ❌ WRONG: Processing outlier "topic"
for topic in topics {
    generateSummary(topic)  // Outlier has no meaningful keywords!
}

// ✅ CORRECT: Skip outliers
for topic in topics where !topic.isOutlierTopic {
    generateSummary(topic)
}
```

### Pitfall 2: Assuming Fixed Keyword Count

```swift
// ⚠️ PROBLEMATIC: Some clusters may have few terms
let config = CTFIDFConfiguration(keywordsPerTopic: 20)

// Very small clusters might not have 20 distinct terms
// Always check: topic.keywords.count might be < 20
```

### Pitfall 3: Not Filtering Stop Words

```swift
// ❌ WRONG: Using default tokenizer without stop words
let tokenizer = Tokenizer(configuration: .default)

// ✅ CORRECT: Use English stop words
let tokenizer = Tokenizer(configuration: .english)

// Or extended set:
let tokenizer = Tokenizer(configuration: .englishExtended)
```

### Pitfall 4: Expecting Perfect Keywords

```swift
// Reality check: Keywords aren't always perfect

// Topic might be about "software development" but keywords show:
// [code, bug, feature, pr, review]

// This is fine! Keywords reflect word usage, not abstract concepts.
// Users will understand from context.
```

---

## Key Takeaways

1. **Clusters are geometric, topics are semantic**: Clustering groups vectors; representation extracts meaning.

2. **Distinctive > frequent**: Good keywords appear often HERE but rarely ELSEWHERE.

3. **Stop word filtering is essential**: Without it, common words dominate.

4. **Keywords have scores**: Higher scores mean more distinctive terms.

5. **Outliers have no keywords**: The noise "topic" shouldn't be labeled.

6. **Topic = keywords + metadata**: Size, coherence, and representative documents provide context.

---

## 💡 Key Insight

The representation step answers: **"If I could only describe this cluster with 10 words, which words would be most informative?"**

This is exactly what a librarian does when labeling a section—they don't say "the books here contain the word 'the' frequently." They say "this is the biography section."

c-TF-IDF automates this intuition by finding words that are:
- Common enough within the topic to be representative
- Rare enough across topics to be distinctive

---

## Next Up

Now that we understand the goal, let's revisit the foundation: **TF-IDF**—the classic algorithm that c-TF-IDF extends.

**[→ 4.2 TF-IDF Refresher](./02-TF-IDF-Refresher.md)**

---

*Guide 4.1 of 4.4 • Chapter 4: Topic Representation*
