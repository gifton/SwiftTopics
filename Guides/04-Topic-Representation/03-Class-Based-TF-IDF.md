# 4.3 Class-Based TF-IDF

> **The key innovation: treating each cluster as a single document.**

---

## The Concept

**Class-based TF-IDF (c-TF-IDF)** adapts traditional TF-IDF for topic modeling. The core insight:

```
Traditional TF-IDF:              c-TF-IDF:
───────────────────              ─────────────────────────────────
Unit of analysis: Document       Unit of analysis: Cluster (class)

Each document has its own        All documents in a cluster merge
TF-IDF scores.                   into ONE virtual document.

Find words important to          Find words important to
THIS document.                   THIS cluster/topic.
```

The merge step is key:

```
Cluster 0 contains 127 documents:

  Doc 0: "Great 5K run this morning"
  Doc 4: "New running shoes arrived"
  Doc 7: "Beat my personal best today"
  ...

c-TF-IDF sees them as ONE document:

  Virtual Doc 0: "Great 5K run this morning
                  New running shoes arrived
                  Beat my personal best today
                  ... (127 docs worth of text)"

Now apply TF-IDF logic at the cluster level.
```

---

## Why It Matters

### Traditional TF-IDF Fails for Topics

```
If we applied traditional TF-IDF to each document:

Doc 0: "Great 5K run this morning"
  Top terms: "5K", "run", "morning"

Doc 4: "New running shoes arrived"
  Top terms: "shoes", "running", "arrived"

Doc 7: "Beat my personal best today"
  Top terms: "beat", "personal", "best"

Each document has DIFFERENT top terms.
How do we summarize the cluster?

Option 1: Average the TF-IDF scores?
  → Dilutes strong signals
  → "running" might not rank highly

Option 2: Most frequent document keywords?
  → Small sample size per document
  → Noisy and inconsistent
```

### c-TF-IDF Solves This

```
c-TF-IDF treats Cluster 0 as one mega-document:

Virtual Doc: (all 127 documents concatenated)
  "running" appears 156 times
  "5K" appears 23 times
  "morning" appears 89 times
  ...

Now compare to the ENTIRE CORPUS:
  "running" appears 162 times total (156 in Cluster 0!)
  → Highly distinctive to this cluster

Top c-TF-IDF terms: "running", "exercise", "5K", "morning", "workout"

These terms characterize the TOPIC, not individual documents.
```

---

## The Mathematics

### The c-TF-IDF Formula

```
c-TF-IDF(t, c) = tf(t, c) × log(1 + A / tf(t, corpus))

Where:
  t = term
  c = cluster (class)
  tf(t, c) = frequency of term t in cluster c (summed across all docs in c)
  tf(t, corpus) = frequency of term t across ALL clusters
  A = average number of tokens per cluster
```

### Breaking Down the Formula

#### Term Frequency in Cluster: tf(t, c)

```
Cluster 0 has 127 documents with these token counts:

  "running": 156 occurrences
  "morning": 89 occurrences
  "fitness": 67 occurrences
  "workout": 45 occurrences

tf("running", Cluster 0) = 156
tf("morning", Cluster 0) = 89
```

#### Corpus Term Frequency: tf(t, corpus)

```
Across ALL clusters:

  "running": 156 + 3 + 2 + 1 = 162 total
  "morning": 89 + 45 + 62 + 51 = 247 total

tf("running", corpus) = 162
tf("morning", corpus) = 247
```

#### Average Tokens Per Cluster: A

```
Cluster 0: 3,420 tokens
Cluster 1: 2,890 tokens
Cluster 2: 1,560 tokens
Cluster 3: 2,130 tokens

A = (3420 + 2890 + 1560 + 2130) / 4 = 2,500
```

#### Putting It Together

```
For "running" in Cluster 0:
  tf("running", Cluster 0) = 156
  tf("running", corpus) = 162
  A = 2500

  c-TF-IDF("running", C0) = 156 × log(1 + 2500/162)
                          = 156 × log(16.43)
                          = 156 × 2.80
                          = 436.8  ← High score!

For "morning" in Cluster 0:
  tf("morning", Cluster 0) = 89
  tf("morning", corpus) = 247
  A = 2500

  c-TF-IDF("morning", C0) = 89 × log(1 + 2500/247)
                          = 89 × log(11.12)
                          = 89 × 2.41
                          = 214.5  ← Lower score

"running" is MORE distinctive because 156/162 = 96% appear in this cluster.
"morning" is LESS distinctive because 89/247 = 36% appear in this cluster.
```

---

## Why log(1 + A/tf)?

### Comparison with Traditional IDF

```
Traditional IDF:
  IDF(t) = log(N / df(t))

  Where N = number of documents
  and df(t) = documents containing term t

c-TF-IDF "IDF-like" term:
  log(1 + A / tf(t, corpus))

  Where A = average tokens per cluster
  and tf(t, corpus) = total occurrences of term
```

### The BERTopic Innovation

The `log(1 + A / tf)` form (rather than `log(N / df)`) provides:

```
1. SMOOTHER WEIGHTING
   ────────────────────
   Traditional: log(1000/1) = 6.9  vs  log(1000/2) = 6.2
   c-TF-IDF:    log(1 + A/1) vs log(1 + A/2) ← gentler curve

2. TERM-LEVEL GRANULARITY
   ────────────────────────
   Traditional uses document frequency (binary: in doc or not)
   c-TF-IDF uses term frequency (counts every occurrence)

3. NATURAL HANDLING OF AGGREGATED DOCS
   ────────────────────────────────────
   When documents merge into clusters, we lose individual doc boundaries.
   Term frequency in corpus is well-defined; document frequency isn't.
```

---

## The Technique: Step by Step

### Step 1: Tokenize Documents

```swift
// 📍 See: Sources/SwiftTopics/Representation/Tokenizer.swift

let tokenizer = Tokenizer(configuration: .english)

// For each document
let allTokens = documents.map { doc in
    tokenizer.tokenize(doc.content)
}

// Result:
// [[["great", "run", "morning"], ["new", "running", "shoes"], ...]
```

### Step 2: Aggregate Tokens by Cluster

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

var clusterTokens = [[String]](repeating: [], count: clusterCount)

for (docIdx, tokens) in allTokens.enumerated() {
    let label = assignment.label(for: docIdx)
    guard label >= 0 else { continue }  // Skip outliers
    clusterTokens[label].append(contentsOf: tokens)
}

// Result:
// clusterTokens[0] = ["great", "run", "morning", "new", "running", "shoes", ...]
// clusterTokens[1] = ["meeting", "project", "deadline", ...]
```

### Step 3: Compute Term Frequencies

```swift
// 📍 See: Sources/SwiftTopics/Representation/cTFIDF.swift

// Per-cluster term frequencies
var clusterTermFreqs: [[Int: Int]] = []

// Corpus-wide term frequencies
var corpusTermFreq = [Int: Int]()

for (clusterIdx, tokens) in clusterTokens.enumerated() {
    var termFreq = [Int: Int]()
    for token in tokens {
        if let termIdx = vocabulary.index(for: token) {
            termFreq[termIdx, default: 0] += 1
            corpusTermFreq[termIdx, default: 0] += 1
        }
    }
    clusterTermFreqs.append(termFreq)
}
```

### Step 4: Compute Average Tokens Per Cluster

```swift
let totalTokens = clusterTokens.reduce(0) { $0 + $1.count }
let avgTokensPerCluster = Float(totalTokens) / Float(max(1, clusterCount))
```

### Step 5: Apply c-TF-IDF Formula

```swift
// 📍 See: Sources/SwiftTopics/Representation/cTFIDF.swift

for (termIdx, tf) in termFreq {
    let tfCorpus = corpusTermFreq[termIdx, default: 1]

    // c-TF-IDF formula: tf(t,c) × log(1 + A / tf(t,corpus))
    let idf = log(1.0 + avgTokensPerCluster / Float(tfCorpus))
    let score = Float(tf) * idf

    scores.append(TermScore(
        term: vocabulary.term(at: termIdx)!,
        termIndex: termIdx,
        score: score,
        frequency: tf
    ))
}

// Sort by score descending
scores.sort { $0.score > $1.score }
```

---

## Visualizing c-TF-IDF Scores

```
Cluster 0: Fitness topic
────────────────────────────────────────────────────────────────────

Term        tf(t,c)    tf(t,corpus)    A/corpus    log(1+A/c)   c-TF-IDF
──────────────────────────────────────────────────────────────────────────
running       156          162           15.4         2.80        436.8
exercise       89           95           26.3         3.30        293.7
fitness        67           71           35.2         3.58        239.9
workout        45           52           48.1         3.90        175.5
morning        89          247           10.1         2.41        214.5
day            34          890            2.8         1.34         45.6
the            12         2340            1.1         0.74          8.9

Notice:
  - "running" ranks highest: very frequent here, rare elsewhere
  - "morning" ranks lower despite high tf: common in other clusters too
  - "day" and "the" rank low: appear everywhere
```

---

## In SwiftTopics

### The CTFIDFComputer

```swift
// 📍 See: Sources/SwiftTopics/Representation/cTFIDF.swift

/// Computes class-based TF-IDF scores for topic keyword extraction.
public struct CTFIDFComputer: Sendable {

    /// Minimum score to consider a term as a keyword.
    public let minScore: Float

    /// Whether to normalize scores to [0, 1] range per cluster.
    public let normalizeScores: Bool

    /// Computes c-TF-IDF scores for all clusters.
    public func compute(
        clusterTokens: [[String]],
        vocabulary: Vocabulary
    ) -> [ClusterTermScores] {
        // Step 1: Compute term frequencies per cluster
        // Step 2: Compute average tokens per cluster
        // Step 3: Apply c-TF-IDF formula
        // Step 4: Sort and return top terms
    }
}
```

### The ClusterTermScores Result

```swift
// 📍 See: Sources/SwiftTopics/Representation/cTFIDF.swift

/// c-TF-IDF scores for all terms in a cluster.
public struct ClusterTermScores: Sendable {

    /// The cluster index.
    public let clusterIndex: Int

    /// Term scores sorted by score (descending).
    public let scores: [TermScore]

    /// Total token count in this cluster.
    public let tokenCount: Int

    /// Gets the top-K terms.
    public func topK(_ k: Int) -> [TermScore] {
        Array(scores.prefix(k))
    }

    /// Gets the top-K terms as strings.
    public func topKTerms(_ k: Int) -> [String] {
        topK(k).map(\.term)
    }
}
```

### Complete Usage Example

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

// Given documents and cluster assignments...
let representer = CTFIDFRepresenter(configuration: CTFIDFConfiguration(
    keywordsPerTopic: 10,
    minDocumentFrequency: 2,
    maxDocumentFrequencyRatio: 0.9
))

let topics = try await representer.represent(
    documents: documents,
    assignment: clusterAssignment
)

// Examine the results
for topic in topics {
    print("Topic \(topic.id.value):")
    for keyword in topic.keywords {
        print("  \(keyword.term): \(keyword.score)")
    }
}

// Output:
// Topic 0:
//   running: 1.00
//   exercise: 0.67
//   fitness: 0.55
//   workout: 0.40
//   morning: 0.49
```

---

## Score Normalization

### Why Normalize?

```
Raw scores vary widely across clusters:

Cluster 0 (3420 tokens):
  Top score: 436.8

Cluster 2 (1560 tokens):
  Top score: 198.2

Raw scores aren't comparable across clusters.
```

### Normalization in SwiftTopics

```swift
// 📍 See: Sources/SwiftTopics/Representation/cTFIDF.swift

if normalizeScores && !scores.isEmpty {
    let maxScore = scores[0].score
    if maxScore > 0 {
        scores = scores.map { score in
            TermScore(
                term: score.term,
                termIndex: score.termIndex,
                score: score.score / maxScore,  // Normalize to [0, 1]
                frequency: score.frequency
            )
        }
    }
}

// After normalization:
// Cluster 0: running=1.00, exercise=0.67, fitness=0.55, ...
// Cluster 2: anxiety=1.00, stress=0.81, feeling=0.73, ...
//
// Now all top keywords have score 1.0, allowing comparison.
```

---

## Handling Edge Cases

### Empty Clusters

```swift
// A cluster with no documents (shouldn't happen with proper HDBSCAN)
guard k > 0 && vocabulary.size > 0 else {
    return []
}

// Skip clusters with no tokens
if clusterTokens[label].isEmpty {
    continue
}
```

### Singleton Clusters

```swift
// Cluster with only one document
// c-TF-IDF still works, but keywords may be noisy

Cluster 5: [Doc 42]
Doc 42: "Unexpected raccoon sighting in the backyard"

Keywords: "raccoon", "sighting", "backyard", "unexpected"

These are valid keywords, but the cluster might be too small
to represent a meaningful topic. Consider merging or filtering.
```

### Rare Terms

```swift
// A term that appears only once in one cluster
tf("ultramarathon", C0) = 1
tf("ultramarathon", corpus) = 1

c-TF-IDF = 1 × log(1 + 2500/1) = 1 × 7.82 = 7.82

High IDF but low TF → moderate score.
The minDocumentFrequency filter can exclude such terms.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Forgetting to Aggregate by Cluster

```swift
// ❌ WRONG: Computing TF-IDF per document
for doc in documents {
    let scores = computeTfIdf(doc)  // Document-level, not cluster-level!
}

// ✅ CORRECT: Aggregate tokens by cluster first
var clusterTokens = [[String]](repeating: [], count: k)
for (docIdx, tokens) in allTokens.enumerated() {
    let label = assignment.label(for: docIdx)
    if label >= 0 {
        clusterTokens[label].append(contentsOf: tokens)
    }
}
let scores = compute(clusterTokens: clusterTokens, vocabulary: vocab)
```

### Pitfall 2: Including Outliers in Corpus Statistics

```swift
// ⚠️ DECISION: Should outliers contribute to corpus frequency?

// Option A: Exclude outliers
// Pros: Keywords reflect only clustered documents
// Cons: Might miss important terms

// Option B: Include outliers (SwiftTopics approach)
// Pros: Full corpus statistics
// Cons: Outlier terms dilute cluster distinctiveness

// SwiftTopics excludes outliers from cluster aggregation
// but could include them in corpus stats.
```

### Pitfall 3: Not Filtering Common Terms

```swift
// ❌ Without maxDocumentFrequencyRatio
// Terms appearing in 95%+ of documents still get scores

"day" appears in 90% of documents → still computed
"thing" appears in 85% of documents → still computed

// ✅ With proper filtering
let config = VocabularyConfiguration(
    maxDocumentFrequencyRatio: 0.8  // Exclude terms in >80% of docs
)
```

### Pitfall 4: Small Clusters with Generic Keywords

```swift
// Cluster 3 has only 8 documents about miscellaneous topics

Keywords: ["day", "thing", "time", "people", "way"]

These are too generic! The cluster might be:
  - Too small for c-TF-IDF to find distinctive terms
  - A "catch-all" cluster of unrelated documents
  - A sign that clustering parameters need adjustment
```

---

## Key Takeaways

1. **Clusters become virtual documents**: Merge all documents in a cluster before scoring.

2. **The formula is tf × log(1 + A/corpus_freq)**: Frequency HERE times rarity ELSEWHERE.

3. **A is the average tokens per cluster**: Normalizes for cluster size variation.

4. **High c-TF-IDF = distinctive keyword**: Frequent in this topic, rare in others.

5. **Normalization enables comparison**: Divide by max score per cluster.

6. **Edge cases require handling**: Empty clusters, singletons, rare terms.

---

## 💡 Key Insight

c-TF-IDF asks: **"If I merged all documents in this cluster into one giant document, what words would make it distinctive?"**

This is exactly what topic labeling requires. We don't want words that are common to ALL topics (like "day" or "thing"). We want words that are CONCENTRATED in this particular cluster.

The math formalizes the intuition: multiply how often a word appears HERE by how rare it is EVERYWHERE ELSE.

```
                     ┌──────────────────────────────┐
                     │   c-TF-IDF Intuition         │
                     ├──────────────────────────────┤
                     │                              │
                     │   frequent    ×    rare      │
                     │      HERE          ELSEWHERE │
                     │       ↓              ↓       │
                     │    tf(t,c)    ×  log(A/tf)   │
                     │                              │
                     │        =  distinctive term   │
                     │                              │
                     └──────────────────────────────┘
```

---

## Next Up

We have distinctive keywords, but they might be redundant. Next, let's learn how **MMR diversification** ensures our keywords cover different aspects of the topic.

**[→ 4.4 Keyword Diversification](./04-Keyword-Diversification.md)**

---

*Guide 4.3 of 4.4 • Chapter 4: Topic Representation*
