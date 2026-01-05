# 5.1 Why Quality Matters

> **Not all topics are created equal—learn to tell good from bad.**

---

## The Concept

You've extracted topics from your corpus. Each topic has keywords. But consider these two examples:

```
Topic A: [running, exercise, fitness, morning, workout]
Topic B: [day, thing, time, people, really]

Which topic is more useful?
```

**Topic A** clearly represents a concept: fitness/exercise routines. A user can understand what documents in this topic are about.

**Topic B** is noise. These are common words that appear everywhere. They don't characterize any specific concept.

**Quality evaluation** gives us objective metrics to distinguish good topics from bad ones.

---

## Why It Matters

### Without Quality Metrics

```
You tune your topic model blindly:

  "Maybe min_cluster_size = 10 is better than 20?"
  "Should I use bigrams?"
  "Is 10 topics or 15 topics better?"

Without metrics, you're guessing.
You can look at topics manually, but:
  - It's subjective
  - It doesn't scale to many topics
  - You might miss subtle issues
```

### With Quality Metrics

```
Now you can measure:

  Config A: coherence = 0.42, diversity = 0.85
  Config B: coherence = 0.38, diversity = 0.91
  Config C: coherence = 0.45, diversity = 0.78

Config A gives the best balance.

You can also:
  - Flag low-coherence topics for review
  - Detect redundant topics that should be merged
  - Track quality over time as your corpus grows
```

---

## What Makes a Good Topic?

### Characteristic 1: Coherent Keywords

```
Good: [running, exercise, fitness, workout, gym]
       ↓
       All words relate to the same concept.
       If you see one, you expect to see the others.

Bad:  [running, meeting, anxiety, coffee, travel]
       ↓
       Words are unrelated.
       They don't co-occur in natural text.
       This topic is probably clustering noise.
```

Coherence measures whether keywords **co-occur in the corpus**. Words that frequently appear together in documents are considered coherent.

### Characteristic 2: Distinctive Keywords

```
Good: [machine, learning, neural, training, model]
       ↓
       These words are specific to a concept.
       They don't describe every document.

Bad:  [the, and, is, to, a]
       ↓
       These appear in EVERY document.
       They're not distinctive at all.
       (c-TF-IDF should filter these, but check!)
```

Distinctive keywords are handled by c-TF-IDF (Chapter 4), but quality metrics can catch failures.

### Characteristic 3: Non-Redundant Topics

```
Good topics:
  Topic 0: [running, exercise, fitness]
  Topic 1: [meeting, project, deadline]
  Topic 2: [anxiety, stress, sleep]
            ↓
            Each topic has unique vocabulary.

Redundant topics:
  Topic 0: [running, exercise, fitness]
  Topic 1: [exercise, workout, training]
  Topic 2: [fitness, gym, health]
            ↓
            All three are about the same thing!
            This wastes "topic slots" and confuses users.
```

Diversity metrics detect when topics share too many keywords.

---

## The Two Dimensions of Quality

### Coherence: Within-Topic Quality

```
Question: "Do this topic's keywords belong together?"

Coherence looks INWARD at a single topic.

High coherence (0.3+):
  Keywords frequently co-occur in documents.
  The topic represents a real concept.

Low coherence (<0):
  Keywords rarely appear together.
  The topic is probably noise or misclustering.
```

### Diversity: Between-Topic Quality

```
Question: "Are topics distinct from each other?"

Diversity looks OUTWARD across all topics.

High diversity (0.9+):
  Topics have unique vocabularies.
  Each topic adds new information.

Low diversity (<0.7):
  Topics share many keywords.
  Some topics might be redundant.
```

### The Trade-off

```
                    Coherence
                       ↑
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         │   IDEAL     │  REDUNDANT  │
         │             │             │
         │  Topics are │  Topics are │
         │  coherent   │  coherent   │
         │  AND        │  but too    │
         │  distinct   │  similar    │
         │             │             │
    ←────┼─────────────┼─────────────┼────→ Diversity
         │             │             │
         │   NOISY     │   CHAOTIC   │
         │             │             │
         │  Topics are │  Topics are │
         │  distinct   │  neither    │
         │  but        │  coherent   │
         │  incoherent │  nor distinct│
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ↓

The goal: Top-left quadrant (high coherence + high diversity)
```

---

## Real-World Use Cases

### Use Case 1: Hyperparameter Tuning

```swift
// 📍 Example: Finding optimal min_cluster_size

var bestConfig: HDBSCANConfiguration?
var bestScore: Float = 0

for minSize in [5, 10, 15, 20, 25] {
    let config = HDBSCANConfiguration(minClusterSize: minSize)
    let hdbscan = HDBSCAN(configuration: config)
    let assignment = try await hdbscan.fit(embeddings)

    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    let quality = await topics.evaluateQuality(documents: documents)

    print("minSize=\(minSize): coherence=\(quality.coherence.meanCoherence), diversity=\(quality.diversity.diversity)")

    if quality.qualityScore > bestScore {
        bestScore = quality.qualityScore
        bestConfig = config
    }
}

print("Best: \(bestConfig!)")
```

### Use Case 2: Flagging Bad Topics

```swift
// 📍 Example: Finding topics that need attention

let coherence = await topics.evaluateCoherence(documents: documents)

let badTopics = coherence.lowCoherenceTopics(threshold: 0.0)

if !badTopics.isEmpty {
    print("⚠️ Low-coherence topics detected:")
    for idx in badTopics {
        print("  Topic \(idx): \(topics[idx].keywordSummary())")
        print("  Score: \(coherence.topicScores[idx])")
    }
}
```

### Use Case 3: Detecting Redundancy

```swift
// 📍 Example: Finding topics that might need merging

let diversity = topics.evaluateDiversity(
    topKeywords: 10,
    computeOverlapMatrix: true
)

let overlapping = diversity.overlappingPairs(threshold: 0.5)

if !overlapping.isEmpty {
    print("⚠️ Overlapping topics detected:")
    for pair in overlapping {
        print("  Topics \(pair.topic1) and \(pair.topic2)")
        print("  Overlap: \(pair.overlap * 100)%")
        print("  Consider merging or adjusting clustering parameters")
    }
}
```

### Use Case 4: A/B Testing Topic Models

```swift
// 📍 Example: Comparing two approaches

// Approach A: Default embedding
let topicsA = try await modelA.fit(documents: documents)
let qualityA = await topicsA.topics.evaluateQuality(documents: documents)

// Approach B: Domain-specific embedding
let topicsB = try await modelB.fit(documents: documents)
let qualityB = await topicsB.topics.evaluateQuality(documents: documents)

print("Model A: coherence=\(qualityA.coherence.meanCoherence), diversity=\(qualityA.diversity.diversity)")
print("Model B: coherence=\(qualityB.coherence.meanCoherence), diversity=\(qualityB.diversity.diversity)")

if qualityB.qualityScore > qualityA.qualityScore {
    print("Model B produces higher quality topics")
}
```

---

## Quality Metrics in SwiftTopics

### The Coherence Result

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

public struct CoherenceResult: Sendable {

    /// Per-topic coherence scores (NPMI mean).
    public let topicScores: [Float]

    /// Mean coherence across all topics.
    public let meanCoherence: Float

    /// Median coherence across all topics.
    public let medianCoherence: Float

    /// Minimum coherence (worst topic).
    public let minCoherence: Float

    /// Maximum coherence (best topic).
    public let maxCoherence: Float

    /// Standard deviation of coherence scores.
    public let stdCoherence: Float

    /// Number of topics with positive coherence (> 0).
    public let positiveCoherenceCount: Int
}
```

### The Diversity Result

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

public struct DiversityResult: Sendable {

    /// Percentage of unique keywords (0-1).
    public let diversity: Float

    /// Number of unique keywords across all topics.
    public let uniqueKeywordCount: Int

    /// Total number of keywords (sum across all topics).
    public let totalKeywordCount: Int

    /// Per-topic redundancy with other topics.
    public let topicRedundancy: [Float]

    /// Mean redundancy across topics.
    public let meanRedundancy: Float

    /// Pairwise overlap matrix between topics.
    public let overlapMatrix: [[Float]]?
}
```

### The Combined Quality Score

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

public struct TopicQualityResult: Sendable {

    /// Coherence evaluation result.
    public let coherence: CoherenceResult

    /// Diversity evaluation result.
    public let diversity: DiversityResult

    /// Combined quality score (coherence × diversity).
    public var qualityScore: Float {
        // Normalize coherence from [-1, 1] to [0, 1]
        let normalizedCoherence = (coherence.meanCoherence + 1) / 2
        return normalizedCoherence * diversity.diversity
    }
}
```

---

## Interpreting Scores

### Coherence Benchmarks

```
Score Range     Interpretation            Typical Causes
────────────────────────────────────────────────────────────────
  > 0.5         Excellent                 Strong, focused topics
  0.3 - 0.5     Good                      Clear themes
  0.1 - 0.3     Acceptable                Some noise, but usable
  0.0 - 0.1     Marginal                  Weak coherence, review needed
  < 0.0         Poor                      Topics likely invalid

Note: These are rough guidelines. Optimal thresholds depend on:
  - Corpus size (larger corpora tend toward higher coherence)
  - Domain (technical corpora may have higher coherence)
  - Use case (some applications tolerate lower coherence)
```

### Diversity Benchmarks

```
Score Range     Interpretation            Action
────────────────────────────────────────────────────────────────
  > 0.95        Excellent                 Topics are well-separated
  0.85 - 0.95   Good                      Minimal overlap
  0.70 - 0.85   Acceptable                Some redundancy
  0.50 - 0.70   Low                       Consider merging topics
  < 0.50        Poor                      Significant redundancy

Note: With 10 topics × 10 keywords = 100 total keywords,
      diversity = 0.9 means 90 unique keywords.
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Optimizing One Metric at the Expense of the Other

```swift
// ❌ WRONG: Only looking at coherence
let bestConfig = configs.max { coherence($0) < coherence($1) }
// Might produce one giant topic with 100% coherence but 0% diversity

// ✅ CORRECT: Balance both metrics
let bestConfig = configs.max { qualityScore($0) < qualityScore($1) }
// where qualityScore = normalizedCoherence × diversity
```

### Pitfall 2: Expecting Perfect Scores

```swift
// ⚠️ Reality check: Perfect coherence is unrealistic
// Human-labeled topics typically score 0.3-0.5 NPMI
// Don't chase 1.0—it might indicate overfitting or trivial topics

// Example: A topic with only one keyword has "perfect" coherence
// but is useless for understanding the corpus
```

### Pitfall 3: Ignoring the Outlier Topic

```swift
// ⚠️ The outlier "topic" (id = -1) shouldn't be evaluated
// It contains unclassified documents, not a coherent concept

let realTopics = topics.filter { !$0.isOutlierTopic }
let coherence = await realTopics.evaluateCoherence(documents: documents)
```

### Pitfall 4: Not Using the Same Corpus

```swift
// ❌ WRONG: Evaluating on different documents
let coherence1 = await topics.evaluateCoherence(documents: trainingDocs)
let coherence2 = await topics.evaluateCoherence(documents: testDocs)
// Can't compare these scores—different corpora!

// ✅ CORRECT: Use the same corpus for fair comparison
let coherence1 = await topicsA.evaluateCoherence(documents: corpus)
let coherence2 = await topicsB.evaluateCoherence(documents: corpus)
// Now we can compare
```

---

## Key Takeaways

1. **Quality metrics are essential**: Don't guess—measure.

2. **Two dimensions**: Coherence (within-topic) + Diversity (between-topics).

3. **Use for tuning**: Compare configurations objectively.

4. **Flag problems**: Low-coherence topics need attention.

5. **Balance trade-offs**: High coherence + high diversity is the goal.

6. **Context matters**: Benchmarks depend on corpus and use case.

---

## 💡 Key Insight

Quality evaluation turns subjective judgment into objective measurement:

```
Before: "This topic looks okay, I guess..."
After:  "This topic has coherence 0.45 (good) and 12% redundancy (acceptable)"

Before: "I wonder if more topics would be better..."
After:  "10 topics: coherence=0.42, diversity=0.85"
        "15 topics: coherence=0.38, diversity=0.91"
        "10 topics provides better balance"
```

The goal isn't to hit specific numbers—it's to make **informed decisions** based on measurable criteria.

---

## Next Up

Now that you understand why quality matters, let's dive into how coherence is actually measured.

**[→ 5.2 Coherence Metrics](./02-Coherence-Metrics.md)**

---

*Guide 5.1 of 5.4 • Chapter 5: Quality Evaluation*
