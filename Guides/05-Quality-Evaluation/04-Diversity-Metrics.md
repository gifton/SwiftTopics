# 5.4 Diversity Metrics

> **Ensuring topics are distinct from each other.**

---

## The Concept

Coherence tells us if keywords *within* a topic belong together. **Diversity** tells us if topics are *distinct* from each other.

```
Good diversity:
  Topic 0: [running, exercise, fitness, workout, gym]
  Topic 1: [meeting, project, deadline, client, review]
  Topic 2: [anxiety, stress, sleep, worried, feeling]

  Each topic has unique vocabulary → High diversity!

Bad diversity:
  Topic 0: [running, exercise, fitness, workout, gym]
  Topic 1: [exercise, workout, training, fitness, health]
  Topic 2: [fitness, gym, health, exercise, running]

  All topics share the same words → Low diversity!
  These should probably be merged into one topic.
```

---

## Why Diversity Matters

### Problem 1: Wasted Topics

```
With 10 topics, you expect 10 distinct themes.

If 3 topics are about the same thing:
  - You've wasted 2 topic slots
  - Users see redundant categories
  - Document assignment is arbitrary between similar topics

Example:
  Topic 2: [coffee, morning, caffeine, espresso, brew]
  Topic 5: [coffee, cup, morning, drink, bean]
  Topic 8: [espresso, latte, coffee, café, barista]

  These are ALL about coffee!
  You could have used those slots for distinct themes.
```

### Problem 2: User Confusion

```
In a topic browser:

┌─────────────────────────────────────┐
│ Your Journal Topics                 │
├─────────────────────────────────────┤
│ • Exercise & Fitness                │
│ • Workout & Training                │  ← Users: "What's the difference?"
│ • Gym & Health                      │
│ • Morning Running                   │
│ • Anxiety & Stress                  │
│ • Work Projects                     │
└─────────────────────────────────────┘

Users can't distinguish the first four topics.
They might assign entries inconsistently.
```

### Problem 3: Poor Clustering Signal

```
Redundant topics suggest clustering issues:

Possible causes:
  1. min_cluster_size too small → Too many clusters
  2. Dimensionality reduction too aggressive → Topics split
  3. Vocabulary overlap in corpus → Topics entangled

Diversity metrics help diagnose these issues.
```

---

## Diversity Metrics

### Metric 1: Unique Keyword Ratio

```
Diversity = |unique keywords| / |total keywords|

Example:
  5 topics × 10 keywords each = 50 total keywords
  If 45 are unique: Diversity = 45/50 = 0.90 (Good!)
  If 25 are unique: Diversity = 25/50 = 0.50 (Poor!)

Interpretation:
  1.0 = Perfect diversity (no overlap)
  0.9+ = High diversity (minimal overlap)
  0.7-0.9 = Moderate diversity (acceptable)
  <0.7 = Low diversity (investigate)
```

### Metric 2: Per-Topic Redundancy

```
Redundancy(topic) = |keywords ∩ other_topics| / |keywords|

For each topic, what fraction of its keywords appear elsewhere?

Example:
  Topic 0: [running, exercise, fitness, morning, workout]

  Other topics contain: {exercise, fitness, morning}

  Redundancy(Topic 0) = 3/5 = 0.60

Interpretation:
  0.0 = No overlap (perfectly unique)
  0.0-0.2 = Low redundancy (good)
  0.2-0.5 = Moderate redundancy (acceptable)
  0.5+ = High redundancy (consider merging)
```

### Metric 3: Pairwise Overlap

```
Overlap(i, j) = |keywords(i) ∩ keywords(j)| / |keywords(i)|

How much of topic i's vocabulary appears in topic j?

              Topic 0  Topic 1  Topic 2
  Topic 0      1.00      0.10     0.20
  Topic 1      0.10      1.00     0.15
  Topic 2      0.20      0.15     1.00

Note: This matrix is NOT symmetric!
  - Topic i may share 50% of its keywords with j
  - But j may only share 10% of its keywords with i
  - (If i has few keywords and j has many)
```

---

## In SwiftTopics

### The DiversityMetrics Calculator

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

/// Computes diversity and redundancy metrics for topics.
public struct DiversityMetrics: Sendable {

    /// Whether to compute the full overlap matrix.
    public let computeOverlapMatrix: Bool

    /// Creates diversity metrics calculator.
    public init(computeOverlapMatrix: Bool = false)

    /// Evaluates diversity metrics for topics.
    public func evaluate(topics: [Topic], topKeywords: Int = 10) -> DiversityResult
}
```

### The DiversityResult

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

/// Result of topic diversity evaluation.
public struct DiversityResult: Sendable {

    /// Percentage of unique keywords across all topics (0-1).
    public let diversity: Float

    /// Number of unique keywords across all topics.
    public let uniqueKeywordCount: Int

    /// Total number of keywords (sum across all topics).
    public let totalKeywordCount: Int

    /// Per-topic redundancy with other topics.
    public let topicRedundancy: [Float]

    /// Mean redundancy across topics.
    public let meanRedundancy: Float

    /// Pairwise overlap matrix between topics (optional).
    public let overlapMatrix: [[Float]]?

    /// Whether topics are highly diverse (diversity > 0.9).
    public var isHighlyDiverse: Bool

    /// Whether topics have low redundancy (mean < 0.1).
    public var isLowRedundancy: Bool

    /// Gets the most redundant topics.
    public func redundantTopics(threshold: Float = 0.5) -> [Int]

    /// Gets pairs of highly overlapping topics.
    public func overlappingPairs(threshold: Float = 0.5) -> [(topic1: Int, topic2: Int, overlap: Float)]
}
```

### Basic Usage

```swift
// Simple diversity evaluation
let result = topics.evaluateDiversity()

print("Diversity: \(result.diversity)")
print("Unique keywords: \(result.uniqueKeywordCount) of \(result.totalKeywordCount)")
print("Mean redundancy: \(result.meanRedundancy)")

// Check quality
if result.isHighlyDiverse {
    print("✓ Topics are well-separated")
}
```

### Finding Redundant Topics

```swift
let result = topics.evaluateDiversity(topKeywords: 10)

// Find topics with high redundancy
let redundant = result.redundantTopics(threshold: 0.4)
for idx in redundant {
    print("Topic \(idx) has \(result.topicRedundancy[idx] * 100)% redundancy")
    print("Keywords: \(topics[idx].keywordSummary())")
}
```

### Finding Overlapping Pairs

```swift
// Enable overlap matrix computation
let result = topics.evaluateDiversity(
    topKeywords: 10,
    computeOverlapMatrix: true
)

// Find pairs that might need merging
let overlapping = result.overlappingPairs(threshold: 0.5)
for pair in overlapping {
    print("Topics \(pair.topic1) and \(pair.topic2) have \(pair.overlap * 100)% overlap")
    print("  Topic \(pair.topic1): \(topics[pair.topic1].keywordSummary())")
    print("  Topic \(pair.topic2): \(topics[pair.topic2].keywordSummary())")
    print("  Consider merging or adjusting clustering parameters")
}
```

---

## Implementation Details

### Computing Diversity

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

public func evaluate(topicKeywords: [[String]]) -> DiversityResult {
    // Flatten all keywords
    let allKeywords = topicKeywords.flatMap { $0 }
    let uniqueKeywords = Set(allKeywords)

    let totalCount = allKeywords.count
    let uniqueCount = uniqueKeywords.count

    // Diversity = unique / total
    let diversity: Float = totalCount > 0 ?
        Float(uniqueCount) / Float(totalCount) : 1.0

    // ... rest of computation
}
```

### Computing Redundancy

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

private func computeRedundancy(topicKeywords: [[String]]) -> [Float] {
    let keywordSets = topicKeywords.map(Set.init)
    var redundancy = [Float]()

    for i in 0..<keywordSets.count {
        let topicKeywordSet = keywordSets[i]

        guard !topicKeywordSet.isEmpty else {
            redundancy.append(0)
            continue
        }

        // Keywords from all OTHER topics
        var otherKeywords = Set<String>()
        for j in 0..<keywordSets.count where j != i {
            otherKeywords.formUnion(keywordSets[j])
        }

        // Count overlap
        let overlap = topicKeywordSet.intersection(otherKeywords).count
        let redundancyScore = Float(overlap) / Float(topicKeywordSet.count)

        redundancy.append(redundancyScore)
    }

    return redundancy
}
```

### Computing Overlap Matrix

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

private func computeOverlapMatrix(topicKeywords: [[String]]) -> [[Float]] {
    let k = topicKeywords.count
    let keywordSets = topicKeywords.map(Set.init)

    var matrix = [[Float]](repeating: [Float](repeating: 0, count: k), count: k)

    for i in 0..<k {
        for j in 0..<k {
            if i == j {
                matrix[i][j] = 1.0  // Self-overlap is 1
            } else if keywordSets[i].isEmpty {
                matrix[i][j] = 0
            } else {
                // Fraction of i's keywords that appear in j
                let overlap = keywordSets[i].intersection(keywordSets[j]).count
                matrix[i][j] = Float(overlap) / Float(keywordSets[i].count)
            }
        }
    }

    return matrix
}
```

---

## Visualizing Diversity

### Overlap Matrix Visualization

```
10 topics, 10 keywords each:

Overlap Matrix:
        T0   T1   T2   T3   T4   T5   T6   T7   T8   T9
  T0  1.00 0.10 0.00 0.20 0.00 0.10 0.00 0.10 0.00 0.00
  T1  0.10 1.00 0.10 0.00 0.60 0.00 0.00 0.10 0.00 0.00
  T2  0.00 0.10 1.00 0.00 0.10 0.20 0.00 0.00 0.10 0.00
  T3  0.30 0.00 0.00 1.00 0.00 0.00 0.20 0.10 0.00 0.00
  T4  0.00 0.70 0.10 0.00 1.00 0.00 0.00 0.00 0.00 0.00  ← High overlap with T1!
  T5  0.10 0.00 0.20 0.00 0.00 1.00 0.10 0.00 0.10 0.00
  T6  0.00 0.00 0.00 0.20 0.00 0.10 1.00 0.00 0.00 0.10
  T7  0.10 0.10 0.00 0.20 0.00 0.00 0.00 1.00 0.00 0.00
  T8  0.00 0.00 0.20 0.00 0.00 0.10 0.00 0.00 1.00 0.10
  T9  0.00 0.00 0.00 0.00 0.00 0.00 0.10 0.00 0.10 1.00

Topics T1 and T4 have 60-70% overlap → Candidates for merging!

Summary:
  Diversity: 0.87 (87 unique keywords of 100 total)
  Mean redundancy: 0.12
  Overlapping pairs (>50%): [(T1, T4, 0.65)]
```

### Redundancy Bar Chart (Conceptual)

```
Per-Topic Redundancy:

T0  ████████████░░░░░░░░░░░░░░░░  0.24
T1  ██████████████████░░░░░░░░░░  0.36
T2  ██████░░░░░░░░░░░░░░░░░░░░░░  0.12
T3  ████████████░░░░░░░░░░░░░░░░  0.24
T4  ██████████████████████████░░  0.52  ← High redundancy!
T5  ████████░░░░░░░░░░░░░░░░░░░░  0.16
T6  ██████░░░░░░░░░░░░░░░░░░░░░░  0.12
T7  ████████████░░░░░░░░░░░░░░░░  0.24
T8  ████░░░░░░░░░░░░░░░░░░░░░░░░  0.08
T9  ██░░░░░░░░░░░░░░░░░░░░░░░░░░  0.04

    └── 0.0 ────── 0.5 ────── 1.0 ──┘

Topic T4 has 52% of its keywords appearing in other topics.
Investigate whether T4 should be merged with T1.
```

---

## Combined Quality Score

### The TopicQualityResult

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/DiversityMetrics.swift

/// Combined coherence and diversity evaluation result.
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

### Using Combined Quality

```swift
// Evaluate both coherence and diversity
let quality = await topics.evaluateQuality(documents: documents)

print("Coherence: \(quality.coherence.meanCoherence)")
print("Diversity: \(quality.diversity.diversity)")
print("Combined Score: \(quality.qualityScore)")

// Quality score interpretation:
// - Coherence: 0.4 → normalized to (0.4 + 1) / 2 = 0.7
// - Diversity: 0.85
// - Combined: 0.7 × 0.85 = 0.595
```

### Why Multiply?

```
Both dimensions must be high for good topics:

                    Coherence
                       ↑
              High C │ Low C
                     │
         ┌───────────┼───────────┐
High D   │   0.7×0.9 │ 0.3×0.9   │ High D
         │   = 0.63  │ = 0.27    │
         ├───────────┼───────────┤
Low D    │   0.7×0.5 │ 0.3×0.5   │ Low D
         │   = 0.35  │ = 0.15    │
         └───────────┼───────────┘
                     │
         ←───────────┼───────────→ Diversity

Only the top-left quadrant (high both) scores well.
Multiplication penalizes weakness in either dimension.
```

---

## Tuning Based on Metrics

### Too Many Topics (Low Diversity)

```
Symptom: Diversity < 0.7, many overlapping pairs

Diagnosis: Topics are too fine-grained, splitting related concepts

Fix:
  - Increase min_cluster_size in HDBSCAN
  - Reduce number of topics (if using k-means)
  - Merge similar topics post-hoc

Before: 15 topics, diversity = 0.65
After:  10 topics, diversity = 0.88
```

### Too Few Topics (Low Coherence)

```
Symptom: High diversity but low coherence

Diagnosis: Topics are too broad, mixing unrelated concepts

Fix:
  - Decrease min_cluster_size in HDBSCAN
  - Increase number of topics
  - Use more specific embeddings

Before: 5 topics, coherence = 0.18, diversity = 0.95
After:  10 topics, coherence = 0.42, diversity = 0.87
```

### Both Low

```
Symptom: Both coherence and diversity are low

Diagnosis: Fundamental issues with clustering or representation

Investigate:
  - Are embeddings capturing semantics? (Try different embedding model)
  - Is UMAP preserving structure? (Check n_neighbors, min_dist)
  - Is the corpus too noisy? (Improve preprocessing)
  - Are stop words leaking through? (Check tokenizer configuration)
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Ignoring Overlap Direction

```swift
// ⚠️ Overlap is NOT symmetric
// Topic A may share 80% of keywords with Topic B,
// but B may only share 20% with A (if B has more keywords)

// ✅ Check both directions when investigating pairs
if let matrix = result.overlapMatrix {
    let overlap_i_j = matrix[i][j]  // i's keywords in j
    let overlap_j_i = matrix[j][i]  // j's keywords in i
    let maxOverlap = max(overlap_i_j, overlap_j_i)
}
```

### Pitfall 2: Using Wrong Keyword Count

```swift
// ⚠️ More keywords = naturally lower diversity
// (More chance of overlap)

let result10 = topics.evaluateDiversity(topKeywords: 10)
let result20 = topics.evaluateDiversity(topKeywords: 20)

// result20.diversity < result10.diversity (expected!)

// ✅ Use consistent topKeywords when comparing
// The default (10) is a good balance
```

### Pitfall 3: Expecting Perfect Diversity

```swift
// ⚠️ Some overlap is normal and acceptable
// Related topics may share vocabulary

// Example: "fitness" and "health" topics
// Both might have: [exercise, wellness, body]
// This doesn't mean they should merge!

// ✅ Use overlap as a signal, not a binary decision
// High overlap (>60%) warrants investigation
// Moderate overlap (20-40%) is often acceptable
```

### Pitfall 4: Forgetting Context

```swift
// ⚠️ Diversity depends on topic count and corpus

// 5 topics in a narrow domain (e.g., cooking blog):
// Expected diversity might be 0.7-0.8 (topics related)

// 20 topics in a broad domain (e.g., general news):
// Expected diversity might be 0.9+ (topics distinct)

// ✅ Interpret relative to your specific use case
```

---

## Best Practices

### 1. Evaluate Both Dimensions

```swift
// Always check coherence AND diversity
let quality = await topics.evaluateQuality(documents: documents)

print("Quality Report:")
print("  Coherence: \(quality.coherence.meanCoherence)")
print("  Diversity: \(quality.diversity.diversity)")
print("  Combined:  \(quality.qualityScore)")
```

### 2. Flag Outliers

```swift
// Find topics needing attention
let lowCoherence = quality.coherence.lowCoherenceTopics(threshold: 0.0)
let highRedundancy = quality.diversity.redundantTopics(threshold: 0.5)

if !lowCoherence.isEmpty {
    print("⚠️ Low coherence: \(lowCoherence)")
}
if !highRedundancy.isEmpty {
    print("⚠️ High redundancy: \(highRedundancy)")
}
```

### 3. Compare Configurations

```swift
var bestQuality: TopicQualityResult?
var bestConfig: HDBSCANConfiguration?

for minSize in [5, 10, 15, 20] {
    let config = HDBSCANConfiguration(minClusterSize: minSize)
    let topics = try await model.fit(documents, config: config)
    let quality = await topics.evaluateQuality(documents: documents)

    if bestQuality == nil || quality.qualityScore > bestQuality!.qualityScore {
        bestQuality = quality
        bestConfig = config
    }
}

print("Best config: \(bestConfig!) with score \(bestQuality!.qualityScore)")
```

### 4. Track Over Time

```swift
// As your corpus grows, re-evaluate quality
func monthlyQualityCheck() async {
    let quality = await currentTopics.evaluateQuality(documents: allDocuments)

    metrics.record("topic_coherence", quality.coherence.meanCoherence)
    metrics.record("topic_diversity", quality.diversity.diversity)

    if quality.qualityScore < 0.5 {
        alert("Topic quality degraded—consider retraining")
    }
}
```

---

## Key Takeaways

1. **Diversity complements coherence**: Both are needed for good topics.

2. **Unique keyword ratio**: Simple but effective diversity metric.

3. **Redundancy is per-topic**: Some topics may be more redundant than others.

4. **Overlap matrix finds pairs**: Identify which specific topics might merge.

5. **Combined score balances both**: Multiplication ensures both matter.

6. **Use for tuning**: Metrics guide hyperparameter selection.

---

## 💡 Key Insight

Quality evaluation transforms topic modeling from art to science:

```
Without metrics:
  "These topics seem okay, I think?"
  "Maybe we should try more topics?"
  "I'm not sure if this configuration is better."

With metrics:
  "Mean coherence improved from 0.35 to 0.45"
  "Diversity dropped below 0.8—we have redundant topics"
  "Configuration B scores 15% higher than A on combined quality"

The metrics don't make decisions for you,
but they give you the data to make informed decisions.
```

---

## Chapter Summary

You've learned how to evaluate topic quality:

1. **Why Quality Matters**: Objective metrics vs. subjective judgment.

2. **Coherence Metrics**: Co-occurrence as a proxy for semantic relatedness.

3. **NPMI Deep Dive**: The normalized PMI formula and its interpretation.

4. **Diversity Metrics**: Ensuring topics don't overlap excessively.

Together, coherence and diversity give you a complete picture of topic quality:
- **Coherence**: Are keywords *within* each topic related?
- **Diversity**: Are topics *distinct* from each other?

High-quality topic models score well on both dimensions.

---

## Next Up

With all the core concepts covered, we're ready for the capstone: putting everything together into a complete topic modeling pipeline.

**[→ Chapter 6: Capstone](../06-Capstone/README.md)**

---

*Guide 5.4 of 5.4 • Chapter 5: Quality Evaluation*
