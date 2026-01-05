# 5.3 NPMI Deep Dive

> **The math behind Normalized Pointwise Mutual Information.**

---

## The Concept

**NPMI (Normalized Pointwise Mutual Information)** is the industry-standard metric for topic coherence. It measures how strongly word pairs co-occur, normalized to a [-1, +1] scale.

```
NPMI answers: "Do these two words appear together more or less
              than we'd expect by chance?"

NPMI = +1: Words ALWAYS appear together (perfect association)
NPMI =  0: Words appear together as often as chance predicts
NPMI = -1: Words NEVER appear together (perfect dissociation)
```

---

## Why NPMI?

### The Problem with Raw Counts

```
Given co-occurrence counts:

  running + exercise:  150 windows
  running + meeting:   12 windows

Is 150 > 12 enough to say (running, exercise) is better?

Not necessarily!

If "exercise" appears in 400 windows total and
   "meeting" appears in 4000 windows total,
then:
  - exercise appearing 150 times with running is significant
  - meeting appearing only 12 times with running is unusual!

We need to account for word frequencies.
```

### The Problem with Raw PMI

```
PMI(w₁, w₂) = log(P(w₁, w₂) / (P(w₁) × P(w₂)))

PMI handles base rates, but:

  - Range is (-∞, +∞) — hard to interpret
  - Sensitive to rare events
  - Can't compare across different corpora

Example:
  PMI(running, exercise) = 2.3
  PMI(quantum, entanglement) = 4.1

  Is quantum-entanglement "twice as good"?
  Not really—it's just that both words are rarer.
```

### NPMI Solves These Problems

```
NPMI normalizes PMI to [-1, +1]:

  NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))

Now:
  NPMI(running, exercise) = 0.6
  NPMI(quantum, entanglement) = 0.7

Both are comparable: 0.7 > 0.6 means stronger association.
```

---

## The Mathematics

### Step 1: Compute Probabilities

From co-occurrence counts:

```
Total windows: N = 10,000

Windows containing "running": 500
  P(running) = 500 / 10,000 = 0.05

Windows containing "exercise": 400
  P(exercise) = 400 / 10,000 = 0.04

Windows containing BOTH: 150
  P(running, exercise) = 150 / 10,000 = 0.015
```

### Step 2: Compute PMI

```
PMI(w₁, w₂) = log(P(w₁, w₂) / (P(w₁) × P(w₂)))

Expected probability if independent:
  P(running) × P(exercise) = 0.05 × 0.04 = 0.002

Observed probability:
  P(running, exercise) = 0.015

PMI = log(0.015 / 0.002)
    = log(7.5)
    ≈ 2.01

Interpretation: These words co-occur 7.5× more than expected!
```

### Step 3: Normalize to Get NPMI

```
NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))

The denominator -log(P(w₁, w₂)) is the maximum possible PMI:
  - If P(w₁, w₂) = P(w₁) = P(w₂), then PMI = -log(P(w₁, w₂))
  - This happens when the words ALWAYS co-occur

For our example:
  -log(P(running, exercise)) = -log(0.015) ≈ 4.20

  NPMI = 2.01 / 4.20 ≈ 0.48

Interpretation: 48% of the maximum possible association strength.
```

### The Complete Formula

```
Given:
  N = total windows
  c(w) = count of windows containing word w
  c(w₁, w₂) = count of windows containing both words

Probabilities (with smoothing ε):
  P(w₁) = (c(w₁) + ε) / (N + ε)
  P(w₂) = (c(w₂) + ε) / (N + ε)
  P(w₁, w₂) = (c(w₁, w₂) + ε) / (N + ε)

PMI:
  PMI(w₁, w₂) = log(P(w₁, w₂) / (P(w₁) × P(w₂)))

NPMI:
  NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))

Result: NPMI ∈ [-1, +1]
```

---

## Interpreting NPMI Values

### The Range

```
NPMI = +1.0
───────────
The words ALWAYS appear together.
P(w₁, w₂) = P(w₁) = P(w₂)
Every occurrence of w₁ also has w₂ and vice versa.

Example: Rare phrase components
  "prime" and "minister" in political corpus
  They almost always co-occur as "prime minister"


NPMI = 0.0
──────────
The words are statistically INDEPENDENT.
P(w₁, w₂) = P(w₁) × P(w₂)
They co-occur exactly as often as chance predicts.

Example: Unrelated words
  "running" and "taxes" in a general corpus
  Sometimes co-occur by coincidence, nothing more


NPMI = -1.0
───────────
The words NEVER appear together.
P(w₁, w₂) = 0
Mutually exclusive words.

Example: Antonyms in structured text
  "male" and "female" in a demographic form
  (Each row is one or the other, never both)
```

### Typical Score Ranges for Topics

```
Topic Coherence     Typical Mean NPMI    Quality
────────────────────────────────────────────────────
Excellent           > 0.30               Strong semantic theme
Good                0.15 - 0.30          Clear topic identity
Acceptable          0.05 - 0.15          Usable, some noise
Marginal            0.00 - 0.05          Weak coherence
Poor                < 0.00               Likely noise

Note: Human-labeled topics typically score 0.20-0.40.
```

---

## Topic Coherence from NPMI

### Averaging Over Pairs

```
Given topic keywords: [w₁, w₂, w₃, ..., wₖ]

Compute NPMI for all pairs (wᵢ, wⱼ) where i < j:

For k = 5 keywords:
  (w₁, w₂), (w₁, w₃), (w₁, w₄), (w₁, w₅)  = 4 pairs
  (w₂, w₃), (w₂, w₄), (w₂, w₅)             = 3 pairs
  (w₃, w₄), (w₃, w₅)                        = 2 pairs
  (w₄, w₅)                                   = 1 pair
                                       Total = 10 pairs

Topic Coherence = mean(NPMI(wᵢ, wⱼ)) for all pairs
```

### Example Calculation

```
Topic: [running, exercise, fitness, morning, workout]

NPMI scores for each pair:
  (running, exercise):   0.52
  (running, fitness):    0.48
  (running, morning):    0.31
  (running, workout):    0.55
  (exercise, fitness):   0.61
  (exercise, morning):   0.28
  (exercise, workout):   0.54
  (fitness, morning):    0.22
  (fitness, workout):    0.49
  (morning, workout):    0.35

Mean NPMI = (0.52 + 0.48 + 0.31 + 0.55 + 0.61 +
             0.28 + 0.54 + 0.22 + 0.49 + 0.35) / 10
          = 4.35 / 10
          = 0.435

Topic Coherence = 0.435 (Good!)
```

---

## In SwiftTopics

### The NPMIScorer

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/NPMIScorer.swift

/// Computes Normalized Pointwise Mutual Information scores.
public struct NPMIScorer: Sendable {

    /// Configuration for scoring.
    public let configuration: NPMIConfiguration

    /// Creates an NPMI scorer.
    public init(configuration: NPMIConfiguration = .default)

    /// Computes NPMI for a single word pair.
    public func score(
        word1: String,
        word2: String,
        counts: CooccurrenceCounts
    ) -> NPMIPairScore?

    /// Computes NPMI for a topic's keywords.
    public func score(
        keywords: [String],
        counts: CooccurrenceCounts
    ) -> TopicNPMIResult

    /// Computes NPMI for multiple topics.
    public func score(
        topics: [Topic],
        counts: CooccurrenceCounts,
        topKeywords: Int = 10
    ) -> [TopicNPMIResult]
}
```

### The NPMIPairScore

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/NPMIScorer.swift

/// NPMI score for a single word pair.
public struct NPMIPairScore: Sendable, Hashable {

    /// First word.
    public let word1: String

    /// Second word.
    public let word2: String

    /// NPMI score in range [-1, +1].
    public let npmi: Float

    /// PMI score (unnormalized).
    public let pmi: Float

    /// Probability of word1.
    public let pWord1: Float

    /// Probability of word2.
    public let pWord2: Float

    /// Joint probability of word pair.
    public let pPair: Float
}
```

### The TopicNPMIResult

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/NPMIScorer.swift

/// NPMI scores for a topic's keywords.
public struct TopicNPMIResult: Sendable {

    /// The topic's keywords that were scored.
    public let keywords: [String]

    /// NPMI scores for each word pair.
    public let pairScores: [NPMIPairScore]

    /// Mean NPMI across all word pairs.
    /// This is the topic's coherence score.
    public let meanNPMI: Float

    /// Number of word pairs evaluated.
    public var pairCount: Int { pairScores.count }
}
```

### Implementation Details

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/NPMIScorer.swift

public func score(
    word1: String,
    word2: String,
    counts: CooccurrenceCounts
) -> NPMIPairScore? {
    guard counts.totalWindows > 0 else { return nil }
    guard word1 != word2 else { return nil }

    let totalWindows = Float(counts.totalWindows)
    let epsilon = configuration.epsilon

    // Get raw counts
    let countW1 = counts.count(for: word1)
    let countW2 = counts.count(for: word2)
    let countPair = counts.count(for: word1, word2)

    // If either word doesn't appear, we can't compute meaningful NPMI
    guard countW1 > 0 && countW2 > 0 else { return nil }

    // Compute probabilities with smoothing
    let pW1 = (Float(countW1) + epsilon) / (totalWindows + epsilon)
    let pW2 = (Float(countW2) + epsilon) / (totalWindows + epsilon)
    let pPair = (Float(countPair) + epsilon) / (totalWindows + epsilon)

    // Compute PMI = log(P(w1,w2) / (P(w1) × P(w2)))
    let pmi = log(pPair / (pW1 * pW2))

    // Compute NPMI = PMI / -log(P(w1,w2))
    let denominator = -log(pPair)
    let npmi: Float
    if denominator.isNaN || denominator.isInfinite || denominator == 0 {
        npmi = 0
    } else {
        npmi = pmi / denominator
    }

    // Clamp to [-1, +1] for numerical stability
    let clampedNPMI = max(-1.0, min(1.0, npmi))

    return NPMIPairScore(
        word1: word1, word2: word2,
        npmi: clampedNPMI, pmi: pmi,
        pWord1: pW1, pWord2: pW2, pPair: pPair
    )
}
```

---

## Edge Cases and Handling

### Case 1: Zero Co-occurrence

```
If two words NEVER co-occur:
  c(w₁, w₂) = 0
  P(w₁, w₂) ≈ ε (smoothing only)

This makes:
  log(P(w₁, w₂)) → very negative (close to -∞)
  PMI → very negative
  NPMI → approaches -1

Interpretation: Strong negative association.
The words actively avoid each other.
```

### Case 2: Unknown Word

```
If a word doesn't appear in the corpus:
  c(w) = 0
  P(w) ≈ ε

SwiftTopics returns nil for such pairs:

guard countW1 > 0 && countW2 > 0 else { return nil }

Rationale: We can't meaningfully evaluate words
that don't appear in the reference corpus.
```

### Case 3: Very Rare Words

```
If words are very rare, NPMI can be noisy:

c(w₁) = 2, c(w₂) = 3, c(w₁, w₂) = 2

With N = 10,000:
  P(w₁) = 0.0002
  P(w₂) = 0.0003
  P(w₁, w₂) = 0.0002

This gives high NPMI because the rare pair
occurs more than chance, but it's based on
very few observations.

Mitigation: Use larger corpora or increase window size.
```

### Case 4: Smoothing

```swift
// Epsilon prevents log(0) and division by zero
let epsilon = configuration.epsilon  // Default: 1e-12

// Applied to all probabilities
let pW1 = (Float(countW1) + epsilon) / (totalWindows + epsilon)

// Without smoothing:
//   P = 0 / 10000 = 0
//   log(0) = -∞  ← Problem!

// With smoothing:
//   P = (0 + 1e-12) / (10000 + 1e-12) ≈ 1e-16
//   log(1e-16) ≈ -36.8  ← Finite, usable
```

---

## Using the NPMICoherenceEvaluator

### Basic Usage

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

let evaluator = NPMICoherenceEvaluator(configuration: .default)

let result = await evaluator.evaluate(
    topics: topics,
    documents: documents
)

print("Mean coherence: \(result.meanCoherence)")
print("Median coherence: \(result.medianCoherence)")
print("Min coherence: \(result.minCoherence)")
print("Max coherence: \(result.maxCoherence)")
```

### With Pre-computed Counts

```swift
// Useful when evaluating multiple models on same corpus
let counts = documents.precomputeCooccurrences(
    configuration: .default
)

// Evaluate first model
let result1 = evaluator.evaluate(topics: topicsA, counts: counts)

// Evaluate second model (reuses counts)
let result2 = evaluator.evaluate(topics: topicsB, counts: counts)

// Compare
print("Model A: \(result1.meanCoherence)")
print("Model B: \(result2.meanCoherence)")
```

### Getting Detailed Results

```swift
let evaluator = NPMICoherenceEvaluator(
    configuration: .default,
    includeDetailedResults: true  // Include per-pair scores
)

let result = await evaluator.evaluate(topics: topics, documents: documents)

// Access detailed results
if let detailed = result.detailedResults {
    for (idx, topicResult) in detailed.enumerated() {
        print("Topic \(idx): mean NPMI = \(topicResult.meanNPMI)")

        // See individual pair scores
        for pairScore in topicResult.pairScores {
            print("  \(pairScore.word1) + \(pairScore.word2): \(pairScore.npmi)")
        }
    }
}
```

### Finding Problem Topics

```swift
let result = await evaluator.evaluate(topics: topics, documents: documents)

// Get topics sorted by coherence
let sorted = result.topicsByCoherence()
print("Best topic: \(sorted.first!.index) with score \(sorted.first!.score)")
print("Worst topic: \(sorted.last!.index) with score \(sorted.last!.score)")

// Find low-coherence topics
let lowCoherence = result.lowCoherenceTopics(threshold: 0.0)
if !lowCoherence.isEmpty {
    print("⚠️ Topics with negative coherence: \(lowCoherence)")
}

// Check if all topics are positive
if result.allPositive {
    print("✓ All topics have positive coherence")
}
```

---

## Walkthrough Example

```
Corpus: 1000 journal entries about daily life
Topics extracted: 5 topics, 10 keywords each

Step 1: Tokenize documents
  → 1000 documents, ~50,000 total tokens

Step 2: Count co-occurrences (window size 10)
  → 45,000 windows
  → 8,500 unique words
  → 127,000 unique word pairs with count > 0

Step 3: Compute NPMI for Topic 0

  Keywords: [running, exercise, fitness, morning, workout,
             health, gym, training, daily, routine]

  Pairs to evaluate: 10 × 9 / 2 = 45 pairs

  Sample pairs:
    running + exercise:
      c(running) = 312, c(exercise) = 287, c(both) = 189
      P(running) = 0.007, P(exercise) = 0.006, P(both) = 0.004
      PMI = log(0.004 / (0.007 × 0.006)) = log(95.2) = 4.56
      NPMI = 4.56 / -log(0.004) = 4.56 / 5.52 = 0.83

    running + routine:
      c(running) = 312, c(routine) = 156, c(both) = 42
      P(running) = 0.007, P(routine) = 0.003, P(both) = 0.0009
      PMI = log(0.0009 / (0.007 × 0.003)) = log(42.9) = 3.76
      NPMI = 3.76 / -log(0.0009) = 3.76 / 7.01 = 0.54

  Mean NPMI for Topic 0: 0.62 (Excellent!)

Step 4: Aggregate across topics
  Topic 0: 0.62
  Topic 1: 0.51
  Topic 2: 0.38
  Topic 3: 0.45
  Topic 4: -0.08  ← Problem topic!

  Mean coherence: 0.38
  Median coherence: 0.45
  Min coherence: -0.08
  Max coherence: 0.62

Topic 4 has negative coherence → Investigate!
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Forgetting to Filter Outliers

```swift
// ❌ WRONG: Including outlier topic
let result = await evaluator.evaluate(topics: allTopics, documents: docs)
// Outlier topic (id = -1) will likely have low/negative coherence
// and drag down the mean

// ✅ CORRECT: Filter outliers first
let realTopics = allTopics.filter { !$0.isOutlierTopic }
let result = await evaluator.evaluate(topics: realTopics, documents: docs)
```

### Pitfall 2: Too Few Keywords

```swift
// ⚠️ With only 3 keywords, you have 3 pairs
// That's not enough for a stable mean

let config = CoherenceConfiguration(topKeywords: 3)
// May produce noisy coherence scores

// ✅ Use at least 10 keywords
let config = CoherenceConfiguration(topKeywords: 10)
// 45 pairs → more stable average
```

### Pitfall 3: Mismatched Tokenization

```swift
// ⚠️ If topic keywords are: ["Running", "Exercise"]
// But documents tokenize to: ["running", "exercise"]
// → No matches, NPMI will be undefined

// ✅ Ensure consistent lowercasing in both:
// 1. Topic extraction (Tokenizer in CTFIDFRepresenter)
// 2. Coherence evaluation (Tokenizer in NPMICoherenceEvaluator)
```

### Pitfall 4: Expecting Perfect Scores

```swift
// ⚠️ NPMI = 1.0 is extremely rare in practice
// It would mean words ALWAYS appear together

// Real-world benchmarks:
// - LDA topics: typically 0.05 - 0.25 NPMI
// - Neural topic models: typically 0.15 - 0.40 NPMI
// - Excellent topics: 0.40 - 0.60 NPMI

// Don't expect 0.9+; that's unrealistic for most corpora
```

---

## Key Takeaways

1. **NPMI normalizes PMI to [-1, +1]**: Easy to interpret and compare.

2. **The formula**: `NPMI = PMI / -log(P(w₁, w₂))`.

3. **Positive NPMI = positive association**: Words co-occur more than chance.

4. **Topic coherence = mean NPMI**: Average over all keyword pairs.

5. **0.3+ is good**: Typical threshold for quality topics.

6. **Smoothing prevents edge cases**: Epsilon avoids log(0).

---

## 💡 Key Insight

NPMI captures a simple but powerful intuition:

```
If a topic is meaningful, its keywords should appear together
in the corpus more often than random chance would predict.

This is because meaningful topics represent real concepts,
and real concepts are discussed coherently in text.

NPMI quantifies this: How much more often do these words
co-occur than we'd expect if they were independent?

High NPMI → Strong co-occurrence → Coherent topic → Real concept
Low NPMI  → Weak co-occurrence  → Incoherent topic → Noise
```

The beauty is that NPMI requires no external knowledge—it uses only the corpus itself to evaluate whether extracted topics are meaningful.

---

## Next Up

We've covered how to evaluate coherence *within* topics. Now let's look at diversity *between* topics.

**[→ 5.4 Diversity Metrics](./04-Diversity-Metrics.md)**

---

*Guide 5.3 of 5.4 • Chapter 5: Quality Evaluation*
