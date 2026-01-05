# Chapter 5: Quality Evaluation

> **How do we know if our topics are any good?**

---

## The Challenge

After clustering and keyword extraction, you have topics. But are they *good* topics?

```
Topic 0: [running, exercise, fitness, morning, workout]
Topic 1: [meeting, project, deadline, client, review]
Topic 2: [day, thing, time, really, just]

Are these all equally good? Intuitively:
- Topics 0 and 1 seem coherent (words relate to each other)
- Topic 2 seems noisy (generic words with no clear theme)

How do we quantify this intuition?
```

**Quality evaluation** answers this question with rigorous metrics.

---

## Why Evaluate Quality?

### 1. Compare Topic Models

```
Model A (k=5):  Mean coherence = 0.42
Model B (k=10): Mean coherence = 0.38
Model C (k=15): Mean coherence = 0.29

Model A produces the most coherent topics.
(But fewer topics means less granularity—trade-offs exist)
```

### 2. Tune Hyperparameters

```
min_cluster_size = 5:  Coherence = 0.35, Diversity = 0.72
min_cluster_size = 10: Coherence = 0.41, Diversity = 0.85
min_cluster_size = 20: Coherence = 0.48, Diversity = 0.91

Larger clusters are more coherent but might miss niche topics.
```

### 3. Identify Problem Topics

```
Topic coherence scores:
  Topic 0: 0.52 ✓ Good
  Topic 1: 0.44 ✓ Good
  Topic 2: -0.12 ✗ Low coherence (investigate!)
  Topic 3: 0.48 ✓ Good

Topic 2 might need to be merged with another or filtered.
```

### 4. Validate Changes

```
Before adding bigrams: Mean coherence = 0.38
After adding bigrams:  Mean coherence = 0.45

Bigrams improved topic quality by 18%.
```

---

## The Metrics

### Coherence: Do Keywords Belong Together?

```
"running" and "exercise" → Often co-occur → High coherence
"running" and "deadline" → Rarely co-occur → Low coherence

Coherence = Average co-occurrence strength across keyword pairs
```

SwiftTopics uses **NPMI (Normalized Pointwise Mutual Information)**:
- Range: [-1, +1]
- +1: Perfect co-occurrence
- 0: Statistical independence
- -1: Never co-occur

### Diversity: Are Topics Distinct?

```
Topic 0: [running, exercise, fitness]
Topic 1: [running, workout, morning]
                ↑
         "running" appears in both → Reduces diversity

Diversity = Unique keywords / Total keywords
```

High diversity means topics have distinct vocabularies.

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Quality Evaluation Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  COHERENCE EVALUATION                                                   │
│  ────────────────────                                                   │
│                                                                         │
│  Documents      Tokenized       Co-occurrence     NPMI          Score  │
│  ─────────  →   Documents   →   Counts        →   Scoring   →   ────── │
│                                                                         │
│  ┌───────┐     ┌─────────┐     ┌─────────┐      ┌─────────┐    ┌─────┐ │
│  │ Doc 0 │     │ [run,   │     │ P(run)  │      │ NPMI    │    │0.42 │ │
│  │ Doc 1 │  →  │  fit,   │  →  │ P(fit)  │  →   │ for all │  → │0.38 │ │
│  │ Doc 2 │     │  gym]   │     │P(run,fit)│     │ pairs   │    │0.51 │ │
│  └───────┘     └─────────┘     └─────────┘      └─────────┘    └─────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DIVERSITY EVALUATION                                                   │
│  ────────────────────                                                   │
│                                                                         │
│  Topics          Keyword Sets        Unique Count        Ratio         │
│  ──────      →   ────────────   →    ───────────   →    ─────         │
│                                                                         │
│  ┌─────────┐     ┌─────────────┐     ┌──────────┐      ┌──────┐       │
│  │Topic 0  │     │{run,fit,gym}│     │ 27 unique│      │ 0.90 │       │
│  │Topic 1  │  →  │{meet,proj}  │  →  │ of 30    │  →   │      │       │
│  │Topic 2  │     │{stress,anx} │     │ total    │      │      │       │
│  └─────────┘     └─────────────┘     └──────────┘      └──────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What You'll Learn

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [5.1 Why Quality Matters](./01-Why-Quality-Matters.md) | Motivation for evaluation | Good vs. bad topics, use cases |
| [5.2 Coherence Metrics](./02-Coherence-Metrics.md) | Co-occurrence & PMI | Sliding windows, document-level counting |
| [5.3 NPMI Deep Dive](./03-NPMI-Deep-Dive.md) | Normalized PMI | The formula, interpretation, edge cases |
| [5.4 Diversity Metrics](./04-Diversity-Metrics.md) | Topic diversity | Redundancy, overlap, combined quality |

---

## The SwiftTopics Implementation

### Core Components

```
Sources/SwiftTopics/Evaluation/
├── CooccurrenceCounter.swift  ← Word pair counting
├── NPMIScorer.swift           ← NPMI computation
├── CoherenceEvaluator.swift   ← Topic coherence scoring
└── DiversityMetrics.swift     ← Topic diversity & redundancy
```

### Quick Start

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

let evaluator = NPMICoherenceEvaluator(configuration: .default)

let result = await evaluator.evaluate(
    topics: topics,
    documents: documents
)

print("Mean coherence: \(result.meanCoherence)")
print("Topics with positive coherence: \(result.positiveCoherenceCount)/\(result.topicCount)")

// Check diversity
let diversityResult = topics.evaluateDiversity()
print("Diversity: \(diversityResult.diversity)")
```

---

## Key Formulas

### NPMI (Normalized Pointwise Mutual Information)

```
PMI(w₁, w₂) = log(P(w₁, w₂) / (P(w₁) × P(w₂)))

NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))

Topic Coherence = mean(NPMI(wᵢ, wⱼ)) for all i < j
```

### Diversity

```
Diversity = |unique keywords| / |total keywords|

Redundancy(topic) = |keywords ∩ other_topics| / |keywords|
```

---

## Interpretation Guide

### Coherence Scores

```
Score Range     Quality        Action
──────────────────────────────────────────────────────
  > 0.3         High           Topics are meaningful
  0.1 - 0.3     Medium         Acceptable, room to improve
  0 - 0.1       Low            Consider parameter tuning
  < 0           Negative       Topic may be noise
```

### Diversity Scores

```
Score Range     Quality        Action
──────────────────────────────────────────────────────
  > 0.9         High           Topics are well-separated
  0.7 - 0.9     Medium         Some overlap, acceptable
  0.5 - 0.7     Low            Significant redundancy
  < 0.5         Very Low       Consider merging topics
```

---

## Configuration Options

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

public struct CoherenceConfiguration {
    /// Sliding window size for co-occurrence counting.
    public let windowSize: Int                    // Default: 10

    /// Use document-level counting instead of sliding window.
    public let useDocumentCooccurrence: Bool      // Default: false

    /// Number of top keywords to evaluate per topic.
    public let topKeywords: Int                   // Default: 10

    /// Smoothing epsilon for probability calculations.
    public let epsilon: Float                     // Default: 1e-12
}

// Pre-defined configurations
CoherenceConfiguration.default   // Sliding window of 10
CoherenceConfiguration.document  // Document-level counting
CoherenceConfiguration.semantic  // Larger window (50) for semantic relationships
CoherenceConfiguration.concise   // Fewer keywords (5) for faster evaluation
```

---

## Example Output

```swift
// Real output from a journal topic model:

Coherence Evaluation:
  Mean:   0.42
  Median: 0.45
  Min:    -0.08 (Topic 4 - needs investigation)
  Max:    0.61 (Topic 0 - very coherent)
  Std:    0.15

Per-Topic Coherence:
  Topic 0 (fitness):  0.61 ✓
  Topic 1 (work):     0.53 ✓
  Topic 2 (anxiety):  0.48 ✓
  Topic 3 (family):   0.41 ✓
  Topic 4 (misc):    -0.08 ✗

Diversity Evaluation:
  Diversity: 0.87
  Unique keywords: 43 of 50
  Mean redundancy: 0.13

Combined Quality Score: 0.62
```

---

## Prerequisites

Before starting this chapter, you should understand:

- ✅ How topics are represented with keywords (Chapter 4)
- ✅ Basic probability concepts (P(A), P(A,B))
- ✅ Logarithms and their properties

---

## 💡 Key Insight

Topic quality has two dimensions that must be balanced:

1. **Coherence**: Do keywords *within* a topic belong together?
2. **Diversity**: Are topics *distinct* from each other?

```
                    Coherence
                       ↑
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         │   IDEAL:    │   REDUNDANT:│
         │   High C,   │   High C,   │
         │   High D    │   Low D     │
         │             │             │
    ←────┼─────────────┼─────────────┼────→ Diversity
         │             │             │
         │   NOISY:    │   CHAOS:    │
         │   Low C,    │   Low C,    │
         │   High D    │   Low D     │
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ↓

Ideal topics: High coherence + High diversity
  → Keywords make sense together
  → Topics don't overlap
```

---

## Let's Begin

Ready to learn how to measure topic quality?

**[→ 5.1 Why Quality Matters](./01-Why-Quality-Matters.md)**

---

*Chapter 5 of 6 • SwiftTopics Learning Guide*
