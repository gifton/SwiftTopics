# 4.4 Keyword Diversification

> **Maximal Marginal Relevance—balancing relevance with coverage.**

---

## The Concept

After c-TF-IDF, you have ranked keywords. But the top keywords might be **redundant**:

```
Topic about fitness, top 5 keywords by c-TF-IDF:

1. running    (score: 1.00)
2. run        (score: 0.89)
3. runs       (score: 0.82)
4. runner     (score: 0.76)
5. jogging    (score: 0.71)

Problem: These are all variations of the same concept!
         We're not learning anything new after "running".
```

**Keyword diversification** ensures keywords cover different aspects of the topic:

```
After MMR diversification:

1. running    (kept: highest score)
2. exercise   (kept: different from "running")
3. morning    (kept: time aspect)
4. fitness    (kept: broader category)
5. workout    (kept: different activity)

Better! Now we see the topic from multiple angles.
```

---

## Why It Matters

### Redundancy Wastes Keyword Slots

```
You extract 10 keywords per topic.
If 4 are synonyms, you've wasted 30% of your slots.

With redundancy:
[running, run, jog, jogging, morning, fitness, ...]
└──────── 4 slots for same idea ────────┘

Without redundancy:
[running, morning, fitness, health, gym, training, ...]
└──────── 6 distinct concepts ────────────────────┘
```

### Users Need Comprehensive Understanding

```
A user looking at topic keywords wants to understand:
- What is this topic about? → "running"
- When does it occur? → "morning"
- What category? → "fitness"
- What activities? → "workout", "gym"
- What goals? → "health", "training"

Redundant keywords answer the same question repeatedly.
Diverse keywords answer DIFFERENT questions.
```

---

## The Mathematics

### Maximal Marginal Relevance (MMR)

MMR is a classic algorithm from information retrieval. It selects items that are:
- **Relevant** to the query (high score)
- **Novel** compared to already-selected items (low similarity)

```
MMR(item) = λ × relevance(item) - (1-λ) × max_similarity(item, selected)

Where:
  λ = weight balancing relevance vs. diversity (0 to 1)
  relevance = c-TF-IDF score (normalized)
  max_similarity = highest similarity to any already-selected keyword
```

### The Trade-off

```
λ = 1.0: Only relevance matters
         → Pure c-TF-IDF ranking (no diversification)

λ = 0.5: Balance relevance and diversity
         → Diverse but still relevant keywords

λ = 0.0: Only diversity matters
         → Random keywords that are different from each other

Typical λ: 0.3 to 0.7 (SwiftTopics default: diversityWeight = 0.3)
Note: SwiftTopics uses (1-λ) for relevance, so diversityWeight=0.3 means λ=0.3
```

---

## The Technique: MMR Algorithm

### Step-by-Step

```
Input: Keywords ranked by c-TF-IDF score
Output: Diversified keywords

1. SELECT the top keyword (highest c-TF-IDF)
   → This is always the most relevant term

2. FOR each remaining slot:
   a. FOR each candidate keyword not yet selected:
      - Compute relevance = c-TF-IDF score
      - Compute max_similarity to any selected keyword
      - Compute MMR = (1-λ)×relevance + λ×(1-max_similarity)

   b. SELECT the candidate with highest MMR

3. RETURN selected keywords in order
```

### Walkthrough Example

```
Keywords by c-TF-IDF:
  running    1.00
  run        0.89
  jogging    0.82
  exercise   0.78
  morning    0.71
  fitness    0.65

λ = 0.3 (diversity weight)

Step 1: Select "running" (top score)
Selected: [running]

Step 2: Evaluate candidates
  run:      relevance=0.89, sim_to_running=0.8
            MMR = 0.7×0.89 + 0.3×(1-0.8) = 0.623 + 0.06 = 0.68

  jogging:  relevance=0.82, sim_to_running=0.6
            MMR = 0.7×0.82 + 0.3×(1-0.6) = 0.574 + 0.12 = 0.69

  exercise: relevance=0.78, sim_to_running=0.3
            MMR = 0.7×0.78 + 0.3×(1-0.3) = 0.546 + 0.21 = 0.76  ← Winner!

  morning:  relevance=0.71, sim_to_running=0.1
            MMR = 0.7×0.71 + 0.3×(1-0.1) = 0.497 + 0.27 = 0.77  ← Winner!

Select "morning" (highest MMR)
Selected: [running, morning]

Step 3: Evaluate remaining candidates
  (Now must consider similarity to BOTH running AND morning)
  ...continue until enough keywords selected
```

---

## Similarity Measures

### String Similarity

SwiftTopics uses **character n-gram Jaccard similarity**:

```
For strings "running" and "jogging":

3-grams of "running": {run, unn, nni, nin, ing}
3-grams of "jogging": {jog, ogg, ggi, gin, ing}

Intersection: {ing}
Union: {run, unn, nni, nin, ing, jog, ogg, ggi, gin}

Jaccard = |intersection| / |union| = 1/9 ≈ 0.11

For "running" and "run":
3-grams of "run": {run}  (too short, use whole word)
3-grams of "running": {run, unn, nni, nin, ing}

Jaccard = 1/5 = 0.20

Higher similarity = more redundant.
```

### Why Character N-grams?

```
Alternative: Exact string matching
  "running" vs "run" → 0 similarity (different strings)
  Problem: Misses morphological variants

Alternative: Embedding-based similarity
  Compute cosine similarity of word embeddings
  Better semantic matching but requires embeddings
  More expensive to compute

Character n-grams:
  ✓ Catches morphological variants (run/running/runner)
  ✓ Fast to compute
  ✓ No external resources needed
  ✗ Misses semantic synonyms (jog/run)
```

---

## In SwiftTopics

### Configuration

```swift
// 📍 See: Sources/SwiftTopics/Protocols/TopicRepresenter.swift

public struct CTFIDFConfiguration: RepresentationConfiguration {

    /// Whether to apply MMR diversification to keywords.
    public let diversify: Bool             // Default: false

    /// Diversity weight for MMR (0 = only relevance, 1 = only diversity).
    public let diversityWeight: Float      // Default: 0.3
}

// Enable diversification
let config = CTFIDFConfiguration(
    keywordsPerTopic: 10,
    diversify: true,
    diversityWeight: 0.3
)
```

### The Diversification Implementation

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

/// Diversifies keywords using Maximal Marginal Relevance.
private func diversifyKeywords(
    _ keywords: [TopicKeyword],
    count: Int,
    weight: Float  // diversity weight (λ in the formula)
) -> [TopicKeyword] {
    guard keywords.count > 1 else { return keywords }

    var selected = [TopicKeyword]()
    var remaining = keywords

    // Always select the top keyword first
    selected.append(remaining.removeFirst())

    while selected.count < count && !remaining.isEmpty {
        var bestIdx = 0
        var bestScore: Float = -.infinity

        for (idx, candidate) in remaining.enumerated() {
            // Relevance component (normalized c-TF-IDF score)
            let relevance = candidate.score

            // Diversity component (1 - max similarity to selected)
            let maxSimilarity = selected.map { selected in
                stringSimilarity(candidate.term, selected.term)
            }.max() ?? 0

            let diversity = 1.0 - maxSimilarity

            // MMR score: balance relevance and diversity
            let mmrScore = (1.0 - weight) * relevance + weight * diversity

            if mmrScore > bestScore {
                bestScore = mmrScore
                bestIdx = idx
            }
        }

        selected.append(remaining.remove(at: bestIdx))
    }

    return selected
}
```

### String Similarity Helper

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

/// Computes string similarity using Jaccard on character n-grams.
private func stringSimilarity(_ a: String, _ b: String) -> Float {
    let ngramSize = 3

    func ngrams(_ s: String) -> Set<String> {
        guard s.count >= ngramSize else { return [s] }
        var result = Set<String>()
        let chars = Array(s)
        for i in 0...(chars.count - ngramSize) {
            result.insert(String(chars[i..<(i + ngramSize)]))
        }
        return result
    }

    let aNgrams = ngrams(a)
    let bNgrams = ngrams(b)

    let intersection = aNgrams.intersection(bNgrams).count
    let union = aNgrams.union(bNgrams).count

    guard union > 0 else { return 0 }
    return Float(intersection) / Float(union)
}
```

---

## Visualizing MMR Selection

```
Initial c-TF-IDF ranking:
┌─────────────────────────────────────────────────────────┐
│ Rank  Term       Score   Similarity to Selected        │
├─────────────────────────────────────────────────────────┤
│  1    running    1.00    (none yet)          → SELECT  │
│  2    run        0.89    0.20 to "running"             │
│  3    jogging    0.82    0.11 to "running"             │
│  4    exercise   0.78    0.05 to "running"             │
│  5    morning    0.71    0.11 to "running"             │
│  6    fitness    0.65    0.00 to "running"             │
└─────────────────────────────────────────────────────────┘

After selecting "running", compute MMR for remaining:
λ = 0.3

┌──────────────────────────────────────────────────────────────────┐
│ Term       Relevance  MaxSim  Diversity  MMR = 0.7×R + 0.3×D    │
├──────────────────────────────────────────────────────────────────┤
│ run        0.89       0.20    0.80       0.623 + 0.24 = 0.86    │
│ jogging    0.82       0.11    0.89       0.574 + 0.27 = 0.84    │
│ exercise   0.78       0.05    0.95       0.546 + 0.29 = 0.83    │
│ morning    0.71       0.11    0.89       0.497 + 0.27 = 0.77    │
│ fitness    0.65       0.00    1.00       0.455 + 0.30 = 0.76    │
└──────────────────────────────────────────────────────────────────┘

"run" still wins (high relevance outweighs similarity penalty).
But as more keywords are selected, diverse terms catch up.
```

---

## When to Use Diversification

### Use Diversification When:

```
✓ Topics have morphological variants
  "run", "running", "runner", "runs"

✓ You want comprehensive coverage
  Users should understand the topic from 10 keywords

✓ Keywords will be displayed to users
  Redundant keywords look unprofessional

✓ You're generating topic summaries
  Diverse keywords make better summaries
```

### Skip Diversification When:

```
✗ You need pure relevance ranking
  For search results, most relevant first

✗ Processing speed is critical
  MMR adds O(k²) comparisons where k = keywords

✗ Keywords are for internal use only
  Machine learning doesn't mind redundancy

✗ Very few keywords (< 5)
  Little benefit from diversification
```

---

## Tuning the Diversity Weight

### Low Weight (0.1 - 0.3)

```
diversityWeight = 0.2

Strong preference for relevance.
Diversity is a tiebreaker, not a major factor.

Result: Mostly c-TF-IDF order, slight reranking.
[running, run, jogging, exercise, morning, fitness]
         ↑    ↑
      Similar terms still near the top
```

### Medium Weight (0.3 - 0.5)

```
diversityWeight = 0.4

Balanced relevance and diversity.
Good for most use cases.

Result: Mix of high-scoring and diverse terms.
[running, exercise, morning, run, fitness, jogging]
              ↑       ↑
         Diverse terms promoted
```

### High Weight (0.5 - 0.7)

```
diversityWeight = 0.6

Strong preference for diversity.
Risk: May include less relevant terms.

Result: Maximally diverse keywords.
[running, morning, fitness, exercise, gym, health]
                                           ↑
                    Lower-relevance but unique term included
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Forgetting to Enable Diversification

```swift
// ❌ Default config has diversify = false
let config = CTFIDFConfiguration.default
// Keywords may be redundant!

// ✅ Explicitly enable diversification
let config = CTFIDFConfiguration(
    keywordsPerTopic: 10,
    diversify: true,
    diversityWeight: 0.3
)
```

### Pitfall 2: Too High Diversity Weight

```swift
// ⚠️ diversity > 0.7 can include irrelevant terms
let config = CTFIDFConfiguration(
    diversify: true,
    diversityWeight: 0.8  // Too aggressive!
)

// Keywords might include terms with low c-TF-IDF scores
// just because they're "different"
```

### Pitfall 3: Expecting Semantic Diversity

```swift
// ⚠️ Character n-grams don't catch semantic synonyms
// "jog" and "run" have low character similarity (0.0)
// They will BOTH be selected despite being synonyms

// For semantic diversity, you'd need:
// - Word embeddings
// - Synonym dictionaries
// - More sophisticated similarity measures
```

### Pitfall 4: Not Enough Keywords to Diversify

```swift
// ⚠️ With only 3 keywords, diversification has minimal effect
let config = CTFIDFConfiguration(
    keywordsPerTopic: 3,
    diversify: true
)

// Top 3 are likely to be selected regardless
// Diversification matters more for 10+ keywords
```

---

## Alternative Diversification Strategies

### Embedding-Based Similarity

```swift
// Instead of character n-grams, use word embeddings:
func embeddingSimilarity(_ a: String, _ b: String) -> Float {
    let embA = wordEmbedding(a)
    let embB = wordEmbedding(b)
    return cosineSimilarity(embA, embB)
}

// Pros: Catches semantic similarity ("run" ≈ "jog")
// Cons: Requires word embeddings, slower
```

### Submodular Optimization

```swift
// More sophisticated than greedy MMR:
// Optimize a submodular function that naturally
// balances relevance and coverage.

// Beyond scope for SwiftTopics, but used in
// advanced summarization systems.
```

### Topic Modeling Constraint

```swift
// Ensure keywords come from different "sub-topics":
// 1. Cluster keywords within the topic
// 2. Select one keyword per sub-cluster

// Example: Fitness topic has sub-clusters:
//   {running, jogging, marathon}
//   {gym, weights, training}
//   {morning, routine, schedule}
// Select: [running, gym, morning]
```

---

## Key Takeaways

1. **MMR balances relevance and diversity**: (1-λ)×relevance + λ×diversity.

2. **First keyword is always the top scorer**: Diversification starts from slot 2.

3. **Character n-grams catch morphological variants**: "run" ≈ "running".

4. **Diversity weight is tunable**: 0.3-0.5 works for most cases.

5. **Diversification improves human comprehension**: Distinct keywords tell a fuller story.

6. **Trade-off exists**: Too much diversity may include less relevant terms.

---

## 💡 Key Insight

Diversification asks: **"Given what I've already selected, what NEW information does this keyword add?"**

A word can be highly relevant but LOW in marginal value if it's redundant with previous selections. MMR formalizes this intuition:

```
Marginal Value = Relevance - Redundancy

         ┌─────────────────────────────────────────┐
         │                                         │
         │  "running" selected first (most relevant)
         │         ↓                               │
         │  "run" is relevant but redundant        │
         │         ↓                               │
         │  "morning" is less relevant but NOVEL   │
         │         ↓                               │
         │  "morning" wins! Higher marginal value  │
         │                                         │
         └─────────────────────────────────────────┘
```

---

## Chapter Summary

You've learned how to extract human-readable keywords from document clusters:

1. **From Clusters to Topics**: Clusters are geometric; topics need labels.

2. **TF-IDF Basics**: Balance frequency (here) with rarity (elsewhere).

3. **c-TF-IDF Innovation**: Treat clusters as documents for topic-level scoring.

4. **MMR Diversification**: Ensure keywords cover different aspects.

The result: given a cluster of similar documents, you can now produce a ranked list of diverse, distinctive keywords that characterize the topic.

---

## Next Up

We have topics with keywords. But how do we know if they're *good* topics? Next, we'll learn how to **evaluate topic quality** using coherence metrics.

**[→ Chapter 5: Quality Evaluation](../05-Quality-Evaluation/README.md)**

---

*Guide 4.4 of 4.4 • Chapter 4: Topic Representation*
