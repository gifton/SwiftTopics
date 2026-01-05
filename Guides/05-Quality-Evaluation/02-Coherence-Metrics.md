# 5.2 Coherence Metrics

> **Measuring whether keywords belong together using co-occurrence.**

---

## The Concept

**Coherence** measures whether a topic's keywords are semantically related. The key insight:

```
Words that are semantically related tend to appear together in text.

"running" and "exercise" → Related → Often co-occur
"running" and "database" → Unrelated → Rarely co-occur

If a topic's keywords frequently co-occur in the corpus,
the topic is likely capturing a real semantic concept.
```

This leads to a measurable definition:

```
Coherence = How strongly do keyword pairs co-occur?
```

---

## Why Co-occurrence?

### The Distributional Hypothesis

Linguists have observed that words appearing in similar contexts have similar meanings:

```
"You shall know a word by the company it keeps."
    — J.R. Firth, 1957

Context of "running":
  "I went running this morning"
  "Running improves cardiovascular health"
  "My running shoes are worn out"

Context of "exercise":
  "Daily exercise is important"
  "Exercise reduces stress"
  "I need new exercise shoes"

Both words appear with: "morning", "health", "shoes", "daily"
→ They're semantically related
→ They'll have high co-occurrence in a large corpus
```

### From Intuition to Measurement

```
Given topic keywords: [running, exercise, fitness, workout]

Count co-occurrences in the corpus:
  running + exercise:  847 windows
  running + fitness:   623 windows
  running + workout:   712 windows
  exercise + fitness:  891 windows
  exercise + workout:  756 windows
  fitness + workout:   802 windows

All pairs co-occur frequently → Topic is coherent!

Compare with: [running, meeting, anxiety, coffee]

  running + meeting:   12 windows
  running + anxiety:    8 windows
  running + coffee:    15 windows
  meeting + anxiety:   34 windows
  meeting + coffee:    89 windows  ← These might co-occur (coffee at meetings)
  anxiety + coffee:    28 windows

Most pairs rarely co-occur → Topic is incoherent!
```

---

## Co-occurrence Counting

### Two Approaches

SwiftTopics supports two methods for counting co-occurrences:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CO-OCCURRENCE MODES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SLIDING WINDOW (default)                                           │
│  ────────────────────────                                           │
│                                                                     │
│  Document: "I went running this morning for exercise"               │
│                                                                     │
│  Window 1: [I, went, running, this, morning]                        │
│  Window 2: [went, running, this, morning, for]                      │
│  Window 3: [running, this, morning, for, exercise]                  │
│                     ↓                       ↓                       │
│            "running" and "exercise" co-occur in Window 3            │
│                                                                     │
│  Pros: Captures local context (words near each other)               │
│  Cons: May miss long-range dependencies                             │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DOCUMENT-LEVEL                                                     │
│  ──────────────                                                     │
│                                                                     │
│  Document: "I went running this morning for exercise"               │
│                                                                     │
│  All words co-occur: (running, exercise), (running, morning), ...   │
│                                                                     │
│  Pros: Simpler, captures document-level themes                      │
│  Cons: Loses locality (words at opposite ends still "co-occur")     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Sliding Window Algorithm

```
Input: Tokenized document, window size W
Output: Word and pair counts

Document tokens: [w₀, w₁, w₂, w₃, w₄, w₅, w₆, ...]

For each position i from 0 to len(tokens) - W:
    window = tokens[i : i + W]

    For each unique word in window:
        wordCount[word] += 1

    For each unique pair (w_a, w_b) in window where a < b:
        pairCount[(w_a, w_b)] += 1

    totalWindows += 1
```

Example with window size 5:

```
Document: "Had a great run this morning felt good after exercise"
Tokens:   [had, great, run, morning, felt, good, after, exercise]
          (stop words removed)

Window 0: [had, great, run, morning, felt]
  Words: had, great, run, morning, felt (each counted once)
  Pairs: (great, run), (great, morning), (run, morning), ...

Window 1: [great, run, morning, felt, good]
  Words: great, run, morning, felt, good
  Pairs: (great, run), (run, felt), (morning, good), ...

Window 2: [run, morning, felt, good, after]
  ...and so on

Final counts:
  wordCount["run"] = 3 (appears in windows 0, 1, 2)
  wordCount["exercise"] = 1 (appears in window 3 only)
  pairCount[("run", "morning")] = 2 (windows 0, 1)
  pairCount[("run", "exercise")] = 0 (never in same window)
```

---

## The Mathematics of Co-occurrence

### Probabilities from Counts

Given co-occurrence counts, we compute probabilities:

```
P(word) = windowsContainingWord / totalWindows

P(word₁, word₂) = windowsContainingBoth / totalWindows

Example:
  Total windows: 10,000
  Windows with "running": 500
  Windows with "exercise": 400
  Windows with both: 150

  P(running) = 500 / 10,000 = 0.05
  P(exercise) = 400 / 10,000 = 0.04
  P(running, exercise) = 150 / 10,000 = 0.015
```

### Expected Co-occurrence

If words were independent, we'd expect:

```
P_expected(running, exercise) = P(running) × P(exercise)
                              = 0.05 × 0.04
                              = 0.002

But we observed:
P_observed(running, exercise) = 0.015

Observed > Expected → Words co-occur MORE than chance
                   → They're likely related!
```

### Pointwise Mutual Information (PMI)

PMI quantifies this relationship:

```
PMI(w₁, w₂) = log(P(w₁, w₂) / (P(w₁) × P(w₂)))

For our example:
PMI(running, exercise) = log(0.015 / 0.002)
                       = log(7.5)
                       ≈ 2.01

Interpretation:
  PMI > 0: Words co-occur more than expected (positive association)
  PMI = 0: Words are independent (no association)
  PMI < 0: Words co-occur less than expected (negative association)
```

---

## In SwiftTopics

### The CooccurrenceCounter

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift

/// Counts word and word-pair co-occurrences in a corpus.
public struct CooccurrenceCounter: Sendable {

    /// The co-occurrence mode.
    public let mode: CooccurrenceMode

    /// Creates a co-occurrence counter.
    public init(mode: CooccurrenceMode = .default)

    /// Counts co-occurrences from tokenized documents.
    public func count(tokenizedDocuments: [[String]]) -> CooccurrenceCounts
}
```

### The CooccurrenceMode

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift

/// Mode for counting word co-occurrences.
public enum CooccurrenceMode: Sendable, Codable, Hashable {

    /// Sliding window mode: count pairs within a fixed-size window.
    case slidingWindow(size: Int)

    /// Document mode: count pairs that appear in the same document.
    case document

    /// Default mode: sliding window of 10 tokens.
    public static let `default`: CooccurrenceMode = .slidingWindow(size: 10)
}
```

### The CooccurrenceCounts

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift

/// Counts of word and word-pair occurrences in a corpus.
public struct CooccurrenceCounts: Sendable {

    /// Number of windows containing each word.
    public let wordCounts: [String: Int]

    /// Number of windows containing each word pair.
    public let pairCounts: [WordPair: Int]

    /// Total number of windows counted.
    public let totalWindows: Int

    /// Gets the count for a word.
    public func count(for word: String) -> Int

    /// Gets the count for a word pair.
    public func count(for word1: String, _ word2: String) -> Int
}
```

### The WordPair

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CooccurrenceCounter.swift

/// A pair of words for co-occurrence counting.
/// Words are stored lexicographically to ensure symmetry.
public struct WordPair: Sendable, Hashable {

    /// First word (lexicographically smaller).
    public let word1: String

    /// Second word (lexicographically larger).
    public let word2: String

    /// Creates a word pair, ordering words lexicographically.
    public init(_ a: String, _ b: String) {
        if a <= b {
            self.word1 = a
            self.word2 = b
        } else {
            self.word1 = b
            self.word2 = a
        }
    }
}
```

### Usage Example

```swift
// Count co-occurrences with sliding window
let counter = CooccurrenceCounter(mode: .slidingWindow(size: 10))
let counts = counter.count(tokenizedDocuments: tokenizedDocs)

print("Vocabulary size: \(counts.vocabularySize)")
print("Total windows: \(counts.totalWindows)")

// Query specific counts
let runningCount = counts.count(for: "running")
let pairCount = counts.count(for: "running", "exercise")

print("'running' appears in \(runningCount) windows")
print("'running' + 'exercise' co-occur in \(pairCount) windows")
```

---

## Window Size Selection

### Small Windows (5-10 tokens)

```
Captures: Local context, syntactic relationships

"I went running yesterday morning"
           ↑        ↑      ↑
     These words are syntactically close.

Good for: Technical coherence, precise associations
Risk: May miss thematic relationships spanning paragraphs
```

### Large Windows (20-50 tokens)

```
Captures: Thematic relationships, semantic associations

"I love running. The morning air is fresh.
 Exercise makes me feel energized."

"running" and "exercise" are in same paragraph
→ Thematically related even if not syntactically adjacent

Good for: Semantic coherence, topical associations
Risk: May include spurious co-occurrences
```

### Document-Level

```
Captures: Document themes, broad associations

Entire document = one window.

Good for: Thematic topics, when documents are short
Risk: Loses locality; unrelated words may "co-occur"
```

### SwiftTopics Configurations

```swift
// 📍 See: Sources/SwiftTopics/Evaluation/CoherenceEvaluator.swift

// Default: Sliding window of 10 (balanced)
CoherenceConfiguration.default

// Document-level counting
CoherenceConfiguration.document

// Larger window for semantic relationships
CoherenceConfiguration.semantic  // windowSize: 50

// Faster evaluation with fewer keywords
CoherenceConfiguration.concise   // topKeywords: 5
```

---

## Coherence Formulas

### Topic Coherence

Given a topic's keywords [w₁, w₂, ..., wₖ], coherence is typically computed as:

```
Coherence = (2 / (k × (k-1))) × Σ score(wᵢ, wⱼ)  for all i < j

Where:
  k = number of keywords
  score(wᵢ, wⱼ) = some measure of association (e.g., PMI, NPMI)

Example with 5 keywords:
  Number of pairs = 5 × 4 / 2 = 10
  Coherence = average score across 10 pairs
```

### Why Pairs?

```
We evaluate ALL pairs, not just adjacent keywords:

Keywords: [running, exercise, fitness, morning, workout]

Pairs evaluated:
  (running, exercise)   ←
  (running, fitness)    ←
  (running, morning)    ← These aren't adjacent in the keyword list
  (running, workout)    ← but we still check their co-occurrence
  (exercise, fitness)
  (exercise, morning)
  (exercise, workout)
  (fitness, morning)
  (fitness, workout)
  (morning, workout)

This is thorough: a coherent topic should have
ALL keywords co-occurring, not just some.
```

---

## From PMI to NPMI

### The Problem with Raw PMI

```
PMI has unbounded range:

PMI(w₁, w₂) ∈ (-∞, +∞)

  - Very rare pairs: PMI can be very negative
  - Very common pairs: PMI can be very positive
  - Hard to interpret: Is PMI = 3.5 good? What about 2.1?

Also, PMI is sensitive to rare events:

If P(w₁, w₂) = 0.0001 (very rare co-occurrence):
  PMI might be very high just because the words are rare,
  not because they're strongly associated.
```

### Normalized PMI (NPMI)

```
NPMI normalizes PMI to [-1, +1]:

NPMI(w₁, w₂) = PMI(w₁, w₂) / -log(P(w₁, w₂))

Properties:
  NPMI = +1: Perfect positive association (always co-occur)
  NPMI = 0:  Statistical independence
  NPMI = -1: Perfect negative association (never co-occur)

Much easier to interpret!
```

We'll dive deep into NPMI in the next guide.

---

## Visualizing Coherence

```
High-Coherence Topic: [running, exercise, fitness, workout]

Co-occurrence matrix (imagined):

              running  exercise  fitness  workout
    running      -       0.7       0.6      0.8
    exercise   0.7        -        0.8      0.7
    fitness    0.6       0.8        -       0.6
    workout    0.8       0.7       0.6       -

All pairs have high scores → Coherent topic!

──────────────────────────────────────────────────

Low-Coherence Topic: [running, meeting, anxiety, coffee]

              running  meeting  anxiety  coffee
    running      -       -0.2      0.0     -0.1
    meeting   -0.2        -        0.2      0.5
    anxiety    0.0       0.2        -       0.1
    coffee    -0.1       0.5       0.1       -

Most pairs have low/negative scores → Incoherent topic!
(meeting + coffee co-occur somewhat, but others don't)
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Too Small Window Size

```swift
// ⚠️ Window of 3 is very small
let config = CoherenceConfiguration(windowSize: 3)

// Problem: Words must be immediately adjacent to co-occur
// "I went running yesterday morning"
// "running" and "morning" won't co-occur (too far apart)

// ✅ Use at least 10 for reasonable context
let config = CoherenceConfiguration(windowSize: 10)
```

### Pitfall 2: Not Tokenizing Consistently

```swift
// ⚠️ Topic keywords: ["running", "exercise"]
// But documents tokenized differently: ["Running", "Exercise"]
// → No matches found!

// ✅ Ensure consistent lowercasing
let tokenizer = Tokenizer(configuration: .english)
// Both topic extraction and coherence evaluation should use same tokenizer
```

### Pitfall 3: Stop Words in Keywords

```swift
// ⚠️ If stop words slip into keywords:
// Keywords: ["the", "running", "and", "exercise"]
// "the" and "and" will inflate co-occurrence (appear everywhere)

// ✅ Ensure stop words are filtered during keyword extraction
// and during co-occurrence counting
```

### Pitfall 4: Very Small Corpus

```swift
// ⚠️ With only 50 documents:
// Co-occurrence counts are noisy
// PMI/NPMI may have high variance

// ✅ For reliable coherence:
// - Aim for 1000+ documents
// - Or use larger window sizes to capture more context
// - Consider smoothing (epsilon in NPMI)
```

---

## Key Takeaways

1. **Coherence = co-occurrence strength**: Words that belong together appear together.

2. **Sliding window captures locality**: Words near each other in text are more related.

3. **Document-level captures themes**: Same document = thematically related.

4. **PMI quantifies association**: log(observed / expected).

5. **NPMI normalizes to [-1, +1]**: Easier to interpret and compare.

6. **Window size matters**: Small = syntactic, large = semantic.

---

## 💡 Key Insight

Coherence evaluation exploits a fundamental property of natural language:

```
Meaning lives in context.

If two words have similar meanings, they:
  - Appear in similar contexts
  - Are used to discuss similar topics
  - Co-occur in documents about those topics

Topic coherence asks:
"Do the words we extracted tend to appear together
 in the very documents we extracted them from?"

If yes → The topic is capturing a real semantic pattern.
If no  → The topic is likely clustering noise.
```

This is why co-occurrence works: it's a proxy for semantic similarity that doesn't require any external knowledge—just the corpus itself.

---

## Next Up

Now let's dive deep into NPMI—the specific coherence measure used by SwiftTopics.

**[→ 5.3 NPMI Deep Dive](./03-NPMI-Deep-Dive.md)**

---

*Guide 5.2 of 5.4 • Chapter 5: Quality Evaluation*
