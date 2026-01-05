# 4.2 TF-IDF Refresher

> **The classic algorithm for finding important words in documents.**

---

## The Concept

**TF-IDF** (Term Frequency–Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document within a collection.

The core insight:

```
Important words are:
  ✓ Frequent in this document (high TF)
  ✓ Rare across all documents (high IDF)

Unimportant words are:
  ✗ Rare in this document (low TF), OR
  ✗ Common everywhere (low IDF)
```

---

## Why It Matters

TF-IDF is foundational to information retrieval:

- **Search engines** use it to rank documents by query relevance
- **Document similarity** is often computed using TF-IDF vectors
- **Feature extraction** for machine learning on text
- **Keyword extraction** — the basis for our topic representation

Understanding TF-IDF is essential to understanding c-TF-IDF.

---

## The Mathematics

### Term Frequency (TF)

How often does this term appear in this document?

```
TF(t, d) = count of term t in document d

Document: "The quick brown fox jumps over the lazy dog"

TF("the", doc) = 2
TF("quick", doc) = 1
TF("fox", doc) = 1
TF("cat", doc) = 0
```

Often normalized to prevent bias toward longer documents:

```
TF_normalized(t, d) = TF(t, d) / total_terms_in_d

Document has 9 terms:
TF_normalized("the", doc) = 2/9 ≈ 0.22
TF_normalized("quick", doc) = 1/9 ≈ 0.11
```

### Inverse Document Frequency (IDF)

How rare is this term across all documents?

```
IDF(t) = log(N / df(t))

Where:
  N = total number of documents
  df(t) = number of documents containing term t
```

Example with 1000 documents:

```
"the" appears in 950 documents:
  IDF("the") = log(1000 / 950) ≈ 0.02   (very common → low IDF)

"algorithm" appears in 50 documents:
  IDF("algorithm") = log(1000 / 50) ≈ 3.0  (rarer → higher IDF)

"quantum" appears in 5 documents:
  IDF("quantum") = log(1000 / 5) ≈ 5.3   (rare → high IDF)
```

### TF-IDF Score

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

For a document about algorithms:
  TF("algorithm", doc) = 5
  IDF("algorithm") = 3.0
  TF-IDF = 5 × 3.0 = 15.0  ← High score (important term)

  TF("the", doc) = 8
  IDF("the") = 0.02
  TF-IDF = 8 × 0.02 = 0.16  ← Low score (unimportant term)
```

---

## Visual Intuition

```
                          Document Frequency
                          ─────────────────►
                     Low (rare)    |    High (common)
                          |        |        |
                    ┌─────────────────────────────┐
                    │                             │
     High (frequent)│  ★★★        │    ●●●       │
                    │  Important! │   Possibly   │
       Term         │  These are  │   stop words │
    Frequency       │  the words  │              │
        │           │  we want    │              │
        │           ├─────────────┼──────────────┤
        │           │             │              │
        ▼           │    ○○○      │    ○○○       │
     Low (rare)     │   Too rare  │  Common but  │
                    │   to matter │  not here    │
                    │             │              │
                    └─────────────────────────────┘

Legend:
  ★★★ = High TF-IDF (important to this document)
  ●●● = Low TF-IDF (common everywhere)
  ○○○ = Low TF-IDF (not significant)
```

---

## The Technique: Computing TF-IDF

### Step 1: Build Document-Term Matrix

```
Documents:
  D0: "I love running in the morning"
  D1: "Running makes me happy"
  D2: "Morning coffee is essential"
  D3: "I love coffee"

Term Frequency Matrix:
         running  morning  love  coffee  happy  essential  makes
    D0      1        1       1      0       0        0        0
    D1      1        0       0      0       1        0        1
    D2      0        1       0      1       0        1        0
    D3      0        0       1      1       0        0        0
```

### Step 2: Compute IDF for Each Term

```
N = 4 documents

Term        Documents Containing    IDF = log(N/df)
─────────────────────────────────────────────────────
running     2                       log(4/2) = 0.69
morning     2                       log(4/2) = 0.69
love        2                       log(4/2) = 0.69
coffee      2                       log(4/2) = 0.69
happy       1                       log(4/1) = 1.39
essential   1                       log(4/1) = 1.39
makes       1                       log(4/1) = 1.39
```

### Step 3: Compute TF-IDF Matrix

```
TF-IDF = TF × IDF

         running  morning  love  coffee  happy  essential  makes
    D0    0.69     0.69    0.69   0.00    0.00    0.00      0.00
    D1    0.69     0.00    0.00   0.00    1.39    0.00      1.39
    D2    0.00     0.69    0.00   0.69    0.00    1.39      0.00
    D3    0.00     0.00    0.69   0.69    0.00    0.00      0.00

Top term per document:
  D0: running, morning, love (tied at 0.69)
  D1: happy, makes (1.39) ← unique terms get highest scores
  D2: essential (1.39)
  D3: love, coffee (tied at 0.69)
```

---

## TF-IDF Variants

### Log-Normalized TF

Reduces the impact of high term frequencies:

```
TF_log(t, d) = 1 + log(TF(t, d))  if TF > 0
             = 0                   if TF = 0

If "running" appears 10 times:
  Raw TF: 10
  Log TF: 1 + log(10) ≈ 3.3

This prevents a word appearing 10x from being 10x more important.
```

### Smoothed IDF

Prevents division by zero and smooths rare terms:

```
IDF_smooth(t) = log(1 + N / df(t))

Or with +1 smoothing:
IDF_smooth(t) = log((N + 1) / (df(t) + 1))

This is closer to what c-TF-IDF uses.
```

### BM25

A sophisticated variant used in search engines:

```
BM25(t, d) = IDF(t) × [TF(t,d) × (k₁ + 1)] / [TF(t,d) + k₁ × (1 - b + b × |d|/avgdl)]

Where:
  k₁ = term frequency saturation parameter (typically 1.2-2.0)
  b = length normalization (typically 0.75)
  |d| = document length
  avgdl = average document length

More complex but better for search ranking.
```

---

## TF-IDF for Document Representation

### As a Vector Space Model

```
Each document becomes a vector of TF-IDF scores:

D0 = [0.69, 0.69, 0.69, 0.00, 0.00, 0.00, 0.00]
D1 = [0.69, 0.00, 0.00, 0.00, 1.39, 0.00, 1.39]
D2 = [0.00, 0.69, 0.00, 0.69, 0.00, 1.39, 0.00]
D3 = [0.00, 0.00, 0.69, 0.69, 0.00, 0.00, 0.00]

Now documents are points in term-space.
We can compute similarity using cosine distance.
```

### Similarity Between Documents

```
cosine_similarity(D0, D2) = (D0 · D2) / (||D0|| × ||D2||)

D0 · D2 = 0.69×0 + 0.69×0.69 + ... = 0.48
||D0|| = √(0.69² + 0.69² + 0.69²) = 1.19
||D2|| = √(0.69² + 0.69² + 1.39²) = 1.69

cosine_sim = 0.48 / (1.19 × 1.69) ≈ 0.24

D0 and D2 share "morning" → some similarity.
```

---

## Limitations of TF-IDF

### Problem 1: No Semantic Understanding

```
"dog" and "canine" are treated as completely different terms.

Document about dogs: TF-IDF("dog") = high
                     TF-IDF("canine") = 0 (if not mentioned)

TF-IDF doesn't know they're synonyms.
```

### Problem 2: No Word Order

```
"Dog bites man" and "Man bites dog" have identical TF-IDF vectors.

Both contain: {dog: 1, bites: 1, man: 1}

The meaning is completely different!
```

### Problem 3: Sparse Vectors

```
Vocabulary size: 50,000 terms
Document vector: [0, 0, 0, 0.8, 0, 0, 0.3, 0, ..., 0]
                 └── Mostly zeros ──────────────────┘

99%+ of entries are zero.
Storage and computation inefficient.
```

This is why embeddings (Chapter 1) are preferred for modern NLP—they provide dense, semantic representations.

---

## TF-IDF vs. Embeddings

```
┌────────────────────────────────────────────────────────────────────┐
│                    TF-IDF vs. Embeddings                           │
├────────────────────┬───────────────────────────────────────────────┤
│      TF-IDF        │              Embeddings                       │
├────────────────────┼───────────────────────────────────────────────┤
│ Sparse vectors     │ Dense vectors                                 │
│ (vocabulary-size)  │ (fixed size, e.g., 768)                       │
├────────────────────┼───────────────────────────────────────────────┤
│ Bag-of-words       │ Semantic meaning                              │
│ (no word order)    │ (captures context)                            │
├────────────────────┼───────────────────────────────────────────────┤
│ Interpretable      │ Not directly interpretable                    │
│ (each dim = term)  │ (learned dimensions)                          │
├────────────────────┼───────────────────────────────────────────────┤
│ Fast to compute    │ Requires neural network                       │
├────────────────────┼───────────────────────────────────────────────┤
│ Good for keywords  │ Good for similarity                           │
└────────────────────┴───────────────────────────────────────────────┘

SwiftTopics uses BOTH:
  - Embeddings for clustering (Chapter 1)
  - TF-IDF concepts for keywords (this chapter)
```

---

## In SwiftTopics

### The Vocabulary

```swift
// 📍 See: Sources/SwiftTopics/Representation/Vocabulary.swift

/// A vocabulary mapping terms to indices with frequency statistics.
public struct Vocabulary: Sendable {

    /// Number of unique terms in the vocabulary.
    public var size: Int

    /// Gets the index for a term.
    public func index(for term: String) -> Int?

    /// Gets the document frequency for a term.
    public func documentFrequency(for term: String) -> Int

    /// Gets the total frequency for a term.
    public func totalFrequency(for term: String) -> Int

    /// Gets the document frequency ratio for a term.
    public func documentFrequencyRatio(for term: String) -> Float
}
```

### Building a Vocabulary

```swift
// 📍 See: Sources/SwiftTopics/Representation/Vocabulary.swift

let config = VocabularyConfiguration(
    minDocumentFrequency: 2,       // Terms must appear in ≥2 docs
    maxDocumentFrequencyRatio: 0.9 // Terms can't appear in >90% of docs
)

let builder = VocabularyBuilder(configuration: config)
let vocabulary = builder.build(from: tokenizedDocuments)

print("Vocabulary size: \(vocabulary.size)")
// Output: "Vocabulary size: 1847"
```

### Document Frequency Filtering

```swift
// 📍 See: Sources/SwiftTopics/Representation/Vocabulary.swift

// minDocumentFrequency filters rare terms (typos, names)
// maxDocumentFrequencyRatio filters common terms (like stop words)

let vocabConfig = VocabularyConfiguration(
    minDocumentFrequency: 5,        // Must appear in at least 5 docs
    maxDocumentFrequencyRatio: 0.95 // Can't appear in >95% of docs
)

// These filters perform a similar role to IDF:
// - Rare terms (low df) are excluded entirely
// - Common terms (high df) are excluded entirely
// - The middle ground remains for scoring
```

---

## Connecting to c-TF-IDF

Traditional TF-IDF compares **documents**:
- TF: How often in THIS document?
- IDF: How rare across ALL documents?

c-TF-IDF compares **clusters**:
- TF: How often in THIS cluster?
- IDF: How rare across ALL clusters?

```
TF-IDF (documents):                 c-TF-IDF (clusters):
───────────────────                 ─────────────────────
Document 0: "running..."            Cluster 0 (all docs):
Document 1: "meeting..."              "running morning 5k..."
Document 2: "anxiety..."            Cluster 1 (all docs):
                                      "meeting project deadline..."

TF = freq in document               TF = freq in cluster
IDF = log(N_docs / df)              "IDF" = log(A / corpus_freq)

Each document gets keywords         Each CLUSTER gets keywords
```

This shift from document-level to cluster-level is the key innovation of c-TF-IDF, which we'll explore next.

---

## ⚠️ Common Pitfalls

### Pitfall 1: Not Removing Stop Words Before TF-IDF

```swift
// ❌ WRONG: Stop words will dominate
let terms = text.split(separator: " ")
let tfIdf = computeTfIdf(terms)
// Top terms: "the", "and", "to", "a" ← useless!

// ✅ CORRECT: Filter stop words first
let tokenizer = Tokenizer(configuration: .english)
let tokens = tokenizer.tokenize(text)
let tfIdf = computeTfIdf(tokens)
```

### Pitfall 2: Ignoring Document Length

```swift
// ⚠️ PROBLEMATIC: Long documents dominate
// Raw TF in a 10,000-word doc will be higher than in a 100-word doc

// Solution 1: Normalize by document length
let normalizedTf = tf / totalTerms

// Solution 2: Use log TF
let logTf = tf > 0 ? 1 + log(Float(tf)) : 0
```

### Pitfall 3: Very Small Corpora

```swift
// ⚠️ With only 10 documents, IDF is noisy
// A term appearing in 1 document has IDF = log(10/1) = 2.3
// A term appearing in 2 documents has IDF = log(10/2) = 1.6
// Not much dynamic range!

// For small corpora, consider:
// - Using smoothed IDF
// - Lowering minDocumentFrequency
// - Being less aggressive with maxDocumentFrequencyRatio
```

---

## Key Takeaways

1. **TF-IDF balances frequency and rarity**: High TF × High IDF = important term.

2. **IDF penalizes common terms**: Words appearing everywhere get low scores.

3. **TF-IDF creates sparse vectors**: Most entries are zero.

4. **Filtering is crucial**: Stop words and rare terms should be excluded.

5. **TF-IDF lacks semantics**: It's bag-of-words, not meaning.

6. **c-TF-IDF adapts TF-IDF for clusters**: Same intuition, different granularity.

---

## 💡 Key Insight

TF-IDF asks a simple but powerful question:

> **"Is this word special to this document, or does everyone use it?"**

A word that appears frequently HERE but rarely ELSEWHERE is distinctive—it tells you something specific about this document.

c-TF-IDF asks the same question at the cluster level:

> **"Is this word special to this topic, or does every topic use it?"**

The math is similar, but the unit of analysis changes from document to cluster.

---

## Next Up

Now let's see how c-TF-IDF adapts this classic algorithm for topic modeling.

**[→ 4.3 Class-Based TF-IDF](./03-Class-Based-TF-IDF.md)**

---

*Guide 4.2 of 4.4 • Chapter 4: Topic Representation*
