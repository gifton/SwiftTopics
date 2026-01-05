# SwiftTopics Learning Guide

> **From text to topics—understanding the mathematics and algorithms behind on-device topic discovery.**

Welcome to the SwiftTopics Learning Guide. This guide teaches the **theory and practice of topic modeling**—the process of automatically discovering themes in a collection of documents.

---

## What You'll Learn

By completing this guide, you'll understand:

- How text becomes vectors (embeddings) and why that enables clustering
- Why high-dimensional spaces are problematic and how to reduce them
- How HDBSCAN discovers clusters without knowing how many to find
- How c-TF-IDF extracts representative keywords from clusters
- How to evaluate whether your topics are meaningful
- How to build production systems with incremental updates

More importantly, you'll gain intuition for **why** each step exists and **when** to adjust parameters.

---

## Prerequisites

This guide assumes:

| Skill | Level | What You Should Know |
|-------|-------|----------------------|
| **Swift** | Intermediate | async/await, actors, protocols |
| **Linear Algebra** | Basic | Vectors, dot products, matrices |
| **Statistics** | Basic | Mean, variance, probability |
| **ML Concepts** | Familiarity | What embeddings are (conceptually) |

You **don't** need:
- Prior experience with topic modeling
- Deep understanding of neural networks
- Experience with Python/sklearn topic modeling libraries

---

## The Topic Modeling Problem

Imagine you have 10,000 journal entries. You want to answer:

- "What themes do I write about?"
- "Which entries are about similar things?"
- "Has my focus shifted over time?"

Reading 10,000 entries manually is impractical. **Topic modeling automates this discovery.**

```
Input:                           Output:
┌─────────────────────┐          ┌─────────────────────────────────┐
│ "Had a great run    │          │ Topic 0: fitness, running,      │
│  this morning..."   │ ──────►  │          exercise, health        │
│                     │          │                                  │
│ "The new Swift      │          │ Topic 1: swift, programming,    │
│  concurrency..."    │          │          code, async             │
│                     │          │                                  │
│ "Feeling anxious    │          │ Topic 2: anxiety, stress,       │
│  about tomorrow..." │          │          feeling, mental         │
│                     │          │                                  │
│ ... 9,997 more      │          │ ... + document assignments       │
└─────────────────────┘          └─────────────────────────────────┘
```

---

## The BERTopic Approach

SwiftTopics implements a pipeline inspired by [BERTopic](https://maartengr.github.io/BERTopic/), the state-of-the-art topic modeling library. Unlike classical approaches (LDA, NMF), BERTopic leverages **neural embeddings** to capture semantic meaning.

### Classical vs. Modern Topic Modeling

| Aspect | Classical (LDA) | Modern (BERTopic/SwiftTopics) |
|--------|-----------------|-------------------------------|
| **Text representation** | Bag of words | Dense embeddings |
| **Semantic understanding** | Word co-occurrence only | Full semantic meaning |
| **"Bank" in "river bank" vs "bank account"** | Same word | Different vectors |
| **Number of topics** | Must specify K | Discovered automatically |
| **Outlier handling** | Forced into topics | Marked as noise |

### Why Embeddings Matter

Classical topic models see text as bags of words—losing word order, context, and meaning. The sentences:

- "The bank rejected my loan application"
- "I sat on the river bank watching the sunset"

Would confuse a bag-of-words model (same word "bank"), but embedding models place them in completely different regions of vector space.

---

## The SwiftTopics Pipeline

SwiftTopics processes documents through four stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SwiftTopics Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐│
│  │  EMBEDDING   │───►│  REDUCTION   │───►│  CLUSTERING  │───►│ REPRE- ││
│  │              │    │              │    │              │    │ SENTA- ││
│  │ Text → Vec   │    │ 768D → 15D   │    │ HDBSCAN      │    │ TION   ││
│  │              │    │ PCA/UMAP     │    │              │    │c-TF-IDF││
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────┘│
│        │                   │                   │                  │    │
│        ▼                   ▼                   ▼                  ▼    │
│   "Great run"         [0.2, -0.1,         Cluster 0           "fitness"│
│   "Swift async"        0.8, ...]          Cluster 1           "swift"  │
│   "Feeling anxious"   (15 values)         Cluster 2           "anxiety"│
│                                           Noise: -1                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Embedding

Convert text to dense vectors using neural embedding models.

```swift
// Input: Raw text
let document = Document(content: "Had a great run this morning")

// Output: Dense vector (e.g., 768 dimensions)
let embedding: Embedding = [0.023, -0.156, 0.089, ..., 0.041]  // 768 values
```

SwiftTopics is **embedding-agnostic**—you provide embeddings from any source:
- Apple's NLContextualEmbedding (via SwiftTopicsApple)
- CoreML models
- Remote APIs (OpenAI, Cohere, etc.)

### Stage 2: Dimensionality Reduction

Compress 768-dimensional vectors to ~15 dimensions.

```
768D embedding space:
  - Sparse (points far apart)
  - Distance becomes meaningless
  - Clustering fails

15D reduced space:
  - Dense (points cluster naturally)
  - Distance is meaningful
  - Clustering works
```

Why? The **curse of dimensionality**—in high dimensions, all points become equidistant. Chapter 2 explains this in depth.

### Stage 3: Clustering (HDBSCAN)

Group similar documents into clusters. HDBSCAN is special because:

- **No predefined K**: Discovers natural groupings
- **Handles noise**: Outliers marked as `-1`, not forced into clusters
- **Variable density**: Finds clusters of different sizes/densities

```
Traditional K-Means:           HDBSCAN:
"I need exactly 5 topics"      "Find natural groupings"
Forces all docs into clusters  Allows outliers
Spherical clusters only        Arbitrary shapes
```

### Stage 4: Topic Representation (c-TF-IDF)

Extract keywords that characterize each cluster.

```
Cluster 0 documents:
  "Had a great run this morning"
  "5K personal best today"
  "New running shoes arrived"

c-TF-IDF identifies distinctive terms:
  → "running", "run", "5K", "shoes", "morning"

These become Topic 0's keywords.
```

---

## Chapter Overview

| Chapter | You'll Learn | Why It Matters |
|---------|-------------|----------------|
| [1. Embeddings Foundation](./01-Embeddings-Foundation/README.md) | How text becomes vectors | The semantic foundation |
| [2. Dimensionality Reduction](./02-Dimensionality-Reduction/README.md) | PCA and UMAP algorithms | Making clustering possible |
| [3. Density-Based Clustering](./03-Density-Based-Clustering/README.md) | HDBSCAN deep dive | Automatic topic discovery |
| [4. Topic Representation](./04-Topic-Representation/README.md) | c-TF-IDF keyword extraction | Human-readable topics |
| [5. Quality Evaluation](./05-Quality-Evaluation/README.md) | NPMI coherence scoring | Measuring topic quality |
| [6. Capstone](./06-Capstone/README.md) | Full pipeline & incremental updates | Production implementation |

---

## How to Use This Guide

### The Sequential Path

Each chapter builds on previous concepts:

```
Chapter 1 ──► Chapter 2 ──► Chapter 3 ──► Chapter 4 ──► Chapter 5 ──► Chapter 6
Embeddings   Reduction    Clustering    Keywords     Evaluation    Capstone
```

If you're new to topic modeling, follow this path.

### The Reference Path

If you have specific questions:

- **"Why do I get poor topics?"** → [Chapter 3: Clustering Parameters](./03-Density-Based-Clustering/README.md)
- **"What do coherence scores mean?"** → [Chapter 5: Quality Evaluation](./05-Quality-Evaluation/README.md)
- **"How do I handle streaming data?"** → [Chapter 6: Incremental Updates](./06-Capstone/README.md)

### Each Guide Follows This Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  THE CONCEPT                                                    │
│  What is this? Plain English with diagrams.                     │
├─────────────────────────────────────────────────────────────────┤
│  WHY IT MATTERS                                                 │
│  What problem does this solve?                                  │
├─────────────────────────────────────────────────────────────────┤
│  THE MATHEMATICS                                                │
│  Formulas with intuitive explanations.                          │
├─────────────────────────────────────────────────────────────────┤
│  THE TECHNIQUE                                                  │
│  Step-by-step implementation walkthrough.                       │
├─────────────────────────────────────────────────────────────────┤
│  IN SWIFTTOPICS                                                 │
│  📍 Links to actual source code.                                │
├─────────────────────────────────────────────────────────────────┤
│  KEY TAKEAWAYS                                                  │
│  What should stick from this guide.                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## SwiftTopics Source Locations

Throughout this guide, we reference actual implementation code:

| Topic | File Path |
|-------|-----------|
| TopicModel orchestrator | `Sources/SwiftTopics/Model/TopicModel.swift` |
| PCA reduction | `Sources/SwiftTopics/Reduction/PCA.swift` |
| UMAP reduction | `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift` |
| HDBSCAN clustering | `Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift` |
| Core distance | `Sources/SwiftTopics/Clustering/HDBSCAN/CoreDistance.swift` |
| Mutual reachability | `Sources/SwiftTopics/Clustering/HDBSCAN/MutualReachability.swift` |
| MST construction | `Sources/SwiftTopics/Clustering/HDBSCAN/MinimumSpanningTree.swift` |
| c-TF-IDF | `Sources/SwiftTopics/Representation/cTFIDF.swift` |
| NPMI scorer | `Sources/SwiftTopics/Evaluation/NPMIScorer.swift` |
| Incremental updater | `Sources/SwiftTopics/Incremental/IncrementalTopicUpdater.swift` |

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| `N` | Number of documents |
| `D` | Embedding dimension |
| `d` | Reduced dimension |
| `K` | Number of topics (discovered, not specified) |
| `k` | Neighborhood size (k-NN, minPts) |
| `📍 See:` | Link to SwiftTopics source code |
| `⚠️` | Common mistake or pitfall |
| `💡` | Key insight or tip |

---

## A Note on Mathematics

This guide includes mathematical formulas. Don't panic.

Each formula is accompanied by:
1. **Intuitive explanation** in plain English
2. **Visual diagram** where helpful
3. **Code implementation** showing how it works

You don't need to memorize formulas. The goal is **intuition**—understanding *why* the math works, not just *what* it computes.

---

## Let's Begin

Ready to understand how text becomes topics?

**[→ Chapter 1: Embeddings Foundation](./01-Embeddings-Foundation/README.md)**

---

*SwiftTopics Learning Guide • Version 0.1.0-beta.1 • January 2026*
